//! Packed weight blocks used by the pinned Qwen checkpoints.
//! These are data formats. Arithmetic and storage remain owned by the backend.
use alloc::vec::Vec;

use crate::{Error, Result, element_count};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Quantization {
    Q8_0 = 8,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
}

impl Quantization {
    pub fn from_raw(value: u32) -> Result<Self> {
        match value {
            8 => Ok(Self::Q8_0),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            _ => Err(Error::InvalidArchive),
        }
    }

    pub const fn block_elements(self) -> usize {
        if matches!(self, Self::Q8_0) { 32 } else { 256 }
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    pub fn byte_len(self, elements: usize) -> Result<usize> {
        element_count(&[elements])?;
        if !elements.is_multiple_of(self.block_elements()) {
            return Err(Error::InvalidShape);
        }
        (elements / self.block_elements()).checked_mul(self.block_bytes()).ok_or(Error::SizeOverflow)
    }

    /// Check every scale before a backend receives an allocation or upload.
    pub fn validate(self, elements: usize, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.byte_len(elements)? {
            return Err(Error::ShapeMismatch);
        }
        let scale = match self {
            Self::Q8_0 | Self::Q4K | Self::Q5K => 0,
            Self::Q3K => 108,
            Self::Q6K => 208,
        };
        for block in bytes.chunks_exact(self.block_bytes()) {
            let finite = |offset| u16::from_le_bytes([block[offset], block[offset + 1]]) & 0x7c00 != 0x7c00;
            if !finite(scale) || (matches!(self, Self::Q4K | Self::Q5K) && !finite(2)) {
                return Err(Error::NonFinite);
            }
        }
        Ok(())
    }

    pub fn decode(self, elements: usize, bytes: &[u8]) -> Result<Vec<f32>> {
        self.validate(elements, bytes)?;
        let mut values = Vec::new();
        values.try_reserve_exact(elements).map_err(|_| Error::AllocationFailed)?;
        for block in bytes.chunks_exact(self.block_bytes()) {
            for index in 0..self.block_elements() {
                values.push(self.value(block, index));
            }
        }
        Ok(values)
    }

    fn value(self, b: &[u8], i: usize) -> f32 {
        let half = |offset| half_to_f32(u16::from_le_bytes([b[offset], b[offset + 1]]));
        match self {
            Self::Q8_0 => half(0) * (b[2 + i] as i8 as f32),
            Self::Q3K => {
                let group = i / 16;
                let low = (b[96 + group % 8] >> (4 * (group / 8))) & 15;
                let high = (b[104 + group % 4] >> (2 * (group / 4))) & 3;
                let scale = i32::from(low | (high << 4)) - 32;
                let quant = (b[32 + i / 128 * 32 + i % 32] >> (2 * (i / 32 % 4))) & 3;
                let high = if b[i % 32] & (1 << (i / 32)) == 0 { 4 } else { 0 };
                (half(108) * scale as f32) * (i32::from(quant) - high) as f32
            }
            Self::Q4K | Self::Q5K => {
                let group = i / 32;
                let (scale, minimum) = if group < 4 {
                    (b[4 + group] & 63, b[8 + group] & 63)
                } else {
                    ((b[8 + group] & 15) | ((b[group] >> 6) << 4), (b[8 + group] >> 4) | ((b[4 + group] >> 6) << 4))
                };
                let offset = if self == Self::Q5K { 48 } else { 16 };
                let mut quant = (b[offset + i / 64 * 32 + i % 32] >> (4 * (group % 2))) & 15;
                if self == Self::Q5K {
                    quant |= ((b[16 + i % 32] >> group) & 1) << 4;
                }
                (half(0) * f32::from(scale)) * f32::from(quant) - half(2) * f32::from(minimum)
            }
            Self::Q6K => {
                let segment = i % 128 / 32;
                let low = (b[i / 128 * 64 + (segment % 2) * 32 + i % 32] >> (4 * (segment / 2))) & 15;
                let high = (b[128 + i / 128 * 32 + i % 32] >> (2 * segment)) & 3;
                let scale = b[192 + i / 16] as i8;
                (half(208) * f32::from(scale)) * (i32::from(low | (high << 4)) - 32) as f32
            }
        }
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 31;
    let fraction = u32::from(bits & 1023);
    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        let shift = fraction.leading_zeros() - 21;
        f32::from_bits(sign | ((113 - shift) << 23) | ((fraction << shift) & 1023) << 13)
    } else {
        f32::from_bits(sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13))
    }
}
