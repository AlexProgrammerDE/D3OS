//! Row-major matrix encodings, independent of an execution backend.
use crate::{Error, Result, element_count, quantized::Quantization};
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixFormat {
    F32,
    Bf16,
    Quantized(Quantization),
}

impl MatrixFormat {
    pub fn byte_len(self, shape: [usize; 2]) -> Result<usize> {
        let elements = element_count(&shape)?;
        match self {
            Self::Quantized(format) => {
                // Each row starts at a block boundary in the GPU projection ABI.
                format.byte_len(shape[1])?;
                format.byte_len(elements)
            }
            Self::F32 => elements.checked_mul(4).ok_or(Error::SizeOverflow),
            Self::Bf16 => elements.checked_mul(2).ok_or(Error::SizeOverflow),
        }
    }

    pub fn validate(self, shape: [usize; 2], bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.byte_len(shape)? {
            return Err(Error::ShapeMismatch);
        }
        match self {
            Self::Quantized(format) => format.validate(element_count(&shape)?, bytes),
            Self::F32 => {
                for word in bytes.chunks_exact(4) {
                    if !f32::from_le_bytes(word.try_into().unwrap()).is_finite() {
                        return Err(Error::NonFinite);
                    }
                }
                Ok(())
            }
            Self::Bf16 => {
                for word in bytes.chunks_exact(2) {
                    if u16::from_le_bytes(word.try_into().unwrap()) & 0x7f80 == 0x7f80 {
                        return Err(Error::NonFinite);
                    }
                }
                Ok(())
            }
        }
    }

    pub fn decode(self, shape: [usize; 2], bytes: &[u8]) -> Result<Vec<f32>> {
        self.validate(shape, bytes)?;
        if let Self::Quantized(format) = self {
            return format.decode(element_count(&shape)?, bytes);
        }
        let mut values = Vec::new();
        values.try_reserve_exact(element_count(&shape)?).map_err(|_| Error::AllocationFailed)?;
        match self {
            Self::F32 => values.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))),
            Self::Bf16 => values.extend(
                bytes
                    .chunks_exact(2)
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16)),
            ),
            Self::Quantized(_) => unreachable!(),
        }
        Ok(values)
    }
}
