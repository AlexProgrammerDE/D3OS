//! Row-major matrix encodings, independent of an execution backend.
use crate::{Error, Result, element_count, quantized::Quantization};
use alloc::borrow::Cow;
use alloc::vec::Vec;

/// Physical placement of complete encoding blocks, independent of their format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixLayout {
    RowMajor,
    /// Q8_0 blocks in [output / 32, input / 32, output % 32] order.
    /// All 34 bytes of each block are retained, including the original scale.
    Q8Columns32,
}

impl MatrixLayout {
    pub fn byte_len(self, shape: [usize; 2], format: MatrixFormat) -> Result<usize> {
        if self == Self::Q8Columns32 && (format != MatrixFormat::Quantized(Quantization::Q8_0) || !shape[0].is_multiple_of(32)) {
            return Err(Error::InvalidShape);
        }
        format.byte_len(shape)
    }

    /// Copy validated file bytes into the requested physical storage order.
    pub fn pack(self, shape: [usize; 2], format: MatrixFormat, bytes: &[u8]) -> Result<Vec<u8>> {
        let length = self.byte_len(shape, format)?;
        format.validate(shape, bytes)?;
        let mut output = Vec::new();
        output.try_reserve_exact(length).map_err(|_| Error::AllocationFailed)?;
        output.resize(length, 0);
        if self == Self::RowMajor {
            output.copy_from_slice(bytes);
        } else {
            self.copy_blocks(shape, bytes, &mut output, false);
        }
        Ok(output)
    }

    /// Borrow ordinary storage, or restore file order for a portable consumer.
    pub fn row_major_bytes(self, shape: [usize; 2], format: MatrixFormat, bytes: &[u8]) -> Result<Cow<'_, [u8]>> {
        let length = self.byte_len(shape, format)?;
        if bytes.len() != length {
            return Err(Error::ShapeMismatch);
        }
        if self == Self::RowMajor {
            return Ok(Cow::Borrowed(bytes));
        }
        let mut output = Vec::new();
        output.try_reserve_exact(length).map_err(|_| Error::AllocationFailed)?;
        output.resize(length, 0);
        self.copy_blocks(shape, bytes, &mut output, true);
        Ok(Cow::Owned(output))
    }

    fn copy_blocks(self, shape: [usize; 2], input: &[u8], output: &mut [u8], inverse: bool) {
        let blocks = shape[1] / 32;
        for column_tile in 0..shape[0] / 32 {
            for block in 0..blocks {
                for lane in 0..32 {
                    let row_major = ((column_tile * 32 + lane) * blocks + block) * 34;
                    let tiled = ((column_tile * blocks + block) * 32 + lane) * 34;
                    let (source, destination) = if inverse { (tiled, row_major) } else { (row_major, tiled) };
                    output[destination..destination + 34].copy_from_slice(&input[source..source + 34]);
                }
            }
        }
    }
}

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
