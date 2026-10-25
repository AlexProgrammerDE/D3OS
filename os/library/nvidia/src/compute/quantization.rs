//! Packed Burn quantization with GPU-resident view metadata and scales.
use crate::{Error, Result};

pub const PARAMETERS: usize = 8;
pub const METADATA_WORDS: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Operation {
    Quantize = 0,
    Dequantize = 1,
    Copy = 2,
}

impl Operation {
    pub fn from_raw(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Quantize),
            1 => Ok(Self::Dequantize),
            2 => Ok(Self::Copy),
            _ => Err(Error::InvalidField),
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Quantize => "burn_quantize",
            Self::Dequantize => "burn_dequantize",
            Self::Copy => "burn_quantized_copy",
        }
    }
}

/// Logical output count, source storage elements, scale storage words, field
/// bits, operation, symmetric-range flag, source offset, and reserved zero.
/// View metadata is untrusted: kernels mask all derived data/scale addresses.
pub fn validate(parameters: &[u32]) -> Result<([u64; 4], u32)> {
    let &[count, source, scales, bits, operation, symmetric, offset, reserved] = parameters else {
        return Err(Error::InvalidField);
    };
    let operation = Operation::from_raw(operation)?;
    if count == 0
        || count > i32::MAX as u32
        || source == 0
        || source > i32::MAX as u32
        || scales == 0
        || scales > i32::MAX as u32
        || !matches!(bits, 2 | 4 | 8)
        || symmetric > 1
        || offset >= source
        || reserved != 0
    {
        return Err(Error::InvalidField);
    }
    let packed = |elements: u32| u64::from(elements).div_ceil(u64::from(32 / bits));
    let input = if operation == Operation::Quantize {
        u64::from(source)
    } else {
        packed(source)
    };
    let output = if operation == Operation::Dequantize {
        u64::from(count)
    } else {
        packed(count)
    };
    Ok(([input, u64::from(scales), METADATA_WORDS, output], output as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_words_and_strided_source_extents() {
        for bits in [2, 4, 8] {
            let p = [17, 41, 3, bits, 0, 1, 7, 0];
            let packed = 17u64.div_ceil(32 / u64::from(bits));
            assert_eq!(validate(&p), Ok(([41, 3, 32, packed], packed as u32)));
            let mut p = p;
            p[4] = 1;
            assert_eq!(validate(&p), Ok(([41u64.div_ceil(32 / u64::from(bits)), 3, 32, 17], 17)));
        }
    }

    #[test]
    fn reject_invalid_launches_before_accessing_buffers() {
        let valid = [17, 41, 3, 8, 0, 1, 7, 0];
        for (index, value) in [(0, 0), (0, u32::MAX), (1, 0), (1, u32::MAX), (2, 0), (3, 3), (4, 3), (5, 2), (6, 41), (7, 1)] {
            let mut p = valid;
            p[index] = value;
            assert_eq!(validate(&p), Err(Error::InvalidField));
        }
        assert_eq!(validate(&valid[..7]), Err(Error::InvalidField));
    }
}
