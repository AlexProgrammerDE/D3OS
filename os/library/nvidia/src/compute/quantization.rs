//! Packed Burn quantization with GPU-resident view metadata and scales.
use crate::{Error, Result};

pub const PARAMETERS: usize = 8;
pub const METADATA_WORDS: u64 = 32;
pub const GLOBAL_SCALE: u32 = 1 << 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ScaleFormat {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    UE8M0 = 3,
    UE4M3 = 4,
}

impl ScaleFormat {
    pub fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::BF16),
            3 => Ok(Self::UE8M0),
            4 => Ok(Self::UE4M3),
            _ => Err(Error::InvalidField),
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::UE8M0 | Self::UE4M3 => 8,
        }
    }

    pub fn from_flags(flags: u32) -> Result<Self> {
        if flags & !(GLOBAL_SCALE | 7) != 0 {
            return Err(Error::InvalidField);
        }
        Self::from_raw(flags & 7)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ScaleOperation {
    Encode = 0,
    Pack = 1,
    Unpack = 2,
}

impl ScaleOperation {
    pub fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            0 => Ok(Self::Encode),
            1 => Ok(Self::Pack),
            2 => Ok(Self::Unpack),
            _ => Err(Error::InvalidField),
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Encode => "burn_scale_encode",
            Self::Pack => "burn_scale_pack",
            Self::Unpack => "burn_scale_unpack",
        }
    }
}

/// Scales are held as one raw code per word on the GPU. Pack/unpack only
/// change the serialized representation and preserve every encoded bit.
pub fn validate_scales(p: &[u32]) -> Result<([u64; 2], u32)> {
    let &[count, format, operation] = p else { return Err(Error::InvalidField) };
    let format = ScaleFormat::from_raw(format)?;
    let operation = ScaleOperation::from_raw(operation)?;
    if count == 0 || count > i32::MAX as u32 {
        return Err(Error::InvalidField);
    }
    let packed = u64::from(count).div_ceil(u64::from(32 / format.bits()));
    let input = if operation == ScaleOperation::Unpack { packed } else { u64::from(count) };
    let output = if operation == ScaleOperation::Pack { packed } else { u64::from(count) };
    Ok(([input, output], output as u32))
}

/// Numeric interpretation of each packed field. Independent of storage width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Codec {
    IntegerFull = 0,
    IntegerSymmetric = 1,
    E4M3 = 2,
    E5M2 = 3,
    E2M1 = 4,
}

impl Codec {
    pub fn from_raw(value: u32, bits: u32) -> Result<Self> {
        match (value, bits) {
            (0, 2 | 4 | 8) => Ok(Self::IntegerFull),
            (1, 2 | 4 | 8) => Ok(Self::IntegerSymmetric),
            (2, 8) => Ok(Self::E4M3),
            (3, 8) => Ok(Self::E5M2),
            (4, 4) => Ok(Self::E2M1),
            _ => Err(Error::InvalidField),
        }
    }
}

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
/// bits, operation, codec, source offset, and scale format/global flags.
/// View metadata is untrusted: kernels mask all derived data/scale addresses.
pub fn validate(parameters: &[u32]) -> Result<([u64; 5], u32)> {
    let &[count, source, scales, bits, operation, codec, offset, flags] = parameters else {
        return Err(Error::InvalidField);
    };
    let operation = Operation::from_raw(operation)?;
    Codec::from_raw(codec, bits)?;
    ScaleFormat::from_flags(flags)?;
    if count == 0 || count > i32::MAX as u32 || source == 0 || source > i32::MAX as u32 || scales == 0 || scales > i32::MAX as u32 || offset >= source {
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
    Ok(([input, u64::from(scales), METADATA_WORDS, 1, output], output as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_words_and_strided_source_extents() {
        for bits in [2, 4, 8] {
            let p = [17, 41, 3, bits, 0, 1, 7, 0];
            let packed = 17u64.div_ceil(32 / u64::from(bits));
            assert_eq!(validate(&p), Ok(([41, 3, 32, 1, packed], packed as u32)));
            let mut p = p;
            p[4] = 1;
            assert_eq!(validate(&p), Ok(([41u64.div_ceil(32 / u64::from(bits)), 3, 32, 1, 17], 17)));
        }
    }

    #[test]
    fn reject_invalid_launches_before_accessing_buffers() {
        let valid = [17, 41, 3, 8, 0, 1, 7, 0];
        for (index, value) in [
            (0, 0),
            (0, u32::MAX),
            (1, 0),
            (1, u32::MAX),
            (2, 0),
            (3, 3),
            (4, 3),
            (5, 5),
            (6, 41),
            (7, 5),
            (7, 512),
        ] {
            let mut p = valid;
            p[index] = value;
            assert_eq!(validate(&p), Err(Error::InvalidField));
        }
        assert_eq!(validate(&valid[..7]), Err(Error::InvalidField));
    }

    #[test]
    fn codecs_require_their_exact_field_width() {
        for codec in 0..=5 {
            for bits in [2, 4, 8] {
                let expected = codec < 2 || ((codec == 2 || codec == 3) && bits == 8) || (codec == 4 && bits == 4);
                assert_eq!(validate(&[17, 41, 1, bits, 0, codec, 0, 0]).is_ok(), expected);
            }
        }
    }

    #[test]
    fn scales_validate_partial_words_and_global_flags() {
        for (format, factor) in [(0, 1u64), (1, 2), (2, 2), (3, 4), (4, 4)] {
            let packed = 7u64.div_ceil(factor);
            assert_eq!(validate_scales(&[7, format, 0]), Ok(([7, 7], 7)));
            assert_eq!(validate_scales(&[7, format, 1]), Ok(([7, packed], packed as u32)));
            assert_eq!(validate_scales(&[7, format, 2]), Ok(([packed, 7], 7)));
            assert!(ScaleFormat::from_flags(format | GLOBAL_SCALE).is_ok());
        }
        for p in [[0, 0, 0], [u32::MAX, 0, 0], [1, 5, 0], [1, 0, 3]] {
            assert!(validate_scales(&p).is_err());
        }
        assert!(validate_scales(&[1, 0]).is_err());
        assert!(ScaleFormat::from_flags(8).is_err());
        assert!(ScaleFormat::from_flags(GLOBAL_SCALE | 5).is_err());
    }
}
