//! Exact integer values and checked scalar metadata over packed GPU storage.
use crate::{NativeTensor, layout::Layout, storage::format};
use alloc::sync::Arc;
use burn_backend::{DType, Scalar};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise, integer_arithmetic as abi, words::WordOp};

fn signed_scalar(value: Scalar) -> Result<i64> {
    match value {
        Scalar::Int(value) => Ok(value),
        Scalar::UInt(value) => i64::try_from(value).map_err(|_| Error::InvalidParameter),
        Scalar::Bool(value) => Ok(i64::from(value)),
        // The upper bound is exclusive: i64::MAX rounds to 2^63 in F64.
        Scalar::Float(value) if value.is_finite() && value >= i64::MIN as f64 && value < -(i64::MIN as f64) => Ok(value as i64),
        _ => Err(Error::InvalidParameter),
    }
}
fn unsigned_scalar(value: Scalar) -> Result<u64> {
    match value {
        Scalar::UInt(value) => Ok(value),
        Scalar::Int(value) => u64::try_from(value).map_err(|_| Error::InvalidParameter),
        Scalar::Bool(value) => Ok(u64::from(value)),
        // Fractional values in (-1, 0) truncate to zero, matching ToElement.
        Scalar::Float(value) if value.is_finite() && value > -1.0 && value < u64::MAX as f64 => Ok(value as u64),
        _ => Err(Error::InvalidParameter),
    }
}

pub(crate) fn integer_scalar_bits(dtype: DType, scalar: Scalar) -> Result<u64> {
    Ok(match dtype {
        DType::I64 => signed_scalar(scalar)? as u64,
        DType::I32 => u64::from(i32::try_from(signed_scalar(scalar)?).map_err(|_| Error::InvalidParameter)? as u32),
        DType::I16 => u64::from(i16::try_from(signed_scalar(scalar)?).map_err(|_| Error::InvalidParameter)? as u16),
        DType::I8 => u64::from(i8::try_from(signed_scalar(scalar)?).map_err(|_| Error::InvalidParameter)? as u8),
        DType::U64 => unsigned_scalar(scalar)?,
        DType::U32 => u64::from(u32::try_from(unsigned_scalar(scalar)?).map_err(|_| Error::InvalidParameter)?),
        DType::U16 => u64::from(u16::try_from(unsigned_scalar(scalar)?).map_err(|_| Error::InvalidParameter)?),
        DType::U8 => u64::from(u8::try_from(unsigned_scalar(scalar)?).map_err(|_| Error::InvalidParameter)?),
        _ => return Err(Error::InvalidParameter),
    })
}

impl NativeTensor {
    pub fn integer_binary(&self, rhs: &Self, operation: WordOp) -> Result<Self> {
        let operation = abi::Operation::from_word(operation).map_err(|_| Error::InvalidParameter)?;
        if !operation.binary() {
            return Err(Error::InvalidParameter);
        }
        if self.dtype == DType::I32 && rhs.dtype == DType::I32 {
            return self.word_binary(rhs, WordOp::from_raw(operation as u32).map_err(|_| Error::InvalidParameter)?);
        }
        self.typed_integer(rhs, operation, None)
    }

    pub fn integer_scalar(&self, operation: WordOp, scalar: Scalar) -> Result<Self> {
        let operation = abi::Operation::from_word(operation).map_err(|_| Error::InvalidParameter)?;
        if !operation.binary() {
            return Err(Error::InvalidParameter);
        }
        let bits = integer_scalar_bits(self.dtype, scalar)?;
        if self.dtype == DType::I32 {
            return self.word_scalar(WordOp::from_raw(operation as u32).map_err(|_| Error::InvalidParameter)?, bits as u32);
        }
        self.typed_integer(self, operation, Some(bits))
    }

    pub fn integer_unary(&self, operation: WordOp) -> Result<Self> {
        let operation = abi::Operation::from_word(operation).map_err(|_| Error::InvalidParameter)?;
        if !operation.unary() {
            return Err(Error::InvalidParameter);
        }
        if self.dtype == DType::I32 {
            return self.word_unary(WordOp::from_raw(operation as u32).map_err(|_| Error::InvalidParameter)?);
        }
        self.typed_integer(self, operation, None)
    }

    /// Burn probes the I32 exponent before normalizing other scalar powers to
    /// the base format. Shortcuts therefore precede that second conversion.
    pub fn integer_powi_scalar(&self, scalar: Scalar) -> Result<Self> {
        let exponent = integer_scalar_bits(DType::I32, scalar)? as u32 as i32;
        // Validate the base dtype even when the shortcut performs no work.
        integer_scalar_bits(self.dtype, Scalar::Int(1))?;
        match exponent {
            0 => Self::full_scalar(self.layout.shape().clone(), self.dtype, Scalar::Int(1)),
            1 => Ok(self.clone()),
            2 => self.integer_binary(self, WordOp::Multiply),
            _ => self.integer_scalar(WordOp::Power, scalar),
        }
    }

    fn typed_integer(&self, rhs: &Self, operation: abi::Operation, scalar: Option<u64>) -> Result<Self> {
        let lhs_format = format(self.dtype)?;
        let rhs_format = if operation.unary() || scalar.is_some() {
            lhs_format
        } else {
            format(rhs.dtype)?
        };
        if !operation.supports(lhs_format, rhs_format) {
            return Err(Error::InvalidParameter);
        }
        let shape = if operation.unary() || scalar.is_some() {
            self.layout.shape().clone()
        } else {
            crate::layout::broadcast(self.layout.shape(), rhs.layout.shape())?
        };
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        let left = self.layout.expand(shape.clone())?;
        let right = if operation.unary() || scalar.is_some() {
            left.clone()
        } else {
            rhs.layout.expand(shape)?
        };
        let mut p = left.elementwise_parameters(&right, Elementwise::Add, 0, false)?;
        p[2] = operation as usize | ((lhs_format as usize) << 8) | ((rhs_format as usize) << 16);
        if operation.unary() || scalar.is_some() {
            p[5] = 0;
            p[7] = 0;
            p[24..32].fill(0);
        }
        if let Some(bits) = scalar {
            p[2] |= abi::SCALAR_RHS as usize;
            p[3] = bits as u32 as usize;
            p[5] = (bits >> 32) as usize;
        }
        abi::validate(&p.map(|v| v as u32)).map_err(|_| Error::InvalidParameter)?;
        let buffer = Nvidia.produce_words(
            lhs_format.width().words(layout.len() as u32) as usize,
            Kernel::IntegerArithmetic,
            &[(&self.buffer, 0), (&rhs.buffer, 0)],
            &p,
        )?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype: self.dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_and_unsigned_scalars_preserve_high_words_and_canonical_narrow_fields() {
        for (dtype, input, expected) in [
            (DType::I64, Scalar::Int(i64::MIN), 1u64 << 63),
            (DType::U64, Scalar::UInt(u64::MAX), u64::MAX),
            (DType::I32, Scalar::Int(-1), u32::MAX as u64),
            (DType::I16, Scalar::Int(-1), u16::MAX as u64),
            (DType::I8, Scalar::Int(-1), u8::MAX as u64),
            (DType::U32, Scalar::UInt(u32::MAX as u64), u32::MAX as u64),
            (DType::U16, Scalar::UInt(u16::MAX as u64), u16::MAX as u64),
            (DType::U8, Scalar::UInt(u8::MAX as u64), u8::MAX as u64),
        ] {
            assert_eq!(integer_scalar_bits(dtype, input), Ok(expected));
        }
    }

    #[test]
    fn scalar_range_checks_happen_before_representation_conversion() {
        for (dtype, value) in [
            (DType::I64, Scalar::UInt(1u64 << 63)),
            (DType::U64, Scalar::Int(-1)),
            (DType::I8, Scalar::Int(-129)),
            (DType::I8, Scalar::Int(128)),
            (DType::I16, Scalar::Int(32768)),
            (DType::U8, Scalar::UInt(256)),
            (DType::U16, Scalar::UInt(65536)),
            (DType::U32, Scalar::UInt(1u64 << 32)),
            (DType::I32, Scalar::UInt(1u64 << 31)),
            (DType::F32, Scalar::Int(1)),
        ] {
            assert_eq!(integer_scalar_bits(dtype, value), Err(Error::InvalidParameter));
        }
    }

    #[test]
    fn fractional_scalars_truncate_then_check_the_destination_range() {
        for (dtype, value, expected) in [
            (DType::I8, -128.75, 128),
            (DType::I8, 127.75, 127),
            (DType::U8, -0.75, 0),
            (DType::U8, 255.75, 255),
            (DType::I32, 2147483647.75, i32::MAX as u64),
        ] {
            assert_eq!(integer_scalar_bits(dtype, Scalar::Float(value)), Ok(expected));
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 18446744073709551616.0] {
            assert_eq!(integer_scalar_bits(DType::U64, Scalar::Float(value)), Err(Error::InvalidParameter));
        }
        assert_eq!(integer_scalar_bits(DType::I64, Scalar::Float(-9223372036854775808.0)), Ok(1u64 << 63));
        assert_eq!(
            integer_scalar_bits(DType::I64, Scalar::Float(9223372036854775808.0)),
            Err(Error::InvalidParameter)
        );
        assert_eq!(integer_scalar_bits(DType::U64, Scalar::Float(-1.0)), Err(Error::InvalidParameter));
    }
}
