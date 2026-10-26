//! Basic floating arithmetic retains destination precision and scalar metadata.
use crate::{NativeTensor, layout::Layout, storage::format};
use alloc::sync::Arc;
use burn_backend::{DType, Scalar, bf16, f16};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise, float_arithmetic as abi};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PowerShortcut {
    Ones,
    Identity,
    Square,
    Reciprocal,
    SquareReciprocal,
}

/// Inspect only exact shortcuts, without converting arbitrary exponents to I64.
pub(crate) fn power_scalar_shortcut(scalar: Scalar) -> Option<PowerShortcut> {
    match scalar {
        Scalar::Float(0.0) | Scalar::Int(0) | Scalar::UInt(0) | Scalar::Bool(false) => Some(PowerShortcut::Ones),
        Scalar::Float(1.0) | Scalar::Int(1) | Scalar::UInt(1) | Scalar::Bool(true) => Some(PowerShortcut::Identity),
        Scalar::Float(2.0) | Scalar::Int(2) | Scalar::UInt(2) => Some(PowerShortcut::Square),
        Scalar::Float(-1.0) | Scalar::Int(-1) => Some(PowerShortcut::Reciprocal),
        Scalar::Float(-2.0) | Scalar::Int(-2) => Some(PowerShortcut::SquareReciprocal),
        _ => None,
    }
}

/// Accept integer variants and finite integral floats across their full ranges.
/// Fractional and nonfinite floats are invalid integer exponents, not truncations.
pub(crate) fn validate_powi_scalar(scalar: Scalar) -> Result<()> {
    if let Scalar::Float(value) = scalar
        && (!value.is_finite() || value % 1.0 != 0.0)
    {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

/// Match the pinned Linux InputScalar conversion schedule through F64 and F32.
/// Half scalars round to their storage format before the arithmetic operation.
pub(crate) fn float_scalar_bits(dtype: DType, scalar: Scalar) -> Result<u64> {
    let value = match scalar {
        Scalar::Float(value) => value,
        Scalar::Int(value) => value as f64,
        Scalar::UInt(value) => value as f64,
        Scalar::Bool(value) => f64::from(u8::from(value)),
    };
    Ok(match dtype {
        DType::F64 => value.to_bits(),
        DType::F32 | DType::Flex32 => u64::from((value as f32).to_bits()),
        DType::F16 => u64::from(f16::from_f32(value as f32).to_bits()),
        DType::BF16 => u64::from(bf16::from_f32(value as f32).to_bits()),
        _ => return Err(Error::InvalidParameter),
    })
}

impl NativeTensor {
    pub fn float_binary(&self, rhs: &Self, operation: Elementwise) -> Result<Self> {
        let operation = abi::Operation::from_elementwise(operation).map_err(|_| Error::InvalidParameter)?;
        if !operation.binary() || self.dtype != rhs.dtype {
            return Err(Error::InvalidParameter);
        }
        if self.dtype == DType::F32 {
            return self.binary(rhs, Elementwise::from_raw(operation as u32).map_err(|_| Error::InvalidParameter)?);
        }
        self.typed_float(rhs, operation, None)
    }

    pub fn float_scalar(&self, operation: Elementwise, scalar: Scalar) -> Result<Self> {
        let operation = abi::Operation::from_elementwise(operation).map_err(|_| Error::InvalidParameter)?;
        if !operation.binary() {
            return Err(Error::InvalidParameter);
        }
        let bits = float_scalar_bits(self.dtype, scalar)?;
        if self.dtype == DType::F32 {
            return self.scalar(
                Elementwise::from_raw(operation as u32).map_err(|_| Error::InvalidParameter)?,
                f32::from_bits(bits as u32),
            );
        }
        self.typed_float(self, operation, Some(bits))
    }

    pub fn float_unary(&self, operation: Elementwise) -> Result<Self> {
        let operation = abi::Operation::from_elementwise(operation).map_err(|_| Error::InvalidParameter)?;
        if !operation.unary() {
            return Err(Error::InvalidParameter);
        }
        if self.dtype == DType::F32 {
            return self.unary(Elementwise::from_raw(operation as u32).map_err(|_| Error::InvalidParameter)?);
        }
        self.typed_float(self, operation, None)
    }

    fn typed_float(&self, rhs: &Self, operation: abi::Operation, scalar: Option<u64>) -> Result<Self> {
        let source_format = format(self.dtype)?;
        if !abi::FORMATS.contains(&source_format) || self.dtype != rhs.dtype {
            return Err(Error::InvalidParameter);
        }
        let unused_rhs = operation.unary() || scalar.is_some();
        let shape = if unused_rhs {
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
        let right = if unused_rhs { left.clone() } else { rhs.layout.expand(shape)? };
        let mut parameters = left.elementwise_parameters(&right, Elementwise::Add, 0, false)?;
        parameters[2] = operation as usize | ((source_format as usize) << 8);
        if unused_rhs {
            parameters[5] = 0;
            parameters[7] = 0;
            parameters[24..32].fill(0);
        }
        if let Some(bits) = scalar {
            parameters[2] |= abi::SCALAR_RHS as usize;
            parameters[3] = bits as u32 as usize;
            parameters[5] = (bits >> 32) as usize;
        }
        abi::validate(&parameters.map(|v| v as u32)).map_err(|_| Error::InvalidParameter)?;
        let buffer = Nvidia.produce_words(
            source_format.width().words(layout.len() as u32) as usize,
            Kernel::FloatArithmetic,
            &[(&self.buffer, 0), (&rhs.buffer, 0)],
            &parameters,
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
    fn power_shortcuts_match_exact_exponents_in_every_scalar_variant() {
        for (exponent, shortcut) in [
            (0, PowerShortcut::Ones),
            (1, PowerShortcut::Identity),
            (2, PowerShortcut::Square),
            (-1, PowerShortcut::Reciprocal),
            (-2, PowerShortcut::SquareReciprocal),
        ] {
            assert_eq!(power_scalar_shortcut(Scalar::Int(exponent)), Some(shortcut));
            assert_eq!(power_scalar_shortcut(Scalar::Float(exponent as f64)), Some(shortcut));
            if exponent >= 0 {
                assert_eq!(power_scalar_shortcut(Scalar::UInt(exponent as u64)), Some(shortcut));
            }
        }
        assert_eq!(power_scalar_shortcut(Scalar::Float(-0.0)), Some(PowerShortcut::Ones));
        assert_eq!(power_scalar_shortcut(Scalar::Bool(false)), Some(PowerShortcut::Ones));
        assert_eq!(power_scalar_shortcut(Scalar::Bool(true)), Some(PowerShortcut::Identity));
    }

    #[test]
    fn arbitrary_power_exponents_never_narrow_into_shortcuts() {
        for exponent in [
            Scalar::Int(i64::MIN),
            Scalar::Int(i64::MAX),
            Scalar::UInt(u64::MAX),
            Scalar::Float(f64::MAX),
            Scalar::Float(f64::MIN),
            Scalar::Float(f64::NAN),
            Scalar::Float(f64::INFINITY),
            Scalar::Float(f64::NEG_INFINITY),
            Scalar::Float(f64::from_bits(1)),
            Scalar::Float(f64::from_bits(1.0f64.to_bits() - 1)),
            Scalar::Float(f64::from_bits(1.0f64.to_bits() + 1)),
            Scalar::Float(f64::from_bits(2.0f64.to_bits() - 1)),
            Scalar::Float(f64::from_bits(2.0f64.to_bits() + 1)),
            Scalar::Float(f64::from_bits((-1.0f64).to_bits() + 1)),
            Scalar::Float(f64::from_bits((-2.0f64).to_bits() + 1)),
        ] {
            assert_eq!(power_scalar_shortcut(exponent), None);
        }
    }

    #[test]
    fn integer_power_validation_accepts_full_ranges_without_truncating_weak_floats() {
        for exponent in [
            Scalar::Int(i64::MIN),
            Scalar::Int(i64::MAX),
            Scalar::UInt(u64::MAX),
            Scalar::Bool(false),
            Scalar::Bool(true),
            Scalar::Float(-0.0),
            Scalar::Float(-2.0),
            Scalar::Float(3.0),
            Scalar::Float(i64::MIN as f64),
            Scalar::Float(u64::MAX as f64),
            Scalar::Float(f64::MIN),
            Scalar::Float(f64::MAX),
        ] {
            assert_eq!(validate_powi_scalar(exponent), Ok(()));
        }
        for exponent in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5, -0.5, 2.5, -2.5, f64::from_bits(1)] {
            assert_eq!(validate_powi_scalar(Scalar::Float(exponent)), Err(Error::InvalidParameter));
        }
    }

    #[test]
    fn generic_power_scalars_keep_the_original_precision_and_variant_conversion() {
        let exponent = f64::from_bits(2.0f64.to_bits() + 1);
        assert_eq!(power_scalar_shortcut(Scalar::Float(exponent)), None);
        assert_eq!(float_scalar_bits(DType::F64, Scalar::Float(exponent)), Ok(exponent.to_bits()));
        assert_eq!(float_scalar_bits(DType::F32, Scalar::Float(exponent)), Ok(u64::from(2.0f32.to_bits())));
        // All variants use the canonical conversion, including Bool and UInt above I64.
        for scalar in [Scalar::Float(3.0), Scalar::Int(3), Scalar::UInt(3)] {
            assert_eq!(float_scalar_bits(DType::F64, scalar), Ok(3.0f64.to_bits()));
        }
        assert_eq!(float_scalar_bits(DType::F64, Scalar::Bool(true)), Ok(1.0f64.to_bits()));
        assert_eq!(float_scalar_bits(DType::F64, Scalar::UInt(u64::MAX)), Ok((u64::MAX as f64).to_bits()));
        // (-1)^(2^24 + 1) needs the odd F64 exponent; an F32 intermediate makes it even.
        let odd_exponent = (1i64 << 24) + 1;
        for scalar in [
            Scalar::Float(odd_exponent as f64),
            Scalar::Int(odd_exponent),
            Scalar::new(odd_exponent, &DType::F64),
        ] {
            assert_eq!(power_scalar_shortcut(scalar), None);
            assert_eq!(float_scalar_bits(DType::F64, scalar), Ok(0x4170_0000_1000_0000));
            assert_eq!(float_scalar_bits(DType::F32, scalar), Ok(0x4b80_0000));
        }
    }

    #[test]
    fn f64_scalar_bits_preserve_precision_range_and_subnormals() {
        for bits in [0x3ff0_0000_0000_1000, 0x54b2_49ad_2594_c37d, 1, 0x8000_0000_0000_0000, 0x7ff0_0000_0000_0000] {
            assert_eq!(float_scalar_bits(DType::F64, Scalar::Float(f64::from_bits(bits))), Ok(bits));
        }
        assert_eq!(float_scalar_bits(DType::F64, Scalar::UInt(u64::MAX)), Ok(0x43f0_0000_0000_0000));
        assert_eq!(float_scalar_bits(DType::F64, Scalar::Int((1i64 << 53) + 1)), Ok(0x4340_0000_0000_0000));
    }

    #[test]
    fn scalars_follow_the_staged_linux_rounding_schedule() {
        // Direct I64 -> F32 rounds upward; the pinned F64 intermediate rounds down.
        let midpoint_neighbor = Scalar::Int((1i64 << 62) + (1i64 << 38) + 1);
        for dtype in [DType::F32, DType::Flex32] {
            assert_eq!(float_scalar_bits(dtype, midpoint_neighbor), Ok(0x5e80_0000));
        }
        let half_midpoint_neighbor = f64::from_bits(1.00048828125f64.to_bits() + 1);
        let bfloat_midpoint_neighbor = f64::from_bits(1.00390625f64.to_bits() + 1);
        assert_eq!(float_scalar_bits(DType::F16, Scalar::Float(half_midpoint_neighbor)), Ok(0x3c00));
        assert_eq!(float_scalar_bits(DType::BF16, Scalar::Float(bfloat_midpoint_neighbor)), Ok(0x3f80));
        for dtype in [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16] {
            assert_eq!(float_scalar_bits(dtype, Scalar::Bool(false)), Ok(0));
        }
    }

    #[test]
    fn nonfinite_scalars_remain_valid_and_other_dtype_families_are_rejected() {
        for dtype in [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16] {
            assert!(float_scalar_bits(dtype, Scalar::Float(f64::NAN)).is_ok());
            assert!(float_scalar_bits(dtype, Scalar::Float(f64::INFINITY)).is_ok());
            assert!(float_scalar_bits(dtype, Scalar::Float(f64::NEG_INFINITY)).is_ok());
        }
        for dtype in [DType::I32, DType::U64] {
            assert_eq!(float_scalar_bits(dtype, Scalar::Int(1)), Err(Error::InvalidParameter));
        }
    }
}
