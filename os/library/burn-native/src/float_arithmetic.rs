//! Basic floating arithmetic retains destination precision and scalar metadata.
use crate::{NativeTensor, layout::Layout, storage::format};
use alloc::sync::Arc;
use burn_backend::{DType, Scalar, bf16, f16};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise, float_arithmetic as abi};

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
