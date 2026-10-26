//! Common reduction routing, legacy staging, and resident value/index pairs.
use crate::{NativeTensor, integer_arithmetic::integer_scalar_bits, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, Scalar, Shape};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, reduction::Reduction};

impl NativeTensor {
    /// Reduce one dimension, preserving its rank and integer dtype. Extrema
    /// select the first logical index for equal values. Integer arithmetic
    /// wraps in the declared accumulator; means divide before narrowing.
    pub fn reduce(&self, axis: usize, operation: Reduction) -> Result<Self> {
        if matches!(self.dtype, DType::F64 | DType::Flex32 | DType::F16 | DType::BF16) {
            if matches!(operation, Reduction::Sum | Reduction::Mean | Reduction::Product) {
                return self.float_reduce(axis, operation);
            }
            let dtype = if operation.index() { DType::I32 } else { self.dtype };
            return self.float_extrema(Some(axis), operation, dtype);
        }
        if self.dtype.is_int() || self.dtype.is_uint() {
            // Validation precedes empty handling, including zero-axis means
            // and extrema. Keep the cooperative legacy I32 kernels intact.
            if self.dtype != DType::I32 {
                return self.integer_reduce(Some(axis), operation);
            }
            crate::integer_reduction::plan(&self.layout, self.dtype, Some(axis), operation)?;
        }
        let operation = match self.dtype {
            DType::F32 if !operation.is_integer() => operation,
            DType::I32 => operation.integer(),
            _ => return Err(Error::InvalidParameter),
        };
        let layout = Layout::contiguous(self.layout.reduction_shape(axis)?)?;
        let dtype = if operation.index() { DType::I32 } else { self.dtype };
        if self.layout.shape()[axis] == 0 && !operation.allows_empty() {
            return Err(Error::InvalidParameter);
        }
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype,
            });
        }
        let parameters = self.layout.reduction_parameters(axis, operation)?;
        let buffer = Nvidia.produce_words(layout.len(), Kernel::TensorReduction, &[(&self.buffer, 0)], &parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    pub(crate) fn reduce_all(&self, operation: Reduction) -> Result<Self> {
        if matches!(self.dtype, DType::F64 | DType::Flex32 | DType::F16 | DType::BF16) {
            if matches!(operation, Reduction::Sum | Reduction::Mean | Reduction::Product) {
                return self.float_reduce_all(operation);
            }
            return self.float_extrema(None, operation, self.dtype);
        }
        if self.dtype.is_int() || self.dtype.is_uint() {
            // Validate ALL metadata before either a legacy stage or a scalar
            // reshape. Public globals never expose an arg reduction.
            if self.dtype != DType::I32 {
                return self.integer_reduce(None, operation);
            }
            crate::integer_reduction::plan(&self.layout, self.dtype, None, operation)?;
        } else if self.dtype != DType::F32 || operation.is_integer() || operation.index() {
            return Err(Error::InvalidParameter);
        }
        Nvidia.record_commands(|| {
            let mut tensor = self.clone();
            for axis in (0..tensor.layout.shape().num_dims()).rev() {
                tensor = tensor.reduce(axis, operation)?;
            }
            tensor.reshape(Shape::new([1]))
        })
    }

    pub(crate) fn reduction_pair(&self, axis: usize, operation: Reduction, index_dtype: DType) -> Result<(Self, Self)> {
        let index_operation = match operation {
            Reduction::Min | Reduction::IntMin => Reduction::ArgMin,
            Reduction::Max | Reduction::IntMax => Reduction::ArgMax,
            _ => return Err(Error::InvalidParameter),
        };
        integer_scalar_bits(index_dtype, Scalar::Int(0))?;
        // Validate the common geometry before either fresh-output reduction.
        if self.dtype == DType::F32 && !operation.is_integer() {
            let shape = self.layout.reduction_shape(axis)?;
            if self.layout.shape()[axis] == 0 {
                return Err(Error::InvalidParameter);
            }
            if !shape.contains(&0) {
                self.layout.reduction_parameters(axis, operation)?;
                self.layout.reduction_parameters(axis, index_operation)?;
            }
        } else if matches!(self.dtype, DType::F64 | DType::Flex32 | DType::F16 | DType::BF16) {
            crate::float_extrema::plan(&self.layout, self.dtype, Some(axis), operation, self.dtype)?;
            crate::float_extrema::plan(&self.layout, self.dtype, Some(axis), index_operation, index_dtype)?;
            return Nvidia.record_commands(|| {
                let values = self.float_extrema(Some(axis), operation, self.dtype)?;
                let indices = self.float_extrema(Some(axis), index_operation, index_dtype)?;
                Ok((values, indices))
            });
        } else {
            crate::integer_reduction::plan(&self.layout, self.dtype, Some(axis), operation)?;
            crate::integer_reduction::plan(&self.layout, self.dtype, Some(axis), index_operation)?;
        }
        if matches!(self.dtype, DType::F32 | DType::I32) && self.layout.is_empty() {
            return Ok((self.reduce(axis, operation)?, self.reduce(axis, index_operation)?.cast(index_dtype)?));
        }
        Nvidia.record_commands(|| {
            // Both folds select from the immutable original source. Exposed
            // index narrowing cannot change the independently selected value.
            let values = self.reduce(axis, operation)?;
            let indices = self.reduce(axis, index_operation)?.cast(index_dtype)?;
            Ok((values, indices))
        })
    }
}
