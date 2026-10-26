//! Broadcast products with the legacy F32 and wrapping I32 graphs preserved.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, Shape};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::Kernel;

impl NativeTensor {
    /// Broadcast matrix products: FP32 Tensor Cores or exact wrapping I32 arithmetic.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if crate::mixed_matmul::supported(self.dtype) {
            return self.mixed_matmul(rhs);
        }
        if !matches!(self.dtype, DType::F32 | DType::I32) || rhs.dtype != self.dtype {
            return Err(Error::InvalidParameter);
        }
        let shape = crate::layout::matmul_shape(self.layout.shape(), rhs.layout.shape())?;
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        let rank = shape.num_dims();
        let rows = shape[rank - 2];
        let cols = shape[rank - 1];
        let inner = self.layout.shape()[self.layout.shape().num_dims() - 1];
        if inner == 0 {
            return Self::empty(shape, self.dtype);
        }
        let matrix_contiguous = |tensor: &Self| {
            let rank = tensor.layout.shape().num_dims();
            let rows = tensor.layout.shape()[rank - 2];
            let cols = tensor.layout.shape()[rank - 1];
            (rows == 1 || tensor.layout.strides()[rank - 2] == cols as i64) && (cols == 1 || tensor.layout.strides()[rank - 1] == 1)
        };
        // Materialize only an input with a noncontiguous matrix, before batch
        // expansion. Broadcast batches then reuse its original resident storage.
        let left = if matrix_contiguous(self) { self.clone() } else { self.copy()? };
        let right = if matrix_contiguous(rhs) { rhs.clone() } else { rhs.copy()? };
        let mut left_shape = shape.to_vec();
        left_shape[rank - 1] = inner;
        let mut right_shape = shape.to_vec();
        right_shape[rank - 2] = inner;
        let left_layout = left.layout.expand(Shape::from(left_shape))?;
        let right_layout = right.layout.expand(Shape::from(right_shape))?;
        let batches = layout.len() / (rows * cols);
        // Thin, long reductions otherwise leave most of the GPU idle. Target
        // 128 output blocks, limited to one split per 256 K elements. Each
        // split and the final merge use compensated FP32 sums.
        let tiles = rows.div_ceil(16) * cols.div_ceil(32);
        let splits = if self.dtype == DType::F32 && inner >= 4096 && tiles < 128 {
            128usize.div_ceil(tiles).next_power_of_two().min(inner / 256)
        } else {
            1
        };
        let parameters = [rows, inner, cols, splits];
        let kernel = if self.dtype == DType::F32 { Kernel::Matmul } else { Kernel::IntMatmul };
        let product = |left_offset, right_offset| {
            Nvidia.produce_words(
                rows * cols * splits,
                kernel,
                &[(&left.buffer, left_offset), (&right.buffer, right_offset)],
                &parameters[..if splits == 1 { 3 } else { 4 }],
            )
        };
        let buffer = if batches == 1 {
            let result = product(left_layout.offset(), right_layout.offset())?;
            if splits == 1 {
                result
            } else {
                Nvidia.produce_words(rows * cols, Kernel::SumProjectionPartials, &[(&result, 0)], &[rows, cols, splits])?
            }
        } else {
            let buffer = Nvidia.allocate_words(layout.len())?;
            for batch in 0..batches {
                let left_offset = left_layout.element_offset(batch * rows * inner)?;
                let right_offset = right_layout.element_offset(batch * inner * cols)?;
                if splits == 1 {
                    Nvidia.dispatch_words(
                        kernel,
                        &[(&left.buffer, left_offset), (&right.buffer, right_offset), (&buffer, batch * rows * cols)],
                        &parameters[..3],
                    )?;
                } else {
                    let partials = product(left_offset, right_offset)?;
                    Nvidia.dispatch_words(
                        Kernel::SumProjectionPartials,
                        &[(&partials, 0), (&buffer, batch * rows * cols)],
                        &[rows, cols, splits],
                    )?;
                }
            }
            buffer
        };
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype: self.dtype,
        })
    }
}
