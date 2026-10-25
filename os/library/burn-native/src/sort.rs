//! Resident merge sorting with bounded ping-pong storage and original-axis indices.
use crate::{NativeTensor, layout::Layout};
use alloc::{sync::Arc, vec};
use burn_backend::{DType, Shape, Slice};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, sort::Mode};

impl NativeTensor {
    pub fn sort(&self, axis: usize, descending: bool) -> Result<(Self, Self)> {
        self.ordered(axis, descending, false)
    }

    /// Maximum ordering places NaNs first and resolves numeric ties by index.
    pub fn topk(&self, axis: usize, k: usize) -> Result<(Self, Self)> {
        if !matches!(self.dtype, DType::F32 | DType::I32) || axis >= self.layout.shape().num_dims() || k > self.layout.shape()[axis] {
            return Err(Error::InvalidParameter);
        }
        let mut shape = self.layout.shape().to_vec();
        shape[axis] = k;
        let layout = Layout::contiguous(Shape::from(shape))?;
        if layout.is_empty() {
            return Ok((
                Self {
                    buffer: self.buffer.clone(),
                    layout: layout.clone(),
                    dtype: self.dtype,
                },
                Self {
                    buffer: self.buffer.clone(),
                    layout,
                    dtype: DType::I32,
                },
            ));
        }
        let (values, indices) = self.ordered(axis, true, true)?;
        let mut slices = vec![Slice::full(); self.layout.shape().num_dims()];
        slices[axis] = Slice::new(0, Some(k as isize), 1);
        Ok((values.slice(&slices)?, indices.slice(&slices)?))
    }

    fn ordered(&self, axis: usize, descending: bool, topk: bool) -> Result<(Self, Self)> {
        if !matches!(self.dtype, DType::F32 | DType::I32) || axis >= self.layout.shape().num_dims() {
            return Err(Error::InvalidParameter);
        }
        let dense = Layout::contiguous(self.layout.shape().clone())?;
        let count = dense.len();
        let storage = count.checked_mul(2).ok_or(Error::SizeOverflow)?;
        let values = Layout::from_parts(dense.shape().clone(), dense.strides().to_vec(), 0, storage)?;
        let indices = Layout::from_parts(dense.shape().clone(), dense.strides().to_vec(), count as i64, storage)?;
        let buffer = if dense.is_empty() {
            self.buffer.clone()
        } else {
            let buffer = Nvidia.record_commands(|| {
                let mode = |initial| {
                    if topk {
                        Mode::for_topk(self.dtype == DType::I32, initial)
                    } else {
                        Mode::new(self.dtype == DType::I32, descending, initial)
                    }
                };
                let initial = self.layout.sort_parameters(axis, mode(true), 1)?;
                let mut current = Nvidia.produce_words(storage, Kernel::TensorSort, &[(&self.buffer, 0), (&self.buffer, 0)], &initial)?;
                let mut scratch = None;
                let mut width = 2;
                while width < self.layout.shape()[axis] {
                    let parameters = dense.sort_parameters(axis, mode(false), width)?;
                    if let Some(next) = &mut scratch {
                        Nvidia.dispatch_words(Kernel::TensorSort, &[(&current, 0), (&current, count), (next, 0)], &parameters)?;
                        core::mem::swap(&mut current, next);
                    } else {
                        let next = Nvidia.produce_words(storage, Kernel::TensorSort, &[(&current, 0), (&current, count)], &parameters)?;
                        scratch = Some(core::mem::replace(&mut current, next));
                    }
                    width *= 2;
                }
                Ok::<_, Error>(current)
            })?;
            Arc::new(buffer)
        };
        Ok((
            Self {
                buffer: buffer.clone(),
                layout: values,
                dtype: self.dtype,
            },
            Self {
                buffer,
                layout: indices,
                dtype: DType::I32,
            },
        ))
    }
}
