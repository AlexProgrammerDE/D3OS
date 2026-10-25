//! Vector products composed from resident tensor operations.
use crate::{
    NativeTensor,
    layout::{Layout, broadcast},
};
use alloc::{vec, vec::Vec};
use burn_backend::{DType, Slice};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::elementwise::Elementwise;

impl NativeTensor {
    pub fn cross(&self, rhs: &Self, axis: usize) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        let left = self.layout.shape();
        let right = rhs.layout.shape();
        if left.num_dims() != right.num_dims() || left.get(axis) != Some(&3) || right.get(axis) != Some(&3) {
            return Err(Error::ShapeMismatch);
        }
        let layout = Layout::contiguous(broadcast(left, right)?)?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        let component = |tensor: &Self, index: usize| {
            let mut slices = vec![Slice::full(); axis + 1];
            slices[axis] = Slice::new(index as isize, Some(index as isize + 1), 1);
            tensor.clone().slice(&slices)
        };
        Nvidia.record_commands(|| {
            let mut components = Vec::with_capacity(3);
            for i in 0..3 {
                let j = (i + 1) % 3;
                let k = (i + 2) % 3;
                let positive = component(self, j)?.binary(&component(rhs, k)?, Elementwise::Multiply)?;
                let negative = component(self, k)?.binary(&component(rhs, j)?, Elementwise::Multiply)?;
                components.push(positive.binary(&negative, Elementwise::Subtract)?);
            }
            Self::cat(&components, axis)
        })
    }
}
