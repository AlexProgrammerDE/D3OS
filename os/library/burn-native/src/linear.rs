//! Dense linear layers with one resident product and a private bias epilogue.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, Shape};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::Kernel;

pub(crate) fn output_shape(input: &Shape, weight: &Shape, bias: Option<&Shape>) -> Result<Shape> {
    let Some(&inner) = input.as_slice().last() else {
        return Err(Error::ShapeMismatch);
    };
    let [width, outputs] = *weight.as_slice() else {
        return Err(Error::ShapeMismatch);
    };
    if inner != width || bias.is_some_and(|b| b.as_slice() != [outputs]) {
        return Err(Error::ShapeMismatch);
    }
    let mut shape = input.to_vec();
    *shape.last_mut().unwrap() = outputs;
    let shape = Shape::from(shape);
    Layout::contiguous(shape.clone())?;
    Ok(shape)
}

impl NativeTensor {
    pub fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        if self.dtype != DType::F32 || weight.dtype != DType::F32 || bias.is_some_and(|b| b.dtype != DType::F32) {
            return Err(Error::InvalidParameter);
        }
        let shape = output_shape(self.layout.shape(), weight.layout.shape(), bias.map(|b| b.layout.shape()))?;
        if shape.as_slice().contains(&0) {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout: Layout::contiguous(shape)?,
                dtype: DType::F32,
            });
        }
        let input = if self.layout.is_contiguous() { self.clone() } else { self.copy()? };
        let dims = input.layout.shape().as_slice();
        let width = dims[dims.len() - 1];
        let rows = dims[..dims.len() - 1]
            .iter()
            .try_fold(1usize, |rows, &n| rows.checked_mul(n).ok_or(Error::SizeOverflow))?;
        let input = input.reshape(Shape::new([rows, width]))?;
        Nvidia.record_commands(|| {
            // A transposed dense weight already has the projection kernel's
            // [output, input] storage order. Preserve it instead of copying
            // every parameter into a row-major [input, output] matrix.
            let output = if width != 0 && weight.layout.swap_dims(0, 1)?.is_contiguous() {
                let outputs = weight.layout.shape()[1];
                let layout = Layout::contiguous(Shape::new([rows, outputs]))?;
                let buffer = Nvidia.produce_words(
                    layout.len(),
                    Kernel::Project,
                    &[(&input.buffer, input.layout.offset()), (&weight.buffer, weight.layout.offset())],
                    &[rows, width, outputs],
                )?;
                Self {
                    buffer: Arc::new(buffer),
                    layout,
                    dtype: DType::F32,
                }
            } else {
                input.matmul(weight)?
            };
            // Both products create private storage. Bias cannot overwrite an
            // input, saved activation, shared parameter, or neighboring view.
            if let Some(bias) = bias.filter(|_| !output.layout.is_empty()) {
                let bias = if bias.layout.is_contiguous() { bias.clone() } else { bias.copy()? };
                Nvidia.dispatch_words(
                    Kernel::BiasRow,
                    &[(&output.buffer, 0), (&bias.buffer, bias.layout.offset()), (&output.buffer, 0)],
                    &[output.layout.len(), weight.layout.shape()[1]],
                )?;
            }
            output.reshape(shape)
        })
    }
}
