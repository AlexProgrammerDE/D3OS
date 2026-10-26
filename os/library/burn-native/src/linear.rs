//! Dense linear layers with one resident product and a private bias epilogue.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, Shape};
use d3os_compute::{Buffer, Error, Result, nvidia::Nvidia};
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

pub(crate) fn flattened_shape(shape: &Shape) -> Result<Shape> {
    Layout::contiguous(shape.clone())?;
    let Some(&width) = shape.as_slice().last() else {
        return Err(Error::ShapeMismatch);
    };
    let leading = &shape[..shape.num_dims() - 1];
    let rows = if leading.contains(&0) {
        0
    } else {
        leading.iter().try_fold(1usize, |rows, &n| {
            rows.checked_mul(n).filter(|&n| n <= i32::MAX as usize).ok_or(Error::SizeOverflow)
        })?
    };
    let flat = Shape::new([rows, width]);
    Layout::contiguous(flat.clone())?;
    Ok(flat)
}

pub(crate) fn flattened_layout(tensor: &NativeTensor) -> Result<Layout> {
    let shape = flattened_shape(tensor.layout.shape())?;
    let layout = if tensor.layout.is_contiguous() {
        tensor.layout.clone()
    } else {
        Layout::contiguous(tensor.layout.shape().clone())?
    };
    layout.reshape(shape)
}

pub(crate) fn weight_gradient_shape(input: &Shape, grad: &Shape) -> Result<Shape> {
    let input_flat = flattened_shape(input)?;
    let grad_flat = flattened_shape(grad)?;
    if input.num_dims() != grad.num_dims() || input[..input.num_dims() - 1] != grad[..grad.num_dims() - 1] {
        return Err(Error::ShapeMismatch);
    }
    let shape = Shape::new([input_flat[1], grad_flat[1]]);
    Layout::contiguous(shape.clone())?;
    Ok(shape)
}

impl NativeTensor {
    pub fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        if crate::mixed_matmul::supported(self.dtype) {
            if weight.dtype != self.dtype || bias.is_some_and(|b| b.dtype != self.dtype) {
                return Err(Error::InvalidParameter);
            }
            let shape = output_shape(self.layout.shape(), weight.layout.shape(), bias.map(|b| b.layout.shape()))?;
            let flat = flattened_shape(self.layout.shape())?;
            // Validate all sources before an empty product can alias an input.
            crate::mixed_matmul::validate_tensor(self)?;
            crate::mixed_matmul::validate_tensor(weight)?;
            if let Some(bias) = bias {
                crate::mixed_matmul::validate_tensor(bias)?;
            }
            let flat_layout = flattened_layout(self)?;
            let words = crate::storage::width(self.dtype)?.words(flat_layout.storage_elements() as u32) as usize;
            crate::mixed_matmul::validate_product(&flat_layout, self.dtype, words.max(1), &weight.layout, weight.dtype, weight.buffer.len())?;
            if shape.as_slice().contains(&0) {
                return Ok(Self {
                    buffer: self.buffer.clone(),
                    layout: Layout::contiguous(shape)?,
                    dtype: self.dtype,
                });
            }
            return Nvidia.record_commands(|| {
                let input = if self.layout.is_contiguous() { self.clone() } else { self.copy()? };
                let output = input.reshape(flat)?.matmul(weight)?.reshape(shape)?;
                // Matmul has already rounded the product to its destination.
                // Typed bias addition therefore retains the default staged graph.
                if let Some(bias) = bias {
                    output.binary(bias, d3os_nvidia::compute::elementwise::Elementwise::Add)
                } else {
                    Ok(output)
                }
            });
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_flatten_handles_rank_one_and_zero_final_dimensions() {
        for (input, flat) in [
            (Shape::new([0]), Shape::new([1, 0])),
            (Shape::new([4]), Shape::new([1, 4])),
            (Shape::new([2, 3, 0]), Shape::new([6, 0])),
            (Shape::new([2, 0, 4]), Shape::new([0, 4])),
        ] {
            assert_eq!(flattened_shape(&input).unwrap(), flat);
        }
        assert!(matches!(flattened_shape(&Shape::new([])), Err(Error::ShapeMismatch)));
        assert_eq!(weight_gradient_shape(&Shape::new([0, 3]), &Shape::new([0, 5])).unwrap(), Shape::new([3, 5]));
        assert_eq!(
            weight_gradient_shape(&Shape::new([2, 3, 0]), &Shape::new([2, 3, 5])).unwrap(),
            Shape::new([0, 5])
        );
        assert!(matches!(
            weight_gradient_shape(&Shape::new([2, 0]), &Shape::new([3, 0])),
            Err(Error::ShapeMismatch)
        ));
        assert!(matches!(
            output_shape(&Shape::new([0, 3]), &Shape::new([3, 0]), Some(&Shape::new([1]))),
            Err(Error::ShapeMismatch)
        ));
    }
}
