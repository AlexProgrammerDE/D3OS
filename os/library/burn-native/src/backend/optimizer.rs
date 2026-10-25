//! Burn optimizer steps over resident FP32 parameter and moment storage.
use super::{BackendTensor, Native};
use crate::{NativeTensor, layout::Layout};
use alloc::{sync::Arc, vec::Vec};
use burn_backend::{DType, Shape, Slice, TensorMetadata};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, optimizer};

fn update<const N: usize>(inputs: &[&NativeTensor], flags: u32, coefficients: [f32; 7]) -> Result<[NativeTensor; N]> {
    let mut contract = [0u32; optimizer::PARAMETERS];
    contract[..2].copy_from_slice(&[1, flags]);
    let extents = optimizer::validate(&contract).map_err(|_| Error::InvalidParameter)?;
    let expected_inputs = 2 + usize::from(flags & optimizer::STATE != 0) * 2 + usize::from(flags & optimizer::MAXIMUM != 0);
    if inputs.len() != expected_inputs || extents[5] as usize != N {
        return Err(Error::InvalidParameter);
    }
    let shape = inputs[0].shape();
    if inputs.iter().any(|tensor| tensor.dtype() != DType::F32 || tensor.shape() != shape) {
        return Err(Error::ShapeMismatch);
    }
    let count = shape.num_elements();
    if count == 0 {
        let empty: Vec<_> = (0..N).map(|_| NativeTensor::empty(shape.clone(), DType::F32)).collect::<Result<_>>()?;
        return empty.try_into().map_err(|_| Error::ShapeMismatch);
    }
    let words = N.checked_mul(count).ok_or(Error::SizeOverflow)?;
    let layout = Layout::contiguous(Shape::new([words]))?;
    Nvidia.record_commands(|| {
        let dense: Vec<_> = inputs
            .iter()
            .map(|tensor| {
                if tensor.layout.is_contiguous() {
                    Ok((*tensor).clone())
                } else {
                    tensor.copy()
                }
            })
            .collect::<Result<_>>()?;
        let state = flags & optimizer::STATE != 0;
        let maximum = flags & optimizer::MAXIMUM != 0;
        let input = |index: usize| {
            let tensor = &dense[index];
            (&*tensor.buffer, tensor.layout.offset())
        };
        let pointers = [
            input(0),
            input(1),
            input(if state { 2 } else { 0 }),
            input(if state { 3 } else { 0 }),
            input(if maximum { 4 } else { 0 }),
        ];
        let mut parameters = [0usize; optimizer::PARAMETERS];
        parameters[..2].copy_from_slice(&[count, flags as usize]);
        for (parameter, coefficient) in parameters[2..].iter_mut().zip(coefficients) {
            *parameter = coefficient.to_bits() as usize;
        }
        let buffer = Nvidia.produce_words(words, Kernel::Adam, &pointers, &parameters)?;
        let slab = NativeTensor {
            buffer: Arc::new(buffer),
            layout,
            dtype: DType::F32,
        };
        let outputs: Vec<_> = (0..N)
            .map(|index| {
                slab.clone()
                    .slice(&[Slice::new((index * count) as isize, Some(((index + 1) * count) as isize), 1)])?
                    .reshape(shape.clone())
            })
            .collect::<Result<_>>()?;
        outputs.try_into().map_err(|_| Error::ShapeMismatch)
    })
}

impl Native {
    /// Update parameters and Adam state in one launch without mutating inputs.
    /// Coefficients retain Burn's FP32 bias correction and epsilon placement.
    pub fn adam(
        parameter: BackendTensor, gradient: BackendTensor, moments: Option<(BackendTensor, BackendTensor)>, maximum: Option<BackendTensor>, amsgrad: bool,
        coefficients: [f32; 7],
    ) -> (BackendTensor, BackendTensor, BackendTensor, Option<BackendTensor>) {
        let shape = parameter.shape();
        let mut inputs = alloc::vec![&parameter, &gradient];
        let mut flags = if amsgrad { optimizer::AMSGRAD } else { 0 };
        if let Some((first, second)) = &moments {
            flags |= optimizer::STATE;
            inputs.extend([first, second]);
        }
        if let Some(maximum) = &maximum {
            flags |= optimizer::MAXIMUM;
            inputs.push(maximum);
        }
        if amsgrad {
            let [parameter, first, second, maximum] =
                BackendTensor::outputs("adam", core::array::from_fn(|_| shape.clone()), [DType::F32; 4], &inputs, |values| {
                    update(values, flags, coefficients)
                });
            (parameter, first, second, Some(maximum))
        } else {
            let [parameter, first, second] = BackendTensor::outputs("adam", core::array::from_fn(|_| shape.clone()), [DType::F32; 3], &inputs, |values| {
                update(values, flags, coefficients)
            });
            (parameter, first, second, None)
        }
    }
}
