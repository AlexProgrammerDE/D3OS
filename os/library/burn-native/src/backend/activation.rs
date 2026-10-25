//! Consume unique activation storage while preserving shared Burn tensors.
use super::{BackendTensor, Native};
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, TensorMetadata, get_or_init_device_settings, ops::ActivationOps, tensor::FloatTensor};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, predicate::Predicate};

impl ActivationOps<Self> for Native {
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        if tensor.dtype() != DType::F32 {
            return BackendTensor::apply("relu", tensor.shape(), tensor.dtype(), &[&tensor], |v| {
                let input = v[0];
                if !matches!(input.dtype(), DType::F64 | DType::F16 | DType::BF16 | DType::Flex32) {
                    return Err(Error::InvalidParameter);
                }
                if input.layout.is_empty() {
                    return Ok(input.clone());
                }
                let bool_dtype = get_or_init_device_settings::<Self>(&input.device()).bool_dtype;
                Nvidia.record_commands(|| {
                    let mask = input.compare_scalar(0f32.into(), Predicate::LessEqual, bool_dtype.into())?;
                    input.mask_fill_scalar(&mask, 0f32.into())
                })
            });
        }
        // Check ownership before apply borrows the input and creates aliases.
        let unique = tensor.can_mut();
        BackendTensor::apply("relu", tensor.shape(), DType::F32, &[&tensor], |v| {
            let input = v[0];
            if input.dtype() != DType::F32 {
                return Err(Error::InvalidParameter);
            }
            if input.layout.is_empty() {
                return Ok(input.clone());
            }
            Nvidia.record_commands(|| {
                if !unique && input.layout.is_contiguous() {
                    let buffer = Nvidia.produce_words(
                        input.layout.len(),
                        Kernel::Relu,
                        &[(&input.buffer, input.layout.offset())],
                        &[input.layout.len()],
                    )?;
                    return Ok(NativeTensor {
                        buffer: Arc::new(buffer),
                        layout: Layout::contiguous(input.shape())?,
                        dtype: DType::F32,
                    });
                }
                let output = if unique { input.clone() } else { input.copy()? };
                Nvidia.dispatch_words(
                    Kernel::Relu,
                    &[(&output.buffer, output.layout.offset()), (&output.buffer, output.layout.offset())],
                    &[output.layout.len()],
                )?;
                Ok(output)
            })
        })
    }
}
