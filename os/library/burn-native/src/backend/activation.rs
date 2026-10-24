//! Consume unique activation storage while preserving shared Burn tensors.
use super::{BackendTensor, Native};
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{DType, TensorMetadata, ops::ActivationOps, tensor::FloatTensor};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute::Kernel;

impl ActivationOps<Self> for Native {
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
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
