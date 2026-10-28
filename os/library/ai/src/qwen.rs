//! Qwen inference operations over public Burn tensors.
mod config;
#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
mod cpu_state;
pub mod gguf;
mod model;
mod ops;
mod packed;
pub mod tokenizer;
pub use config::Config;
pub use model::{Error, Generation, Qwen, Session, StopReason, WeightData, greedy, is_stop, verify, verify_fixture};
mod state;
pub use state::{AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaParameters, DeltaState, DeltaWeights};

use alloc::{format, string::String, vec::Vec};
use burn::{
    module::{Module, Param},
    tensor::{DType, Device, Int, Tensor, TensorData},
};
use d3os_tensor_file::element_count;
pub use d3os_tensor_file::{
    matrix::{MatrixFormat, MatrixLayout},
    quantized::Quantization,
};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefillQuantization {
    #[default]
    Accurate,
    Q8_1,
}

impl PrefillQuantization {
    fn check_device(self, device: &Device) -> Result<()> {
        if self == Self::Q8_1 && !native_matrix_device(device) {
            return Err(Error::from("Q8_1 prefill requires the Native NVIDIA backend"));
        }
        Ok(())
    }
}

#[cfg(feature = "native")]
impl PrefillQuantization {
    fn native(self) -> d3os_burn_native::backend::PrefillQuantization {
        match self {
            Self::Accurate => d3os_burn_native::backend::PrefillQuantization::Accurate,
            Self::Q8_1 => d3os_burn_native::backend::PrefillQuantization::Q8_1,
        }
    }
}

pub(crate) fn device_supported(device: &Device) -> Result<()> {
    if device.is_autodiff() {
        return Err(Error::from("packed Qwen operations support inference devices only"));
    }
    match device.as_dispatch() {
        #[cfg(feature = "native")]
        burn::backend::DispatchDevice::Native(_) => Ok(()),
        #[cfg(feature = "cpu")]
        burn::backend::DispatchDevice::Flex(_) => Ok(()),
        #[allow(unreachable_patterns)]
        _ => Err(Error::from("Qwen requires the native NVIDIA or Flex CPU backend")),
    }
}

fn native_matrix_device(device: &Device) -> bool {
    match device.as_dispatch() {
        #[cfg(feature = "native")]
        burn::backend::DispatchDevice::Native(_) => true,
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// Immutable, validated weights stored as exact packed I32 words.
/// The logical matrix shape is independent of its physical Burn storage shape.
#[derive(Module, Debug)]
pub struct PackedMatrix {
    words: Param<Tensor<1, Int>>,
    #[module(skip)]
    shape: [usize; 2],
    #[module(skip)]
    format: MatrixFormat,
    #[module(skip)]
    layout: MatrixLayout,
    #[module(skip)]
    prefill_quantization: PrefillQuantization,
}

impl PackedMatrix {
    /// Validate the complete encoding before allocating or uploading a device tensor.
    pub fn from_bytes(shape: [usize; 2], format: MatrixFormat, bytes: &[u8], device: &Device) -> Result<Self> {
        device_supported(device)?;
        format.validate(shape, bytes).map_err(|e| format!("Qwen packed matrix: {e}"))?;
        // Reorder only the measured 9B FFN geometries. The file encoding and
        // numerical format stay unchanged; the device owns one packed copy.
        let layout = if cfg!(feature = "native")
            && native_matrix_device(device)
            && matches!(shape, [12288, 4096] | [4096, 12288])
            && format == MatrixFormat::Quantized(Quantization::Q8_0)
        {
            MatrixLayout::Q8Columns32
        } else {
            MatrixLayout::RowMajor
        };
        let count = bytes.len().div_ceil(4);
        let length = count.checked_mul(4).ok_or(Error::AllocationFailed)?;
        let mut packed = if layout == MatrixLayout::Q8Columns32 {
            layout.pack(shape, format, bytes)?
        } else {
            Vec::new()
        };
        packed
            .try_reserve_exact(length - packed.len())
            .map_err(|_| String::from("cannot allocate Qwen packed matrix"))?;
        // Copy the complete encoding once, then initialize only its partial word.
        if layout == MatrixLayout::RowMajor {
            packed.extend_from_slice(bytes);
        }
        packed.resize(length, 0);
        Ok(Self {
            words: packed::parameter(
                Tensor::<1, Int>::from_data(TensorData::from_bytes_vec(packed, [count], DType::I32), device),
                shape,
                format,
                layout,
            ),
            shape,
            format,
            layout,
            prefill_quantization: PrefillQuantization::Accurate,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn format(&self) -> MatrixFormat {
        self.format
    }
    pub fn device(&self) -> Device {
        self.words.val().device()
    }
    pub fn storage_words(&self) -> usize {
        self.words.val().dims()[0]
    }

    pub fn project(&self, input: Tensor<2>) -> Result<Tensor<2>> {
        self.check_input(&input)?;
        Ok(Tensor::from_dispatch(<burn::backend::Dispatch as ops::QwenOps>::project(
            input.into_dispatch(),
            self.primitive(),
        )))
    }

    pub fn project_add(&self, input: Tensor<2>, residual: Tensor<2>) -> Result<Tensor<2>> {
        self.check_input(&input)?;
        let [batch, _] = input.dims();
        if residual.dims() != [batch, self.shape[0]] || residual.device() != input.device() || residual.dtype() != DType::F32 || residual.device().is_autodiff()
        {
            return Err(Error::from("Qwen residual projection shape, dtype, or device differs"));
        }
        Ok(Tensor::from_dispatch(<burn::backend::Dispatch as ops::QwenOps>::project_add(
            input.into_dispatch(),
            self.primitive(),
            residual.into_dispatch(),
        )))
    }

    /// Compute silu(gate(input)) * up(input), preserving packed weight storage.
    pub fn gated_project(&self, up: &Self, input: Tensor<2>) -> Result<Tensor<2>> {
        self.check_input(&input)?;
        if up.shape != self.shape || up.device() != self.device() || up.prefill_quantization != self.prefill_quantization {
            return Err(Error::from("Qwen gated projection matrices differ"));
        }
        Ok(Tensor::from_dispatch(<burn::backend::Dispatch as ops::QwenOps>::gated_project(
            input.into_dispatch(),
            self.primitive(),
            up.primitive(),
        )))
    }

    pub fn embedding(&self, tokens: &[u32]) -> Result<Tensor<2>> {
        device_supported(&self.device())?;
        if tokens.is_empty() || tokens.iter().any(|&id| id as usize >= self.shape[0]) {
            return Err(Error::from("Qwen embedding token is outside the vocabulary"));
        }
        element_count(&[tokens.len(), self.shape[1]]).map_err(|e| format!("Qwen embedding: {e}"))?;
        Ok(Tensor::from_dispatch(<burn::backend::Dispatch as ops::QwenOps>::embedding(
            self.primitive(),
            tokens.to_vec(),
        )))
    }

    fn check_input(&self, input: &Tensor<2>) -> Result<()> {
        device_supported(&input.device())?;
        device_supported(&self.device())?;
        self.prefill_quantization.check_device(&input.device())?;
        let [batch, width] = input.dims();
        if batch == 0 || width != self.shape[1] || input.dtype() != DType::F32 || input.device() != self.device() {
            return Err(Error::from("Qwen projection input shape, dtype, or device differs"));
        }
        element_count(&[batch, self.shape[0]]).map_err(|e| format!("Qwen projection: {e}"))?;
        Ok(())
    }

    fn primitive(&self) -> ops::MatrixPrimitive<burn::backend::Dispatch> {
        ops::MatrixPrimitive {
            words: self.words.val().into_dispatch(),
            shape: self.shape,
            format: self.format,
            layout: self.layout,
            prefill_quantization: self.prefill_quantization,
        }
    }
}

/// Qwen's zero-centered normalization uses (weight + 1), unlike ordinary RMSNorm.
pub fn rms_norm(input: Tensor<2>, weight: Tensor<1>, epsilon: f32) -> Result<Tensor<2>> {
    check_norm(&input, &weight, epsilon)?;
    Ok(Tensor::from_dispatch(<burn::backend::Dispatch as ops::QwenOps>::rms_norm(
        input.into_dispatch(),
        weight.into_dispatch(),
        epsilon,
    )))
}

/// Return both the residual sum and its zero-centered normalization.
pub fn residual_norm(input: Tensor<2>, residual: Tensor<2>, weight: Tensor<1>, epsilon: f32) -> Result<(Tensor<2>, Tensor<2>)> {
    check_norm(&input, &weight, epsilon)?;
    if residual.dims() != input.dims() || residual.device() != input.device() || residual.dtype() != DType::F32 || residual.device().is_autodiff() {
        return Err(Error::from("Qwen normalization residual shape, dtype, or device differs"));
    }
    let (sum, normalized) =
        <burn::backend::Dispatch as ops::QwenOps>::residual_norm(input.into_dispatch(), residual.into_dispatch(), weight.into_dispatch(), epsilon);
    Ok((Tensor::from_dispatch(sum), Tensor::from_dispatch(normalized)))
}

fn check_norm(input: &Tensor<2>, weight: &Tensor<1>, epsilon: f32) -> Result<()> {
    device_supported(&input.device())?;
    if input.dims()[0] == 0
        || input.dims()[1] == 0
        || weight.dims() != [input.dims()[1]]
        || input.device() != weight.device()
        || weight.device().is_autodiff()
        || input.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || !epsilon.is_finite()
        || epsilon <= 0.
    {
        return Err(Error::from("invalid Qwen RMS normalization input"));
    }
    Ok(())
}
