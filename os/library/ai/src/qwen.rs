//! Qwen inference operations over public Burn tensors.
#[cfg(feature = "cpu")]
mod cpu;
mod ops;

use alloc::{format, string::String, vec::Vec};
use burn::{
    module::{Module, Param, ParamId},
    tensor::{DType, Device, Int, Tensor, TensorData},
};
use d3os_tensor_file::element_count;
pub use d3os_tensor_file::{matrix::MatrixFormat, quantized::Quantization};

pub type Result<T> = core::result::Result<T, String>;

pub(crate) fn device_supported(device: &Device) -> Result<()> {
    if device.is_autodiff() {
        return Err(String::from("packed Qwen operations support inference devices only"));
    }
    match device.as_dispatch() {
        #[cfg(feature = "native")]
        burn::backend::DispatchDevice::Native(_) => Ok(()),
        #[cfg(feature = "cpu")]
        burn::backend::DispatchDevice::Flex(_) => Ok(()),
        #[allow(unreachable_patterns)]
        _ => Err(String::from("Qwen requires the native NVIDIA or Flex CPU backend")),
    }
}

/// Immutable, validated row-major weights stored as exact packed I32 words.
/// The logical matrix shape is independent of its physical Burn storage shape.
#[derive(Module, Debug)]
pub struct PackedMatrix {
    words: Param<Tensor<1, Int>>,
    #[module(skip)]
    shape: [usize; 2],
    #[module(skip)]
    format: MatrixFormat,
}

impl PackedMatrix {
    /// Validate the complete encoding before allocating or uploading a device tensor.
    pub fn from_bytes(shape: [usize; 2], format: MatrixFormat, bytes: &[u8], device: &Device) -> Result<Self> {
        device_supported(device)?;
        format.validate(shape, bytes).map_err(|e| format!("Qwen packed matrix: {e}"))?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(bytes.len().div_ceil(4))
            .map_err(|_| String::from("cannot allocate Qwen packed matrix"))?;
        for chunk in bytes.chunks(4) {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            words.push(i32::from_le_bytes(word));
        }
        let count = words.len();
        Ok(Self {
            words: Param::initialized(ParamId::new(), Tensor::<1, Int>::from_data(TensorData::new(words, [count]), device)),
            shape,
            format,
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
            return Err(String::from("Qwen residual projection shape, dtype, or device differs"));
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
        if up.shape != self.shape || up.device() != self.device() {
            return Err(String::from("Qwen gated projection matrices differ"));
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
            return Err(String::from("Qwen embedding token is outside the vocabulary"));
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
        let [batch, width] = input.dims();
        if batch == 0 || width != self.shape[1] || input.dtype() != DType::F32 || input.device() != self.device() {
            return Err(String::from("Qwen projection input shape, dtype, or device differs"));
        }
        element_count(&[batch, self.shape[0]]).map_err(|e| format!("Qwen projection: {e}"))?;
        Ok(())
    }

    fn primitive(&self) -> ops::MatrixPrimitive<burn::backend::Dispatch> {
        ops::MatrixPrimitive {
            words: self.words.val().into_dispatch(),
            shape: self.shape,
            format: self.format,
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
        return Err(String::from("Qwen normalization residual shape, dtype, or device differs"));
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
        return Err(String::from("invalid Qwen RMS normalization input"));
    }
    Ok(())
}
