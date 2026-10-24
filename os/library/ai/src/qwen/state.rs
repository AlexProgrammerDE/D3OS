//! Owned session tensors. Clones are snapshots, including after fused cache updates.
use super::{Error, Result, device_supported, ops};
use alloc::format;
use burn::{
    backend::Dispatch,
    tensor::{DType, Device, Tensor},
};
use d3os_tensor_file::element_count;

fn count(shape: &[usize]) -> Result<usize> {
    element_count(shape).map_err(|e| Error::from(format!("Qwen state geometry: {e}")))
}
fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.
}
fn tensor<const D: usize>(value: &Tensor<D>, shape: [usize; D], device: &Device) -> Result<()> {
    if value.dims() != shape || value.dtype() != DType::F32 || value.device() != *device || value.device().is_autodiff() {
        return Err(Error::from("Qwen state input shape, dtype, or device differs"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub epsilon: f32,
}
impl AttentionConfig {
    pub(crate) fn widths(self) -> Result<(usize, usize)> {
        let query = count(&[self.heads, self.head_dim])?;
        let key = count(&[self.kv_heads, self.head_dim])?;
        count(&[2, query])?;
        if !self.heads.is_multiple_of(self.kv_heads)
            || self.rotary_dim == 0
            || !self.rotary_dim.is_multiple_of(2)
            || self.rotary_dim > self.head_dim
            || !positive(self.rope_theta)
            || !positive(self.epsilon)
        {
            return Err(Error::from("invalid Qwen attention configuration"));
        }
        Ok((query, key))
    }
    #[cfg(feature = "native")]
    pub(crate) fn geometry(self) -> [usize; 4] {
        [self.heads, self.kv_heads, self.head_dim, self.rotary_dim]
    }
}
#[derive(Clone, Debug)]
pub struct AttentionState {
    keys: Tensor<3>,
    values: Tensor<3>,
    position: usize,
}
pub struct AttentionInputs {
    pub query_gate: Tensor<2>,
    pub key: Tensor<2>,
    pub value: Tensor<2>,
    pub query_norm: Tensor<1>,
    pub key_norm: Tensor<1>,
}
impl AttentionState {
    pub fn new(config: AttentionConfig, capacity: usize, device: &Device) -> Result<Self> {
        device_supported(device)?;
        config.widths()?;
        count(&[capacity, config.kv_heads, config.head_dim])?;
        Ok(Self {
            keys: Tensor::zeros([capacity, config.kv_heads, config.head_dim], device),
            values: Tensor::zeros([capacity, config.kv_heads, config.head_dim], device),
            position: 0,
        })
    }
    /// Restore resident tensors without a host readback. Unused cache rows retain their values.
    pub fn from_tensors(keys: Tensor<3>, values: Tensor<3>, position: usize, config: AttentionConfig) -> Result<Self> {
        config.widths()?;
        let device = keys.device();
        device_supported(&device)?;
        let capacity = keys.dims()[0];
        count(&[capacity, config.kv_heads, config.head_dim])?;
        tensor(&keys, [capacity, config.kv_heads, config.head_dim], &device)?;
        tensor(&values, keys.dims(), &device)?;
        if position > capacity {
            return Err(Error::from("Qwen attention cache position exceeds capacity"));
        }
        Ok(Self { keys, values, position })
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn keys(&self) -> &Tensor<3> {
        &self.keys
    }
    pub fn values(&self) -> &Tensor<3> {
        &self.values
    }
    /// Consume the current state and return its continuation with the output batch.
    /// A backend error remains attached to every result until readback or synchronization.
    pub fn forward(self, input: AttentionInputs, config: AttentionConfig) -> Result<(Self, Tensor<2>)> {
        let device = self.keys.device();
        device_supported(&device)?;
        let (query, key) = config.widths()?;
        let rows = input.query_gate.dims()[0];
        count(&[rows, query])?;
        tensor(&input.query_gate, [rows, 2 * query], &device)?;
        tensor(&input.key, [rows, key], &device)?;
        tensor(&input.value, [rows, key], &device)?;
        tensor(&input.query_norm, [config.head_dim], &device)?;
        tensor(&input.key_norm, [config.head_dim], &device)?;
        let capacity = self.keys.dims()[0];
        tensor(&self.keys, [capacity, config.kv_heads, config.head_dim], &device)?;
        tensor(&self.values, self.keys.dims(), &device)?;
        let position = self
            .position
            .checked_add(rows)
            .filter(|&end| end <= capacity)
            .ok_or("Qwen attention cache is full")?;
        let result = <Dispatch as ops::QwenOps>::attention(
            ops::AttentionPrimitive {
                query_gate: input.query_gate.into_dispatch(),
                key: input.key.into_dispatch(),
                value: input.value.into_dispatch(),
                query_norm: input.query_norm.into_dispatch(),
                key_norm: input.key_norm.into_dispatch(),
                keys: self.keys.into_dispatch(),
                values: self.values.into_dispatch(),
                position: self.position,
            },
            config,
        );
        Ok((
            Self {
                keys: Tensor::from_dispatch(result.keys),
                values: Tensor::from_dispatch(result.values),
                position,
            },
            Tensor::from_dispatch(result.output),
        ))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeltaConfig {
    pub key_heads: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kernel: usize,
    pub epsilon: f32,
}
impl DeltaConfig {
    pub(crate) fn widths(self) -> Result<(usize, usize)> {
        let keys = count(&[self.key_heads, self.key_dim, 2])?;
        let values = count(&[self.heads, self.value_dim])?;
        let channels = keys.checked_add(values).ok_or("Qwen convolution width overflow")?;
        count(&[channels, self.kernel])?;
        count(&[self.heads, self.key_dim, self.value_dim])?;
        if !self.heads.is_multiple_of(self.key_heads) || !positive(self.epsilon) {
            return Err(Error::from("invalid Qwen recurrent configuration"));
        }
        Ok((channels, values))
    }
    #[cfg(feature = "native")]
    pub(crate) fn geometry(self) -> [usize; 5] {
        [self.key_heads, self.heads, self.key_dim, self.value_dim, self.kernel]
    }
}
#[derive(Clone, Debug)]
pub struct DeltaState {
    convolution: Tensor<2>,
    recurrent: Tensor<3>,
}
#[derive(Clone, Debug)]
pub struct DeltaParameters {
    pub convolution: Tensor<3>,
    pub log_decay: Tensor<1>,
    pub time_bias: Tensor<1>,
    pub norm: Tensor<1>,
}
pub struct DeltaInputs {
    pub qkv: Tensor<2>,
    pub gate: Tensor<2>,
    pub beta: Tensor<2>,
    pub alpha: Tensor<2>,
}
impl DeltaState {
    pub fn new(config: DeltaConfig, device: &Device) -> Result<Self> {
        device_supported(device)?;
        let (channels, _) = config.widths()?;
        Ok(Self {
            convolution: Tensor::zeros([channels, config.kernel], device),
            recurrent: Tensor::zeros([config.heads, config.key_dim, config.value_dim], device),
        })
    }
    pub fn from_tensors(convolution: Tensor<2>, recurrent: Tensor<3>, config: DeltaConfig) -> Result<Self> {
        let (channels, _) = config.widths()?;
        let device = convolution.device();
        device_supported(&device)?;
        tensor(&convolution, [channels, config.kernel], &device)?;
        tensor(&recurrent, [config.heads, config.key_dim, config.value_dim], &device)?;
        Ok(Self { convolution, recurrent })
    }
    pub fn convolution(&self) -> &Tensor<2> {
        &self.convolution
    }
    pub fn recurrent(&self) -> &Tensor<3> {
        &self.recurrent
    }
    fn check(&self, parameters: &DeltaParameters, config: DeltaConfig) -> Result<(usize, usize, Device)> {
        let device = self.convolution.device();
        device_supported(&device)?;
        let (channels, values) = config.widths()?;
        tensor(&self.convolution, [channels, config.kernel], &device)?;
        tensor(&self.recurrent, [config.heads, config.key_dim, config.value_dim], &device)?;
        tensor(&parameters.convolution, [channels, 1, config.kernel], &device)?;
        tensor(&parameters.log_decay, [config.heads], &device)?;
        tensor(&parameters.time_bias, [config.heads], &device)?;
        tensor(&parameters.norm, [config.value_dim], &device)?;
        Ok((channels, values, device))
    }
    pub fn forward(self, input: DeltaInputs, parameters: DeltaParameters, config: DeltaConfig) -> Result<(Self, Tensor<2>)> {
        let (channels, values, device) = self.check(&parameters, config)?;
        let rows = input.qkv.dims()[0];
        count(&[rows, channels])?;
        tensor(&input.qkv, [rows, channels], &device)?;
        tensor(&input.gate, [rows, values], &device)?;
        tensor(&input.beta, [rows, config.heads], &device)?;
        tensor(&input.alpha, [rows, config.heads], &device)?;
        let result = <Dispatch as ops::QwenOps>::delta(
            ops::DeltaPrimitive {
                qkv: input.qkv.into_dispatch(),
                gate: input.gate.into_dispatch(),
                beta: input.beta.into_dispatch(),
                alpha: input.alpha.into_dispatch(),
                parameters: parameters.primitive(),
                convolution: self.convolution.into_dispatch(),
                recurrent: self.recurrent.into_dispatch(),
            },
            config,
        );
        Ok(Self::result(result))
    }
    pub fn project(self, input: Tensor<2>, weights: DeltaWeights<'_>, parameters: DeltaParameters, config: DeltaConfig) -> Result<(Self, Tensor<2>)> {
        let (channels, values, device) = self.check(&parameters, config)?;
        let [rows, width] = input.dims();
        count(&[rows, width])?;
        tensor(&input, [rows, width], &device)?;
        for (matrix, outputs) in [
            (weights.qkv, channels),
            (weights.gate, values),
            (weights.beta, config.heads),
            (weights.alpha, config.heads),
        ] {
            matrix.check_input(&input)?;
            if matrix.shape() != [outputs, width] || matrix.device() != device {
                return Err(Error::from("Qwen recurrent projection matrix differs"));
            }
        }
        let result = <Dispatch as ops::QwenOps>::project_delta(
            input.into_dispatch(),
            ops::DeltaWeightPrimitive {
                qkv: weights.qkv.primitive(),
                gate: weights.gate.primitive(),
                beta: weights.beta.primitive(),
                alpha: weights.alpha.primitive(),
            },
            parameters.primitive(),
            ops::DeltaCachePrimitive {
                convolution: self.convolution.into_dispatch(),
                recurrent: self.recurrent.into_dispatch(),
            },
            config,
        );
        Ok(Self::result(result))
    }
    fn result(result: ops::DeltaResult<Dispatch>) -> (Self, Tensor<2>) {
        (
            Self {
                convolution: Tensor::from_dispatch(result.convolution),
                recurrent: Tensor::from_dispatch(result.recurrent),
            },
            Tensor::from_dispatch(result.output),
        )
    }
}
pub struct DeltaWeights<'a> {
    pub qkv: &'a super::PackedMatrix,
    pub gate: &'a super::PackedMatrix,
    pub beta: &'a super::PackedMatrix,
    pub alpha: &'a super::PackedMatrix,
}
impl DeltaParameters {
    fn primitive(self) -> ops::DeltaParameterPrimitive<Dispatch> {
        ops::DeltaParameterPrimitive {
            convolution: self.convolution.into_dispatch(),
            log_decay: self.log_decay.into_dispatch(),
            time_bias: self.time_bias.into_dispatch(),
            norm: self.norm.into_dispatch(),
        }
    }
}
