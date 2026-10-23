//! Single-sequence transformer operations. Persistent state stays in backend tensors.
use crate::{Backend, Error, Result, Tensor, element_count};

#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub epsilon: f32,
}

pub struct AttentionState<B: Backend> {
    /// [capacity, kv_heads, head_dim]; only the first `position` entries are valid.
    pub keys: Tensor<B>,
    pub values: Tensor<B>,
    pub position: usize,
}

pub struct AttentionInputs<'a, B: Backend> {
    /// [rows, 2 * heads * head_dim], query followed by gate within each head.
    pub query_gate: &'a Tensor<B>,
    pub key: &'a Tensor<B>,
    pub value: &'a Tensor<B>,
    pub query_norm: &'a Tensor<B>,
    pub key_norm: &'a Tensor<B>,
}

#[derive(Clone, Copy, Debug)]
pub struct DeltaConfig {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kernel: usize,
    pub epsilon: f32,
}

pub struct DeltaState<B: Backend> {
    /// [2 * heads * key_dim + heads * value_dim, kernel], oldest to newest.
    pub convolution: Tensor<B>,
    /// [heads, key_dim, value_dim].
    pub recurrent: Tensor<B>,
}

pub struct DeltaInputs<'a, B: Backend> {
    /// [rows, heads * (2 * key_dim + value_dim)].
    pub qkv: &'a Tensor<B>,
    /// [rows, heads * value_dim].
    pub gate: &'a Tensor<B>,
    /// [rows, heads].
    pub beta: &'a Tensor<B>,
    /// [rows, heads].
    pub alpha: &'a Tensor<B>,
    pub convolution: &'a Tensor<B>,
    pub log_decay: &'a Tensor<B>,
    pub time_bias: &'a Tensor<B>,
    /// Ordinary RMSNorm weights, unlike the zero-centered transformer norms.
    pub norm: &'a Tensor<B>,
}

/// No operation downloads model activations to the host. Implementations own all arithmetic.
/// Attention methods validate shapes and capacity before changing their cache.
pub trait TransformerBackend: Backend {
    fn embedding(&self, table: &Tensor<Self>, token: usize) -> Result<Tensor<Self>>;
    fn embedding_batch(&self, table: &Tensor<Self>, tokens: &[u32]) -> Result<Tensor<Self>>;
    /// Return an independently owned row, with shape [1, width].
    fn row(&self, input: &Tensor<Self>, row: usize) -> Result<Tensor<Self>>;
    fn project(&self, input: &Tensor<Self>, weights: &Tensor<Self>) -> Result<Tensor<Self>>;
    fn project_add(&self, input: &Tensor<Self>, weights: &Tensor<Self>, residual: Tensor<Self>) -> Result<Tensor<Self>> {
        let projected = self.project(input, weights)?;
        self.add(residual, &projected)
    }
    /// SiLU(input * gateᵀ) * (input * upᵀ), with no host activation transfer.
    fn gated_project(&self, input: &Tensor<Self>, gate: &Tensor<Self>, up: &Tensor<Self>) -> Result<Tensor<Self>> {
        let gate = self.project(input, gate)?;
        let up = self.project(input, up)?;
        self.silu_mul(gate, &up)
    }
    fn rms_norm(&self, input: &Tensor<Self>, weights: &Tensor<Self>, epsilon: f32) -> Result<Tensor<Self>>;
    fn add(&self, input: Tensor<Self>, residual: &Tensor<Self>) -> Result<Tensor<Self>>;
    /// Return the updated residual stream and its zero-centered RMSNorm.
    fn residual_norm(&self, input: Tensor<Self>, residual: &Tensor<Self>, weights: &Tensor<Self>, epsilon: f32) -> Result<(Tensor<Self>, Tensor<Self>)> {
        let input = self.add(input, residual)?;
        let normalized = self.rms_norm(&input, weights, epsilon)?;
        Ok((input, normalized))
    }
    fn silu_mul(&self, input: Tensor<Self>, other: &Tensor<Self>) -> Result<Tensor<Self>>;
    fn attention_step(&self, input: AttentionInputs<'_, Self>, state: &mut AttentionState<Self>, config: AttentionConfig) -> Result<Tensor<Self>> {
        if input.query_gate.shape().first() != Some(&1) {
            return Err(Error::ShapeMismatch);
        }
        self.attention_batch(input, state, config)
    }
    fn delta_step(&self, input: DeltaInputs<'_, Self>, state: &mut DeltaState<Self>, config: DeltaConfig) -> Result<Tensor<Self>> {
        if input.qkv.shape().first() != Some(&1) {
            return Err(Error::ShapeMismatch);
        }
        self.delta_batch(input, state, config)
    }
    /// Consecutive token rows from one sequence; cache updates remain ordered.
    fn attention_batch(&self, input: AttentionInputs<'_, Self>, state: &mut AttentionState<Self>, config: AttentionConfig) -> Result<Tensor<Self>>;
    fn delta_batch(&self, input: DeltaInputs<'_, Self>, state: &mut DeltaState<Self>, config: DeltaConfig) -> Result<Tensor<Self>>;
}

pub(crate) fn attention_shape<B: Backend>(input: &AttentionInputs<'_, B>, state: &AttentionState<B>, config: AttentionConfig) -> Result<(usize, usize, usize)> {
    let AttentionConfig {
        heads,
        kv_heads,
        head_dim: dim,
        rotary_dim,
        rope_theta,
        epsilon,
    } = config;
    let q_width = element_count(&[heads, dim])?;
    let kv_width = element_count(&[kv_heads, dim])?;
    let qg_width = element_count(&[2, q_width])?;
    let [rows, width] = *input.query_gate.shape() else {
        return Err(Error::ShapeMismatch);
    };
    let [capacity, cache_heads, cache_dim] = *state.keys.shape() else {
        return Err(Error::ShapeMismatch);
    };
    if heads % kv_heads != 0
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > dim
        || !rope_theta.is_finite()
        || rope_theta <= 0.
        || !epsilon.is_finite()
        || epsilon <= 0.
    {
        return Err(Error::InvalidParameter);
    }
    if width != qg_width
        || input.key.shape() != [rows, kv_width]
        || input.value.shape() != [rows, kv_width]
        || input.query_norm.shape() != [dim]
        || input.key_norm.shape() != [dim]
        || cache_heads != kv_heads
        || cache_dim != dim
        || state.values.shape() != state.keys.shape()
    {
        return Err(Error::ShapeMismatch);
    }
    if state.position.checked_add(rows).is_none_or(|end| end > capacity) {
        return Err(Error::InvalidParameter);
    }
    Ok((rows, q_width, kv_width))
}

pub(crate) fn delta_shape<B: Backend>(input: &DeltaInputs<'_, B>, state: &DeltaState<B>, config: DeltaConfig) -> Result<(usize, usize, usize)> {
    let DeltaConfig {
        heads,
        key_dim,
        value_dim,
        kernel,
        epsilon,
    } = config;
    let key_width = element_count(&[heads, key_dim])?;
    let value_width = element_count(&[heads, value_dim])?;
    let channels = key_width.checked_mul(2).and_then(|n| n.checked_add(value_width)).ok_or(Error::SizeOverflow)?;
    element_count(&[channels, kernel])?;
    element_count(&[heads, key_dim, value_dim])?;
    let [rows, width] = *input.qkv.shape() else {
        return Err(Error::ShapeMismatch);
    };
    if !epsilon.is_finite() || epsilon <= 0. {
        return Err(Error::InvalidParameter);
    }
    if width != channels
        || input.gate.shape() != [rows, value_width]
        || input.beta.shape() != [rows, heads]
        || input.alpha.shape() != [rows, heads]
        || input.convolution.shape() != [channels, 1, kernel]
        || input.log_decay.shape() != [heads]
        || input.time_bias.shape() != [heads]
        || input.norm.shape() != [value_dim]
        || state.convolution.shape() != [channels, kernel]
        || state.recurrent.shape() != [heads, key_dim, value_dim]
    {
        return Err(Error::ShapeMismatch);
    }
    Ok((rows, channels, value_width))
}
