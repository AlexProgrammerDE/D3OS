//! Single-sequence transformer operations. Persistent state stays in backend tensors.
use crate::{Backend, Result, Tensor};

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
    /// Per head, query followed by gate: [heads, 2 * head_dim].
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
    pub qkv: &'a Tensor<B>,
    pub gate: &'a Tensor<B>,
    pub beta: &'a Tensor<B>,
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
    fn project(&self, input: &Tensor<Self>, weights: &Tensor<Self>) -> Result<Tensor<Self>>;
    /// SiLU(input * gateᵀ) * (input * upᵀ), with no host activation transfer.
    fn gated_project(&self, input: &Tensor<Self>, gate: &Tensor<Self>, up: &Tensor<Self>) -> Result<Tensor<Self>> {
        let gate = self.project(input, gate)?;
        let up = self.project(input, up)?;
        self.silu_mul(gate, &up)
    }
    fn rms_norm(&self, input: &Tensor<Self>, weights: &Tensor<Self>, epsilon: f32) -> Result<Tensor<Self>>;
    fn add(&self, input: Tensor<Self>, residual: &Tensor<Self>) -> Result<Tensor<Self>>;
    fn silu_mul(&self, input: Tensor<Self>, other: &Tensor<Self>) -> Result<Tensor<Self>>;
    fn attention_step(&self, input: AttentionInputs<'_, Self>, state: &mut AttentionState<Self>, config: AttentionConfig) -> Result<Tensor<Self>>;
    fn delta_step(&self, input: DeltaInputs<'_, Self>, state: &mut DeltaState<Self>, config: DeltaConfig) -> Result<Tensor<Self>>;
}
