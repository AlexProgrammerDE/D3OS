//! Qwen3.5 text decoder, with per-session convolution, recurrent, and KV caches.
//! The backend performs every numerical operation; the model only schedules them.
use alloc::{format, vec::Vec};
use d3os_compute::transformer::{AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaState, TransformerBackend};
use d3os_compute::{Backend, Tensor, element_count};

use crate::{Error, Result, archive::Archive};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub linear_heads: usize,
    pub linear_key_dim: usize,
    pub linear_value_dim: usize,
    pub convolution_kernel: usize,
    pub full_attention_interval: usize,
    pub max_positions: usize,
    pub rope_theta: f32,
    pub epsilon: f32,
}

impl Config {
    pub const SMALL: Self = Self {
        hidden: 1024,
        intermediate: 3584,
        layers: 24,
        vocab: 248320,
        heads: 8,
        kv_heads: 2,
        head_dim: 256,
        rotary_dim: 64,
        linear_heads: 16,
        linear_key_dim: 128,
        linear_value_dim: 128,
        convolution_kernel: 4,
        full_attention_interval: 4,
        max_positions: 262144,
        rope_theta: 10000000.,
        epsilon: 1e-6,
    };

    fn parse(values: &[f32]) -> Result<Self> {
        if values.len() != 16 {
            return Err(Error::InvalidConfig);
        }
        let mut dims = [0usize; 14];
        for (dim, &value) in dims.iter_mut().zip(values) {
            if !value.is_finite() || !(1. ..=1_048_576.).contains(&value) || value != (value as usize) as f32 {
                return Err(Error::InvalidConfig);
            }
            *dim = value as usize;
        }
        let [
            hidden,
            intermediate,
            layers,
            vocab,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            linear_heads,
            linear_key_dim,
            linear_value_dim,
            convolution_kernel,
            full_attention_interval,
            max_positions,
        ] = dims;
        let config = Self {
            hidden,
            intermediate,
            layers,
            vocab,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            linear_heads,
            linear_key_dim,
            linear_value_dim,
            convolution_kernel,
            full_attention_interval,
            max_positions,
            rope_theta: values[14],
            epsilon: values[15],
        };
        if layers > 128
            || heads % kv_heads != 0
            || rotary_dim > head_dim
            || rotary_dim % 2 != 0
            || !config.rope_theta.is_finite()
            || config.rope_theta <= 0.
            || !config.epsilon.is_finite()
            || config.epsilon <= 0.
        {
            return Err(Error::InvalidConfig);
        }
        element_count(&[vocab, hidden])?;
        element_count(&[heads, head_dim, 2])?;
        element_count(&[linear_heads, linear_key_dim, linear_value_dim])?;
        Ok(config)
    }

    fn attention(self) -> AttentionConfig {
        AttentionConfig {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
            rope_theta: self.rope_theta,
            epsilon: self.epsilon,
        }
    }
    fn delta(self) -> DeltaConfig {
        DeltaConfig {
            heads: self.linear_heads,
            key_dim: self.linear_key_dim,
            value_dim: self.linear_value_dim,
            kernel: self.convolution_kernel,
            epsilon: self.epsilon,
        }
    }
    fn linear_width(self) -> usize {
        self.linear_heads * self.linear_value_dim
    }
    fn conv_width(self) -> usize {
        self.linear_heads * (2 * self.linear_key_dim + self.linear_value_dim)
    }
}

struct FullAttention<B: Backend> {
    q: Tensor<B>,
    k: Tensor<B>,
    v: Tensor<B>,
    out: Tensor<B>,
    q_norm: Tensor<B>,
    k_norm: Tensor<B>,
}
struct LinearAttention<B: Backend> {
    qkv: Tensor<B>,
    z: Tensor<B>,
    beta: Tensor<B>,
    alpha: Tensor<B>,
    conv: Tensor<B>,
    log_decay: Tensor<B>,
    time_bias: Tensor<B>,
    norm: Tensor<B>,
    out: Tensor<B>,
}
enum Mixer<B: Backend> {
    Full(FullAttention<B>),
    Linear(LinearAttention<B>),
}
struct Layer<B: Backend> {
    input_norm: Tensor<B>,
    post_norm: Tensor<B>,
    gate: Tensor<B>,
    up: Tensor<B>,
    down: Tensor<B>,
    mixer: Mixer<B>,
}

pub struct Qwen<B: TransformerBackend> {
    config: Config,
    embedding: Tensor<B>,
    norm: Tensor<B>,
    layers: Vec<Layer<B>>,
}

fn load<B: Backend>(archive: &mut Archive<'_>, backend: &B, name: &str, shape: &[usize]) -> Result<Tensor<B>> {
    let record = archive.take(name)?;
    if record.shape() != shape {
        return Err(Error::InvalidModelShape);
    }
    record.load(backend)
}

fn zeros<B: Backend>(backend: &B, shape: &[usize]) -> Result<Tensor<B>> {
    let count = element_count(shape)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| d3os_compute::Error::AllocationFailed)?;
    values.resize(count, 0.);
    Ok(Tensor::from_vec(backend, shape, values)?)
}

impl<B: TransformerBackend> Qwen<B> {
    /// Load text-only Qwen3.5 weights. Word embeddings and the output head are tied.
    pub fn load(backend: &B, bytes: &[u8]) -> Result<Self> {
        let mut archive = Archive::parse(bytes)?;
        let metadata = archive.take("qwen3_5.config")?;
        if metadata.shape() != [16] {
            return Err(Error::InvalidConfig);
        }
        let config = Config::parse(&metadata.to_vec()?)?;
        let embedding = load(&mut archive, backend, "embed_tokens.weight", &[config.vocab, config.hidden])?;
        let norm = load(&mut archive, backend, "norm.weight", &[config.hidden])?;
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let prefix = format!("layers.{index}");
            let mut weight = |suffix: &str, shape: &[usize]| load(&mut archive, backend, &format!("{prefix}.{suffix}"), shape);
            let input_norm = weight("input_layernorm.weight", &[config.hidden])?;
            let post_norm = weight("post_attention_layernorm.weight", &[config.hidden])?;
            let gate = weight("mlp.gate_proj.weight", &[config.intermediate, config.hidden])?;
            let up = weight("mlp.up_proj.weight", &[config.intermediate, config.hidden])?;
            let down = weight("mlp.down_proj.weight", &[config.hidden, config.intermediate])?;
            let mixer = if (index + 1) % config.full_attention_interval == 0 {
                Mixer::Full(FullAttention {
                    q: weight("self_attn.q_proj.weight", &[2 * config.heads * config.head_dim, config.hidden])?,
                    k: weight("self_attn.k_proj.weight", &[config.kv_heads * config.head_dim, config.hidden])?,
                    v: weight("self_attn.v_proj.weight", &[config.kv_heads * config.head_dim, config.hidden])?,
                    out: weight("self_attn.o_proj.weight", &[config.hidden, config.heads * config.head_dim])?,
                    q_norm: weight("self_attn.q_norm.weight", &[config.head_dim])?,
                    k_norm: weight("self_attn.k_norm.weight", &[config.head_dim])?,
                })
            } else {
                Mixer::Linear(LinearAttention {
                    qkv: weight("linear_attn.in_proj_qkv.weight", &[config.conv_width(), config.hidden])?,
                    z: weight("linear_attn.in_proj_z.weight", &[config.linear_width(), config.hidden])?,
                    beta: weight("linear_attn.in_proj_b.weight", &[config.linear_heads, config.hidden])?,
                    alpha: weight("linear_attn.in_proj_a.weight", &[config.linear_heads, config.hidden])?,
                    conv: weight("linear_attn.conv1d.weight", &[config.conv_width(), 1, config.convolution_kernel])?,
                    log_decay: weight("linear_attn.A_log", &[config.linear_heads])?,
                    time_bias: weight("linear_attn.dt_bias", &[config.linear_heads])?,
                    norm: weight("linear_attn.norm.weight", &[config.linear_value_dim])?,
                    out: weight("linear_attn.out_proj.weight", &[config.hidden, config.linear_width()])?,
                })
            };
            layers.push(Layer {
                input_norm,
                post_norm,
                gate,
                up,
                down,
                mixer,
            });
        }
        archive.finish()?;
        Ok(Self {
            config,
            embedding,
            norm,
            layers,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }

    pub fn session<'a>(&'a self, backend: &'a B, capacity: usize) -> Result<Session<'a, B>> {
        Session::new(self, backend, capacity)
    }

    /// Greedy decoding. Call `Session::step` directly to implement other samplers.
    pub fn generate(&self, backend: &B, prompt: &[u32], max_new_tokens: usize) -> Result<Generation> {
        self.generate_stream(backend, prompt, max_new_tokens, |_| Ok(()))
    }

    /// Deliver each selected non-stop token before computing the next one.
    /// Callback errors stop generation and release the session's caches.
    pub fn generate_stream(&self, backend: &B, prompt: &[u32], max_new_tokens: usize, mut emit: impl FnMut(u32) -> Result<()>) -> Result<Generation> {
        if prompt.is_empty() || max_new_tokens == 0 {
            return Err(Error::InvalidToken);
        }
        let capacity = prompt.len().checked_add(max_new_tokens).ok_or(Error::ContextFull)?;
        let mut session = self.session(backend, capacity)?;
        let mut logits = session.prefill(prompt)?;
        let mut tokens = Vec::new();
        for index in 0..max_new_tokens {
            let token = greedy(backend, &logits)?;
            if is_stop(token) {
                return Ok(Generation {
                    tokens,
                    stop: StopReason::EndOfSequence,
                });
            }
            tokens.push(token);
            emit(token)?;
            if index + 1 < max_new_tokens {
                logits = session.step(token)?;
            }
        }
        Ok(Generation {
            tokens,
            stop: StopReason::TokenLimit,
        })
    }
}

enum Cache<B: Backend> {
    Full(AttentionState<B>),
    Linear(DeltaState<B>),
}

/// Independent state for one sequence. A failed backend call invalidates the session.
/// Recreate the session after an error; the model weights remain reusable.
pub struct Session<'a, B: TransformerBackend> {
    model: &'a Qwen<B>,
    backend: &'a B,
    caches: Vec<Cache<B>>,
    capacity: usize,
    position: usize,
    failed: bool,
}

impl<'a, B: TransformerBackend> Session<'a, B> {
    fn new(model: &'a Qwen<B>, backend: &'a B, capacity: usize) -> Result<Self> {
        let config = model.config;
        if capacity == 0 || capacity > config.max_positions {
            return Err(Error::ContextFull);
        }
        let mut caches = Vec::with_capacity(config.layers);
        for layer in &model.layers {
            caches.push(match layer.mixer {
                Mixer::Full(_) => Cache::Full(AttentionState {
                    keys: zeros(backend, &[capacity, config.kv_heads, config.head_dim])?,
                    values: zeros(backend, &[capacity, config.kv_heads, config.head_dim])?,
                    position: 0,
                }),
                Mixer::Linear(_) => Cache::Linear(DeltaState {
                    convolution: zeros(backend, &[config.conv_width(), config.convolution_kernel])?,
                    recurrent: zeros(backend, &[config.linear_heads, config.linear_key_dim, config.linear_value_dim])?,
                }),
            });
        }
        Ok(Self {
            model,
            backend,
            caches,
            capacity,
            position: 0,
            failed: false,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Consume prompt tokens in ordered batches. Project vocabulary logits only for the final token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor<B>> {
        self.prefill_observed(tokens, |_, _, _| Ok(()))
    }

    /// Observe each layer's token batch, with its starting sequence position.
    pub fn prefill_observed(&mut self, tokens: &[u32], mut observe: impl FnMut(usize, usize, &Tensor<B>) -> Result<()>) -> Result<Tensor<B>> {
        if self.failed {
            return Err(Error::SessionFailed);
        }
        if tokens.is_empty() || tokens.len() > self.capacity - self.position {
            return Err(Error::ContextFull);
        }
        if tokens.iter().any(|&token| token as usize >= self.model.config.vocab) {
            return Err(Error::InvalidToken);
        }
        let mut hidden = None;
        for chunk in tokens.chunks(32) {
            let start = self.position;
            hidden = Some(self.hidden_batch(chunk, |layer, output| observe(start, layer, output))?);
        }
        let hidden = hidden.ok_or(Error::InvalidToken)?;
        self.failed = true;
        let last = if hidden.shape()[0] == 1 {
            hidden
        } else {
            let rows = hidden.shape()[0];
            self.backend.row(&hidden, rows - 1)?
        };
        self.logits(&last)
    }

    pub fn step(&mut self, token: u32) -> Result<Tensor<B>> {
        self.step_observed(token, |_, _| Ok(()))
    }

    /// Observe the residual stream after each decoder layer, before the final RMSNorm.
    pub fn step_observed(&mut self, token: u32, observe: impl FnMut(usize, &Tensor<B>) -> Result<()>) -> Result<Tensor<B>> {
        let hidden = self.hidden_batch(&[token], observe)?;
        self.logits(&hidden)
    }

    fn logits(&mut self, hidden: &Tensor<B>) -> Result<Tensor<B>> {
        self.failed = true;
        let normalized = self.backend.rms_norm(hidden, &self.model.norm, self.model.config.epsilon)?;
        let logits = self.backend.project(&normalized, &self.model.embedding)?;
        self.failed = false;
        Ok(logits)
    }

    fn hidden_batch(&mut self, tokens: &[u32], mut observe: impl FnMut(usize, &Tensor<B>) -> Result<()>) -> Result<Tensor<B>> {
        if self.failed {
            return Err(Error::SessionFailed);
        }
        if tokens.is_empty() || tokens.len() > self.capacity - self.position {
            return Err(Error::ContextFull);
        }
        if tokens.iter().any(|&token| token as usize >= self.model.config.vocab) {
            return Err(Error::InvalidToken);
        }
        self.failed = true;
        let backend = self.backend;
        let config = self.model.config;
        let hidden = backend.with_commands(|| -> Result<Tensor<B>> {
            let mut hidden = if tokens.len() == 1 {
                backend.embedding(&self.model.embedding, tokens[0] as usize)?
            } else {
                backend.embedding_batch(&self.model.embedding, tokens)?
            };
            for (index, (layer, cache)) in self.model.layers.iter().zip(&mut self.caches).enumerate() {
                let normalized = backend.rms_norm(&hidden, &layer.input_norm, config.epsilon)?;
                let mixed = match (&layer.mixer, cache) {
                    (Mixer::Full(weights), Cache::Full(state)) => {
                        let q = backend.project(&normalized, &weights.q)?;
                        let k = backend.project(&normalized, &weights.k)?;
                        let v = backend.project(&normalized, &weights.v)?;
                        let attention = backend.attention_batch(
                            AttentionInputs {
                                query_gate: &q,
                                key: &k,
                                value: &v,
                                query_norm: &weights.q_norm,
                                key_norm: &weights.k_norm,
                            },
                            state,
                            config.attention(),
                        )?;
                        backend.project(&attention, &weights.out)?
                    }
                    (Mixer::Linear(weights), Cache::Linear(state)) => {
                        let qkv = backend.project(&normalized, &weights.qkv)?;
                        let z = backend.project(&normalized, &weights.z)?;
                        let beta = backend.project(&normalized, &weights.beta)?;
                        let alpha = backend.project(&normalized, &weights.alpha)?;
                        let attention = backend.delta_batch(
                            DeltaInputs {
                                qkv: &qkv,
                                gate: &z,
                                beta: &beta,
                                alpha: &alpha,
                                convolution: &weights.conv,
                                log_decay: &weights.log_decay,
                                time_bias: &weights.time_bias,
                                norm: &weights.norm,
                            },
                            state,
                            config.delta(),
                        )?;
                        backend.project(&attention, &weights.out)?
                    }
                    _ => return Err(Error::InvalidConfig),
                };
                let (updated, normalized) = backend.residual_norm(hidden, &mixed, &layer.post_norm, config.epsilon)?;
                hidden = updated;
                let activation = backend.gated_project(&normalized, &layer.gate, &layer.up)?;
                hidden = backend.project_add(&activation, &layer.down, hidden)?;
                observe(index, &hidden)?;
            }
            Ok(hidden)
        })?;
        self.position += tokens.len();
        self.failed = false;
        Ok(hidden)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    EndOfSequence,
    TokenLimit,
}
pub struct Generation {
    pub tokens: Vec<u32>,
    pub stop: StopReason,
}
pub fn is_stop(token: u32) -> bool {
    matches!(token, 248044 | 248046)
}

pub fn greedy<B: Backend>(backend: &B, logits: &Tensor<B>) -> Result<u32> {
    if logits.shape().len() != 2 || logits.shape()[0] != 1 {
        return Err(Error::InvalidModelShape);
    }
    u32::try_from(backend.argmax(logits)?).map_err(|_| Error::InvalidToken)
}

/// Compare sequential and batched execution with every fixture layer and logit,
/// then compare their final caches. Upstream fixtures can mix prefill and decode.
pub fn verify_fixture<B: TransformerBackend>(model: &Qwen<B>, backend: &B, bytes: &[u8]) -> Result<usize> {
    let mut archive = Archive::parse(bytes)?;
    let tokens = archive.take("tokens")?;
    let values = tokens.to_vec()?;
    if tokens.shape() != [values.len()] || values.is_empty() {
        return Err(Error::InvalidModelShape);
    }
    let mut ids = Vec::with_capacity(values.len());
    for value in values {
        if value < 0. || value >= model.config.vocab as f32 || value != (value as u32) as f32 {
            return Err(Error::InvalidToken);
        }
        ids.push(value as u32);
    }
    let logits = archive.take("logits")?;
    if logits.shape() != [ids.len(), model.config.vocab] {
        return Err(Error::InvalidModelShape);
    }
    let logits = logits.to_vec()?;
    let mut references = Vec::with_capacity(model.config.layers);
    for layer in 0..model.config.layers {
        let record = archive.take(&format!("layer.{layer}"))?;
        if record.shape() != [ids.len(), model.config.hidden] {
            return Err(Error::InvalidModelShape);
        }
        references.push(record.to_vec()?);
    }
    archive.finish()?;
    let reference_token = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .map(|(i, _)| i)
    };
    let mut session = model.session(backend, ids.len())?;
    for (position, token) in ids.iter().enumerate() {
        let output = session.step_observed(*token, |layer, output| {
            let expected = &references[layer][position * model.config.hidden..(position + 1) * model.config.hidden];
            crate::verify(&backend.download(output)?, expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt { position, layer })
        })?;
        let expected = &logits[position * model.config.vocab..(position + 1) * model.config.vocab];
        let actual = backend.download(&output)?;
        crate::verify(&actual, expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt {
            position,
            layer: model.config.layers,
        })?;
        // Check greedy decisions too, including closely ranked vocabulary entries.
        let expected_best = reference_token(expected);
        if Some(greedy(backend, &output)? as usize) != expected_best {
            return Err(Error::VerificationAt {
                position,
                layer: model.config.layers,
            });
        }
    }
    let mut batched = model.session(backend, ids.len())?;
    // Different chunk sizes exercise continuation after a nonzero cache prefix.
    for chunk in ids.chunks(17) {
        let start = batched.position();
        let final_logits = batched.prefill_observed(chunk, |position, layer, output| {
            let values = backend.download(output)?;
            for (row, actual) in values.chunks_exact(model.config.hidden).enumerate() {
                let position = position + row;
                crate::verify(
                    actual,
                    &references[layer][position * model.config.hidden..(position + 1) * model.config.hidden],
                    2e-3,
                    5e-4,
                )
                .map_err(|_| Error::VerificationAt { position, layer })?;
            }
            if layer + 1 == model.config.layers {
                let normalized = backend.rms_norm(output, &model.norm, model.config.epsilon)?;
                let batch_logits = backend.project(&normalized, &model.embedding)?;
                let actual = backend.download(&batch_logits)?;
                for (row, actual) in actual.chunks_exact(model.config.vocab).enumerate() {
                    let position = position + row;
                    let expected = &logits[position * model.config.vocab..(position + 1) * model.config.vocab];
                    crate::verify(actual, expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt {
                        position,
                        layer: model.config.layers,
                    })?;
                    let expected_best = reference_token(expected);
                    let selected = backend.row(&batch_logits, row)?;
                    if Some(greedy(backend, &selected)? as usize) != expected_best {
                        return Err(Error::VerificationAt {
                            position,
                            layer: model.config.layers,
                        });
                    }
                }
            }
            Ok(())
        })?;
        let position = start + chunk.len() - 1;
        let expected = &logits[position * model.config.vocab..(position + 1) * model.config.vocab];
        crate::verify(&backend.download(&final_logits)?, expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt {
            position,
            layer: model.config.layers,
        })?;
        if Some(greedy(backend, &final_logits)? as usize) != reference_token(expected) {
            return Err(Error::VerificationAt {
                position,
                layer: model.config.layers,
            });
        }
    }
    for (layer, (actual, expected)) in batched.caches.iter().zip(&session.caches).enumerate() {
        let tensors = match (actual, expected) {
            (Cache::Full(a), Cache::Full(e)) if a.position == e.position => [(&a.keys, &e.keys), (&a.values, &e.values)],
            (Cache::Linear(a), Cache::Linear(e)) => [(&a.convolution, &e.convolution), (&a.recurrent, &e.recurrent)],
            _ => {
                return Err(Error::VerificationAt {
                    position: ids.len() - 1,
                    layer,
                });
            }
        };
        for (actual, expected) in tensors {
            crate::verify(&backend.download(actual)?, &backend.download(expected)?, 2e-3, 5e-4).map_err(|_| Error::VerificationAt {
                position: ids.len() - 1,
                layer,
            })?;
        }
    }
    Ok(ids.len())
}
