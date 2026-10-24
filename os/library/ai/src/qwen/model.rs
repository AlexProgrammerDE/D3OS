//! Burn model composition and independent per-sequence state.
use super::{
    AttentionInputs, AttentionState, Config, DeltaParameters, DeltaState, DeltaWeights, MatrixFormat, PackedMatrix, device_supported, residual_norm, rms_norm,
};
use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    vec::Vec,
};
use burn::{
    module::{Module, Param},
    tensor::{Device, Tensor, TensorData},
};
use d3os_tensor_file::{Archive, Record};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    InvalidArchive,
    MissingTensor(String),
    UnexpectedTensor,
    InvalidModelShape,
    InvalidConfig,
    InvalidToken,
    InvalidTokenizer,
    ContextFull,
    SessionFailed,
    VerificationFailed,
    VerificationAt { position: usize, layer: usize },
    SourceRead,
    NonFinite,
    AllocationFailed,
    Execution(String),
    DevicePoisoned(String),
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidArchive => f.write_str("invalid or unsupported model file"),
            Self::MissingTensor(name) => write!(f, "model is missing tensor {name}"),
            Self::UnexpectedTensor => f.write_str("model contains unexpected tensors"),
            Self::InvalidModelShape => f.write_str("model tensor has an invalid shape"),
            Self::InvalidConfig => f.write_str("invalid Qwen configuration"),
            Self::InvalidToken => f.write_str("invalid token ID or generation length"),
            Self::InvalidTokenizer => f.write_str("invalid or unsupported tokenizer"),
            Self::ContextFull => f.write_str("sequence exceeds the session capacity"),
            Self::SessionFailed => f.write_str("session failed; create a new session before continuing"),
            Self::VerificationFailed => f.write_str("inference output differs from the reference"),
            Self::VerificationAt { position, layer } => write!(f, "Qwen reference mismatch at position {position}, layer {layer}"),
            Self::SourceRead => f.write_str("cannot read model data"),
            Self::NonFinite => f.write_str("model data or logits contain a nonfinite value"),
            Self::AllocationFailed => f.write_str("cannot allocate model data"),
            Self::Execution(reason) | Self::DevicePoisoned(reason) => f.write_str(reason),
        }
    }
}
impl core::error::Error for Error {}
impl From<burn::tensor::TensorReadError> for Error {
    fn from(value: burn::tensor::TensorReadError) -> Self {
        match value {
            burn::tensor::TensorReadError::Execution(error) => error.into(),
            error => Self::Execution(format!("Qwen tensor read: {error:?}")),
        }
    }
}
impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Self::Execution(value.into())
    }
}
impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::Execution(value)
    }
}
impl From<burn::backend::ExecutionError> for Error {
    fn from(value: burn::backend::ExecutionError) -> Self {
        let reason = format!("Qwen execution: {value:?}");
        if value.is_device_poisoned() {
            Self::DevicePoisoned(reason)
        } else {
            Self::Execution(reason)
        }
    }
}
#[cfg(feature = "native")]
impl From<crate::native::NativeError> for Error {
    fn from(value: crate::native::NativeError) -> Self {
        let reason = format!("Qwen native execution: {value}");
        if value == crate::native::NativeError::DeviceFault {
            Self::DevicePoisoned(reason)
        } else {
            Self::Execution(reason)
        }
    }
}
impl From<Error> for String {
    fn from(value: Error) -> Self {
        value.to_string()
    }
}
impl From<d3os_tensor_file::Error> for Error {
    fn from(value: d3os_tensor_file::Error) -> Self {
        use d3os_tensor_file::Error as E;
        match value {
            E::MissingTensor(name) => Self::MissingTensor(name),
            E::UnexpectedTensor => Self::UnexpectedTensor,
            E::InvalidShape | E::ShapeMismatch | E::SizeOverflow => Self::InvalidModelShape,
            E::NonFinite => Self::NonFinite,
            E::AllocationFailed => Self::AllocationFailed,
            E::InvalidArchive => Self::InvalidArchive,
        }
    }
}
pub type Result<T> = core::result::Result<T, Error>;

/// One encoded weight from a streaming source. Model loading owns the Burn upload.
pub struct WeightData<'a> {
    pub shape: Vec<usize>,
    pub format: MatrixFormat,
    pub bytes: Cow<'a, [u8]>,
}
impl<'a> WeightData<'a> {
    pub fn from_record(record: Record<'a>) -> Self {
        let (shape, dtype, bytes) = record.into_parts();
        Self {
            shape,
            format: match dtype {
                d3os_tensor_file::DType::F32 => MatrixFormat::F32,
                d3os_tensor_file::DType::Bf16 => MatrixFormat::Bf16,
            },
            bytes: Cow::Borrowed(bytes),
        }
    }
    fn float<const D: usize>(self, device: &Device) -> Result<Tensor<D>> {
        if self.shape.len() != D || matches!(self.format, MatrixFormat::Quantized(_)) {
            return Err(Error::InvalidModelShape);
        }
        let elements = d3os_tensor_file::element_count(&self.shape)?;
        let values = self.format.decode([1, elements], &self.bytes)?;
        Ok(Tensor::from_data(TensorData::new(values, self.shape), device))
    }
}

#[derive(Module, Debug)]
struct FullAttention {
    q: PackedMatrix,
    k: PackedMatrix,
    v: PackedMatrix,
    out: PackedMatrix,
    q_norm: Param<Tensor<1>>,
    k_norm: Param<Tensor<1>>,
}
#[derive(Module, Debug)]
struct LinearAttention {
    #[module(skip)]
    output_heads_tiled: bool,
    qkv: PackedMatrix,
    z: PackedMatrix,
    beta: PackedMatrix,
    alpha: PackedMatrix,
    conv: Param<Tensor<3>>,
    log_decay: Param<Tensor<1>>,
    time_bias: Param<Tensor<1>>,
    norm: Param<Tensor<1>>,
    out: PackedMatrix,
}
// Burn 0.22.1 does not implement Module for Box. Keep the typed module variants
// inline so all parameters participate in records, device moves, and visitors.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum Mixer {
    Full(FullAttention),
    Linear(LinearAttention),
}
#[derive(Module, Debug)]
struct Layer {
    input_norm: Param<Tensor<1>>,
    post_norm: Param<Tensor<1>>,
    gate: PackedMatrix,
    up: PackedMatrix,
    down: PackedMatrix,
    mixer: Mixer,
}

#[derive(Module, Debug)]
pub struct Qwen {
    #[module(skip)]
    config: Config,
    embedding: PackedMatrix,
    output: Option<PackedMatrix>,
    norm: Param<Tensor<1>>,
    layers: Vec<Layer>,
}

struct Loader<'a, F> {
    source: F,
    device: &'a Device,
}
impl<'a, 'data, F: FnMut(&str, &[usize]) -> Result<WeightData<'data>>> Loader<'a, F> {
    fn data(&mut self, name: &str, shape: &[usize]) -> Result<WeightData<'data>> {
        let data = (self.source)(name, shape)?;
        if data.shape != shape {
            return Err(Error::InvalidModelShape);
        }
        Ok(data)
    }
    fn matrix(&mut self, name: &str, shape: [usize; 2]) -> Result<PackedMatrix> {
        let data = self.data(name, &shape)?;
        #[cfg(feature = "cpu")]
        if *self.device == Device::flex() && data.format == MatrixFormat::Bf16 {
            // Flex projections accumulate in FP32. Expand constant BF16
            // weights once at import, rather than on every generated token.
            let values = data.format.decode(shape, &data.bytes)?;
            let bytes = TensorData::new(values, shape);
            return PackedMatrix::from_bytes(shape, MatrixFormat::F32, bytes.as_bytes(), self.device);
        }
        PackedMatrix::from_bytes(shape, data.format, &data.bytes, self.device)
    }
    fn tensor<const D: usize>(&mut self, name: &str, shape: [usize; D]) -> Result<Param<Tensor<D>>> {
        let data = self.data(name, &shape)?;
        Ok(Param::from_tensor(data.float(self.device)?))
    }
}
fn scope<T>(device: &Device, work: impl FnOnce() -> Result<T>) -> Result<T> {
    #[cfg(feature = "native")]
    if matches!(device.as_dispatch(), burn::backend::DispatchDevice::Native(_)) {
        return crate::native::NativeDevice.with_commands(work);
    }
    let value = work()?;
    device.sync()?;
    Ok(value)
}
impl Qwen {
    pub fn load(device: &Device, bytes: &[u8]) -> Result<Self> {
        let mut archive = Archive::parse(bytes)?;
        let metadata = archive.take("qwen3_5.config")?;
        if metadata.shape() != [16] && metadata.shape() != [18] {
            return Err(Error::InvalidConfig);
        }
        let config = Config::parse(&metadata.to_vec()?)?;
        let model = Self::load_weights(config, device, |name, _| Ok(WeightData::from_record(archive.take(name)?)))?;
        archive.finish()?;
        Ok(model)
    }
    pub fn load_weights<'a>(config: Config, device: &Device, source: impl FnMut(&str, &[usize]) -> Result<WeightData<'a>>) -> Result<Self> {
        config.validate()?;
        device_supported(device)?;
        let mut load = Loader { source, device };
        let embedding = load.matrix("embed_tokens.weight", [config.vocab, config.hidden])?;
        let output = if config.tied_embeddings {
            None
        } else {
            Some(load.matrix("lm_head.weight", [config.vocab, config.hidden])?)
        };
        let norm = load.tensor("norm.weight", [config.hidden])?;
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let prefix = format!("layers.{index}");
            let input_norm = load.tensor(&format!("{prefix}.input_layernorm.weight"), [config.hidden])?;
            let post_norm = load.tensor(&format!("{prefix}.post_attention_layernorm.weight"), [config.hidden])?;
            let gate = load.matrix(&format!("{prefix}.mlp.gate_proj.weight"), [config.intermediate, config.hidden])?;
            let up = load.matrix(&format!("{prefix}.mlp.up_proj.weight"), [config.intermediate, config.hidden])?;
            let down = load.matrix(&format!("{prefix}.mlp.down_proj.weight"), [config.hidden, config.intermediate])?;
            let mixer = if (index + 1) % config.full_attention_interval == 0 {
                Mixer::Full(FullAttention {
                    q: load.matrix(
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        [2 * config.heads * config.head_dim, config.hidden],
                    )?,
                    k: load.matrix(&format!("{prefix}.self_attn.k_proj.weight"), [config.kv_heads * config.head_dim, config.hidden])?,
                    v: load.matrix(&format!("{prefix}.self_attn.v_proj.weight"), [config.kv_heads * config.head_dim, config.hidden])?,
                    out: load.matrix(&format!("{prefix}.self_attn.o_proj.weight"), [config.hidden, config.heads * config.head_dim])?,
                    q_norm: load.tensor(&format!("{prefix}.self_attn.q_norm.weight"), [config.head_dim])?,
                    k_norm: load.tensor(&format!("{prefix}.self_attn.k_norm.weight"), [config.head_dim])?,
                })
            } else {
                Mixer::Linear(LinearAttention {
                    output_heads_tiled: false,
                    qkv: load.matrix(&format!("{prefix}.linear_attn.in_proj_qkv.weight"), [config.conv_width(), config.hidden])?,
                    z: load.matrix(&format!("{prefix}.linear_attn.in_proj_z.weight"), [config.linear_width(), config.hidden])?,
                    beta: load.matrix(&format!("{prefix}.linear_attn.in_proj_b.weight"), [config.linear_heads, config.hidden])?,
                    alpha: load.matrix(&format!("{prefix}.linear_attn.in_proj_a.weight"), [config.linear_heads, config.hidden])?,
                    conv: load.tensor(
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        [config.conv_width(), 1, config.convolution_kernel],
                    )?,
                    log_decay: load.tensor(&format!("{prefix}.linear_attn.A_log"), [config.linear_heads])?,
                    time_bias: load.tensor(&format!("{prefix}.linear_attn.dt_bias"), [config.linear_heads])?,
                    norm: load.tensor(&format!("{prefix}.linear_attn.norm.weight"), [config.linear_value_dim])?,
                    out: load.matrix(&format!("{prefix}.linear_attn.out_proj.weight"), [config.hidden, config.linear_width()])?,
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
        device.sync()?;
        Ok(Self {
            config,
            embedding,
            output,
            norm,
            layers,
        }
        .valid())
    }
    fn output_weights(&self) -> &PackedMatrix {
        self.output.as_ref().unwrap_or(&self.embedding)
    }
    pub(crate) fn with_tiled_linear_outputs(mut self) -> Self {
        for layer in &mut self.layers {
            if let Mixer::Linear(weights) = &mut layer.mixer {
                weights.output_heads_tiled = true;
            }
        }
        self
    }
    pub fn config(&self) -> Config {
        self.config
    }
    pub fn device(&self) -> Device {
        self.norm.val().device()
    }
    pub fn session(&self, capacity: usize) -> Result<Session<'_>> {
        Session::new(self, capacity)
    }
    pub fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Result<Generation> {
        self.generate_stream(prompt, max_new_tokens, |_| Ok(()))
    }
    pub fn generate_stream(&self, prompt: &[u32], max_new_tokens: usize, mut emit: impl FnMut(u32) -> Result<()>) -> Result<Generation> {
        if prompt.is_empty() || max_new_tokens == 0 {
            return Err(Error::InvalidToken);
        }
        let capacity = prompt.len().checked_add(max_new_tokens).ok_or(Error::ContextFull)?;
        let mut session = self.session(capacity)?;
        let mut logits = session.prefill(prompt)?;
        let mut tokens = Vec::new();
        for index in 0..max_new_tokens {
            let token = greedy(&logits)?;
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
enum Cache {
    Full(AttentionState),
    Linear(DeltaState),
}
pub struct Session<'a> {
    model: &'a Qwen,
    caches: Vec<Option<Cache>>,
    capacity: usize,
    position: usize,
    failed: bool,
}
impl<'a> Session<'a> {
    fn new(model: &'a Qwen, capacity: usize) -> Result<Self> {
        let config = model.config;
        if capacity == 0 || capacity > config.max_positions {
            return Err(Error::ContextFull);
        }
        let device = model.device();
        device_supported(&device)?;
        let caches = scope(&device, || {
            model
                .layers
                .iter()
                .map(|layer| {
                    Ok(Some(match layer.mixer {
                        Mixer::Full(_) => Cache::Full(AttentionState::new(config.attention(), capacity, &device)?),
                        Mixer::Linear(_) => Cache::Linear(DeltaState::new(config.delta(), &device)?),
                    }))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(Self {
            model,
            caches,
            capacity,
            position: 0,
            failed: false,
        })
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor<2>> {
        self.prefill_observed(tokens, |_, _, _| Ok(()))
    }
    pub fn prefill_observed(&mut self, tokens: &[u32], mut observe: impl FnMut(usize, usize, &Tensor<2>) -> Result<()>) -> Result<Tensor<2>> {
        self.validate(tokens)?;
        let mut hidden = None;
        for chunk in tokens.chunks(32) {
            let start = self.position;
            hidden = Some(self.hidden_batch(chunk, |layer, output| observe(start, layer, output))?);
        }
        let hidden = hidden.ok_or(Error::InvalidToken)?;
        let [rows, width] = hidden.dims();
        self.logits(hidden.slice([rows - 1..rows, 0..width]))
    }
    pub fn step(&mut self, token: u32) -> Result<Tensor<2>> {
        self.step_observed(token, |_, _| Ok(()))
    }
    pub fn step_observed(&mut self, token: u32, observe: impl FnMut(usize, &Tensor<2>) -> Result<()>) -> Result<Tensor<2>> {
        let hidden = self.hidden_batch(&[token], observe)?;
        self.logits(hidden)
    }
    fn logits(&mut self, hidden: Tensor<2>) -> Result<Tensor<2>> {
        self.failed = true;
        let device = self.model.device();
        let logits = scope(&device, || {
            self.model
                .output_weights()
                .project(rms_norm(hidden, self.model.norm.val(), self.model.config.epsilon)?)
        })?;
        self.failed = false;
        Ok(logits)
    }
    fn validate(&self, tokens: &[u32]) -> Result<()> {
        if self.failed {
            return Err(Error::SessionFailed);
        }
        if tokens.is_empty() || tokens.len() > self.capacity - self.position {
            return Err(Error::ContextFull);
        }
        if tokens.iter().any(|&token| token as usize >= self.model.config.vocab) {
            return Err(Error::InvalidToken);
        }
        Ok(())
    }
    fn hidden_batch(&mut self, tokens: &[u32], mut observe: impl FnMut(usize, &Tensor<2>) -> Result<()>) -> Result<Tensor<2>> {
        self.validate(tokens)?;
        self.failed = true;
        let config = self.model.config;
        let device = self.model.device();
        let hidden = scope(&device, || {
            let mut hidden = self.model.embedding.embedding(tokens)?;
            for (index, (layer, slot)) in self.model.layers.iter().zip(&mut self.caches).enumerate() {
                let normalized = rms_norm(hidden.clone(), layer.input_norm.val(), config.epsilon)?;
                let cache = slot.take().ok_or(Error::SessionFailed)?;
                let mixed = match (&layer.mixer, cache) {
                    (Mixer::Full(weights), Cache::Full(state)) => {
                        let (state, attention) = state.forward(
                            AttentionInputs {
                                query_gate: weights.q.project(normalized.clone())?,
                                key: weights.k.project(normalized.clone())?,
                                value: weights.v.project(normalized)?,
                                query_norm: weights.q_norm.val(),
                                key_norm: weights.k_norm.val(),
                            },
                            config.attention(),
                        )?;
                        *slot = Some(Cache::Full(state));
                        weights.out.project(attention)?
                    }
                    (Mixer::Linear(weights), Cache::Linear(state)) => {
                        let (state, attention) = state.project(
                            normalized,
                            DeltaWeights {
                                qkv: &weights.qkv,
                                gate: &weights.z,
                                beta: &weights.beta,
                                alpha: &weights.alpha,
                            },
                            DeltaParameters {
                                convolution: weights.conv.val(),
                                log_decay: weights.log_decay.val(),
                                time_bias: weights.time_bias.val(),
                                norm: weights.norm.val(),
                            },
                            config.delta(),
                        )?;
                        *slot = Some(Cache::Linear(state));
                        // GGUF keeps output-projection columns in tiled V-head
                        // order. Permute the small activation instead of
                        // decoding or requantizing packed weight blocks.
                        let attention = if weights.output_heads_tiled {
                            attention
                                .reshape([
                                    tokens.len(),
                                    config.linear_key_heads,
                                    config.linear_heads / config.linear_key_heads,
                                    config.linear_value_dim,
                                ])
                                .swap_dims(1, 2)
                                .reshape([tokens.len(), config.linear_width()])
                        } else {
                            attention
                        };
                        weights.out.project(attention)?
                    }
                    _ => return Err(Error::InvalidConfig),
                };
                let (updated, normalized) = residual_norm(hidden, mixed, layer.post_norm.val(), config.epsilon)?;
                let activation = layer.gate.gated_project(&layer.up, normalized)?;
                hidden = layer.down.project_add(activation, updated)?;
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
pub fn greedy(logits: &Tensor<2>) -> Result<u32> {
    if logits.dims()[0] != 1 || logits.dims()[1] == 0 {
        return Err(Error::InvalidModelShape);
    }
    if !logits.clone().is_finite().all().try_into_scalar::<bool>()? {
        return Err(Error::NonFinite);
    }
    let index = logits.clone().argmax(1).try_into_scalar::<i32>()?;
    u32::try_from(index).map_err(|_| Error::InvalidToken)
}
pub fn verify(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) -> Result<()> {
    if actual.len() != expected.len() || actual.is_empty() || !absolute.is_finite() || !relative.is_finite() || absolute < 0. || relative < 0. {
        return Err(Error::VerificationFailed);
    }
    if actual
        .iter()
        .zip(expected)
        .any(|(&a, &e)| !a.is_finite() || !e.is_finite() || (a - e).abs() > absolute + relative * e.abs())
    {
        return Err(Error::VerificationFailed);
    }
    Ok(())
}

fn read<const D: usize>(tensor: &Tensor<D>) -> Result<Vec<f32>> {
    Ok(tensor.clone().try_into_vec_as::<f32>()?)
}
/// Compare sequential and batched execution with every fixture layer and logit,
/// then compare their final caches. Upstream fixtures can mix prefill and decode.
pub fn verify_fixture(model: &Qwen, bytes: &[u8]) -> Result<usize> {
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
    let mut session = model.session(ids.len())?;
    for (position, token) in ids.iter().enumerate() {
        let output = session.step_observed(*token, |layer, output| {
            let expected = &references[layer][position * model.config.hidden..(position + 1) * model.config.hidden];
            verify(&read(output)?, expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt { position, layer })
        })?;
        let expected = &logits[position * model.config.vocab..(position + 1) * model.config.vocab];
        let actual = read(&output)?;
        verify(&actual, expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt {
            position,
            layer: model.config.layers,
        })?;
        // Check greedy decisions too, including closely ranked vocabulary entries.
        let expected_best = reference_token(expected);
        if Some(greedy(&output)? as usize) != expected_best {
            return Err(Error::VerificationAt {
                position,
                layer: model.config.layers,
            });
        }
    }
    let mut batched = model.session(ids.len())?;
    // Different chunk sizes exercise continuation after a nonzero cache prefix.
    for chunk in ids.chunks(17) {
        let start = batched.position();
        let final_logits = batched.prefill_observed(chunk, |position, layer, output| {
            let values = read(output)?;
            for (row, actual) in values.chunks_exact(model.config.hidden).enumerate() {
                let position = position + row;
                verify(
                    actual,
                    &references[layer][position * model.config.hidden..(position + 1) * model.config.hidden],
                    2e-3,
                    5e-4,
                )
                .map_err(|_| Error::VerificationAt { position, layer })?;
            }
            if layer + 1 == model.config.layers {
                let normalized = rms_norm(output.clone(), model.norm.val(), model.config.epsilon)?;
                let batch_logits = model.output_weights().project(normalized)?;
                let actual = read(&batch_logits)?;
                for (row, actual) in actual.chunks_exact(model.config.vocab).enumerate() {
                    let position = position + row;
                    let expected = &logits[position * model.config.vocab..(position + 1) * model.config.vocab];
                    verify(actual, expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt {
                        position,
                        layer: model.config.layers,
                    })?;
                    let expected_best = reference_token(expected);
                    let selected = Ok::<_, Error>(batch_logits.clone().slice([row..row + 1, 0..model.config.vocab]))?;
                    if Some(greedy(&selected)? as usize) != expected_best {
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
        verify(&read(&final_logits)?, expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt {
            position,
            layer: model.config.layers,
        })?;
        if Some(greedy(&final_logits)? as usize) != reference_token(expected) {
            return Err(Error::VerificationAt {
                position,
                layer: model.config.layers,
            });
        }
    }
    for (layer, (actual, expected)) in batched.caches.iter().zip(&session.caches).enumerate() {
        let tensors = match (actual, expected) {
            (Some(Cache::Full(a)), Some(Cache::Full(e))) if a.position() == e.position() => {
                [(read(a.keys())?, read(e.keys())?), (read(a.values())?, read(e.values())?)]
            }
            (Some(Cache::Linear(a)), Some(Cache::Linear(e))) => {
                [(read(a.convolution())?, read(e.convolution())?), (read(a.recurrent())?, read(e.recurrent())?)]
            }
            _ => {
                return Err(Error::VerificationAt {
                    position: ids.len() - 1,
                    layer,
                });
            }
        };
        for (actual, expected) in tensors {
            verify(&actual, &expected, 2e-3, 5e-4).map_err(|_| Error::VerificationAt {
                position: ids.len() - 1,
                layer,
            })?;
        }
    }
    Ok(ids.len())
}
