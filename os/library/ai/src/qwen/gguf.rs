//! Streaming GGUF v3 reader for the Qwen3.5/3.6 text architecture.
pub mod shards;
use super::model::{Error, Result};
use super::{Config, MatrixFormat, Qwen, WeightData};
use alloc::{borrow::Cow, collections::BTreeMap, format, string::String, vec, vec::Vec};
use burn::tensor::Device;
use d3os_tensor_file::element_count;
use d3os_tensor_file::quantized::Quantization;

/// Fill the complete destination or return an error. Implementations can use
/// files, firmware shards, or a memory slice without retaining the full model.
pub trait ReadAt {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_at(&mut self, offset: u64, output: &mut [u8]) -> Result<()>;
}

impl ReadAt for &[u8] {
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }
    fn read_at(&mut self, offset: u64, output: &mut [u8]) -> Result<()> {
        let start = usize::try_from(offset).map_err(|_| Error::SourceRead)?;
        let end = start.checked_add(output.len()).ok_or(Error::SourceRead)?;
        output.copy_from_slice(self.get(start..end).ok_or(Error::SourceRead)?);
        Ok(())
    }
}

struct Reader<R> {
    source: R,
    position: u64,
    cache_start: u64,
    cache_len: usize,
    cache: Vec<u8>,
}

impl<R: ReadAt> Reader<R> {
    fn new(source: R) -> Self {
        Self {
            source,
            position: 0,
            cache_start: 0,
            cache_len: 0,
            cache: vec![0; 65536],
        }
    }
    fn skip(&mut self, size: u64) -> Result<()> {
        self.position = self
            .position
            .checked_add(size)
            .filter(|&end| end <= self.source.len())
            .ok_or(Error::InvalidArchive)?;
        Ok(())
    }
    fn bytes(&mut self, mut output: &mut [u8]) -> Result<()> {
        if self.position.checked_add(output.len() as u64).is_none_or(|end| end > self.source.len()) {
            return Err(Error::InvalidArchive);
        }
        while !output.is_empty() {
            if self.position < self.cache_start || self.position >= self.cache_start + self.cache_len as u64 {
                self.cache_start = self.position;
                self.cache_len = (self.source.len() - self.position).min(self.cache.len() as u64) as usize;
                self.source.read_at(self.position, &mut self.cache[..self.cache_len])?;
            }
            let start = (self.position - self.cache_start) as usize;
            let count = output.len().min(self.cache_len - start);
            output[..count].copy_from_slice(&self.cache[start..start + count]);
            self.position += count as u64;
            output = &mut output[count..];
        }
        Ok(())
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut result = [0; N];
        self.bytes(&mut result)?;
        Ok(result)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn string(&mut self) -> Result<String> {
        let size = self.u64()?;
        if size == 0 || size > 4096 {
            return Err(Error::InvalidArchive);
        }
        let mut bytes = vec![0; size as usize];
        self.bytes(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| Error::InvalidArchive)
    }
    fn skip_value(&mut self, kind: u32, depth: usize) -> Result<()> {
        let width = match kind {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            8 => {
                let n = self.u64()?;
                return self.skip(n);
            }
            9 if depth < 8 => {
                let element = self.u32()?;
                let count = self.u64()?;
                if count > 1_048_576 {
                    return Err(Error::InvalidArchive);
                }
                let width = match element {
                    0 | 1 | 7 => Some(1),
                    2 | 3 => Some(2),
                    4..=6 => Some(4),
                    10..=12 => Some(8),
                    _ => None,
                };
                if let Some(width) = width {
                    return self.skip(count.checked_mul(width).ok_or(Error::InvalidArchive)?);
                }
                for _ in 0..count {
                    self.skip_value(element, depth + 1)?;
                }
                return Ok(());
            }
            _ => return Err(Error::InvalidArchive),
        };
        self.skip(width)
    }
    fn number(&mut self, kind: u32) -> Result<f64> {
        Ok(match kind {
            0 => self.array::<1>()?[0] as f64,
            1 => self.array::<1>()?[0] as i8 as f64,
            2 => u16::from_le_bytes(self.array()?) as f64,
            3 => i16::from_le_bytes(self.array()?) as f64,
            4 => self.u32()? as f64,
            5 => i32::from_le_bytes(self.array()?) as f64,
            6 => f32::from_le_bytes(self.array()?) as f64,
            10 => self.u64()? as f64,
            11 => i64::from_le_bytes(self.array()?) as f64,
            12 => f64::from_le_bytes(self.array()?),
            _ => return Err(Error::InvalidArchive),
        })
    }
}

#[derive(Clone, Copy)]
enum Encoding {
    F32,
    Bf16,
    Quantized(Quantization),
}
struct Stored {
    shape: Vec<usize>,
    encoding: Encoding,
    offset: u64,
    bytes: usize,
}

/// Metadata stays resident; each tensor's source bytes are released after upload.
pub struct Gguf<R> {
    source: R,
    tensors: BTreeMap<String, Stored>,
    config: Config,
}

const NUMBERS: &[&str] = &[
    "general.alignment",
    "qwen35.embedding_length",
    "qwen35.feed_forward_length",
    "qwen35.block_count",
    "qwen35.attention.head_count",
    "qwen35.attention.head_count_kv",
    "qwen35.attention.key_length",
    "qwen35.attention.value_length",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.group_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.conv_kernel",
    "qwen35.full_attention_interval",
    "qwen35.context_length",
    "qwen35.rope.freq_base",
    "qwen35.attention.layer_norm_rms_epsilon",
];

impl<R: ReadAt> Gguf<R> {
    pub fn parse(source: R) -> Result<Self> {
        let mut reader = Reader::new(source);
        if &reader.array::<4>()? != b"GGUF" || reader.u32()? != 3 {
            return Err(Error::InvalidArchive);
        }
        let tensor_count = reader.u64()?;
        let metadata_count = reader.u64()?;
        if tensor_count == 0 || tensor_count > 4096 || metadata_count > 4096 {
            return Err(Error::InvalidArchive);
        }
        let mut numbers = BTreeMap::new();
        let mut architecture = None;
        for _ in 0..metadata_count {
            let key = reader.string()?;
            let kind = reader.u32()?;
            if key == "general.architecture" {
                if kind != 8 || architecture.is_some() {
                    return Err(Error::InvalidArchive);
                }
                architecture = Some(reader.string()?);
            } else if NUMBERS.contains(&key.as_str()) {
                let value = reader.number(kind)?;
                if !value.is_finite() || numbers.insert(key, value).is_some() {
                    return Err(Error::InvalidArchive);
                }
            } else {
                reader.skip_value(kind, 0)?;
            }
        }
        if architecture.as_deref() != Some("qwen35") {
            return Err(Error::InvalidConfig);
        }
        let number = |key: &str| numbers.get(key).copied().ok_or(Error::InvalidConfig);
        let dimension = |key: &str| -> Result<usize> {
            let value = number(key)?;
            if !(1. ..=1_048_576.).contains(&value) || value != (value as usize) as f64 {
                return Err(Error::InvalidConfig);
            }
            Ok(value as usize)
        };
        let alignment = numbers.get("general.alignment").copied().unwrap_or(32.);
        if !(1. ..=4096.).contains(&alignment) || alignment != (alignment as u64) as f64 || !(alignment as u64).is_power_of_two() {
            return Err(Error::InvalidArchive);
        }
        let alignment = alignment as u64;
        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let rank = reader.u32()?;
            if !(1..=8).contains(&rank) {
                return Err(Error::InvalidArchive);
            }
            let mut shape = Vec::new();
            for _ in 0..rank {
                shape.push(usize::try_from(reader.u64()?).map_err(|_| Error::InvalidArchive)?);
            }
            shape.reverse();
            let elements = element_count(&shape)?;
            let kind = reader.u32()?;
            let offset = reader.u64()?;
            if !offset.is_multiple_of(alignment) {
                return Err(Error::InvalidArchive);
            }
            let encoding = match kind {
                0 => Encoding::F32,
                30 => Encoding::Bf16,
                _ => Encoding::Quantized(Quantization::from_raw(kind)?),
            };
            let bytes = match encoding {
                Encoding::F32 => elements.checked_mul(4).ok_or(Error::InvalidArchive)?,
                Encoding::Bf16 => elements.checked_mul(2).ok_or(Error::InvalidArchive)?,
                Encoding::Quantized(format) => {
                    if !shape[shape.len() - 1].is_multiple_of(format.block_elements()) {
                        return Err(Error::InvalidModelShape);
                    }
                    format.byte_len(elements)?
                }
            };
            if tensors
                .insert(
                    name,
                    Stored {
                        shape,
                        encoding,
                        offset,
                        bytes,
                    },
                )
                .is_some()
            {
                return Err(Error::InvalidArchive);
            }
        }
        let start = reader.position.checked_add(alignment - 1).ok_or(Error::InvalidArchive)? & !(alignment - 1);
        let mut regions = Vec::new();
        for record in tensors.values_mut() {
            record.offset = start.checked_add(record.offset).ok_or(Error::InvalidArchive)?;
            let end = record
                .offset
                .checked_add(record.bytes as u64)
                .filter(|&end| end <= reader.source.len())
                .ok_or(Error::InvalidArchive)?;
            regions.push((record.offset, end));
        }
        regions.sort_unstable();
        if regions.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(Error::InvalidArchive);
        }
        let embedding = tensors
            .get("token_embd.weight")
            .ok_or_else(|| Error::MissingTensor("token_embd.weight".into()))?;
        if embedding.shape.len() != 2 {
            return Err(Error::InvalidModelShape);
        }
        let linear_heads = dimension("qwen35.ssm.time_step_rank")?;
        let inner = dimension("qwen35.ssm.inner_size")?;
        if !inner.is_multiple_of(linear_heads) {
            return Err(Error::InvalidConfig);
        }
        let config = Config {
            hidden: dimension("qwen35.embedding_length")?,
            intermediate: dimension("qwen35.feed_forward_length")?,
            layers: dimension("qwen35.block_count")?,
            vocab: embedding.shape[0],
            heads: dimension("qwen35.attention.head_count")?,
            kv_heads: dimension("qwen35.attention.head_count_kv")?,
            head_dim: dimension("qwen35.attention.key_length")?,
            rotary_dim: dimension("qwen35.rope.dimension_count")?,
            linear_heads,
            linear_key_heads: dimension("qwen35.ssm.group_count")?,
            linear_key_dim: dimension("qwen35.ssm.state_size")?,
            linear_value_dim: inner / linear_heads,
            convolution_kernel: dimension("qwen35.ssm.conv_kernel")?,
            full_attention_interval: dimension("qwen35.full_attention_interval")?,
            max_positions: dimension("qwen35.context_length")?,
            rope_theta: number("qwen35.rope.freq_base")? as f32,
            epsilon: number("qwen35.attention.layer_norm_rms_epsilon")? as f32,
            tied_embeddings: !tensors.contains_key("output.weight"),
        };
        config.validate()?;
        if config.head_dim != dimension("qwen35.attention.value_length")? || embedding.shape[1] != config.hidden {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            source: reader.source,
            tensors,
            config,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }

    pub fn load(mut self, device: &Device) -> Result<Qwen> {
        let model = Qwen::load_weights(self.config, device, |name, shape| self.tensor(name, shape))?;
        if !self.tensors.is_empty() {
            return Err(Error::UnexpectedTensor);
        }
        Ok(if self.config.linear_heads != self.config.linear_key_heads {
            model.with_tiled_linear_outputs()
        } else {
            model
        })
    }
    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<WeightData<'static>> {
        let mapped = tensor_name(name)?;
        let record = self.tensors.remove(&mapped).ok_or_else(|| Error::MissingTensor(mapped.clone()))?;
        let convolution = name.ends_with("linear_attn.conv1d.weight");
        if record.shape != shape && !(convolution && shape.len() == 3 && shape[1] == 1 && record.shape == [shape[0], shape[2]]) {
            return Err(Error::InvalidModelShape);
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(record.bytes).map_err(|_| Error::AllocationFailed)?;
        bytes.resize(record.bytes, 0);
        self.source.read_at(record.offset, &mut bytes)?;
        let format = match record.encoding {
            Encoding::F32 => MatrixFormat::F32,
            Encoding::Bf16 => MatrixFormat::Bf16,
            Encoding::Quantized(format) => MatrixFormat::Quantized(format),
        };
        self.restore_linear_heads(name, shape, format, &mut bytes)?;
        if shape.len() != 2 {
            let elements = element_count(shape)?;
            let mut values = format.decode([1, elements], &bytes)?;
            for value in &mut values {
                if name.ends_with("A_log") {
                    if *value >= 0. {
                        return Err(Error::InvalidArchive);
                    }
                    *value = libm::logf(-*value);
                } else if name.ends_with("norm.weight") && !name.ends_with("linear_attn.norm.weight") {
                    *value -= 1.;
                }
                if !value.is_finite() {
                    return Err(Error::NonFinite);
                }
            }
            bytes.clear();
            bytes
                .try_reserve_exact(elements.checked_mul(4).ok_or(Error::InvalidModelShape)?)
                .map_err(|_| Error::AllocationFailed)?;
            bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
            return Ok(WeightData {
                shape: shape.into(),
                format: MatrixFormat::F32,
                bytes: Cow::Owned(bytes),
            });
        }
        Ok(WeightData {
            shape: shape.into(),
            format,
            bytes: Cow::Owned(bytes),
        })
    }

    fn restore_linear_heads(&self, name: &str, shape: &[usize], format: MatrixFormat, bytes: &mut [u8]) -> Result<()> {
        let config = self.config;
        if config.linear_key_heads == config.linear_heads || !name.contains(".linear_attn.") {
            return Ok(());
        }
        let keys = config.linear_key_heads;
        let values = config.linear_heads;
        let key_channels = 2 * keys * config.linear_key_dim;
        let (offset, head_bytes) = if name.ends_with("in_proj_qkv.weight") {
            let row = format.byte_len([1, shape[1]])?;
            (key_channels * row, config.linear_value_dim * row)
        } else if name.ends_with("in_proj_z.weight") {
            (0, config.linear_value_dim * format.byte_len([1, shape[1]])?)
        } else if name.ends_with("in_proj_a.weight") || name.ends_with("in_proj_b.weight") {
            (0, format.byte_len([1, shape[1]])?)
        } else if name.ends_with("conv1d.weight") {
            let row = format.byte_len([1, config.convolution_kernel])?;
            (key_channels * row, config.linear_value_dim * row)
        } else if name.ends_with("A_log") || name.ends_with("dt_bias") {
            (0, format.byte_len([1, 1])?)
        } else {
            return Ok(());
        };
        restore_tiled_heads(bytes, offset, head_bytes, keys, values)
    }
}

fn restore_tiled_heads(bytes: &mut [u8], offset: usize, head_bytes: usize, keys: usize, values: usize) -> Result<()> {
    let length = head_bytes.checked_mul(values).ok_or(Error::InvalidModelShape)?;
    let end = offset.checked_add(length).ok_or(Error::InvalidModelShape)?;
    let data = bytes.get_mut(offset..end).ok_or(Error::InvalidModelShape)?;
    let mut tiled = Vec::new();
    tiled.try_reserve_exact(length).map_err(|_| Error::AllocationFailed)?;
    tiled.extend_from_slice(data);
    let repeats = values / keys;
    for head in 0..values {
        let source = (head % repeats) * keys + head / repeats;
        data[head * head_bytes..(head + 1) * head_bytes].copy_from_slice(&tiled[source * head_bytes..(source + 1) * head_bytes]);
    }
    Ok(())
}


fn tensor_name(name: &str) -> Result<String> {
    match name {
        "embed_tokens.weight" => return Ok("token_embd.weight".into()),
        "lm_head.weight" => return Ok("output.weight".into()),
        "norm.weight" => return Ok("output_norm.weight".into()),
        _ => {}
    }
    let (layer, suffix) = name
        .strip_prefix("layers.")
        .and_then(|rest| rest.split_once('.'))
        .ok_or(Error::InvalidModelShape)?;
    if layer.parse::<usize>().is_err() {
        return Err(Error::InvalidModelShape);
    }
    let mapped = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "linear_attn.in_proj_z.weight" => "attn_gate.weight",
        "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
        "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
        "linear_attn.A_log" => "ssm_a",
        "linear_attn.dt_bias" => "ssm_dt.bias",
        "linear_attn.norm.weight" => "ssm_norm.weight",
        "linear_attn.out_proj.weight" => "ssm_out.weight",
        _ => return Err(Error::InvalidModelShape),
    };
    Ok(format!("blk.{layer}.{mapped}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restoring_grouped_heads_preserves_packed_blocks_and_neighboring_regions() {
        for (keys, repeats, head_bytes) in [(2, 3, 4), (16, 2, 128 * 34), (16, 3, 128 * 176)] {
            let heads = keys * repeats;
            let canonical: Vec<_> = (0..heads * head_bytes)
                .map(|i| (i as u8).wrapping_mul(117).wrapping_add((i / head_bytes) as u8))
                .collect();
            // Independently export grouped heads by visiting each tile in turn.
            let mut exported = vec![0x53; 17];
            for tile in 0..repeats {
                for key in 0..keys {
                    let start = (key * repeats + tile) * head_bytes;
                    exported.extend_from_slice(&canonical[start..start + head_bytes]);
                }
            }
            exported.extend([0xa7; 23]);
            restore_tiled_heads(&mut exported, 17, head_bytes, keys, heads).unwrap();
            assert_eq!(&exported[..17], &[0x53; 17]);
            assert_eq!(&exported[17..17 + canonical.len()], canonical);
            assert_eq!(&exported[17 + canonical.len()..], &[0xa7; 23]);
            assert!(restore_tiled_heads(&mut exported, usize::MAX, head_bytes, keys, heads).is_err());
        }
    }
}
