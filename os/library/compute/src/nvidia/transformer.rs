use super::{Nvidia, Storage, request};
use crate::transformer::{AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaState, TransformerBackend};
use crate::{Error, Result, Tensor, element_count};
use d3os_nvidia::compute::Kernel;

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.
}

impl TransformerBackend for Nvidia {
    fn embedding(&self, table: &Tensor<Self>, token: usize) -> Result<Tensor<Self>> {
        let [vocab, width] = table.shape() else {
            return Err(Error::ShapeMismatch);
        };
        if token >= *vocab {
            return Err(Error::InvalidParameter);
        }
        let output = self.allocate(&[1, *width])?;
        if table.buffer.storage == Storage::Bf16 {
            self.launch(
                Kernel::DequantizeBf16,
                &[(&table.buffer, token * width / 2), (&output.buffer, 0)],
                &[*width, token * width % 2],
            )?;
        } else {
            self.launch(Kernel::Copy, &[(&table.buffer, token * width), (&output.buffer, 0)], &[*width])?;
        }
        Ok(output)
    }

    fn project(&self, input: &Tensor<Self>, weights: &Tensor<Self>) -> Result<Tensor<Self>> {
        let ([batch, width], [outputs, inner]) = (input.shape(), weights.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if width != inner {
            return Err(Error::ShapeMismatch);
        }
        self.produce(
            &[*batch, *outputs],
            if weights.buffer.storage == Storage::Bf16 {
                Kernel::ProjectBf16
            } else {
                Kernel::Project
            },
            &[(&input.buffer, 0), (&weights.buffer, 0)],
            &[*batch, *width, *outputs],
        )
    }

    fn rms_norm(&self, input: &Tensor<Self>, weights: &Tensor<Self>, epsilon: f32) -> Result<Tensor<Self>> {
        let Some(&width) = input.shape().last() else {
            return Err(Error::ShapeMismatch);
        };
        if weights.shape() != [width] || !positive(epsilon) {
            return Err(Error::InvalidParameter);
        }
        self.produce(
            input.shape(),
            Kernel::RmsNorm,
            &[(&input.buffer, 0), (&weights.buffer, 0)],
            &[input.len() / width, width, epsilon.to_bits() as usize, 1f32.to_bits() as usize],
        )
    }

    fn add(&self, input: Tensor<Self>, residual: &Tensor<Self>) -> Result<Tensor<Self>> {
        if input.shape() != residual.shape() {
            return Err(Error::ShapeMismatch);
        }
        self.launch(Kernel::Add, &[(&input.buffer, 0), (&residual.buffer, 0), (&input.buffer, 0)], &[input.len()])?;
        Ok(input)
    }

    fn silu_mul(&self, input: Tensor<Self>, other: &Tensor<Self>) -> Result<Tensor<Self>> {
        if input.shape() != other.shape() {
            return Err(Error::ShapeMismatch);
        }
        self.launch(Kernel::SiluMul, &[(&input.buffer, 0), (&other.buffer, 0), (&input.buffer, 0)], &[input.len()])?;
        Ok(input)
    }

    fn attention_step(&self, input: AttentionInputs<'_, Self>, state: &mut AttentionState<Self>, config: AttentionConfig) -> Result<Tensor<Self>> {
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
        let [capacity, cache_heads, cache_dim] = *state.keys.shape() else {
            return Err(Error::ShapeMismatch);
        };
        if heads % kv_heads != 0 || rotary_dim == 0 || rotary_dim > dim || rotary_dim % 2 != 0 || !positive(rope_theta) || !positive(epsilon) {
            return Err(Error::InvalidParameter);
        }
        if input.query_gate.shape() != [1, qg_width]
            || input.key.shape() != [1, kv_width]
            || input.value.shape() != [1, kv_width]
            || input.query_norm.shape() != [dim]
            || input.key_norm.shape() != [dim]
            || cache_heads != kv_heads
            || cache_dim != dim
            || state.values.shape() != state.keys.shape()
        {
            return Err(Error::ShapeMismatch);
        }
        if state.position >= capacity {
            return Err(Error::InvalidParameter);
        }
        let positions = state.position + 1;
        let query = self.allocate(&[heads, dim])?;
        let scores = self.allocate(&[heads, positions])?;
        let output = self.allocate(&[1, q_width])?;
        let theta = rope_theta.to_bits() as usize;
        let epsilon = epsilon.to_bits() as usize;
        let cache_offset = state.position * kv_width;
        // Allocate all scratch and validate every dispatch before changing a cache.
        let calls = [
            self.prepare(
                Kernel::AttentionPrepare,
                &[(&input.query_gate.buffer, 0), (&input.query_norm.buffer, 0), (&query.buffer, 0)],
                &[heads, dim, rotary_dim, state.position, theta, epsilon, 2],
            )?,
            self.prepare(
                Kernel::AttentionPrepare,
                &[(&input.key.buffer, 0), (&input.key_norm.buffer, 0), (&state.keys.buffer, cache_offset)],
                &[kv_heads, dim, rotary_dim, state.position, theta, epsilon, 1],
            )?,
            self.prepare(Kernel::Copy, &[(&input.value.buffer, 0), (&state.values.buffer, cache_offset)], &[kv_width])?,
            self.prepare(
                Kernel::AttentionScores,
                &[(&query.buffer, 0), (&state.keys.buffer, 0), (&scores.buffer, 0)],
                &[heads, kv_heads, dim, positions],
            )?,
            self.prepare(Kernel::Softmax, &[(&scores.buffer, 0), (&scores.buffer, 0)], &[heads, positions])?,
            self.prepare(
                Kernel::AttentionOutput,
                &[
                    (&scores.buffer, 0),
                    (&state.values.buffer, 0),
                    (&input.query_gate.buffer, 0),
                    (&output.buffer, 0),
                ],
                &[heads, kv_heads, dim, positions],
            )?,
        ];
        for call in calls {
            request(call)?;
        }
        state.position += 1;
        Ok(output)
    }

    fn delta_step(&self, input: DeltaInputs<'_, Self>, state: &mut DeltaState<Self>, config: DeltaConfig) -> Result<Tensor<Self>> {
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
        if !positive(epsilon) {
            return Err(Error::InvalidParameter);
        }
        if input.qkv.shape() != [1, channels]
            || input.gate.shape() != [1, value_width]
            || input.alpha.shape() != [1, heads]
            || input.beta.shape() != [1, heads]
            || input.convolution.shape() != [channels, 1, kernel]
            || input.log_decay.shape() != [heads]
            || input.time_bias.shape() != [heads]
            || input.norm.shape() != [value_dim]
            || state.convolution.shape() != [channels, kernel]
            || state.recurrent.shape() != [heads, key_dim, value_dim]
        {
            return Err(Error::ShapeMismatch);
        }
        let mixed = self.allocate(&[channels])?;
        let scales = self.allocate(&[heads, 4])?;
        let raw = self.allocate(&[heads, value_dim])?;
        let output = self.allocate(&[1, value_width])?;
        let calls = [
            self.prepare(
                Kernel::DeltaConvolution,
                &[
                    (&input.qkv.buffer, 0),
                    (&input.convolution.buffer, 0),
                    (&state.convolution.buffer, 0),
                    (&mixed.buffer, 0),
                ],
                &[channels, kernel],
            )?,
            self.prepare(
                Kernel::DeltaPrepare,
                &[
                    (&mixed.buffer, 0),
                    (&input.beta.buffer, 0),
                    (&input.alpha.buffer, 0),
                    (&input.log_decay.buffer, 0),
                    (&input.time_bias.buffer, 0),
                    (&scales.buffer, 0),
                ],
                &[heads, key_dim, value_dim],
            )?,
            self.prepare(
                Kernel::DeltaUpdate,
                &[(&mixed.buffer, 0), (&scales.buffer, 0), (&state.recurrent.buffer, 0), (&raw.buffer, 0)],
                &[heads, key_dim, value_dim],
            )?,
            self.prepare(
                Kernel::DeltaOutput,
                &[(&raw.buffer, 0), (&input.norm.buffer, 0), (&input.gate.buffer, 0), (&output.buffer, 0)],
                &[heads, value_dim, epsilon.to_bits() as usize],
            )?,
        ];
        for call in calls {
            request(call)?;
        }
        Ok(output)
    }
}
