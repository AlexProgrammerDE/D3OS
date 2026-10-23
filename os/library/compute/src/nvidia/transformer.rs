use super::{Nvidia, Storage, request};
use crate::transformer::{
    AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaState, DeltaWeights, TransformerBackend, attention_shape, delta_shape,
    project_delta, project_delta_shape,
};
use crate::{Error, Result, Tensor};
use alloc::vec::Vec;
use d3os_nvidia::compute::Kernel;

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.
}

impl TransformerBackend for Nvidia {
    fn project_delta(&self, input: &Tensor<Self>, weights: DeltaWeights<'_, Self>, state: &mut DeltaState<Self>, config: DeltaConfig) -> Result<Tensor<Self>> {
        let (rows, width, channels, values) = project_delta_shape(input, &weights, state, config)?;
        if rows != 1
            || !width.is_multiple_of(2)
            || !channels.is_multiple_of(4)
            || !values.is_multiple_of(4)
            || !config.heads.is_multiple_of(4)
            || config.key_dim > 128
            || config.value_dim > 128
            || config.kernel > 8
            || [weights.qkv, weights.gate, weights.beta, weights.alpha]
                .iter()
                .any(|tensor| tensor.buffer.storage != Storage::Bf16)
        {
            return project_delta(self, input, weights, state, config);
        }
        if [
            weights.convolution,
            weights.log_decay,
            weights.time_bias,
            weights.norm,
            &state.convolution,
            &state.recurrent,
        ]
        .iter()
        .any(|tensor| tensor.buffer.storage != Storage::F32)
        {
            return Err(Error::InvalidParameter);
        }
        let total = channels
            .checked_add(values)
            .and_then(|n| config.heads.checked_mul(2).and_then(|h| n.checked_add(h)))
            .ok_or(Error::SizeOverflow)?;
        let projected = self.produce(
            &[1, total],
            Kernel::GroupedProjectBf16,
            &[
                (&input.buffer, 0),
                (&weights.qkv.buffer, 0),
                (&weights.gate.buffer, 0),
                (&weights.beta.buffer, 0),
                (&weights.alpha.buffer, 0),
            ],
            &[width, channels, values, config.heads],
        )?;
        self.produce(
            &[1, values],
            Kernel::DeltaFused,
            &[
                (&projected.buffer, 0),
                (&projected.buffer, channels),
                (&projected.buffer, channels + values),
                (&projected.buffer, channels + values + config.heads),
                (&weights.convolution.buffer, 0),
                (&weights.log_decay.buffer, 0),
                (&weights.time_bias.buffer, 0),
                (&weights.norm.buffer, 0),
                (&state.convolution.buffer, 0),
                (&state.recurrent.buffer, 0),
            ],
            &[config.heads, config.key_dim, config.value_dim, config.kernel, config.epsilon.to_bits() as usize],
        )
    }
    fn project_add(&self, input: &Tensor<Self>, weights: &Tensor<Self>, residual: Tensor<Self>) -> Result<Tensor<Self>> {
        if let ([1, width], [outputs, inner]) = (input.shape(), weights.shape())
            && width == inner
            && width.is_multiple_of(2)
            && *outputs <= 8192
            && residual.shape() == [1, *outputs]
            && weights.buffer.storage == Storage::Bf16
        {
            self.launch(
                Kernel::ProjectResidualBf16,
                &[(&residual.buffer, 0), (&input.buffer, 0), (&weights.buffer, 0), (&residual.buffer, 0)],
                &[*width, *outputs],
            )?;
            return Ok(residual);
        }
        let projected = self.project(input, weights)?;
        self.add(residual, &projected)
    }

    fn embedding_batch(&self, table: &Tensor<Self>, tokens: &[u32]) -> Result<Tensor<Self>> {
        let [vocab, width] = *table.shape() else {
            return Err(Error::ShapeMismatch);
        };
        if tokens.iter().any(|&token| token as usize >= vocab) {
            return Err(Error::InvalidParameter);
        }
        let ids = self.upload_word_storage::<f32>(tokens)?;
        self.produce(
            &[tokens.len(), width],
            Kernel::Embedding,
            &[(&table.buffer, 0), (&ids, 0)],
            &[vocab, width, tokens.len(), usize::from(table.buffer.storage == Storage::Bf16)],
        )
    }

    fn row(&self, input: &Tensor<Self>, row: usize) -> Result<Tensor<Self>> {
        let [rows, width] = *input.shape() else {
            return Err(Error::ShapeMismatch);
        };
        if row >= rows {
            return Err(Error::InvalidParameter);
        }
        if input.buffer.storage == Storage::Bf16 {
            return self.embedding(input, row);
        }
        self.produce(&[1, width], Kernel::Copy, &[(&input.buffer, row * width)], &[width])
    }

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
        if weights.buffer.storage == Storage::Bf16 && (2..=32).contains(batch) && (256..=2048).contains(outputs) && *width >= 512 {
            let partials = self.produce(
                &[16, *batch, *outputs],
                Kernel::ProjectBf16Split,
                &[(&input.buffer, 0), (&weights.buffer, 0)],
                &[*batch, *width, *outputs, 16],
            )?;
            return self.produce(
                &[*batch, *outputs],
                Kernel::SumProjectionPartials,
                &[(&partials.buffer, 0)],
                &[*batch, *outputs, 16],
            );
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

    fn gated_project(&self, input: &Tensor<Self>, gate: &Tensor<Self>, up: &Tensor<Self>) -> Result<Tensor<Self>> {
        let ([batch, width], [outputs, inner]) = (input.shape(), gate.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if width != inner || up.shape() != gate.shape() {
            return Err(Error::ShapeMismatch);
        }
        if *batch == 1 && gate.buffer.storage == Storage::Bf16 && up.buffer.storage == Storage::Bf16 {
            return self.produce(
                &[1, *outputs],
                Kernel::GatedProjectBf16,
                &[(&input.buffer, 0), (&gate.buffer, 0), (&up.buffer, 0)],
                &[*width, *outputs],
            );
        }
        let gate = self.project(input, gate)?;
        let up = self.project(input, up)?;
        self.silu_mul(gate, &up)
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

    fn residual_norm(&self, input: Tensor<Self>, residual: &Tensor<Self>, weights: &Tensor<Self>, epsilon: f32) -> Result<(Tensor<Self>, Tensor<Self>)> {
        let Some(&width) = input.shape().last() else {
            return Err(Error::ShapeMismatch);
        };
        if input.shape() != residual.shape() || weights.shape() != [width] || !positive(epsilon) {
            return Err(Error::InvalidParameter);
        }
        if width > 1024 {
            let input = self.add(input, residual)?;
            let normalized = self.rms_norm(&input, weights, epsilon)?;
            return Ok((input, normalized));
        }
        let normalized = self.produce(
            input.shape(),
            Kernel::ResidualNorm,
            &[(&input.buffer, 0), (&residual.buffer, 0), (&weights.buffer, 0)],
            &[input.len() / width, width, epsilon.to_bits() as usize],
        )?;
        Ok((input, normalized))
    }

    fn silu_mul(&self, input: Tensor<Self>, other: &Tensor<Self>) -> Result<Tensor<Self>> {
        if input.shape() != other.shape() {
            return Err(Error::ShapeMismatch);
        }
        self.launch(Kernel::SiluMul, &[(&input.buffer, 0), (&other.buffer, 0), (&input.buffer, 0)], &[input.len()])?;
        Ok(input)
    }

    fn attention_batch(&self, input: AttentionInputs<'_, Self>, state: &mut AttentionState<Self>, config: AttentionConfig) -> Result<Tensor<Self>> {
        let AttentionConfig {
            heads,
            kv_heads,
            head_dim: dim,
            rotary_dim,
            rope_theta,
            epsilon,
        } = config;
        let (rows, q_width, kv_width) = attention_shape(&input, state, config)?;
        let positions = state.position + rows;
        let parameters = |heads, stride| {
            [
                heads,
                dim,
                rotary_dim,
                state.position,
                rope_theta.to_bits() as usize,
                epsilon.to_bits() as usize,
                stride,
                rows,
            ]
        };
        let query = self.produce(
            &[rows, q_width],
            Kernel::AttentionPrepareBatch,
            &[(&input.query_gate.buffer, 0), (&input.query_norm.buffer, 0)],
            &parameters(heads, 2),
        )?;
        if dim <= 256 {
            let before = [
                self.prepare(
                    Kernel::AttentionPrepareBatch,
                    &[
                        (&input.key.buffer, 0),
                        (&input.key_norm.buffer, 0),
                        (&state.keys.buffer, state.position * kv_width),
                    ],
                    &parameters(kv_heads, 1),
                )?,
                self.prepare(
                    Kernel::Copy,
                    &[(&input.value.buffer, 0), (&state.values.buffer, state.position * kv_width)],
                    &[rows * kv_width],
                )?,
            ];
            if rows == 1 && heads <= 16 && positions >= 512 {
                const SPLITS: usize = 16;
                let partials = self.produce_storage(
                    rows * heads * SPLITS * (dim + 2),
                    Kernel::AttentionFusedBatch,
                    &[
                        (&query.buffer, 0),
                        (&state.keys.buffer, 0),
                        (&state.values.buffer, 0),
                        (&input.query_gate.buffer, 0),
                    ],
                    &[rows, heads, kv_heads, dim, state.position, SPLITS],
                    Storage::F32,
                    &before,
                )?;
                let buffer = self.produce_storage(
                    rows * q_width,
                    Kernel::AttentionOutput,
                    &[(&partials, 0), (&input.query_gate.buffer, 0)],
                    &[rows * heads, dim, SPLITS],
                    Storage::F32,
                    &[],
                )?;
                state.position = positions;
                return Tensor::from_buffer(&[rows, q_width], buffer);
            }
            let buffer = self.produce_storage(
                rows * q_width,
                Kernel::AttentionFusedBatch,
                &[
                    (&query.buffer, 0),
                    (&state.keys.buffer, 0),
                    (&state.values.buffer, 0),
                    (&input.query_gate.buffer, 0),
                ],
                &[rows, heads, kv_heads, dim, state.position],
                Storage::F32,
                &before,
            )?;
            state.position = positions;
            return Tensor::from_buffer(&[rows, q_width], buffer);
        }
        let scores = self.allocate(&[heads, positions])?;
        let output = self.allocate(&[rows, q_width])?;
        // Validate the entire request list before the first cache write. Future
        // keys may be prepared early; scoring reads only each row's valid prefix.
        let mut calls = Vec::new();
        calls
            .try_reserve(rows.checked_mul(3).and_then(|n| n.checked_add(2)).ok_or(Error::SizeOverflow)?)
            .map_err(|_| Error::AllocationFailed)?;
        calls.push(self.prepare(
            Kernel::AttentionPrepareBatch,
            &[
                (&input.key.buffer, 0),
                (&input.key_norm.buffer, 0),
                (&state.keys.buffer, state.position * kv_width),
            ],
            &parameters(kv_heads, 1),
        )?);
        calls.push(self.prepare(
            Kernel::Copy,
            &[(&input.value.buffer, 0), (&state.values.buffer, state.position * kv_width)],
            &[rows * kv_width],
        )?);
        for row in 0..rows {
            let visible = state.position + row + 1;
            calls.push(self.prepare(
                Kernel::AttentionScores,
                &[(&query.buffer, row * q_width), (&state.keys.buffer, 0), (&scores.buffer, 0)],
                &[heads, kv_heads, dim, visible],
            )?);
            calls.push(self.prepare(Kernel::Softmax, &[(&scores.buffer, 0), (&scores.buffer, 0)], &[heads, visible])?);
            calls.push(self.prepare(
                Kernel::AttentionOutput,
                &[
                    (&scores.buffer, 0),
                    (&state.values.buffer, 0),
                    (&input.query_gate.buffer, row * 2 * q_width),
                    (&output.buffer, row * q_width),
                ],
                &[heads, kv_heads, dim, visible],
            )?);
        }
        for call in calls {
            request(call)?;
        }
        state.position = positions;
        Ok(output)
    }

    fn delta_batch(&self, input: DeltaInputs<'_, Self>, state: &mut DeltaState<Self>, config: DeltaConfig) -> Result<Tensor<Self>> {
        let DeltaConfig {
            heads,
            key_dim,
            value_dim,
            kernel,
            epsilon,
        } = config;
        let (rows, channels, value_width) = delta_shape(&input, state, config)?;
        if rows > 1 && key_dim <= 128 && value_dim <= 128 && kernel <= 8 {
            // Reject every unsupported storage type before the convolution
            // changes history, including inputs consumed only by later stages.
            if [
                &input.qkv.buffer,
                &input.gate.buffer,
                &input.beta.buffer,
                &input.alpha.buffer,
                &input.convolution.buffer,
                &input.log_decay.buffer,
                &input.time_bias.buffer,
                &input.norm.buffer,
                &state.convolution.buffer,
                &state.recurrent.buffer,
            ]
            .iter()
            .any(|buffer| buffer.storage != Storage::F32)
            {
                return Err(Error::InvalidParameter);
            }
            let mixed = self.produce(
                &[rows, channels],
                Kernel::DeltaConvolution,
                &[(&input.qkv.buffer, 0), (&input.convolution.buffer, 0), (&state.convolution.buffer, 0)],
                &[channels, kernel, rows],
            )?;
            let raw = self.produce(
                &[rows, value_width],
                Kernel::DeltaUpdate,
                &[
                    (&mixed.buffer, 0),
                    (&input.beta.buffer, 0),
                    (&input.alpha.buffer, 0),
                    (&input.log_decay.buffer, 0),
                    (&input.time_bias.buffer, 0),
                    (&state.recurrent.buffer, 0),
                ],
                &[heads, key_dim, value_dim, rows],
            )?;
            return self.produce(
                &[rows, value_width],
                Kernel::DeltaOutput,
                &[(&raw.buffer, 0), (&input.norm.buffer, 0), (&input.gate.buffer, 0)],
                &[rows * heads, value_dim, epsilon.to_bits() as usize],
            );
        }
        if key_dim <= 128 && value_dim <= 128 && kernel <= 8 {
            let inputs = [
                (&input.qkv.buffer, 0),
                (&input.gate.buffer, 0),
                (&input.beta.buffer, 0),
                (&input.alpha.buffer, 0),
                (&input.convolution.buffer, 0),
                (&input.log_decay.buffer, 0),
                (&input.time_bias.buffer, 0),
                (&input.norm.buffer, 0),
                (&state.convolution.buffer, 0),
                (&state.recurrent.buffer, 0),
            ];
            return self.produce(
                &[rows, value_width],
                Kernel::DeltaFused,
                &inputs,
                &[heads, key_dim, value_dim, kernel, epsilon.to_bits() as usize],
            );
        }
        let mixed = self.allocate(&[channels])?;
        let scales = self.allocate(&[heads, 4])?;
        let raw = self.allocate(&[heads, value_dim])?;
        let output = self.allocate(&[rows, value_width])?;
        for row in 0..rows {
            let calls = [
                self.prepare(
                    Kernel::DeltaConvolution,
                    &[
                        (&input.qkv.buffer, row * channels),
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
                        (&input.beta.buffer, row * heads),
                        (&input.alpha.buffer, row * heads),
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
                    &[
                        (&raw.buffer, 0),
                        (&input.norm.buffer, 0),
                        (&input.gate.buffer, row * value_width),
                        (&output.buffer, row * value_width),
                    ],
                    &[heads, value_dim, epsilon.to_bits() as usize],
                )?,
            ];
            for call in calls {
                request(call)?;
            }
        }
        Ok(output)
    }
}
