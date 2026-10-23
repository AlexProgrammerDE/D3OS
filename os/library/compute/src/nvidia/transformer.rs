use super::{Nvidia, Storage, request};
use crate::transformer::{
    AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaState, TransformerBackend, attention_shape, delta_shape,
};
use crate::{Error, Result, Tensor};
use alloc::vec::Vec;
use d3os_nvidia::compute::Kernel;

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.
}

impl TransformerBackend for Nvidia {
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
        if key_dim <= 128 && value_dim <= 128 && kernel <= 8 {
            let inputs = |row| {
                [
                    (&input.qkv.buffer, row * channels),
                    (&input.gate.buffer, row * value_width),
                    (&input.beta.buffer, row * heads),
                    (&input.alpha.buffer, row * heads),
                    (&input.convolution.buffer, 0),
                    (&input.log_decay.buffer, 0),
                    (&input.time_bias.buffer, 0),
                    (&input.norm.buffer, 0),
                    (&state.convolution.buffer, 0),
                    (&state.recurrent.buffer, 0),
                ]
            };
            let parameters = [heads, key_dim, value_dim, kernel, epsilon.to_bits() as usize];
            if rows == 1 {
                return self.produce(&[1, value_width], Kernel::DeltaFused, &inputs(0), &parameters);
            }
            let output = self.allocate(&[rows, value_width])?;
            for row in 0..rows {
                let mut buffers = [(&output.buffer, 0); 11];
                buffers[..10].copy_from_slice(&inputs(row));
                buffers[10] = (&output.buffer, row * value_width);
                self.launch(Kernel::DeltaFused, &buffers, &parameters)?;
            }
            return Ok(output);
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
