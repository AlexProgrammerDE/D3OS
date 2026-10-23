use super::{Cpu, zeros};
use crate::transformer::{
    AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaState, TransformerBackend, attention_shape, delta_shape,
};
use crate::{Backend, Error, Result, Tensor, element_count};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.; 8];
    let mut lhs = a.chunks_exact(8);
    let mut rhs = b.chunks_exact(8);
    for (a, b) in lhs.by_ref().zip(rhs.by_ref()) {
        for lane in 0..8 {
            sums[lane] += a[lane] * b[lane];
        }
    }
    sums.iter().sum::<f32>() + lhs.remainder().iter().zip(rhs.remainder()).map(|(a, b)| a * b).sum::<f32>()
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + libm::expf(-x))
    } else {
        let exp = libm::expf(x);
        exp / (1. + exp)
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn norm(row: &mut [f32], weights: &[f32], epsilon: f32, offset: f32) {
    let scale = 1. / libm::sqrtf(dot(row, row) / row.len() as f32 + epsilon);
    for (x, weight) in row.iter_mut().zip(weights) {
        *x = *x * scale * (offset + weight);
    }
}

fn epsilon_valid(epsilon: f32) -> bool {
    epsilon.is_finite() && epsilon > 0.
}

fn rotate(row: &mut [f32], position: usize, config: AttentionConfig) {
    let half = config.rotary_dim / 2;
    for i in 0..half {
        let angle = position as f32 / libm::powf(config.rope_theta, (2 * i) as f32 / config.rotary_dim as f32);
        let (sin, cos) = (libm::sinf(angle), libm::cosf(angle));
        let (a, b) = (row[i], row[i + half]);
        row[i] = a * cos - b * sin;
        row[i + half] = b * cos + a * sin;
    }
}

impl TransformerBackend for Cpu {
    fn embedding_batch(&self, table: &Tensor<Self>, tokens: &[u32]) -> Result<Tensor<Self>> {
        let [vocab, width] = *table.shape() else {
            return Err(Error::ShapeMismatch);
        };
        if tokens.iter().any(|&token| token as usize >= vocab) {
            return Err(Error::InvalidParameter);
        }
        let mut values = zeros(&[tokens.len(), width])?;
        for (&token, output) in tokens.iter().zip(values.chunks_exact_mut(width)) {
            output.copy_from_slice(&self.values(table)[token as usize * width..(token as usize + 1) * width]);
        }
        Tensor::from_vec(self, &[tokens.len(), width], values)
    }

    fn row(&self, input: &Tensor<Self>, row: usize) -> Result<Tensor<Self>> {
        let [rows, width] = *input.shape() else {
            return Err(Error::ShapeMismatch);
        };
        if row >= rows {
            return Err(Error::InvalidParameter);
        }
        Tensor::from_vec(self, &[1, width], self.values(input)[row * width..(row + 1) * width].to_vec())
    }

    fn attention_batch(&self, input: AttentionInputs<'_, Self>, state: &mut AttentionState<Self>, config: AttentionConfig) -> Result<Tensor<Self>> {
        let (rows, width, _) = attention_shape(&input, state, config)?;
        let mut output = zeros(&[rows, width])?;
        for row in 0..rows {
            let query_gate = self.row(input.query_gate, row)?;
            let key = self.row(input.key, row)?;
            let value = self.row(input.value, row)?;
            let result = self.attention_step(
                AttentionInputs {
                    query_gate: &query_gate,
                    key: &key,
                    value: &value,
                    query_norm: input.query_norm,
                    key_norm: input.key_norm,
                },
                state,
                config,
            )?;
            output[row * width..(row + 1) * width].copy_from_slice(self.values(&result));
        }
        Tensor::from_vec(self, &[rows, width], output)
    }

    fn delta_batch(&self, input: DeltaInputs<'_, Self>, state: &mut DeltaState<Self>, config: DeltaConfig) -> Result<Tensor<Self>> {
        let (rows, _, width) = delta_shape(&input, state, config)?;
        let mut output = zeros(&[rows, width])?;
        for row in 0..rows {
            let qkv = self.row(input.qkv, row)?;
            let gate = self.row(input.gate, row)?;
            let beta = self.row(input.beta, row)?;
            let alpha = self.row(input.alpha, row)?;
            let result = self.delta_step(
                DeltaInputs {
                    qkv: &qkv,
                    gate: &gate,
                    beta: &beta,
                    alpha: &alpha,
                    convolution: input.convolution,
                    log_decay: input.log_decay,
                    time_bias: input.time_bias,
                    norm: input.norm,
                },
                state,
                config,
            )?;
            output[row * width..(row + 1) * width].copy_from_slice(self.values(&result));
        }
        Tensor::from_vec(self, &[rows, width], output)
    }

    fn embedding(&self, table: &Tensor<Self>, token: usize) -> Result<Tensor<Self>> {
        let [vocab, width] = table.shape() else {
            return Err(Error::ShapeMismatch);
        };
        if token >= *vocab {
            return Err(Error::InvalidParameter);
        }
        let mut values = zeros(&[1, *width])?;
        values.copy_from_slice(&self.values(table)[token * width..(token + 1) * width]);
        Tensor::from_vec(self, &[1, *width], values)
    }

    fn project(&self, input: &Tensor<Self>, weights: &Tensor<Self>) -> Result<Tensor<Self>> {
        let ([batch, width], [outputs, inner]) = (input.shape(), weights.shape()) else {
            return Err(Error::ShapeMismatch);
        };
        if width != inner {
            return Err(Error::ShapeMismatch);
        }
        let mut output = zeros(&[*batch, *outputs])?;
        for (row, dest) in self.values(input).chunks_exact(*width).zip(output.chunks_exact_mut(*outputs)) {
            for (weights, dest) in self.values(weights).chunks_exact(*width).zip(dest) {
                *dest = dot(row, weights);
            }
        }
        Tensor::from_vec(self, &[*batch, *outputs], output)
    }

    fn rms_norm(&self, input: &Tensor<Self>, weights: &Tensor<Self>, epsilon: f32) -> Result<Tensor<Self>> {
        let Some(&width) = input.shape().last() else {
            return Err(Error::ShapeMismatch);
        };
        if weights.shape() != [width] || !epsilon_valid(epsilon) {
            return Err(Error::InvalidParameter);
        }
        let mut output = self.download(input)?;
        for row in output.chunks_exact_mut(width) {
            norm(row, self.values(weights), epsilon, 1.);
        }
        Tensor::from_vec(self, input.shape(), output)
    }

    fn add(&self, mut input: Tensor<Self>, residual: &Tensor<Self>) -> Result<Tensor<Self>> {
        if input.shape() != residual.shape() {
            return Err(Error::ShapeMismatch);
        }
        for (out, rhs) in input.buffer.0.iter_mut().zip(self.values(residual)) {
            *out += rhs;
        }
        Ok(input)
    }

    fn silu_mul(&self, mut input: Tensor<Self>, other: &Tensor<Self>) -> Result<Tensor<Self>> {
        if input.shape() != other.shape() {
            return Err(Error::ShapeMismatch);
        }
        for (out, rhs) in input.buffer.0.iter_mut().zip(self.values(other)) {
            *out = silu(*out) * rhs;
        }
        Ok(input)
    }

    fn attention_step(&self, input: AttentionInputs<'_, Self>, state: &mut AttentionState<Self>, config: AttentionConfig) -> Result<Tensor<Self>> {
        let AttentionConfig {
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            epsilon,
        } = config;
        let q_width = element_count(&[heads, head_dim])?;
        let kv_width = element_count(&[kv_heads, head_dim])?;
        let qg_width = element_count(&[2, q_width])?;
        let [capacity, cache_heads, cache_dim] = *state.keys.shape() else {
            return Err(Error::ShapeMismatch);
        };
        if heads % kv_heads != 0
            || rotary_dim == 0
            || rotary_dim > head_dim
            || rotary_dim % 2 != 0
            || !rope_theta.is_finite()
            || rope_theta <= 0.
            || !epsilon_valid(epsilon)
        {
            return Err(Error::InvalidParameter);
        }
        if input.query_gate.shape() != [1, qg_width]
            || input.key.shape() != [1, kv_width]
            || input.value.shape() != [1, kv_width]
            || input.query_norm.shape() != [head_dim]
            || input.key_norm.shape() != [head_dim]
            || cache_heads != kv_heads
            || cache_dim != head_dim
            || state.values.shape() != state.keys.shape()
        {
            return Err(Error::ShapeMismatch);
        }
        if state.position >= capacity {
            return Err(Error::InvalidParameter);
        }
        // Allocate scratch before the first cache mutation.
        let mut output = zeros(&[1, q_width])?;
        let mut query = zeros(&[head_dim])?;
        let mut scores = zeros(&[state.position + 1])?;
        let mut key = self.download(input.key)?;
        for row in key.chunks_exact_mut(head_dim) {
            norm(row, self.values(input.key_norm), epsilon, 1.);
            rotate(row, state.position, config);
        }
        let start = state.position * kv_width;
        state.keys.buffer.0[start..start + kv_width].copy_from_slice(&key);
        state.values.buffer.0[start..start + kv_width].copy_from_slice(self.values(input.value));
        for (head, out) in output.chunks_exact_mut(head_dim).enumerate() {
            let qg = &self.values(input.query_gate)[head * 2 * head_dim..(head + 1) * 2 * head_dim];
            query.copy_from_slice(&qg[..head_dim]);
            norm(&mut query, self.values(input.query_norm), epsilon, 1.);
            rotate(&mut query, state.position, config);
            let kv_head = head / (heads / kv_heads);
            for (position, score) in scores.iter_mut().enumerate() {
                let start = position * kv_width + kv_head * head_dim;
                *score = dot(&query, &state.keys.buffer.0[start..start + head_dim]) / libm::sqrtf(head_dim as f32);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.;
            for score in &mut scores {
                *score = libm::expf(*score - max);
                total += *score;
            }
            for (position, score) in scores.iter().enumerate() {
                let start = position * kv_width + kv_head * head_dim;
                for (out, value) in out.iter_mut().zip(&state.values.buffer.0[start..start + head_dim]) {
                    *out += *score / total * value;
                }
            }
            for (out, gate) in out.iter_mut().zip(&qg[head_dim..]) {
                *out *= sigmoid(*gate);
            }
        }
        state.position += 1;
        Tensor::from_vec(self, &[1, q_width], output)
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
        let channels = key_width.checked_mul(2).and_then(|x| x.checked_add(value_width)).ok_or(Error::SizeOverflow)?;
        element_count(&[channels, kernel])?;
        element_count(&[heads, key_dim, value_dim])?;
        if !epsilon_valid(epsilon) {
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
        let mut mixed = zeros(&[channels])?;
        let mut output = zeros(&[1, value_width])?;
        let mut prediction = zeros(&[value_dim])?;
        let mut delta = zeros(&[value_dim])?;
        for (channel, ((history, weights), out)) in state
            .convolution
            .buffer
            .0
            .chunks_exact_mut(kernel)
            .zip(self.values(input.convolution).chunks_exact(kernel))
            .zip(&mut mixed)
            .enumerate()
        {
            history.copy_within(1.., 0);
            history[kernel - 1] = self.values(input.qkv)[channel];
            *out = silu(dot(history, weights));
        }
        let (query, rest) = mixed.split_at_mut(key_width);
        let (key, value) = rest.split_at(key_width);
        for head in 0..heads {
            let q = &mut query[head * key_dim..(head + 1) * key_dim];
            let k = &key[head * key_dim..(head + 1) * key_dim];
            let v = &value[head * value_dim..(head + 1) * value_dim];
            let q_scale = 1. / libm::sqrtf(dot(q, q) + 1e-6) / libm::sqrtf(key_dim as f32);
            let k_scale = 1. / libm::sqrtf(dot(k, k) + 1e-6);
            for x in q.iter_mut() {
                *x *= q_scale;
            }
            let beta = sigmoid(self.values(input.beta)[head]);
            let alpha = self.values(input.alpha)[head] + self.values(input.time_bias)[head];
            let softplus = alpha.max(0.) + libm::log1pf(libm::expf(-alpha.abs()));
            let decay = libm::expf(-libm::expf(self.values(input.log_decay)[head]) * softplus);
            let recurrent = &mut state.recurrent.buffer.0[head * key_dim * value_dim..(head + 1) * key_dim * value_dim];
            prediction.fill(0.);
            for (row, &k) in recurrent.chunks_exact_mut(value_dim).zip(k) {
                for (x, predicted) in row.iter_mut().zip(&mut prediction) {
                    *x *= decay;
                    *predicted += *x * (k * k_scale);
                }
            }
            for ((delta, &value), &predicted) in delta.iter_mut().zip(v).zip(&prediction) {
                *delta = (value - predicted) * beta;
            }
            let out = &mut output[head * value_dim..(head + 1) * value_dim];
            for ((row, &k), &q) in recurrent.chunks_exact_mut(value_dim).zip(k).zip(q.iter()) {
                for ((x, &delta), out) in row.iter_mut().zip(&delta).zip(out.iter_mut()) {
                    *x += k * k_scale * delta;
                    *out += *x * q;
                }
            }
            norm(out, self.values(input.norm), epsilon, 0.);
            let gate = &self.values(input.gate)[head * value_dim..(head + 1) * value_dim];
            for (out, &gate) in out.iter_mut().zip(gate) {
                *out *= silu(gate);
            }
        }
        Tensor::from_vec(self, &[1, value_width], output)
    }
}
