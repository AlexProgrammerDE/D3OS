//! Explicit Flex CPU implementations. Native dispatch never enters this module.
use super::{
    AttentionConfig, DeltaConfig,
    ops::{AttentionPrimitive, AttentionResult, DeltaPrimitive, DeltaResult},
};
use alloc::{vec, vec::Vec};
use burn::{
    backend::{Flex, TensorMetadata, ops::FloatTensorOps, read_sync, tensor::FloatTensor},
    tensor::{Shape, TensorData},
};

fn values(tensor: FloatTensor<Flex>) -> Vec<f32> {
    read_sync(Flex::float_into_data(tensor)).unwrap().try_to_vec::<f32>().unwrap()
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.; 8];
    let mut a = a.chunks_exact(8);
    let mut b = b.chunks_exact(8);
    for (a, b) in a.by_ref().zip(b.by_ref()) {
        for lane in 0..8 {
            lanes[lane] += a[lane] * b[lane];
        }
    }
    lanes.iter().sum::<f32>() + a.remainder().iter().zip(b.remainder()).map(|(a, b)| a * b).sum::<f32>()
}
fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + libm::expf(-x))
    } else {
        let e = libm::expf(x);
        e / (1. + e)
    }
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn norm(row: &mut [f32], weights: &[f32], epsilon: f32, offset: f32) {
    let scale = 1. / libm::sqrtf(dot(row, row) / row.len() as f32 + epsilon);
    for (x, w) in row.iter_mut().zip(weights) {
        *x = *x * scale * (offset + w);
    }
}
fn rotate(row: &mut [f32], position: usize, config: AttentionConfig) {
    let half = config.rotary_dim / 2;
    for i in 0..half {
        let angle = position as f32 / libm::powf(config.rope_theta, (2 * i) as f32 / config.rotary_dim as f32);
        let (a, b) = (row[i], row[i + half]);
        let (sin, cos) = (libm::sinf(angle), libm::cosf(angle));
        row[i] = a * cos - b * sin;
        row[i + half] = b * cos + a * sin;
    }
}

pub(super) fn attention(input: AttentionPrimitive<Flex>, config: AttentionConfig) -> AttentionResult<Flex> {
    let device = input.query_gate.device();
    let rows = input.query_gate.shape()[0];
    let cache_shape = input.keys.shape();
    let (heads, kv_heads, dim) = (config.heads, config.kv_heads, config.head_dim);
    let q_width = heads * dim;
    let kv_width = kv_heads * dim;
    let qg = values(input.query_gate);
    let key = values(input.key);
    let value = values(input.value);
    let qn = values(input.query_norm);
    let kn = values(input.key_norm);
    let mut keys = values(input.keys);
    let mut cached_values = values(input.values);
    let mut output = vec![0.; rows * q_width];
    let mut query = vec![0.; dim];
    let mut prepared = vec![0.; kv_width];
    for row in 0..rows {
        let position = input.position + row;
        prepared.copy_from_slice(&key[row * kv_width..(row + 1) * kv_width]);
        for head in prepared.chunks_exact_mut(dim) {
            norm(head, &kn, config.epsilon, 1.);
            rotate(head, position, config);
        }
        keys[position * kv_width..(position + 1) * kv_width].copy_from_slice(&prepared);
        cached_values[position * kv_width..(position + 1) * kv_width].copy_from_slice(&value[row * kv_width..(row + 1) * kv_width]);
        let mut scores = vec![0.; position + 1];
        for head in 0..heads {
            let qg = &qg[row * 2 * q_width + head * 2 * dim..row * 2 * q_width + (head + 1) * 2 * dim];
            query.copy_from_slice(&qg[..dim]);
            norm(&mut query, &qn, config.epsilon, 1.);
            rotate(&mut query, position, config);
            let kv_head = head / (heads / kv_heads);
            for (p, score) in scores.iter_mut().enumerate() {
                *score = dot(&query, &keys[p * kv_width + kv_head * dim..p * kv_width + (kv_head + 1) * dim]) / libm::sqrtf(dim as f32);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.;
            for score in &mut scores {
                *score = libm::expf(*score - max);
                total += *score;
            }
            let out = &mut output[row * q_width + head * dim..row * q_width + (head + 1) * dim];
            for (p, score) in scores.iter().enumerate() {
                for (out, value) in out
                    .iter_mut()
                    .zip(&cached_values[p * kv_width + kv_head * dim..p * kv_width + (kv_head + 1) * dim])
                {
                    *out += *score / total * value;
                }
            }
            for (out, gate) in out.iter_mut().zip(&qg[dim..]) {
                *out *= sigmoid(*gate);
            }
        }
    }
    AttentionResult {
        output: Flex::float_from_data(TensorData::new(output, [rows, q_width]), &device),
        keys: Flex::float_from_data(TensorData::new(keys, cache_shape.clone()), &device),
        values: Flex::float_from_data(TensorData::new(cached_values, cache_shape), &device),
    }
}

pub(super) fn delta(input: DeltaPrimitive<Flex>, config: DeltaConfig) -> DeltaResult<Flex> {
    let device = input.qkv.device();
    let rows = input.qkv.shape()[0];
    let convolution_shape = input.convolution.shape();
    let recurrent_shape = input.recurrent.shape();
    let DeltaConfig {
        key_heads,
        heads,
        key_dim,
        value_dim,
        kernel,
        epsilon,
    } = config;
    let key_width = key_heads * key_dim;
    let value_width = heads * value_dim;
    let channels = 2 * key_width + value_width;
    let qkv = values(input.qkv);
    let gate = values(input.gate);
    let beta = values(input.beta);
    let alpha = values(input.alpha);
    let params = input.parameters;
    let weights = values(params.convolution);
    let log_decay = values(params.log_decay);
    let time_bias = values(params.time_bias);
    let rms = values(params.norm);
    let mut convolution = values(input.convolution);
    let mut recurrent = values(input.recurrent);
    let mut output = vec![0.; rows * value_width];
    let mut mixed = vec![0.; channels];
    let mut prediction = vec![0.; value_dim];
    let mut delta = vec![0.; value_dim];
    for row in 0..rows {
        for (channel, ((history, weights), out)) in convolution
            .chunks_exact_mut(kernel)
            .zip(weights.chunks_exact(kernel))
            .zip(&mut mixed)
            .enumerate()
        {
            history.copy_within(1.., 0);
            history[kernel - 1] = qkv[row * channels + channel];
            *out = silu(dot(history, weights));
        }
        let (query, rest) = mixed.split_at_mut(key_width);
        let (key, value) = rest.split_at(key_width);
        for q in query.chunks_exact_mut(key_dim) {
            let scale = 1. / libm::sqrtf(dot(q, q) + 1e-6) / libm::sqrtf(key_dim as f32);
            for x in q {
                *x *= scale;
            }
        }
        for head in 0..heads {
            let key_head = head / (heads / key_heads);
            let q = &query[key_head * key_dim..(key_head + 1) * key_dim];
            let k = &key[key_head * key_dim..(key_head + 1) * key_dim];
            let v = &value[head * value_dim..(head + 1) * value_dim];
            let k_scale = 1. / libm::sqrtf(dot(k, k) + 1e-6);
            let beta = sigmoid(beta[row * heads + head]);
            let alpha = alpha[row * heads + head] + time_bias[head];
            let softplus = alpha.max(0.) + libm::log1pf(libm::expf(-alpha.abs()));
            let decay = libm::expf(-libm::expf(log_decay[head]) * softplus);
            let state = &mut recurrent[head * key_dim * value_dim..(head + 1) * key_dim * value_dim];
            prediction.fill(0.);
            for (state, &k) in state.chunks_exact_mut(value_dim).zip(k) {
                for (x, predicted) in state.iter_mut().zip(&mut prediction) {
                    *x *= decay;
                    *predicted += *x * (k * k_scale);
                }
            }
            for ((delta, &value), &predicted) in delta.iter_mut().zip(v).zip(&prediction) {
                *delta = (value - predicted) * beta;
            }
            let out = &mut output[row * value_width + head * value_dim..row * value_width + (head + 1) * value_dim];
            for ((state, &k), &q) in state.chunks_exact_mut(value_dim).zip(k).zip(q) {
                for ((x, &delta), out) in state.iter_mut().zip(&delta).zip(out.iter_mut()) {
                    *x += k * k_scale * delta;
                    *out += *x * q;
                }
            }
            norm(out, &rms, epsilon, 0.);
            for (out, &gate) in out
                .iter_mut()
                .zip(&gate[row * value_width + head * value_dim..row * value_width + (head + 1) * value_dim])
            {
                *out *= silu(gate);
            }
        }
    }
    DeltaResult {
        output: Flex::float_from_data(TensorData::new(output, Shape::new([rows, value_width])), &device),
        convolution: Flex::float_from_data(TensorData::new(convolution, convolution_shape), &device),
        recurrent: Flex::float_from_data(TensorData::new(recurrent, recurrent_shape), &device),
    }
}
