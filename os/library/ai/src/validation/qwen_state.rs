//! Binary64 references cover complete outputs, cache continuations, and snapshots.
use crate::qwen::{
    AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaParameters, DeltaState, DeltaWeights, MatrixFormat, PackedMatrix,
};
use alloc::{format, string::String, vec::Vec};
use burn::tensor::{Device, Tensor, TensorData};
use d3os_tensor_file::{Archive, Record};
type Result<T> = core::result::Result<T, String>;
const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/qwen-state.d3t");

fn record<'a>(archive: &mut Archive<'a>, prefix: &str, name: &str) -> Result<Record<'a>> {
    archive.take(&format!("{prefix}.{name}")).map_err(|e| format!("Qwen state fixture: {e}"))
}
fn tensor<const D: usize>(archive: &mut Archive<'_>, prefix: &str, name: &str, device: &Device) -> Result<Tensor<D>> {
    let record = record(archive, prefix, name)?;
    if record.shape().len() != D {
        return Err(String::from("Qwen state fixture rank differs"));
    }
    let values = record.to_vec().map_err(|e| format!("Qwen state fixture values: {e}"))?;
    Ok(Tensor::from_data(TensorData::new(values, record.shape()), device))
}
fn data(archive: &mut Archive<'_>, prefix: &str, name: &str) -> Result<Vec<f32>> {
    record(archive, prefix, name)?.to_vec().map_err(|e| format!("Qwen state reference: {e}"))
}
fn check<const D: usize>(tensor: Tensor<D>, expected: &[f32], context: &str) -> Result<()> {
    let actual = tensor.try_into_vec_as::<f32>().map_err(|e| format!("Qwen state {context}: {e:?}"))?;
    if actual.len() != expected.len() {
        return Err(format!("Qwen state {context} length differs"));
    }
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() || (actual - expected).abs() > 3e-5 + 3e-5 * expected.abs() {
            return Err(format!("Qwen state {context}[{i}]: actual={actual} expected={expected}"));
        }
    }
    Ok(())
}
fn steps(rows: usize, batched: bool) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while start < rows {
        let count = (if batched { [2, 3, 2][index % 3] } else { 1 }).min(rows - start);
        result.push((start, start + count));
        start += count;
        index += 1;
    }
    result
}
fn attention(device: &Device, case: usize, batched: bool) -> Result<()> {
    let mut archive = Archive::parse(FIXTURE).map_err(|e| format!("Qwen state fixture: {e}"))?;
    let prefix = format!("attention.{case}");
    let c = data(&mut archive, &prefix, "config")?;
    let config = AttentionConfig {
        heads: c[0] as usize,
        kv_heads: c[1] as usize,
        head_dim: c[2] as usize,
        rotary_dim: c[3] as usize,
        rope_theta: c[4],
        epsilon: c[5],
    };
    let position = c[6] as usize;
    let qg = tensor::<2>(&mut archive, &prefix, "qg", device)?;
    let rows = qg.dims()[0];
    let key = tensor::<2>(&mut archive, &prefix, "key", device)?;
    let value = tensor::<2>(&mut archive, &prefix, "value", device)?;
    let qnorm = tensor(&mut archive, &prefix, "qnorm", device)?;
    let knorm = tensor(&mut archive, &prefix, "knorm", device)?;
    let keys_record = record(&mut archive, &prefix, "keys")?;
    let initial_keys = keys_record.to_vec().map_err(|e| format!("cache fixture: {e}"))?;
    // A signed-stride cache must become private dense storage before native mutation.
    let keys = if case == 0 {
        let reversed: Vec<_> = initial_keys.chunks_exact(config.head_dim).flat_map(|row| row.iter().rev().copied()).collect();
        Tensor::<3>::from_data(TensorData::new(reversed, keys_record.shape()), device).flip([2])
    } else {
        Tensor::<3>::from_data(TensorData::new(initial_keys.clone(), keys_record.shape()), device)
    };
    let values_record = record(&mut archive, &prefix, "values")?;
    let initial_values = values_record.to_vec().map_err(|e| format!("cache fixture: {e}"))?;
    let values = Tensor::<3>::from_data(TensorData::new(initial_values.clone(), values_record.shape()), device);
    let expected = data(&mut archive, &prefix, "output")?;
    let final_keys = data(&mut archive, &prefix, "final_keys")?;
    let final_values = data(&mut archive, &prefix, "final_values")?;
    let mut state = AttentionState::from_tensors(keys, values, position, config)?;
    let snapshot = state.clone();
    let inputs = |start, end| AttentionInputs {
        query_gate: qg.clone().slice([start..end, 0..qg.dims()[1]]),
        key: key.clone().slice([start..end, 0..key.dims()[1]]),
        value: value.clone().slice([start..end, 0..value.dims()[1]]),
        query_norm: qnorm.clone(),
        key_norm: knorm.clone(),
    };
    let width = config.heads * config.head_dim;
    for (start, end) in steps(rows, batched) {
        let (next, output) = state.forward(inputs(start, end), config)?;
        state = next;
        check(output, &expected[start * width..end * width], &format!("{prefix} batched={batched} output"))?;
    }
    if state.position() != position + rows || snapshot.position() != position {
        return Err(String::from("Qwen attention state position changed incorrectly"));
    }
    check(state.keys().clone(), &final_keys, &prefix)?;
    check(state.values().clone(), &final_values, &prefix)?;
    check(snapshot.keys().clone(), &initial_keys, "attention snapshot keys")?;
    check(snapshot.values().clone(), &initial_values, "attention snapshot values")?;
    if state.clone().forward(inputs(0, 3), config).is_ok() {
        return Err(String::from("Qwen attention accepted insufficient capacity"));
    }
    check(state.keys().clone(), &final_keys, "rejected attention cache")?;
    Ok(())
}
fn delta(device: &Device, case: usize, batched: bool, encoding: Option<usize>) -> Result<()> {
    let mut archive = Archive::parse(FIXTURE).map_err(|e| format!("Qwen state fixture: {e}"))?;
    let prefix = format!("delta.{case}");
    let c = data(&mut archive, &prefix, "config")?;
    let config = DeltaConfig {
        key_heads: c[0] as usize,
        heads: c[1] as usize,
        key_dim: c[2] as usize,
        value_dim: c[3] as usize,
        kernel: c[4] as usize,
        epsilon: c[5],
    };
    let qkv = tensor::<2>(&mut archive, &prefix, "qkv", device)?;
    let rows = qkv.dims()[0];
    let channels = qkv.dims()[1];
    let gate = tensor::<2>(&mut archive, &prefix, "gate", device)?;
    let beta = tensor::<2>(&mut archive, &prefix, "beta", device)?;
    let alpha = tensor::<2>(&mut archive, &prefix, "alpha", device)?;
    let parameters = DeltaParameters {
        convolution: tensor(&mut archive, &prefix, "weights", device)?,
        log_decay: tensor(&mut archive, &prefix, "decay", device)?,
        time_bias: tensor(&mut archive, &prefix, "bias", device)?,
        norm: tensor(&mut archive, &prefix, "rms", device)?,
    };
    let history_record = record(&mut archive, &prefix, "convolution")?;
    let history = history_record.to_vec().map_err(|e| format!("history: {e}"))?;
    let recurrent_record = record(&mut archive, &prefix, "recurrent")?;
    let recurrent = recurrent_record.to_vec().map_err(|e| format!("recurrent: {e}"))?;
    let mut state = DeltaState::from_tensors(
        Tensor::from_data(TensorData::new(history.clone(), history_record.shape()), device),
        Tensor::from_data(TensorData::new(recurrent.clone(), recurrent_record.shape()), device),
        config,
    )?;
    let snapshot = state.clone();
    let expected = data(&mut archive, &prefix, "output")?;
    let final_history = data(&mut archive, &prefix, "final_convolution")?;
    let final_recurrent = data(&mut archive, &prefix, "final_recurrent")?;
    let mut matrices = Vec::new();
    let mut hidden = None;
    if let Some(encoding) = encoding {
        hidden = Some(tensor::<2>(&mut archive, &prefix, "hidden", device)?);
        for (index, name) in ["qkv", "gate", "beta", "alpha"].iter().enumerate() {
            let record = record(&mut archive, &prefix, &format!("weight_{name}"))?;
            let format = if encoding == 0 || (encoding == 2 && index.is_multiple_of(2)) {
                MatrixFormat::Bf16
            } else {
                MatrixFormat::F32
            };
            let bytes = if format == MatrixFormat::Bf16 {
                record
                    .to_vec()
                    .map_err(|e| format!("matrix: {e}"))?
                    .iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect()
            } else {
                record.bytes().to_vec()
            };
            matrices.push(PackedMatrix::from_bytes([record.shape()[0], record.shape()[1]], format, &bytes, device)?);
        }
    }
    for (start, end) in steps(rows, batched) {
        let (next, output) = if let Some(hidden) = &hidden {
            state.project(
                hidden.clone().slice([start..end, 0..hidden.dims()[1]]),
                DeltaWeights {
                    qkv: &matrices[0],
                    gate: &matrices[1],
                    beta: &matrices[2],
                    alpha: &matrices[3],
                },
                parameters.clone(),
                config,
            )?
        } else {
            state.forward(
                DeltaInputs {
                    qkv: qkv.clone().slice([start..end, 0..channels]),
                    gate: gate.clone().slice([start..end, 0..gate.dims()[1]]),
                    beta: beta.clone().slice([start..end, 0..config.heads]),
                    alpha: alpha.clone().slice([start..end, 0..config.heads]),
                },
                parameters.clone(),
                config,
            )?
        };
        state = next;
        let width = config.heads * config.value_dim;
        check(
            output,
            &expected[start * width..end * width],
            &format!("{prefix} batched={batched} encoding={encoding:?} output"),
        )?;
    }
    check(state.convolution().clone(), &final_history, &prefix)?;
    check(state.recurrent().clone(), &final_recurrent, &prefix)?;
    check(snapshot.convolution().clone(), &history, "delta history snapshot")?;
    check(snapshot.recurrent().clone(), &recurrent, "delta recurrent snapshot")?;
    Ok(())
}

pub fn run(device: &Device) -> Result<()> {
    for batched in [false, true] {
        for case in 0..4 {
            attention(device, case, batched)?;
        }
        for case in 0..5 {
            delta(device, case, batched, None)?;
        }
        for encoding in 0..3 {
            delta(device, 4, batched, Some(encoding))?;
        }
    }
    Ok(())
}
