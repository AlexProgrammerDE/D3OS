use alloc::{format, string::String, vec, vec::Vec};
use d3os_compute::transformer::{AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaState, TransformerBackend};
use d3os_compute::{Backend, Result, Tensor, cpu::Cpu, nvidia::Nvidia};

fn tensor<B: TransformerBackend>(backend: &B, shape: &[usize], seed: usize) -> Result<Tensor<B>> {
    let count = d3os_compute::element_count(shape)?;
    let values = (0..count).map(|i| (((i * 37 + seed * 17) % 127) as f32 - 63.) / 71.).collect();
    Tensor::from_vec(backend, shape, values)
}

fn attention<B: TransformerBackend>(backend: &B, config: AttentionConfig) -> Result<Vec<Vec<f32>>> {
    let (heads, kv, dim) = (config.heads, config.kv_heads, config.head_dim);
    let mut state = AttentionState {
        keys: Tensor::from_vec(backend, &[7, kv, dim], vec![0.; 7 * kv * dim])?,
        values: Tensor::from_vec(backend, &[7, kv, dim], vec![0.; 7 * kv * dim])?,
        position: 0,
    };
    let query_norm = tensor(backend, &[dim], 3)?;
    let key_norm = tensor(backend, &[dim], 5)?;
    let mut trace = Vec::new();
    for step in 0..7 {
        let query_gate = tensor(backend, &[1, 2 * heads * dim], step + 7)?;
        let key = tensor(backend, &[1, kv * dim], step + 11)?;
        let value = tensor(backend, &[1, kv * dim], step + 17)?;
        let output = backend.attention_step(
            AttentionInputs {
                query_gate: &query_gate,
                key: &key,
                value: &value,
                query_norm: &query_norm,
                key_norm: &key_norm,
            },
            &mut state,
            config,
        )?;
        trace.push(backend.download(&output)?);
        trace.push(backend.download(&state.keys)?);
        trace.push(backend.download(&state.values)?);
    }
    // A rejected call must leave both the position and cache bytes unchanged.
    let before = backend.download(&state.keys)?;
    let query_gate = tensor(backend, &[1, 2 * heads * dim], 31)?;
    let key = tensor(backend, &[1, kv * dim], 32)?;
    let value = tensor(backend, &[1, kv * dim], 33)?;
    if backend
        .attention_step(
            AttentionInputs {
                query_gate: &query_gate,
                key: &key,
                value: &value,
                query_norm: &query_norm,
                key_norm: &key_norm,
            },
            &mut state,
            config,
        )
        .is_ok()
        || state.position != 7
        || backend.download(&state.keys)? != before
    {
        return Err(d3os_compute::Error::DeviceFault);
    }
    Ok(trace)
}

fn delta<B: TransformerBackend>(backend: &B, config: DeltaConfig) -> Result<Vec<Vec<f32>>> {
    let (heads, key, value, kernel) = (config.heads, config.key_dim, config.value_dim, config.kernel);
    let channels = heads * (2 * key + value);
    let mut state = DeltaState {
        convolution: Tensor::from_vec(backend, &[channels, kernel], vec![0.; channels * kernel])?,
        recurrent: Tensor::from_vec(backend, &[heads, key, value], vec![0.; heads * key * value])?,
    };
    let convolution = tensor(backend, &[channels, 1, kernel], 3)?;
    let log_decay = tensor(backend, &[heads], 5)?;
    let time_bias = tensor(backend, &[heads], 7)?;
    let norm = tensor(backend, &[value], 11)?;
    let mut trace = Vec::new();
    for step in 0..7 {
        let qkv = tensor(backend, &[1, channels], step + 13)?;
        let gate = tensor(backend, &[1, heads * value], step + 17)?;
        let beta = tensor(backend, &[1, heads], step + 19)?;
        let alpha = tensor(backend, &[1, heads], step + 23)?;
        let output = backend.delta_step(
            DeltaInputs {
                qkv: &qkv,
                gate: &gate,
                beta: &beta,
                alpha: &alpha,
                convolution: &convolution,
                log_decay: &log_decay,
                time_bias: &time_bias,
                norm: &norm,
            },
            &mut state,
            config,
        )?;
        trace.push(backend.download(&output)?);
        trace.push(backend.download(&state.convolution)?);
        trace.push(backend.download(&state.recurrent)?);
    }
    Ok(trace)
}

fn sequence<B: TransformerBackend>(backend: &B, start: usize, rows: usize, width: usize, seed: usize) -> Result<Tensor<B>> {
    Tensor::from_vec(
        backend,
        &[rows, width],
        (0..rows * width)
            .map(|i| (((i % width * 37 + (start + i / width + seed) * 17) % 127) as f32 - 63.) / 71.)
            .collect(),
    )
}

fn batched_trace(trace: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    for (start, rows) in [(0, 2), (2, 3), (5, 2)] {
        result.push((start..start + rows).flat_map(|i| trace[i * 3].iter().copied()).collect());
        result.push(trace[(start + rows - 1) * 3 + 1].clone());
        result.push(trace[(start + rows - 1) * 3 + 2].clone());
    }
    result
}

fn attention_batches<B: TransformerBackend>(backend: &B, config: AttentionConfig) -> Result<Vec<Vec<f32>>> {
    let (heads, kv, dim) = (config.heads, config.kv_heads, config.head_dim);
    let mut state = AttentionState {
        keys: Tensor::from_vec(backend, &[7, kv, dim], vec![0.; 7 * kv * dim])?,
        values: Tensor::from_vec(backend, &[7, kv, dim], vec![0.; 7 * kv * dim])?,
        position: 0,
    };
    let query_norm = tensor(backend, &[dim], 3)?;
    let key_norm = tensor(backend, &[dim], 5)?;
    let mut trace = Vec::new();
    for (start, rows) in [(0, 2), (2, 3), (5, 2)] {
        let query_gate = sequence(backend, start, rows, 2 * heads * dim, 7)?;
        let key = sequence(backend, start, rows, kv * dim, 11)?;
        let value = sequence(backend, start, rows, kv * dim, 17)?;
        let output = backend.attention_batch(
            AttentionInputs {
                query_gate: &query_gate,
                key: &key,
                value: &value,
                query_norm: &query_norm,
                key_norm: &key_norm,
            },
            &mut state,
            config,
        )?;
        trace.push(backend.download(&output)?);
        trace.push(backend.download(&state.keys)?);
        trace.push(backend.download(&state.values)?);
    }
    Ok(trace)
}

fn delta_batches<B: TransformerBackend>(backend: &B, config: DeltaConfig) -> Result<Vec<Vec<f32>>> {
    let (heads, key, value, kernel) = (config.heads, config.key_dim, config.value_dim, config.kernel);
    let channels = heads * (2 * key + value);
    let mut state = DeltaState {
        convolution: Tensor::from_vec(backend, &[channels, kernel], vec![0.; channels * kernel])?,
        recurrent: Tensor::from_vec(backend, &[heads, key, value], vec![0.; heads * key * value])?,
    };
    let convolution = tensor(backend, &[channels, 1, kernel], 3)?;
    let log_decay = tensor(backend, &[heads], 5)?;
    let time_bias = tensor(backend, &[heads], 7)?;
    let norm = tensor(backend, &[value], 11)?;
    let mut trace = Vec::new();
    for (start, rows) in [(0, 2), (2, 3), (5, 2)] {
        let qkv = sequence(backend, start, rows, channels, 13)?;
        let gate = sequence(backend, start, rows, heads * value, 17)?;
        let beta = sequence(backend, start, rows, heads, 19)?;
        let alpha = sequence(backend, start, rows, heads, 23)?;
        let output = backend.delta_batch(
            DeltaInputs {
                qkv: &qkv,
                gate: &gate,
                beta: &beta,
                alpha: &alpha,
                convolution: &convolution,
                log_decay: &log_decay,
                time_bias: &time_bias,
                norm: &norm,
            },
            &mut state,
            config,
        )?;
        trace.push(backend.download(&output)?);
        trace.push(backend.download(&state.convolution)?);
        trace.push(backend.download(&state.recurrent)?);
    }
    Ok(trace)
}

fn pointwise<B: TransformerBackend>(backend: &B) -> Result<Vec<f32>> {
    let table = tensor(backend, &[7, 11], 3)?;
    let weights = tensor(backend, &[11], 5)?;
    let residual = tensor(backend, &[1, 11], 7)?;
    let embedding = backend.embedding(&table, 3)?;
    let normalized = backend.rms_norm(&embedding, &weights, 1e-6)?;
    let gated = backend.silu_mul(normalized, &residual)?;
    backend.download(&backend.add(gated, &residual)?)
}

fn compare_traces(actual: &[Vec<f32>], expected: &[Vec<f32>], name: &str) -> core::result::Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{name}: trace length differs"));
    }
    for (step, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        super::compare(actual, expected, &format!("{name} trace {step}"))?;
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> core::result::Result<(), String> {
    residual_norm(backend).map_err(|e| format!("residual normalization: {e}"))?;
    preparation(backend)?;
    fused_attention(backend)?;
    recurrent_guards(backend).map_err(|e| format!("recurrent batch guards: {e}"))?;
    let table = backend
        .upload_words(&[1f32.to_bits(), 2f32.to_bits(), 3f32.to_bits(), 4f32.to_bits(), 5f32.to_bits(), 6f32.to_bits()])
        .map_err(|e| format!("{e}"))?;
    let ids = backend.upload_words(&[1, u32::MAX, 2, 3]).map_err(|e| format!("{e}"))?;
    let output = backend.upload_words(&[0xd305abcd; 10]).map_err(|e| format!("{e}"))?;
    backend
        .dispatch_words(d3os_nvidia::compute::Kernel::Embedding, &[(&table, 0), (&ids, 0), (&output, 1)], &[3, 2, 4, 0])
        .map_err(|e| format!("{e}"))?;
    let mut actual = [0; 10];
    backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
    if actual
        != [
            0xd305abcd,
            3f32.to_bits(),
            4f32.to_bits(),
            0,
            0,
            5f32.to_bits(),
            6f32.to_bits(),
            0,
            0,
            0xd305abcd,
        ]
    {
        return Err("embedding gather escaped its table or output range".into());
    }
    for (rows, width) in [(1, 1), (7, 33), (3, 129), (5, 1024), (2, 4097)] {
        let gpu_input = tensor(backend, &[rows, width], 3).map_err(|e| format!("{e}"))?;
        let gpu_weights = tensor(backend, &[width], 5).map_err(|e| format!("{e}"))?;
        let cpu_input = tensor(&Cpu, &[rows, width], 3).map_err(|e| format!("{e}"))?;
        let cpu_weights = tensor(&Cpu, &[width], 5).map_err(|e| format!("{e}"))?;
        let gpu = backend
            .rms_norm(&gpu_input, &gpu_weights, 1e-6)
            .and_then(|t| backend.download(&t))
            .map_err(|e| format!("{e}"))?;
        let cpu = Cpu
            .rms_norm(&cpu_input, &cpu_weights, 1e-6)
            .and_then(|t| Cpu.download(&t))
            .map_err(|e| format!("{e}"))?;
        super::compare(&gpu, &cpu, "block RMSNorm")?;
    }
    let pointwise_gpu = pointwise(backend).map_err(|error| format!("{error}"))?;
    let pointwise_cpu = pointwise(&Cpu).map_err(|error| format!("{error}"))?;
    super::compare(&pointwise_gpu, &pointwise_cpu, "transformer pointwise")?;
    for (heads, kv_heads, head_dim, rotary_dim) in [(4, 2, 10, 6), (8, 2, 256, 64), (2, 1, 257, 64)] {
        let config = AttentionConfig {
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rope_theta: 10000000.,
            epsilon: 1e-6,
        };
        let gpu = attention(backend, config).map_err(|error| format!("attention: {error}"))?;
        let cpu = attention(&Cpu, config).map_err(|error| format!("attention reference: {error}"))?;
        compare_traces(&gpu, &cpu, "grouped attention")?;
        let batch = attention_batches(backend, config).map_err(|e| format!("{e}"))?;
        compare_traces(&batch, &batched_trace(&cpu), "batched attention")?;
    }
    for (heads, key_dim, value_dim, kernel) in [(1, 1, 1, 1), (2, 5, 7, 3), (3, 33, 17, 8), (16, 128, 128, 4), (2, 129, 9, 4), (2, 7, 129, 9)] {
        let config = DeltaConfig {
            heads,
            key_dim,
            value_dim,
            kernel,
            epsilon: 1e-6,
        };
        let gpu = delta(backend, config).map_err(|error| format!("delta: {error}"))?;
        let cpu = delta(&Cpu, config).map_err(|error| format!("delta reference: {error}"))?;
        compare_traces(&gpu, &cpu, "gated delta")?;
        let batch = delta_batches(backend, config).map_err(|e| format!("{e}"))?;
        compare_traces(&batch, &batched_trace(&cpu), "batched delta")?;
    }
    super::super::report(format_args!(
        "NVIDIA_TRANSFORMER_OPERATORS_OK attention_cases=21 delta_cases=42 batch_attention_rows=21 batch_delta_rows=42 backend=nvidia"
    ));
    Ok(())
}

fn residual_norm(backend: &Nvidia) -> Result<()> {
    use d3os_nvidia::compute::Kernel;
    const GUARD: u32 = 0xd305abcd;
    for (rows, width) in [(1, 1), (3, 33), (2, 129), (1, 1024), (17, 1024)] {
        let count = rows * width;
        let input: Vec<f32> = (0..count).map(|i| ((i * 37 % 127) as f32 - 63.) / 71.).collect();
        let residual: Vec<f32> = (0..count).map(|i| ((i * 19 % 127) as f32 - 63.) / 53.).collect();
        let weights: Vec<f32> = (0..width).map(|i| ((i * 13 % 127) as f32 - 63.) / 97.).collect();
        let guarded = |values: &[f32]| {
            core::iter::once(GUARD)
                .chain(values.iter().map(|x| x.to_bits()))
                .chain(core::iter::once(GUARD))
                .collect::<Vec<_>>()
        };
        let a = backend.upload_words(&guarded(&input))?;
        let b = backend.upload_words(&guarded(&residual))?;
        let w = backend.upload_words(&weights.iter().map(|x| x.to_bits()).collect::<Vec<_>>())?;
        let output = backend.upload_words(&vec![GUARD; count + 2])?;
        backend.dispatch_words(
            Kernel::ResidualNorm,
            &[(&a, 1), (&b, 1), (&w, 0), (&output, 1)],
            &[rows, width, 1e-6f32.to_bits() as usize],
        )?;
        let mut updated = vec![0; count + 2];
        let mut normalized = vec![0; count + 2];
        let mut untouched = vec![0; count + 2];
        backend.read_words_into(&a, 0, &mut updated)?;
        backend.read_words_into(&b, 0, &mut untouched)?;
        backend.read_words_into(&output, 0, &mut normalized)?;
        if [updated[0], updated[count + 1], normalized[0], normalized[count + 1]] != [GUARD; 4] || untouched != guarded(&residual) {
            return Err(d3os_compute::Error::DeviceFault);
        }
        let sums: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        if updated[1..count + 1].iter().copied().ne(sums.iter().map(|x| x.to_bits())) {
            return Err(d3os_compute::Error::DeviceFault);
        }
        let expected = Cpu.rms_norm(
            &Tensor::from_vec(&Cpu, &[rows, width], sums)?,
            &Tensor::from_vec(&Cpu, &[width], weights)?,
            1e-6,
        )?;
        let actual: Vec<_> = normalized[1..count + 1].iter().map(|&v| f32::from_bits(v)).collect();
        super::compare(&actual, &Cpu.download(&expected)?, "residual normalization").map_err(|_| d3os_compute::Error::DeviceFault)?;
    }
    Ok(())
}

fn preparation(backend: &Nvidia) -> core::result::Result<(), String> {
    use d3os_nvidia::compute::Kernel;
    for (rows, heads, dim, rotary, position, stride) in [
        (3, 5, 11, 8, 5, 1),
        (3, 5, 11, 8, 5, 2),
        (2, 8, 256, 64, 127, 2),
        (1, 1, 2, 2, 0, 1),
        (2, 3, 65, 64, 11, 1),
        (2, 3, 130, 130, 7, 2),
    ] {
        let run = || -> Result<_> {
            const GUARD: u32 = 0xd305abcd;
            let values: Vec<_> = (0..rows * heads * dim * stride)
                .map(|i| (((i * 37 % 127) as f32 - 63.) / 71.).to_bits())
                .collect();
            let weights: Vec<_> = (0..dim).map(|i| (((i * 17 % 127) as f32 - 63.) / 71.).to_bits()).collect();
            let input = backend.upload_words(&values)?;
            let weights = backend.upload_words(&weights)?;
            let length = rows * heads * dim;
            let expected = backend.upload_words(&vec![GUARD; length + 2])?;
            let actual = backend.upload_words(&vec![GUARD; length + 2])?;
            for row in 0..rows {
                backend.dispatch_words(
                    Kernel::AttentionPrepare,
                    &[(&input, row * heads * dim * stride), (&weights, 0), (&expected, 1 + row * heads * dim)],
                    &[
                        heads,
                        dim,
                        rotary,
                        position + row,
                        1e7f32.to_bits() as usize,
                        1e-6f32.to_bits() as usize,
                        stride,
                    ],
                )?;
            }
            backend.dispatch_words(
                Kernel::AttentionPrepareBatch,
                &[(&input, 0), (&weights, 0), (&actual, 1)],
                &[
                    heads,
                    dim,
                    rotary,
                    position,
                    1e7f32.to_bits() as usize,
                    1e-6f32.to_bits() as usize,
                    stride,
                    rows,
                ],
            )?;
            let mut old = vec![0; length + 2];
            let mut new = vec![0; length + 2];
            backend.read_words_into(&expected, 0, &mut old)?;
            backend.read_words_into(&actual, 0, &mut new)?;
            if old[0] != GUARD || new[0] != GUARD || old[length + 1] != GUARD || new[length + 1] != GUARD {
                return Err(d3os_compute::Error::DeviceFault);
            }
            Ok((
                old[1..length + 1].iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
                new[1..length + 1].iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
            ))
        };
        let (expected, actual) = run().map_err(|e| format!("attention preparation {rows}x{heads}x{dim} rotary={rotary}: {e}"))?;
        super::compare(&actual, &expected, "cooperative attention preparation")?;
    }
    Ok(())
}

fn recurrent_guards(backend: &Nvidia) -> Result<()> {
    use d3os_nvidia::compute::Kernel;
    const GUARD: u32 = 0xd305abcd;
    let parameters = [2, 5, 7, 3, 1e-6f32.to_bits(), 3];
    let work = Kernel::DeltaFusedBatch.work(&parameters).map_err(|_| d3os_compute::Error::InvalidParameter)?;
    let mut sequential = Vec::new();
    let mut batched = Vec::new();
    for (i, &extent) in work.extents[..11].iter().enumerate() {
        let mut values = vec![GUARD; extent as usize + 2];
        if i != 10 {
            for (j, value) in values[1..extent as usize + 1].iter_mut().enumerate() {
                *value = ((((j * 37 + i * 17) % 127) as f32 - 63.) / 71.).to_bits();
            }
        }
        sequential.push(backend.upload_words(&values)?);
        batched.push(backend.upload_words(&values)?);
    }
    for row in 0..3 {
        let buffers: Vec<_> = sequential
            .iter()
            .enumerate()
            .map(|(i, buffer)| {
                let width = match i {
                    0 => 34,
                    1 | 10 => 14,
                    2 | 3 => 2,
                    _ => 0,
                };
                (buffer, 1 + row * width)
            })
            .collect();
        backend.dispatch_words(Kernel::DeltaFused, &buffers, &parameters[..5].iter().map(|&p| p as usize).collect::<Vec<_>>())?;
    }
    let buffers: Vec<_> = batched.iter().map(|buffer| (buffer, 1)).collect();
    backend.dispatch_words(Kernel::DeltaFusedBatch, &buffers, &parameters.iter().map(|&p| p as usize).collect::<Vec<_>>())?;
    for (i, &extent) in work.extents[..11].iter().enumerate() {
        let mut expected = vec![0; extent as usize + 2];
        let mut actual = expected.clone();
        backend.read_words_into(&sequential[i], 0, &mut expected)?;
        backend.read_words_into(&batched[i], 0, &mut actual)?;
        if actual[0] != GUARD || actual[actual.len() - 1] != GUARD {
            return Err(d3os_compute::Error::DeviceFault);
        }
        if i < 8 {
            if actual != expected {
                return Err(d3os_compute::Error::DeviceFault);
            }
        } else {
            let expected: Vec<_> = expected[1..expected.len() - 1].iter().map(|&v| f32::from_bits(v)).collect();
            let actual: Vec<_> = actual[1..actual.len() - 1].iter().map(|&v| f32::from_bits(v)).collect();
            super::compare(&actual, &expected, "recurrent batch storage").map_err(|_| d3os_compute::Error::DeviceFault)?;
        }
    }
    Ok(())
}

fn fused_attention(backend: &Nvidia) -> core::result::Result<(), String> {
    use d3os_nvidia::compute::Kernel;
    const GUARD: u32 = 0xd305abcd;
    for (rows, heads, kv_heads, dim, position, large) in [
        (1, 1, 1, 1, 0, false),
        (3, 4, 2, 10, 5, false),
        (3, 4, 2, 129, 127, false),
        (2, 8, 2, 256, 255, false),
        (1, 2, 1, 31, 1024, false),
        (2, 2, 2, 256, 1023, false),
        (1, 1, 1, 1, 128, true),
    ] {
        let run = || -> Result<_> {
            let width = heads * dim;
            let positions = position + rows;
            let upload = |values: Vec<f32>| {
                let mut words = Vec::with_capacity(values.len() + 2);
                words.push(GUARD);
                words.extend(values.into_iter().map(f32::to_bits));
                words.push(GUARD);
                backend.upload_words(&words)
            };
            let query = upload((0..rows * width).map(|i| 0.25 + (i * 11 % 31) as f32 / 32.).collect())?;
            let keys = upload(
                (0..positions * kv_heads * dim)
                    .map(|i| {
                        if large {
                            0.
                        } else {
                            let sign = if i / dim % kv_heads % 2 == 0 { 1. } else { -1. };
                            ((i * 37 % 127) as f32 - 63.) / 127. + (i / (kv_heads * dim)) as f32 * 0.004 * sign
                        }
                    })
                    .collect(),
            )?;
            let values = upload(
                (0..positions * kv_heads * dim)
                    .map(|i| if large { 1e37 } else { ((i * 73 % 251) as f32 - 125.) / 127. })
                    .collect(),
            )?;
            let gate = upload((0..rows * width * 2).map(|i| ((i * 13 % 71) as f32 - 35.) / 41.).collect())?;
            let expected = backend.upload_words(&vec![GUARD; rows * width + 2])?;
            let actual = backend.upload_words(&vec![GUARD; rows * width + 2])?;
            let scores = backend.allocate_words(heads * positions)?;
            for row in 0..rows {
                let visible = position + row + 1;
                backend.dispatch_words(
                    Kernel::AttentionScores,
                    &[(&query, 1 + row * width), (&keys, 1), (&scores, 0)],
                    &[heads, kv_heads, dim, visible],
                )?;
                backend.dispatch_words(Kernel::Softmax, &[(&scores, 0), (&scores, 0)], &[heads, visible])?;
                backend.dispatch_words(
                    Kernel::AttentionOutput,
                    &[(&scores, 0), (&values, 1), (&gate, 1 + row * width * 2), (&expected, 1 + row * width)],
                    &[heads, kv_heads, dim, visible],
                )?;
            }
            backend.dispatch_words(
                Kernel::AttentionFusedBatch,
                &[(&query, 1), (&keys, 1), (&values, 1), (&gate, 1), (&actual, 1)],
                &[rows, heads, kv_heads, dim, position],
            )?;
            let mut old = vec![0; rows * width + 2];
            let mut new = old.clone();
            backend.read_words_into(&expected, 0, &mut old)?;
            backend.read_words_into(&actual, 0, &mut new)?;
            if old[0] != GUARD || new[0] != GUARD || old[old.len() - 1] != GUARD || new[new.len() - 1] != GUARD {
                return Err(d3os_compute::Error::DeviceFault);
            }
            Ok((
                old[1..old.len() - 1].iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
                new[1..new.len() - 1].iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
            ))
        };
        let (expected, actual) = run().map_err(|e| format!("fused attention {rows}x{heads}x{dim} position={position}: {e}"))?;
        super::compare(&actual, &expected, "streaming attention")?;
    }
    Ok(())
}
