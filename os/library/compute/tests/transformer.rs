use d3os_compute::transformer::{AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaState, TransformerBackend};
use d3os_compute::{Backend, Tensor, cpu::Cpu};

fn tensor(shape: &[usize], values: &[f32]) -> Tensor<Cpu> {
    Tensor::from_vec(&Cpu, shape, values.to_vec()).unwrap()
}

#[test]
fn batched_attention_rejects_insufficient_cache_before_any_update() {
    let config = AttentionConfig {
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        rotary_dim: 2,
        rope_theta: 10000.,
        epsilon: 1e-6,
    };
    let mut state = AttentionState {
        keys: tensor(&[1, 1, 2], &[7., 8.]),
        values: tensor(&[1, 1, 2], &[9., 10.]),
        position: 0,
    };
    let query = tensor(&[2, 4], &[0.2; 8]);
    let key = tensor(&[2, 2], &[0.3; 4]);
    let value = tensor(&[2, 2], &[0.4; 4]);
    let norm = tensor(&[2], &[0.; 2]);
    assert!(
        Cpu.attention_batch(
            AttentionInputs {
                query_gate: &query,
                key: &key,
                value: &value,
                query_norm: &norm,
                key_norm: &norm
            },
            &mut state,
            config
        )
        .is_err()
    );
    assert_eq!(state.position, 0);
    assert_eq!(Cpu.values(&state.keys), &[7., 8.]);
    assert_eq!(Cpu.values(&state.values), &[9., 10.]);
}
fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a as f64 - b).abs() < 3e-5, "{a} != {b}");
    }
}
fn norm(row: &[f64], weights: &[f32], epsilon: f64, offset: f64) -> Vec<f64> {
    let scale = (row.iter().map(|x| x * x).sum::<f64>() / row.len() as f64 + epsilon).sqrt().recip();
    row.iter().zip(weights).map(|(x, w)| x * scale * (offset + *w as f64)).collect()
}

#[test]
fn projection_normalization_and_gated_activation_handle_non_simd_tails() {
    let values: Vec<_> = (0..22).map(|i| (i as f32 - 7.) / 9.).collect();
    let weights: Vec<_> = (0..33).map(|i| (i as f32 - 20.) / 11.).collect();
    let out = Cpu.project(&tensor(&[2, 11], &values), &tensor(&[3, 11], &weights)).unwrap();
    let expected: Vec<_> = values
        .chunks_exact(11)
        .flat_map(|row| {
            weights
                .chunks_exact(11)
                .map(move |w| row.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum())
        })
        .collect();
    close(Cpu.values(&out), &expected);
    let weights = [0.2, -0.5, 0.7];
    let normalized = Cpu.rms_norm(&out, &tensor(&[3], &weights), 1e-6).unwrap();
    let expected: Vec<_> = expected.chunks_exact(3).flat_map(|row| norm(row, &weights, 1e-6, 1.)).collect();
    close(Cpu.values(&normalized), &expected);
    let other = tensor(&[2, 3], &[1., -1., 2., 0.5, 1., 2.]);
    let actual = Cpu.silu_mul(normalized, &other).unwrap();
    let expected: Vec<_> = expected
        .iter()
        .zip(Cpu.values(&other))
        .map(|(x, y)| x / (1. + (-x).exp()) * *y as f64)
        .collect();
    close(Cpu.values(&actual), &expected);
    assert!(Cpu.rms_norm(&out, &tensor(&[3], &weights), 0.).is_err());
    assert!(Cpu.embedding(&out, 2).is_err());
}

#[test]
fn grouped_attention_cache_matches_independent_full_causal_attention() {
    let config = AttentionConfig {
        heads: 4,
        kv_heads: 2,
        head_dim: 4,
        rotary_dim: 2,
        rope_theta: 10000.,
        epsilon: 1e-6,
    };
    let qn = tensor(&[4], &[0.1, -0.1, 0.3, 0.2]);
    let kn = tensor(&[4], &[-0.1, 0.2, 0.1, 0.4]);
    let mut state = AttentionState {
        keys: tensor(&[3, 2, 4], &[0.; 24]),
        values: tensor(&[3, 2, 4], &[0.; 24]),
        position: 0,
    };
    let mut keys: Vec<Vec<f64>> = Vec::new();
    let mut values: Vec<Vec<f64>> = Vec::new();
    for position in 0..3 {
        let qg: Vec<_> = (0..32).map(|i| ((i * 7 + position * 3) as f32 - 60.) / 100.).collect();
        let k: Vec<_> = (0..8).map(|i| ((i * 3 + position * 2) as f32 - 10.) / 10.).collect();
        let v: Vec<_> = (0..8).map(|i| ((i + position * 4) as f32 - 4.) / 7.).collect();
        let rotate = |mut row: Vec<f64>| {
            let (s, c) = (position as f64).sin_cos();
            let (a, b) = (row[0], row[1]);
            row[0] = a * c - b * s;
            row[1] = b * c + a * s;
            row
        };
        keys.push(
            k.chunks_exact(4)
                .flat_map(|row| rotate(norm(&row.iter().map(|&x| x as f64).collect::<Vec<_>>(), Cpu.values(&kn), 1e-6, 1.)))
                .collect(),
        );
        values.push(v.iter().map(|&x| x as f64).collect());
        let mut expected = Vec::new();
        for head in 0..4 {
            let q = rotate(norm(
                &qg[head * 8..head * 8 + 4].iter().map(|&x| x as f64).collect::<Vec<_>>(),
                Cpu.values(&qn),
                1e-6,
                1.,
            ));
            let kv = head / 2;
            let scores: Vec<_> = keys
                .iter()
                .map(|key| q.iter().zip(&key[kv * 4..kv * 4 + 4]).map(|(q, k)| q * k).sum::<f64>() / 2.)
                .collect();
            let sum: f64 = scores.iter().map(|x| x.exp()).sum();
            for d in 0..4 {
                let value: f64 = scores.iter().zip(&values).map(|(score, value)| score.exp() / sum * value[kv * 4 + d]).sum();
                expected.push(value / (1. + (-qg[head * 8 + 4 + d] as f64).exp()));
            }
        }
        let out = Cpu
            .attention_step(
                AttentionInputs {
                    query_gate: &tensor(&[1, 32], &qg),
                    key: &tensor(&[1, 8], &k),
                    value: &tensor(&[1, 8], &v),
                    query_norm: &qn,
                    key_norm: &kn,
                },
                &mut state,
                config,
            )
            .unwrap();
        close(Cpu.values(&out), &expected);
        assert_eq!(state.position, position + 1);
    }
    let before = Cpu.download(&state.keys).unwrap();
    assert!(
        Cpu.attention_step(
            AttentionInputs {
                query_gate: &tensor(&[1, 32], &[1.; 32]),
                key: &tensor(&[1, 8], &[1.; 8]),
                value: &tensor(&[1, 8], &[1.; 8]),
                query_norm: &qn,
                key_norm: &kn
            },
            &mut state,
            config
        )
        .is_err()
    );
    assert_eq!(Cpu.values(&state.keys), before);
    assert_eq!(state.position, 3);
}

#[test]
fn delta_recurrence_matches_scalar_state_updates_across_convolution_history() {
    let config = DeltaConfig {
        heads: 2,
        key_dim: 2,
        value_dim: 3,
        kernel: 3,
        epsilon: 1e-6,
    };
    let conv: Vec<_> = (0..42).map(|i| (i % 3 + 1) as f32 / 6.).collect();
    let conv = tensor(&[14, 1, 3], &conv);
    let log_decay = tensor(&[2], &[-0.3, 0.2]);
    let time = tensor(&[2], &[0.1, -0.4]);
    let weights = tensor(&[3], &[0.7, 1.2, -0.4]);
    let mut cache = DeltaState {
        convolution: tensor(&[14, 3], &[0.; 42]),
        recurrent: tensor(&[2, 2, 3], &[0.; 12]),
    };
    let mut history = [[0f64; 3]; 14];
    let mut recurrent = [[[0f64; 3]; 2]; 2];
    for step in 0..5 {
        let qkv: Vec<_> = (0..14).map(|i| ((i + step * 2) as f32 - 5.) / 11.).collect();
        let gate: Vec<_> = (0..6).map(|i| (i as f32 - step as f32) / 7.).collect();
        let beta = [0.3, -0.2];
        let alpha = [step as f32 / 10., -0.1];
        let mut mixed = [0f64; 14];
        for channel in 0..14 {
            history[channel].rotate_left(1);
            history[channel][2] = qkv[channel] as f64;
            let x: f64 = history[channel]
                .iter()
                .zip(&Cpu.values(&conv)[channel * 3..channel * 3 + 3])
                .map(|(x, w)| x * *w as f64)
                .sum();
            mixed[channel] = x / (1. + (-x).exp());
        }
        let mut expected = Vec::new();
        for head in 0..2 {
            let mut q = mixed[head * 2..head * 2 + 2].to_vec();
            let mut k = mixed[4 + head * 2..4 + head * 2 + 2].to_vec();
            let q_scale = (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt().recip() / 2f64.sqrt();
            let k_scale = (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt().recip();
            q.iter_mut().for_each(|x| *x *= q_scale);
            k.iter_mut().for_each(|x| *x *= k_scale);
            let decay = (-((Cpu.values(&log_decay)[head] as f64).exp()) * (1. + ((alpha[head] + Cpu.values(&time)[head]) as f64).exp()).ln()).exp();
            let rate = 1. / (1. + (-beta[head] as f64).exp());
            let mut out = [0f64; 3];
            for (d, out) in out.iter_mut().enumerate() {
                for row in &mut recurrent[head] {
                    row[d] *= decay;
                }
                let predicted: f64 = (0..2).map(|r| k[r] * recurrent[head][r][d]).sum();
                let delta = (mixed[8 + head * 3 + d] - predicted) * rate;
                for r in 0..2 {
                    recurrent[head][r][d] += k[r] * delta;
                    *out += q[r] * recurrent[head][r][d];
                }
            }
            let normalized = norm(&out, Cpu.values(&weights), 1e-6, 0.);
            expected.extend(
                normalized
                    .iter()
                    .zip(&gate[head * 3..head * 3 + 3])
                    .map(|(x, g)| x * *g as f64 / (1. + (-*g as f64).exp())),
            );
        }
        let actual = Cpu
            .delta_step(
                DeltaInputs {
                    qkv: &tensor(&[1, 14], &qkv),
                    gate: &tensor(&[1, 6], &gate),
                    beta: &tensor(&[1, 2], &beta),
                    alpha: &tensor(&[1, 2], &alpha),
                    convolution: &conv,
                    log_decay: &log_decay,
                    time_bias: &time,
                    norm: &weights,
                },
                &mut cache,
                config,
            )
            .unwrap();
        close(Cpu.values(&actual), &expected);
        close(Cpu.values(&cache.recurrent), &recurrent.iter().flatten().flatten().copied().collect::<Vec<_>>());
    }
}
