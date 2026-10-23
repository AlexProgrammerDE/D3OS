//! Packed-weight checks include odd rows and tails absent from the Qwen model.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_compute::{Backend, Error, Tensor, nvidia::Nvidia, transformer::TransformerBackend};

fn gated_projection<B: TransformerBackend>(backend: &B, m: usize, k: usize, n: usize, packed: [bool; 2]) -> d3os_compute::Result<Vec<f32>> {
    let input = Tensor::from_vec(backend, &[m, k], (0..m * k).map(|i| ((i * 73 % 251) as f32 - 125.) / 127.).collect())?;
    let weight = |salt: usize, packed: bool| {
        let bits = (0..n * k).map(|i| (((((i * 37 + salt * 17) % 127) as f32 - 63.) / 71.).to_bits() >> 16) as u16);
        let buffer = if packed {
            backend.upload_bf16(bits)?
        } else {
            backend.upload(bits.map(|bits| f32::from_bits(u32::from(bits) << 16)).collect())?
        };
        Tensor::from_buffer(&[n, k], buffer)
    };
    let gate = weight(3, packed[0])?;
    let up = weight(7, packed[1])?;
    backend.download(&backend.gated_project(&input, &gate, &up)?)
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    split_guards(backend).map_err(|e| format!("split projection guards: {e}"))?;
    grouped_guards(backend).map_err(|e| format!("grouped projection guards: {e}"))?;
    for (m, k, n) in [
        (1, 1, 1),
        (1, 2, 1),
        (1, 33, 17),
        (1, 63, 17),
        (1, 64, 17),
        (1, 65, 17),
        (1, 128, 2051),
        (1, 64, 8193),
        (1, 65, 8193),
        (3, 11, 5),
        (17, 33, 9),
        (1, 1024, 1024),
        (17, 1025, 33),
        (17, 1025, 257),
    ] {
        let input: Vec<f32> = (0..m * k).map(|i| ((i * 73 % 251) as f32 - 125.) / 127.).collect();
        let weights: Vec<u16> = (0..n * k).map(|i| ((((i * 37 % 127) as f32 - 63.) / 71.).to_bits() >> 16) as u16).collect();
        let mut expected = vec![0.; m * n];
        for row in 0..m {
            for col in 0..n {
                expected[row * n + col] = (0..k)
                    .map(|j| input[row * k + j] as f64 * f32::from_bits(u32::from(weights[col * k + j]) << 16) as f64)
                    .sum::<f64>() as f32;
            }
        }
        let input = Tensor::from_vec(backend, &[m, k], input).map_err(|e| format!("{e}"))?;
        let weights = Tensor::from_buffer(&[n, k], backend.upload_bf16(weights.into_iter()).map_err(|e| format!("{e}"))?).map_err(|e| format!("{e}"))?;
        let output = backend
            .project(&input, &weights)
            .and_then(|t| backend.download(&t))
            .map_err(|e| format!("{e}"))?;
        super::compare(&output, &expected, "BF16-weight projection")?;
        let residual: Vec<f32> = (0..m * n).map(|i| ((i * 13 % 29) as f32 - 14.) / 17.).collect();
        let expected: Vec<_> = expected.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let residual = Tensor::from_vec(backend, &[m, n], residual).map_err(|e| format!("{e}"))?;
        let output = backend
            .project_add(&input, &weights, residual)
            .and_then(|v| backend.download(&v))
            .map_err(|e| format!("{e}"))?;
        super::compare(&output, &expected, "BF16 residual projection")?;
    }
    let bits: Vec<u16> = (0..77).map(|i| [0, 0x8000, 1, 0x807f, 0x3f80, 0xbf80, 0x7f7f][i % 7]).collect();
    let expected: Vec<_> = bits.iter().map(|&bits| f32::from_bits(u32::from(bits) << 16)).collect();
    let table = Tensor::from_buffer(&[7, 11], backend.upload_bf16(bits.into_iter()).map_err(|e| format!("{e}"))?).map_err(|e| format!("{e}"))?;
    let values = backend.download(&table).map_err(|e| format!("{e}"))?;
    if values.iter().map(|v| v.to_bits()).ne(expected.iter().map(|v| v.to_bits())) {
        return Err("BF16-weight roundtrip changed bits".into());
    }
    for token in 0..7 {
        let row = backend
            .embedding(&table, token)
            .and_then(|t| backend.download(&t))
            .map_err(|e| format!("{e}"))?;
        if row
            .iter()
            .map(|v| v.to_bits())
            .ne(expected[token * 11..(token + 1) * 11].iter().map(|v| v.to_bits()))
        {
            return Err(format!("BF16 embedding row {token} changed bits"));
        }
    }
    let tokens = [6u32, 1, 0, 6, 4];
    let gathered = backend.embedding_batch(&table, &tokens).map_err(|e| format!("{e}"))?;
    let expected_batch: Vec<_> = tokens
        .iter()
        .flat_map(|&token| expected[token as usize * 11..(token as usize + 1) * 11].iter().map(|v| v.to_bits()))
        .collect();
    let row = backend.row(&gathered, 1).and_then(|row| backend.relu(row)).map_err(|e| format!("{e}"))?;
    let actual = backend.download(&gathered).map_err(|e| format!("{e}"))?;
    if actual.iter().map(|v| v.to_bits()).ne(expected_batch) {
        return Err("batched embedding or row ownership changed bits".into());
    }
    drop(row);
    if !matches!(backend.relu(table), Err(Error::InvalidParameter)) {
        return Err("FP32 kernel accepted packed BF16 storage".into());
    }
    for value in [0x7f80, 0xff80, 0x7fc1] {
        if !matches!(backend.upload_bf16([value].into_iter()), Err(Error::NonFinite)) {
            return Err("BF16 upload accepted a non-finite weight".into());
        }
    }
    for (m, k, n, packed) in [
        (1, 1, 1, [true, true]),
        (1, 33, 17, [true, true]),
        (1, 63, 17, [true, true]),
        (1, 64, 17, [true, true]),
        (1, 65, 17, [true, true]),
        (1, 128, 2051, [true, true]),
        (1, 129, 65, [true, true]),
        (1, 33, 2051, [true, true]),
        (1, 1024, 3584, [true, true]),
        (3, 11, 5, [true, true]),
        (1, 33, 17, [false, false]),
        (1, 33, 17, [true, false]),
        (1, 33, 17, [false, true]),
    ] {
        let gpu = gated_projection(backend, m, k, n, packed).map_err(|e| format!("{e}"))?;
        let cpu = gated_projection(&d3os_compute::cpu::Cpu, m, k, n, packed).map_err(|e| format!("{e}"))?;
        super::compare(&gpu, &cpu, "gated projection")?;
    }
    super::super::report(format_args!(
        "NVIDIA_BF16_WEIGHTS_OK projections=14 residual_projections=14 gated_projections=13 embedding_rows=7 roundtrip=exact guards=typed backend=nvidia"
    ));
    Ok(())
}

fn grouped_guards(backend: &Nvidia) -> d3os_compute::Result<()> {
    use d3os_nvidia::compute::Kernel;
    const GUARD: u32 = 0xd305abcd;
    for inner in [6usize, 130] {
        let input: Vec<_> = (0..inner).map(|i| ((i % 3) as i32 - 1) as f32).collect();
        let a = backend.upload_words(&input.iter().map(|x| x.to_bits()).collect::<Vec<_>>())?;
        let weight = |group: usize, col: usize, k: usize| ((group * 13 + col * 3 + k) % 17) as i32 - 8;
        let mut matrices = Vec::new();
        let mut expected = Vec::new();
        for (group, cols) in [4usize, 8, 4, 4].into_iter().enumerate() {
            let bits: Vec<_> = (0..cols)
                .flat_map(|col| (0..inner).map(move |k| (weight(group, col, k) as f32).to_bits() >> 16))
                .collect();
            let mut words = vec![GUARD];
            words.extend(bits.chunks_exact(2).map(|pair| pair[0] | (pair[1] << 16)));
            words.push(GUARD);
            matrices.push(backend.upload_words(&words)?);
            for col in 0..cols {
                expected.push((0..inner).map(|k| input[k] * weight(group, col, k) as f32).sum::<f32>().to_bits());
            }
        }
        let output = backend.upload_words(&vec![GUARD; expected.len() + 2])?;
        backend.dispatch_words(
            Kernel::GroupedProjectBf16,
            &[
                (&a, 0),
                (&matrices[0], 1),
                (&matrices[1], 1),
                (&matrices[2], 1),
                (&matrices[3], 1),
                (&output, 1),
            ],
            &[inner, 4, 8, 4],
        )?;
        let mut actual = vec![0; expected.len() + 2];
        backend.read_words_into(&output, 0, &mut actual)?;
        if actual[0] != GUARD || actual[actual.len() - 1] != GUARD || actual[1..actual.len() - 1] != expected {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

fn split_guards(backend: &Nvidia) -> d3os_compute::Result<()> {
    use d3os_nvidia::compute::Kernel;
    const GUARD: u32 = 0xd305abcd;
    let input = backend.upload_words(&[1f32.to_bits()])?;
    let bits: Vec<_> = (1..=9).map(|i| ((i as f32).to_bits() >> 16) as u16).collect();
    let weights = backend.upload_words(
        &bits
            .chunks(2)
            .map(|pair| u32::from(pair[0]) | (u32::from(*pair.get(1).unwrap_or(&0)) << 16))
            .collect::<Vec<_>>(),
    )?;
    for splits in [1, 4] {
        let partials = backend.upload_words(&vec![GUARD; splits * 9 + 2])?;
        let output = backend.upload_words(&[GUARD; 11])?;
        backend.dispatch_words(Kernel::ProjectBf16Split, &[(&input, 0), (&weights, 0), (&partials, 1)], &[1, 1, 9, splits])?;
        backend.dispatch_words(Kernel::SumProjectionPartials, &[(&partials, 1), (&output, 1)], &[1, 9, splits])?;
        let mut parts = vec![0; splits * 9 + 2];
        let mut actual = [0; 11];
        backend.read_words_into(&partials, 0, &mut parts)?;
        backend.read_words_into(&output, 0, &mut actual)?;
        if parts[0] != GUARD || parts[parts.len() - 1] != GUARD || actual[0] != GUARD || actual[10] != GUARD {
            return Err(Error::DeviceFault);
        }
        for i in 0..9 {
            if parts[i + 1] != ((i + 1) as f32).to_bits() || actual[i + 1] != parts[i + 1] {
                return Err(Error::DeviceFault);
            }
        }
        if parts[10..parts.len() - 1].iter().any(|&v| v != 0) {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}
