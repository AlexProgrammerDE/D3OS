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
    for (m, k, n) in [(1, 1, 1), (1, 33, 17), (3, 11, 5), (17, 33, 9), (1, 1024, 1024)] {
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
        "NVIDIA_BF16_WEIGHTS_OK projections=5 gated_projections=9 embedding_rows=7 roundtrip=exact guards=typed backend=nvidia"
    ));
    Ok(())
}
