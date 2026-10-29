use alloc::{format, string::String, vec::Vec};
use d3os_compute::{Backend, Tensor, cpu::Cpu, nvidia::Nvidia, transformer::TransformerBackend};
use d3os_tensor_file::quantized::Quantization;

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let mut fixture = &include_bytes!("../../../../library/compute/tests/fixtures/quantized.bin")[8..];
    for _ in 0..5 {
        let word = |offset| u32::from_le_bytes(fixture[offset..offset + 4].try_into().expect("fixture header")) as usize;
        let format = Quantization::from_raw(word(0) as u32).map_err(|e| format!("quantized format: {e}"))?;
        let elements = word(4);
        let size = word(8);
        let bytes = &fixture[12..12 + size];
        let expected: Vec<f32> = fixture[12 + size..12 + size + elements * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("fixture float")))
            .collect();
        verify(backend, format, bytes, &expected).map_err(|e| format!("quantized {format:?}: {e}"))?;
        fixture = &fixture[12 + size + elements * 4..];
    }
    super::super::report(format_args!("NVIDIA_QUANTIZED_OK formats=5 source=checkpoint_blocks backend=nvidia"));
    Ok(())
}

fn verify(backend: &Nvidia, format: Quantization, bytes: &[u8], expected: &[f32]) -> Result<(), String> {
    guards(backend, format, bytes, expected).map_err(|e| format!("packed buffer guards: {e}"))?;
    let check = || -> d3os_compute::Result<()> {
        // Exercise borrowed sources at every word alignment, including a
        // single block whose final two bytes need a padded upload word.
        for offset in 0..4 {
            for blocks in [1, 3, expected.len() / format.block_elements()] {
                let size = blocks * format.block_bytes();
                let n = blocks * format.block_elements();
                let mut unaligned = alloc::vec![0u8; offset];
                unaligned.extend_from_slice(&bytes[..size]);
                let weights = Tensor::from_buffer(&[blocks, format.block_elements()], backend.upload_quantized(format, n, &unaligned[offset..])?)?;
                let decoded = backend.download(&weights)?;
                if decoded.iter().zip(expected).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    return Err(d3os_compute::Error::DeviceFault);
                }
                let tokens = [0, blocks as u32 - 1, 0];
                let gathered = backend.download(&backend.embedding_batch(&weights, &tokens)?)?;
                for (token, row) in tokens.iter().zip(gathered.chunks_exact(format.block_elements())) {
                    let start = *token as usize * format.block_elements();
                    if row.iter().zip(&expected[start..]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        return Err(d3os_compute::Error::DeviceFault);
                    }
                }
            }
        }
        Ok(())
    };
    check().map_err(|e| format!("weight upload, decode, or embedding: {e}"))?;
    for (rows, inner, outputs) in [
        (1, format.block_elements(), 1),
        (1, format.block_elements() * 3, 5),
        (1, format.block_elements() * 32, 13),
        (3, format.block_elements(), 5),
        (17, format.block_elements() * 2, 33),
        (33, format.block_elements() * 3, 65),
    ] {
        let raw: Vec<u8> = bytes
            .iter()
            .copied()
            .cycle()
            .take(format.byte_len(inner * outputs).map_err(|e| format!("weight size: {e}"))?)
            .collect();
        let input: Vec<f32> = (0..rows * inner).map(|i| ((i * 37 % 127) as f32 - 63.) / 71.).collect();
        let run = |gpu: bool| -> d3os_compute::Result<Vec<f32>> {
            if gpu {
                let x = Tensor::from_vec(backend, &[rows, inner], input.clone())?;
                let w = Tensor::from_buffer(&[outputs, inner], backend.upload_quantized(format, inner * outputs, &raw)?)?;
                backend.download(&backend.project(&x, &w)?)
            } else {
                let x = Tensor::from_vec(&Cpu, &[rows, inner], input.clone())?;
                let w = Tensor::from_buffer(&[outputs, inner], Cpu.upload_quantized(format, inner * outputs, &raw)?)?;
                Cpu.download(&Cpu.project(&x, &w)?)
            }
        };
        let actual = run(true).map_err(|e| format!("projection {rows}x{inner}x{outputs}: {e}"))?;
        let expected = run(false).map_err(|e| format!("reference projection: {e}"))?;
        super::compare(&actual, &expected, "quantized projection")?;
    }
    Ok(())
}

fn guards(backend: &Nvidia, format: Quantization, bytes: &[u8], expected: &[f32]) -> d3os_compute::Result<()> {
    use d3os_compute::Error;
    use d3os_nvidia::compute::Kernel;
    const GUARD: u32 = 0xd305abcd;
    let n = format.block_elements();
    let mut packed = alloc::vec![GUARD];
    for chunk in bytes[..format.block_bytes()].chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        packed.push(u32::from_le_bytes(word));
    }
    packed.push(GUARD);
    let weights = backend.upload_words(&packed)?;
    let output = backend.upload_words(&alloc::vec![GUARD; n + 2])?;
    backend.dispatch_words(Kernel::Dequantize, &[(&weights, 1), (&output, 1)], &[n, format as usize])?;
    let mut actual = alloc::vec![0; n + 2];
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual[0] != GUARD || actual[n + 1] != GUARD || actual[1..n + 1].iter().zip(expected).any(|(a, b)| *a != b.to_bits()) {
        return Err(Error::DeviceFault);
    }
    for rows in [1, 17, 33] {
        let input = backend.upload_words(&alloc::vec![1f32.to_bits(); rows * n])?;
        let output = backend.upload_words(&alloc::vec![GUARD; rows + 2])?;
        backend.dispatch_words(
            Kernel::ProjectQuantized,
            &[(&input, 0), (&weights, 1), (&output, 1)],
            &[rows, n, 1, format as usize],
        )?;
        let mut actual = alloc::vec![0; rows + 2];
        backend.read_words_into(&output, 0, &mut actual)?;
        let expected = expected[..n].iter().sum::<f32>();
        if actual[0] != GUARD
            || actual[rows + 1] != GUARD
            || actual[1..rows + 1]
                .iter()
                .any(|a| (f32::from_bits(*a) - expected).abs() > 2e-4 + 2e-4 * expected.abs())
        {
            return Err(Error::DeviceFault);
        }
    }
    let ids = backend.upload_words(&[0, u32::MAX, 0])?;
    let output = backend.upload_words(&alloc::vec![GUARD; n * 3 + 2])?;
    backend.dispatch_words(
        Kernel::EmbeddingQuantized,
        &[(&weights, 1), (&ids, 0), (&output, 1)],
        &[1, n, 3, format as usize],
    )?;
    let mut actual = alloc::vec![0; n * 3 + 2];
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual[0] != GUARD || actual[n * 3 + 1] != GUARD || actual[n + 1..n * 2 + 1].iter().any(|a| *a != 0) {
        return Err(Error::DeviceFault);
    }
    for row in [0, 2] {
        if actual[row * n + 1..(row + 1) * n + 1].iter().zip(expected).any(|(a, b)| *a != b.to_bits()) {
            return Err(Error::DeviceFault);
        }
    }
    let mut unchanged = alloc::vec![0; packed.len()];
    backend.read_words_into(&weights, 0, &mut unchanged)?;
    if unchanged != packed {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

/// Exercise the optional PTX module through guarded SDK dispatch and public Burn storage.
pub(super) fn components(backend: &Nvidia) -> Result<(), String> {
    use d3os_ai::{
        burn::tensor::Tensor as BurnTensor,
        qwen::{MatrixFormat, PackedMatrix, Q3VectorKernel},
    };
    use d3os_nvidia::compute::Kernel;
    let format = Quantization::Q3K;
    let mut fixture = &include_bytes!("../../../../library/compute/tests/fixtures/quantized.bin")[8..];
    let blocks = loop {
        let word = |offset| u32::from_le_bytes(fixture[offset..offset + 4].try_into().unwrap()) as usize;
        let (kind, elements, bytes) = (word(0), word(4), word(8));
        if kind == format as usize {
            break &fixture[12..12 + bytes];
        }
        fixture = &fixture[12 + bytes + elements * 4..];
    };
    let device = d3os_ai::native_device().map_err(|e| format!("{e}"))?;
    let mut values = 0;
    for [inner, columns] in [
        [256, 1],
        [256, 3],
        [512, 4],
        [768, 31],
        [1024, 32],
        [5120, 33],
        [5120, 129],
        [17408, 33],
        [5120, 17408],
    ] {
        let raw: Vec<u8> = blocks
            .iter()
            .copied()
            .cycle()
            .take(format.byte_len(inner * columns).map_err(|e| format!("{e}"))?)
            .collect();
        let input: Vec<f32> = (0..inner).map(|i| ((i * 37 % 127) as f32 - 63.) / 71.).collect();
        let reference = (|| -> d3os_compute::Result<Vec<f32>> {
            let x = Tensor::from_vec(&Cpu, &[1, inner], input.clone())?;
            let weights = Tensor::from_buffer(&[columns, inner], Cpu.upload_quantized(format, inner * columns, &raw)?)?;
            Cpu.download(&Cpu.project(&x, &weights)?)
        })()
        .map_err(|e| format!("Q3 CPU projection: {e}"))?;
        let check = |actual: &[f32]| -> Result<(), String> {
            if actual.len() != columns
                || actual
                    .iter()
                    .zip(&reference)
                    .any(|(a, b)| !a.is_finite() || (a - b).abs() > 2e-4 + 2e-4 * b.abs())
            {
                return Err(format!("Q3 component projection mismatch at {inner}x{columns}"));
            }
            Ok(())
        };
        const GUARD: u32 = 0xd305abcd;
        let source: Vec<u32> = core::iter::once(GUARD)
            .chain(input.iter().map(|x| x.to_bits()))
            .chain(core::iter::once(GUARD))
            .collect();
        let weights: Vec<u32> = core::iter::once(GUARD)
            .chain(raw.chunks(4).map(|b| {
                let mut word = [0; 4];
                word[..b.len()].copy_from_slice(b);
                u32::from_le_bytes(word)
            }))
            .chain(core::iter::once(GUARD))
            .collect();
        let scratch_words = inner * 3 / 4 + inner / 32;
        let guarded = (|| -> d3os_compute::Result<()> {
            let x = backend.upload_words(&source)?;
            let weights_gpu = backend.upload_words(&weights)?;
            let scratch = backend.upload_words(&alloc::vec![GUARD;scratch_words+2])?;
            let output = backend.upload_words(&alloc::vec![GUARD;columns+2])?;
            for poison in [0x7fc12345, 0x5a5a5a5a] {
                let poison_data = backend.upload_words(&alloc::vec![poison;columns])?;
                backend.dispatch_words(Kernel::Copy, &[(&poison_data, 0), (&output, 1)], &[columns])?;
                backend.dispatch_words(Kernel::ProjectQuantized, &[(&x, 1), (&scratch, 1)], &[1, inner, columns, 8, 0, 3])?;
                backend.dispatch_words(
                    Kernel::ProjectQuantized,
                    &[(&scratch, 1), (&weights_gpu, 1), (&output, 1)],
                    &[1, inner, columns, 11, 1, 3],
                )?;
                let mut actual = alloc::vec![0;columns+2];
                backend.read_words_into(&output, 0, &mut actual)?;
                if actual[0] != GUARD || actual[columns + 1] != GUARD || actual[1..columns + 1].contains(&poison) {
                    return Err(d3os_compute::Error::DeviceFault);
                }
                check(&actual[1..columns + 1].iter().map(|x| f32::from_bits(*x)).collect::<Vec<_>>()).map_err(|_| d3os_compute::Error::DeviceFault)?;
                let mut storage = alloc::vec![0;scratch_words+2];
                backend.read_words_into(&scratch, 0, &mut storage)?;
                if storage[0] != GUARD || storage[scratch_words + 1] != GUARD {
                    return Err(d3os_compute::Error::DeviceFault);
                }
            }
            for (buffer, expected) in [(&x, &source), (&weights_gpu, &weights)] {
                let mut actual = alloc::vec![0;expected.len()];
                backend.read_words_into(buffer, 0, &mut actual)?;
                if actual != *expected {
                    return Err(d3os_compute::Error::DeviceFault);
                }
            }
            Ok(())
        })();
        guarded.map_err(|e| format!("Q3 guarded dispatch {inner}x{columns}: {e}"))?;
        let mut matrix = PackedMatrix::from_bytes([columns, inner], MatrixFormat::Quantized(format), &raw, &device).map_err(|e| format!("{e}"))?;
        matrix.set_q3_vector_kernel(Q3VectorKernel::Components).map_err(|e| format!("{e}"))?;
        let actual = matrix
            .project(BurnTensor::from_data(d3os_ai::burn::tensor::TensorData::new(input, [1, inner]), &device))
            .map_err(|e| format!("{e}"))?
            .try_into_vec_as::<f32>()
            .map_err(|e| format!("{e}"))?;
        check(&actual)?;
        values += columns * 3;
    }
    super::super::report(format_args!(
        "NVIDIA_Q3_COMPONENTS_OK cases=9 comparisons={values} source=checkpoint_blocks guards=ok public=ok"
    ));
    Ok(())
}
