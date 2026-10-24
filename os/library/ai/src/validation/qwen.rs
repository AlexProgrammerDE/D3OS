//! Independent packed-weight and ownership checks shared by CPU and physical GPU runs.
use crate::qwen::{MatrixFormat, PackedMatrix, Quantization, residual_norm, rms_norm};
use alloc::{format, string::String, vec::Vec};
use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Device, Tensor, TensorData},
};

type Result<T> = core::result::Result<T, String>;

fn check<const D: usize>(tensor: Tensor<D>, expected: &[f32]) -> Result<()> {
    let actual = tensor.try_into_vec_as::<f32>().map_err(|e| format!("Qwen Burn read: {e:?}"))?;
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 3e-5 + 3e-5 * b.abs())
    {
        return Err(format!("Qwen Burn reference differs: actual={actual:?} expected={expected:?}"));
    }
    Ok(())
}

fn bytes(values: &[f32], format: MatrixFormat) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| {
            let word = v.to_le_bytes();
            if format == MatrixFormat::Bf16 { word[2..].to_vec() } else { word.to_vec() }
        })
        .collect()
}

fn reference(input: &[f32], weights: &[f32], batch: usize, width: usize, outputs: usize) -> Vec<f32> {
    (0..batch * outputs)
        .map(|i| {
            (0..width)
                .map(|k| f64::from(input[i / outputs * width + k]) * f64::from(weights[i % outputs * width + k]))
                .sum::<f64>() as f32
        })
        .collect()
}

fn roundtrip(matrix: &PackedMatrix) -> Result<PackedMatrix> {
    let bytes = matrix.clone().into_record().into_bytes().map_err(|e| format!("packed weight record: {e}"))?;
    let record = ModuleRecord::from_bytes(bytes).map_err(|e| format!("packed weight decode: {e}"))?;
    let restored = matrix.clone().try_load_record(record).map_err(|e| format!("packed weight restore: {e}"))?;
    if restored.device() != matrix.device()
        || restored.storage_words() != matrix.storage_words()
        || restored.shape() != matrix.shape()
        || restored.format() != matrix.format()
    {
        return Err(String::from("packed weight restore changed its device or metadata"));
    }
    Ok(restored)
}

fn matrices(device: &Device) -> Result<()> {
    let values: Vec<_> = (0..15).map(|i| (i as f32 - 7.) / 8.).collect();
    let input: Vec<_> = (0..10).map(|i| i as f32 / 4. - 1.).collect();
    // An offset and a negative stride must be materialized on the GPU, without a readback.
    let source = Tensor::<2>::from_data(TensorData::new(input.clone(), [2, 5]), device);
    for format in [MatrixFormat::F32, MatrixFormat::Bf16] {
        let matrix = PackedMatrix::from_bytes([3, 5], format, &bytes(&values, format), device)?;
        if matrix.storage_words() != (15usize * if format == MatrixFormat::F32 { 4 } else { 2 }).div_ceil(4) {
            return Err(String::from("Qwen physical packing differs"));
        }
        let actual = matrix.project(source.clone())?;
        check(actual, &reference(&input, &values, 2, 5, 3))?;
        let restored = roundtrip(&matrix)?;
        check(restored.embedding(&[2, 0, 2])?, &[&values[10..], &values[..5], &values[10..]].concat())?;
        let reversed: Vec<_> = input.chunks_exact(5).flat_map(|row| row.iter().rev().copied()).collect();
        check(matrix.project(source.clone().flip([1]))?, &reference(&reversed, &values, 2, 5, 3))?;
        check(matrix.project(source.clone().slice([1..2, 0..5]))?, &reference(&input[5..], &values, 1, 5, 3))?;
        let gate = reference(&input, &values, 2, 5, 3);
        let expected: Vec<_> = gate.iter().map(|g| g * g / (1. + libm_exp(-g))).collect();
        check(matrix.gated_project(&matrix, source.clone())?, &expected)?;
        if matrix.embedding(&[3]).is_ok() || matrix.embedding(&[]).is_ok() || matrix.project(Tensor::<2>::zeros([1, 4], device)).is_ok() {
            return Err(String::from("Qwen accepted invalid matrix inputs"));
        }
    }
    check(source, &input)?;
    Ok(())
}

// Use Burn-independent scalar references; native and Flex use their own implementations.
fn libm_exp(value: f32) -> f32 {
    libm::expf(value)
}

fn fused_and_ownership(device: &Device) -> Result<()> {
    let weights = [1., 2., -1., 0.5, -0.5, 1., 0.25, -2., 0., 1., 2., 3., -1., 0., 1., 0.];
    let matrix = PackedMatrix::from_bytes([4, 4], MatrixFormat::Bf16, &bytes(&weights, MatrixFormat::Bf16), device)?;
    let original = Tensor::<2>::from_floats([[1., 2., 3., 4.]], device);
    let projected = reference(&[1., 2., 3., 4.], &weights, 1, 4, 4);
    let expected: Vec<_> = projected.iter().zip([1., 2., 3., 4.]).map(|(a, b)| a + b).collect();
    check(matrix.project_add(original.clone(), original.clone())?, &expected)?;
    check(original.clone(), &[1., 2., 3., 4.])?;
    // A surviving row view must retain its old values when the sibling is consumed.
    let storage = Tensor::<2>::from_floats([[17., 18., 19., 20.], [1., 2., 3., 4.]], device);
    let snapshot = storage.clone().slice([1..2, 0..4]);
    check(matrix.project_add(original.clone(), storage.slice([1..2, 0..4]))?, &expected)?;
    check(snapshot, &[1., 2., 3., 4.])?;
    check(
        matrix.project_add(original.clone(), Tensor::<2>::from_floats([[1., 2., 3., 4.]], device))?,
        &expected,
    )?;
    // Also exercise the single-token BF16 fused gate path with an even inner dimension.
    let expected_gate: Vec<_> = projected.iter().map(|g| g * g / (1. + libm_exp(-g))).collect();
    check(matrix.gated_project(&matrix, original.clone())?, &expected_gate)?;
    for width in [4, 1025] {
        let values: Vec<_> = (0..width * 2).map(|i| (i as f32 % 23. - 11.) / 8.).collect();
        let residual: Vec<_> = (0..width * 2).map(|i| (i as f32 % 7. - 3.) / 4.).collect();
        let scales: Vec<_> = (0..width).map(|i| (i as f32 % 5. - 2.) / 8.).collect();
        let input = Tensor::<2>::from_data(TensorData::new(values.clone(), [2, width]), device);
        let other = Tensor::<2>::from_data(TensorData::new(residual.clone(), [2, width]), device);
        let norm = Tensor::<1>::from_data(TensorData::new(scales.clone(), [width]), device);
        let sum: Vec<_> = values.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let expected: Vec<_> = sum
            .chunks_exact(width)
            .flat_map(|row| {
                let mean = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / width as f64;
                row.iter()
                    .zip(&scales)
                    .map(move |(&v, &w)| (f64::from(v) * (1. + f64::from(w)) / libm::sqrt(mean + 1e-6)) as f32)
            })
            .collect();
        let (added, normalized) = residual_norm(input.clone(), other.clone(), norm.clone(), 1e-6)?;
        check(added.clone(), &sum)?;
        check(normalized, &expected)?;
        check(rms_norm(added, norm, 1e-6)?, &expected)?;
        check(input, &values)?;
        check(other, &residual)?;
    }
    Ok(())
}

fn split_projection(device: &Device) -> Result<()> {
    let (batch, width, outputs) = (3, 512, 256);
    let weights: Vec<_> = (0..width * outputs).map(|i| ((i * 17 % 29) as f32 - 14.) / 32.).collect();
    let input: Vec<_> = (0..width * batch).map(|i| ((i * 11 % 31) as f32 - 15.) / 32.).collect();
    let matrix = PackedMatrix::from_bytes([outputs, width], MatrixFormat::Bf16, &bytes(&weights, MatrixFormat::Bf16), device)?;
    check(
        matrix.project(Tensor::from_data(TensorData::new(input.clone(), [batch, width]), device))?,
        &reference(&input, &weights, batch, width, outputs),
    )
}

fn quantized(device: &Device) -> Result<()> {
    let mut fixture = &include_bytes!("../../../compute/tests/fixtures/quantized.bin")[8..];
    for _ in 0..5 {
        let format = Quantization::from_raw(u32::from_le_bytes(fixture[..4].try_into().unwrap())).unwrap();
        let width = u32::from_le_bytes(fixture[4..8].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(fixture[8..12].try_into().unwrap()) as usize;
        let packed = &fixture[12..12 + count];
        let expected: Vec<_> = fixture[12 + count..12 + count + width * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let matrix = PackedMatrix::from_bytes([2, width], MatrixFormat::Quantized(format), &[packed, packed].concat(), device)?;
        let restored = roundtrip(&matrix)?;
        check(restored.embedding(&[1, 0, 1])?, &[&expected[..], &expected[..], &expected[..]].concat())?;
        for batch in [1, 3] {
            let input: Vec<_> = (0..batch * width).map(|i| ((i * 7 % 17) as f32 - 8.) / 32.).collect();
            let result = reference(&input, &[&expected[..], &expected[..]].concat(), batch, width, 2);
            check(matrix.project(Tensor::from_data(TensorData::new(input, [batch, width]), device))?, &result)?;
        }
        fixture = &fixture[12 + count + width * 4..];
    }
    Ok(())
}

pub fn run(device: &Device) -> Result<()> {
    matrices(device)?;
    fused_and_ownership(device)?;
    split_projection(device)?;
    quantized(device)?;
    Ok(())
}
