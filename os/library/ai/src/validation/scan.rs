//! Public cumulative operations and gradients with independent scalar references.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, Int, Tensor, kind::Ordered},
};

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Sum,
    Product,
    Min,
    Max,
}
const OPERATIONS: [Operation; 4] = [Operation::Sum, Operation::Product, Operation::Min, Operation::Max];

fn apply<const D: usize, K: Ordered>(input: Tensor<D, K>, axis: usize, operation: Operation) -> Tensor<D, K> {
    match operation {
        Operation::Sum => input.cumsum(axis),
        Operation::Product => input.cumprod(axis),
        Operation::Min => input.cummin(axis),
        Operation::Max => input.cummax(axis),
    }
}

fn prefix(shape: &[usize], axis: usize, index: usize) -> (usize, usize, usize) {
    let suffix = shape[axis + 1..].iter().product::<usize>();
    let coordinate = index / suffix % shape[axis];
    (index - coordinate * suffix, coordinate + 1, suffix)
}

fn integers(actual: &[i32], input: &[i32], shape: &[usize], axis: usize, op: Operation) -> Result<(), String> {
    if actual.len() != input.len() {
        return Err(String::from("integer scan output length"));
    }
    for (i, &actual) in actual.iter().enumerate() {
        let (base, count, step) = prefix(shape, axis, i);
        let mut expected = input[base];
        for k in 1..count {
            let x = input[base + k * step];
            expected = match op {
                Operation::Sum => (i128::from(expected) + i128::from(x)) as i32,
                Operation::Product => (i128::from(expected) * i128::from(x)) as i32,
                Operation::Min => expected.min(x),
                Operation::Max => expected.max(x),
            };
        }
        if actual != expected {
            return Err(format!("I32 scan {op:?} axis={axis} [{i}]: {actual} != {expected}"));
        }
    }
    Ok(())
}

fn floats(actual: &[f32], input: &[f32], shape: &[usize], axis: usize, op: Operation) -> Result<(), String> {
    if actual.len() != input.len() {
        return Err(String::from("float scan output length"));
    }
    for (i, &actual) in actual.iter().enumerate() {
        let (base, count, step) = prefix(shape, axis, i);
        let mut expected = match op {
            Operation::Sum => 0.,
            Operation::Product => 1.,
            _ => f64::from(input[base]),
        };
        let mut norm = 0.;
        for k in 0..count {
            let x = f64::from(input[base + k * step]);
            norm += x.abs();
            expected = match op {
                Operation::Sum => expected + x,
                Operation::Product => expected * x,
                _ if expected.is_nan() => expected,
                _ if x.is_nan() => x,
                Operation::Min => expected.min(x),
                Operation::Max => expected.max(x),
            };
        }
        let expected = expected as f32;
        let tolerance = match op {
            Operation::Sum => 2e-6 * (1. + norm),
            Operation::Product => 2e-5 * (1. + f64::from(expected.abs())),
            _ => 0.,
        };
        let agrees = if expected.is_nan() {
            actual.is_nan()
        } else if !expected.is_finite() || (expected == 0. && !matches!(op, Operation::Sum)) {
            actual.to_bits() == expected.to_bits()
        } else {
            actual.is_finite() && f64::from((actual - expected).abs()) <= tolerance
        };
        if !agrees {
            return Err(format!("F32 scan {op:?} axis={axis} [{i}]: {actual:?} != {expected:?}, tolerance={tolerance}"));
        }
    }
    Ok(())
}

pub fn gradients(device: &Device) -> Result<(), String> {
    let device = device.clone().autodiff();
    let weights: [f32; 7] = [1., -2., 3., -4., 5., -6., 7.];
    let cases: [[f32; 7]; 3] = [
        [-2., 1., 0., -3., -3., 2., -1.],
        [0., 2., 0., -1., 3., 1., -2.],
        [0.5, 1.25, -0.75, 1.5, 0.9, 1.1, -1.2],
    ];
    for values in cases {
        for op in OPERATIONS {
            let input = Tensor::<1>::from_data(TensorData::new(values.to_vec(), [7]), &device).require_grad();
            let output = apply(input.clone(), 0, op);
            let loss = (output * Tensor::<1>::from_data(TensorData::new(weights.to_vec(), [7]), &device)).sum();
            let gradients = loss.backward();
            let actual = input
                .grad(&gradients)
                .ok_or_else(|| String::from("missing cumulative gradient"))?
                .try_into_vec_as::<f32>()
                .map_err(|e| format!("scan gradient read: {e:?}"))?;
            let mut expected = [0f64; 7];
            let mut winner = 0;
            for j in 0..7 {
                match op {
                    Operation::Sum | Operation::Product => {
                        for (i, value) in expected.iter_mut().enumerate().take(j + 1) {
                            let mut derivative = f64::from(weights[j]);
                            if matches!(op, Operation::Product) {
                                for (k, &x) in values.iter().enumerate().take(j + 1) {
                                    if k != i {
                                        derivative *= f64::from(x);
                                    }
                                }
                            }
                            *value += derivative;
                        }
                    }
                    Operation::Min | Operation::Max => {
                        if (matches!(op, Operation::Min) && values[j] <= values[winner]) || (matches!(op, Operation::Max) && values[j] >= values[winner]) {
                            winner = j;
                        }
                        expected[winner] += f64::from(weights[j]);
                    }
                }
            }
            if actual.len() != 7 {
                return Err(String::from("scan gradient output length"));
            }
            for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                if !actual.is_finite() || (f64::from(actual) - expected).abs() > 2e-5 * (1. + expected.abs()) {
                    return Err(format!("scan gradient {op:?}[{i}]: {actual} != {expected}"));
                }
            }
        }
    }
    Ok(())
}

pub fn run(device: &Device, include_empty: bool) -> Result<(), String> {
    for length in [1, 127, 128, 129, 257, 531] {
        let words: Vec<i32> = (0..3 * length * 5)
            .map(|i| {
                if i % 7 == 0 {
                    i32::MIN
                } else {
                    (i as u32).wrapping_mul(0x9e37_79b9).rotate_left((i % 32) as u32) as i32
                }
            })
            .collect();
        let input = Tensor::<3, Int>::from_data(TensorData::new(words.clone(), [3, length, 5]), device);
        let view = input.clone().permute([1, 2, 0]).flip([0, 2]);
        let ordered: Vec<_> = (0..words.len())
            .map(|i| words[(2 - i % 3) * length * 5 + (length - 1 - i / 15) * 5 + i / 3 % 5])
            .collect();
        for op in OPERATIONS {
            for axis in 0..3 {
                let output = apply(view.clone(), axis, op)
                    .try_into_vec_as::<i32>()
                    .map_err(|e| format!("I32 scan read: {e:?}"))?;
                integers(&output, &ordered, &[length, 5, 3], axis, op)?;
                let values: Vec<f32> = ordered
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if matches!(op, Operation::Product) {
                            (1. + (i % 17) as f32 / 512.) * if i % 37 == 0 { -1. } else { 1. }
                        } else {
                            (v % 1009) as f32 / 37.
                        }
                    })
                    .collect();
                let float = Tensor::<3>::from_data(TensorData::new(values.clone(), [length, 5, 3]), device).flip([0, 2]);
                let logical: Vec<_> = (0..values.len())
                    .map(|i| values[(length - 1 - i / 15) * 15 + i / 3 % 5 * 3 + 2 - i % 3])
                    .collect();
                let result = apply(float.clone(), axis, op)
                    .try_into_vec_as::<f32>()
                    .map_err(|e| format!("F32 scan read: {e:?}"))?;
                floats(&result, &logical, &[length, 5, 3], axis, op)?;
                let preserved = float.try_into_vec_as::<f32>().map_err(|e| format!("shared scan read: {e:?}"))?;
                if preserved.iter().zip(&logical).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    return Err(String::from("scan changed shared float input"));
                }
            }
        }
        let preserved = input.try_into_vec_as::<i32>().map_err(|e| format!("shared scan read: {e:?}"))?;
        if preserved != words {
            return Err(String::from("scan changed shared integer input"));
        }
    }
    for op in OPERATIONS {
        let integers_source = [i32::MIN, -1, i32::MAX];
        let input = Tensor::<2, Int>::from_data(TensorData::new(integers_source.to_vec(), [3, 1]), device).expand([3, 257]);
        let expected: Vec<_> = (0..3 * 257).map(|i| integers_source[i / 257]).collect();
        let actual = apply(input, 1, op).try_into_vec_as::<i32>().map_err(|e| format!("broadcast I32 scan: {e:?}"))?;
        integers(&actual, &expected, &[3, 257], 1, op)?;
        let float_source = [0.5f32, -0., 1.];
        let input = Tensor::<2>::from_data(TensorData::new(float_source.to_vec(), [3, 1]), device).expand([3, 257]);
        let expected: Vec<_> = (0..3 * 257).map(|i| float_source[i / 257]).collect();
        let actual = apply(input, 1, op).try_into_vec_as::<f32>().map_err(|e| format!("broadcast F32 scan: {e:?}"))?;
        floats(&actual, &expected, &[3, 257], 1, op)?;
    }
    for values in [
        vec![f32::from_bits(1); 257],
        vec![f32::INFINITY; 257],
        vec![f32::NEG_INFINITY; 257],
        (0..257).map(|i| if i == 128 { f32::NAN } else { 1. }).collect(),
    ] {
        for op in OPERATIONS {
            let input = Tensor::<1>::from_data(TensorData::new(values.clone(), [values.len()]), device);
            let actual = apply(input, 0, op).try_into_vec_as::<f32>().map_err(|e| format!("scan special read: {e:?}"))?;
            floats(&actual, &values, &[values.len()], 0, op)?;
        }
    }
    if include_empty {
        for axis in 0..3 {
            for op in OPERATIONS {
                let integer = apply(Tensor::<3, Int>::zeros([2, 0, 5], device), axis, op)
                    .try_into_vec_as::<i32>()
                    .map_err(|e| format!("empty I32 scan: {e:?}"))?;
                let float = apply(Tensor::<3>::zeros([2, 0, 5], device), axis, op)
                    .try_into_vec_as::<f32>()
                    .map_err(|e| format!("empty F32 scan: {e:?}"))?;
                if !integer.is_empty() || !float.is_empty() {
                    return Err(String::from("nonempty scan result for empty input"));
                }
            }
        }
    }
    gradients(device)
}
