//! Public I32 matrix products checked against exact I128 sums.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, Int, Tensor},
};

fn values(length: usize, salt: u32) -> Vec<i32> {
    let edges = [i32::MIN, i32::MAX, 16_777_217, -16_777_217, -1, 0, 1];
    (0..length)
        .map(|i| {
            if i % 3 == 0 {
                edges[(i / 3) % edges.len()]
            } else {
                (i as u32).wrapping_mul(salt).rotate_left((i % 32) as u32) as i32
            }
        })
        .collect()
}

fn product(a: &[i32], b: &[i32], rows: usize, inner: usize, cols: usize) -> Vec<i32> {
    let mut output = vec![0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let value: i128 = (0..inner).map(|k| i128::from(a[row * inner + k]) * i128::from(b[k * cols + col])).sum();
            output[row * cols + col] = value as i32;
        }
    }
    output
}

fn check<const D: usize>(tensor: Tensor<D, Int>, shape: [usize; D], expected: &[i32], label: &str) -> Result<(), String> {
    if tensor.dims() != shape {
        return Err(format!("I32 matmul {label}: output shape"));
    }
    let actual = tensor.try_into_vec_as::<i32>().map_err(|e| format!("I32 matmul {label}: {e:?}"))?;
    if actual.len() != expected.len() {
        return Err(format!("I32 matmul {label}: output length"));
    }
    if let Some((index, (a, b))) = actual.iter().zip(expected).enumerate().find(|(_, (a, b))| a != b) {
        return Err(format!("I32 matmul {label}[{index}]: {a} != {b}"));
    }
    Ok(())
}

pub fn run(device: &Device, include_empty: bool) -> Result<(), String> {
    for (rows, inner, cols) in [(1, 1, 1), (3, 7, 5), (17, 33, 129), (7, 513, 17), (16, 0, 5), (0, 7, 5), (3, 7, 0)] {
        if !include_empty && (rows == 0 || inner == 0 || cols == 0) {
            continue;
        }
        let a = values(rows * inner, 0x9e37_79b9);
        let b = values(inner * cols, 0x6a09_e667);
        let expected = product(&a, &b, rows, inner, cols);
        let left = Tensor::<2, Int>::from_data(TensorData::new(a.clone(), [rows, inner]), device);
        let right = Tensor::<2, Int>::from_data(TensorData::new(b.clone(), [inner, cols]), device);
        check(left.clone(), [rows, inner], &a, "uploaded left")?;
        check(right.clone(), [inner, cols], &b, "uploaded right")?;
        check(
            left.clone().matmul(right.clone()),
            [rows, cols],
            &expected,
            &format!("dense [{rows},{inner},{cols}]"),
        )?;
        check(left.clone().flip([1]).matmul(right.clone().flip([0])), [rows, cols], &expected, "signed views")?;
        if rows != 0 && inner != 0 && cols != 0 {
            let transposed: Vec<_> = (0..cols * inner).map(|i| b[(i % inner) * cols + i / inner]).collect();
            let transposed = Tensor::<2, Int>::from_data(TensorData::new(transposed, [cols, inner]), device).transpose();
            check(left.clone().matmul(transposed), [rows, cols], &expected, "transposed right")?;
            let mut padded = vec![0; (rows + 2) * (inner + 2)];
            for row in 0..rows {
                padded[(row + 1) * (inner + 2) + 1..(row + 1) * (inner + 2) + 1 + inner].copy_from_slice(&a[row * inner..(row + 1) * inner]);
            }
            let subview = Tensor::<2, Int>::from_data(TensorData::new(padded, [rows + 2, inner + 2]), device).slice([1..rows + 1, 1..inner + 1]);
            check(subview.matmul(right.clone()), [rows, cols], &expected, "offset subview")?;
        }
        check(left, [rows, inner], &a, "shared left")?;
        check(right, [inner, cols], &b, "shared right")?;
    }
    let (rows, inner, cols) = (3, 17, 129);
    let a = values(2 * rows * inner, 0x9e37_79b9);
    let b = values(3 * inner * cols, 0x6a09_e667);
    let left = Tensor::<4, Int>::from_data(TensorData::new(a.clone(), [1, 2, rows, inner]), device);
    let right = Tensor::<4, Int>::from_data(TensorData::new(b.clone(), [3, 1, inner, cols]), device);
    let mut expected = Vec::new();
    for right_batch in 0..3 {
        for left_batch in 0..2 {
            expected.extend(product(
                &a[left_batch * rows * inner..][..rows * inner],
                &b[right_batch * inner * cols..][..inner * cols],
                rows,
                inner,
                cols,
            ));
        }
    }
    check(
        left.clone().flip([3]).matmul(right.clone().flip([2])),
        [3, 2, rows, cols],
        &expected,
        "broadcast batches",
    )?;
    check(left, [1, 2, rows, inner], &a, "shared batched left")?;
    check(right, [3, 1, inner, cols], &b, "shared batched right")?;
    Ok(())
}
