//! Exact public I32 quotient and floor-modulo checks with binary64-free oracles.
use alloc::{format, string::String, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, Int, Tensor},
};

fn reference(x: i32, y: i32, remainder: bool) -> i32 {
    let (x, y) = (i64::from(x), i64::from(y));
    if !remainder {
        return (x / y) as i32;
    }
    let value = x % y;
    (if value != 0 && (value < 0) != (y < 0) { value + y } else { value }) as i32
}

fn check<const D: usize>(tensor: Tensor<D, Int>, expected: &[i32], operation: &str) -> Result<(), String> {
    let values = tensor.try_into_vec_as::<i32>().map_err(|e| format!("{operation}: {e:?}"))?;
    if values.len() != expected.len() {
        return Err(format!("{operation}: output length"));
    }
    if let Some((index, (actual, expected))) = values.iter().zip(expected).enumerate().find(|(_, (a, b))| a != b) {
        return Err(format!("{operation}[{index}]: {actual} != {expected}"));
    }
    Ok(())
}

pub fn run(device: &Device) -> Result<(), String> {
    let edges = [i32::MIN, i32::MIN + 1, -16_777_217, -257, -7, -1, 0, 1, 7, 257, 16_777_217, i32::MAX];
    let values: Vec<i32> = (0..771)
        .map(|i| {
            if i % 3 == 0 {
                edges[(i / 3) % edges.len()]
            } else {
                (i as u32).wrapping_mul(0x9e37_79b9).rotate_left((i % 32) as u32) as i32
            }
        })
        .collect();
    let input = Tensor::<2, Int>::from_data(TensorData::new(values.clone(), [3, 257]), device);
    let view = input.clone().transpose().flip([0]);
    let ordered: Vec<_> = (0..771).map(|i| values[(i % 3) * 257 + 256 - i / 3]).collect();
    let factors = [i32::MIN, -31, i32::MAX];
    let divisor = Tensor::<1, Int>::from_data(TensorData::new(factors.to_vec(), [3]), device).unsqueeze::<2>();
    for remainder in [false, true] {
        let output = if remainder {
            view.clone().remainder(divisor.clone())
        } else {
            view.clone() / divisor.clone()
        };
        if output.dims() != [257, 3] {
            return Err(String::from("integer broadcast shape"));
        }
        let expected: Vec<_> = ordered.iter().enumerate().map(|(i, &x)| reference(x, factors[i % 3], remainder)).collect();
        check(output, &expected, "strided broadcast integer arithmetic")?;
        for scalar in [i32::MIN, -16_777_217, -257, -7, -2, 1, 2, 7, 257, 16_777_217, i32::MAX] {
            let output = if remainder {
                view.clone().remainder_scalar(scalar)
            } else {
                view.clone() / scalar
            };
            let expected: Vec<_> = ordered.iter().map(|&x| reference(x, scalar, remainder)).collect();
            check(output, &expected, "scalar integer arithmetic")?;
        }
        let lhs: Vec<_> = (0..4099)
            .map(|i| edges[i % edges.len()].wrapping_add((i as i32).wrapping_mul(0x6a09_e667)))
            .collect();
        let rhs: Vec<_> = (0..4099).map(|i| edges[(i / edges.len()) % edges.len()]).collect();
        // Zero divisors and MIN/-1 have no portable result. Cover -1 for all
        // other values, including exact division and every combination of signs.
        let pairs: Vec<_> = lhs.into_iter().zip(rhs).filter(|&(x, y)| y != 0 && !(x == i32::MIN && y == -1)).collect();
        let left = Tensor::<1, Int>::from_data(TensorData::new(pairs.iter().map(|p| p.0).collect::<Vec<_>>(), [pairs.len()]), device);
        let right = Tensor::<1, Int>::from_data(TensorData::new(pairs.iter().map(|p| p.1).collect::<Vec<_>>(), [pairs.len()]), device);
        let output = if remainder { left.remainder(right) } else { left / right };
        let expected: Vec<_> = pairs.iter().map(|&(x, y)| reference(x, y, remainder)).collect();
        check(output, &expected, "integer boundary pairs")?;
        let empty = Tensor::<2, Int>::zeros([0, 7], device);
        let right = Tensor::<2, Int>::ones([1, 7], device);
        let output = if remainder { empty.remainder(right) } else { empty / right };
        check(output, &[], "empty integer arithmetic")?;
    }
    check(input, &values, "shared integer source")?;
    Ok(())
}
