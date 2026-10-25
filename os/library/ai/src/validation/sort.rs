//! Public ordering APIs checked by CPU total ordering and index permutations.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, Int, Tensor, kind::Ordered},
};
use core::cmp::Ordering;

#[derive(Clone, Copy)]
pub enum FloatDomain {
    Ieee,
    Finite,
}

#[derive(Clone, Copy)]
pub enum TopKOrdering {
    TotalOrder,
    NanFirst,
}

fn compare(a: u32, b: u32, integer: bool, descending: bool, policy: TopKOrdering) -> Ordering {
    let order = if integer {
        (a as i32).cmp(&(b as i32))
    } else {
        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
        match policy {
            TopKOrdering::TotalOrder => a.total_cmp(&b),
            TopKOrdering::NanFirst => match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.partial_cmp(&b).unwrap(),
            },
        }
    };
    if descending { order.reverse() } else { order }
}

fn reference(words: &[u32], shape: &[usize], axis: usize, integer: bool, descending: bool, k: usize, policy: TopKOrdering) -> Vec<u32> {
    let length = shape[axis];
    if words.is_empty() || k == 0 {
        return Vec::new();
    }
    let suffix: usize = shape[axis + 1..].iter().product();
    let rows = words.len() / length;
    let mut out = vec![0; rows * k];
    for row in 0..rows {
        let base = row / suffix * length * suffix + row % suffix;
        let mut values: Vec<_> = (0..length).map(|j| (words[base + j * suffix], j)).collect();
        values.sort_by(|a, b| compare(a.0, b.0, integer, descending, policy).then(a.1.cmp(&b.1)));
        let destination = row / suffix * k * suffix + row % suffix;
        for j in 0..k {
            out[destination + j * suffix] = values[j].0;
        }
    }
    out
}

fn words<const D: usize, K: Ordered>(tensor: Tensor<D, K>) -> Result<Vec<u32>, String> {
    let data = tensor.try_into_data().map_err(|e| format!("sort read: {e:?}"))?;
    if data.as_bytes().len() % 4 != 0 {
        return Err(String::from("sort word size"));
    }
    Ok(data.as_bytes().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
}

fn indices(actual: &[i32], sorted: &[u32], input: &[u32], shape: &[usize], axis: usize, k: usize) -> Result<(), String> {
    if actual.len() != sorted.len() {
        return Err(String::from("sort index length"));
    }
    if sorted.is_empty() {
        return Ok(());
    }
    let length = shape[axis];
    let suffix: usize = shape[axis + 1..].iter().product();
    for row in 0..input.len() / length {
        let base = row / suffix * length * suffix + row % suffix;
        let output = row / suffix * k * suffix + row % suffix;
        let mut seen = vec![false; length];
        for j in 0..k {
            let index = actual[output + j * suffix];
            if index < 0 || index as usize >= length || seen[index as usize] {
                return Err(String::from("sort indices are not an axis permutation"));
            }
            seen[index as usize] = true;
            if sorted[output + j * suffix] != input[base + index as usize * suffix] {
                return Err(String::from("sort index does not map to original bits"));
            }
        }
    }
    Ok(())
}

fn check<const D: usize, K: Ordered>(input: Tensor<D, K>, logical: &[u32], axis: usize, integer: bool, policy: TopKOrdering) -> Result<(), String> {
    let shape = input.dims();
    let length = shape[axis];
    for descending in [false, true] {
        let expected = reference(logical, &shape, axis, integer, descending, length, TopKOrdering::TotalOrder);
        let sorted = if descending {
            input.clone().sort_descending(axis)
        } else {
            input.clone().sort(axis)
        };
        if sorted.dims() != shape || words(sorted)? != expected {
            return Err(format!("sort values integer={integer} axis={axis} descending={descending}"));
        }
        let (sorted, index) = if descending {
            input.clone().sort_descending_with_indices(axis)
        } else {
            input.clone().sort_with_indices(axis)
        };
        let actual = words(sorted)?;
        if actual != expected {
            return Err(String::from("sort_with_indices values"));
        }
        let index = index.try_into_vec_as::<i32>().map_err(|e| format!("sort index read: {e:?}"))?;
        indices(&index, &actual, logical, &shape, axis, length)?;
        let args = if descending {
            input.clone().argsort_descending(axis)
        } else {
            input.clone().argsort(axis)
        };
        indices(
            &args.try_into_vec_as::<i32>().map_err(|e| format!("argsort read: {e:?}"))?,
            &expected,
            logical,
            &shape,
            axis,
            length,
        )?;
    }
    let mut counts = vec![0, 1.min(length), length.min(7), length];
    counts.dedup();
    for k in counts {
        let expected = reference(logical, &shape, axis, integer, true, k, policy);
        let mut output_shape = shape;
        output_shape[axis] = k;
        let output = input.clone().topk(k, axis);
        if output.dims() != output_shape {
            return Err(String::from("topk output shape"));
        }
        let actual = words(output)?;
        if actual != expected {
            let mismatch = actual.iter().zip(&expected).enumerate().find(|(_, (a, b))| a != b);
            return Err(format!("topk values integer={integer} shape={shape:?} axis={axis} k={k}: {mismatch:?}"));
        }
        let (output, index) = input.clone().topk_with_indices(k, axis);
        if words(output)? != expected {
            return Err(String::from("topk_with_indices values"));
        }
        indices(
            &index.try_into_vec_as::<i32>().map_err(|e| format!("topk index read: {e:?}"))?,
            &expected,
            logical,
            &shape,
            axis,
            k,
        )?;
        if k < length {
            let index = input
                .clone()
                .argtopk(k, axis)
                .try_into_vec_as::<i32>()
                .map_err(|e| format!("argtopk read: {e:?}"))?;
            indices(&index, &expected, logical, &shape, axis, k)?;
        }
    }
    if words(input)? != logical {
        return Err(String::from("sorting changed shared input bits"));
    }
    Ok(())
}

pub fn gradients(device: &Device) -> Result<(), String> {
    let device = device.clone().autodiff();
    let values: [f32; 7] = [3., -2., 1., 7., -5., 4., 0.];
    for (descending, k) in [(false, 7), (true, 7), (true, 3)] {
        let mut order: Vec<usize> = (0..7).collect();
        order.sort_by(|&a, &b| {
            if descending {
                values[b].total_cmp(&values[a])
            } else {
                values[a].total_cmp(&values[b])
            }
        });
        let weights: Vec<f32> = (0..k).map(|i| i as f32 + 1.).collect();
        let mut expected = [0f32; 7];
        for (i, &index) in order.iter().take(k).enumerate() {
            expected[index] = weights[i];
        }
        for with_indices in [false, true] {
            let input = Tensor::<1>::from_data(TensorData::new(values.to_vec(), [7]), &device).require_grad();
            let output = if k < 7 {
                if with_indices {
                    input.clone().topk_with_indices(k, 0).0
                } else {
                    input.clone().topk(k, 0)
                }
            } else if descending {
                if with_indices {
                    input.clone().sort_descending_with_indices(0).0
                } else {
                    input.clone().sort_descending(0)
                }
            } else if with_indices {
                input.clone().sort_with_indices(0).0
            } else {
                input.clone().sort(0)
            };
            let gradients = (output * Tensor::<1>::from_data(TensorData::new(weights.clone(), [k]), &device))
                .sum()
                .backward();
            let actual = input
                .grad(&gradients)
                .ok_or_else(|| String::from("missing sort gradient"))?
                .try_into_vec_as::<f32>()
                .map_err(|e| format!("sort gradient read: {e:?}"))?;
            if actual != expected {
                return Err(String::from("sort/topk gradient permutation"));
            }
        }
    }
    Ok(())
}

pub fn run(device: &Device, include_empty: bool, policy: TopKOrdering, domain: FloatDomain) -> Result<(), String> {
    for length in [1, 2, 3, 127, 128, 129, 257, 1025] {
        let input_words: Vec<u32> = (0..2 * length * 3)
            .map(|i| {
                if i % 3 == 0 {
                    [i32::MIN as u32, i32::MAX as u32, 0, u32::MAX, 16_777_217][i / 3 % 5]
                } else {
                    (i as u32).wrapping_mul(0x9e37_79b9).rotate_left((i % 32) as u32)
                }
            })
            .collect();
        let integer = Tensor::<3, Int>::from_data(
            TensorData::new(input_words.iter().map(|&x| x as i32).collect::<Vec<_>>(), [2, length, 3]),
            device,
        )
        .permute([1, 2, 0])
        .flip([0, 2]);
        let logical: Vec<_> = (0..input_words.len())
            .map(|i| input_words[(1 - i % 2) * length * 3 + (length - 1 - i / 6) * 3 + i / 2 % 3])
            .collect();
        let floats: Vec<_> = (0..input_words.len())
            .map(|i| {
                if i % 3 == 0 {
                    [
                        0,
                        0x8000_0000,
                        0x7f80_0000,
                        0xff80_0000,
                        0x7fc0_1234,
                        0xffc0_0035,
                        0x7f80_0001,
                        0xff80_0001,
                        1,
                        0x8000_0001,
                    ][i / 3 % 10]
                } else {
                    input_words[i]
                }
            })
            .collect();
        let floats: Vec<_> = match domain {
            FloatDomain::Ieee => floats,
            FloatDomain::Finite => floats.into_iter().map(|bits| ((bits as i32 % 1009) as f32 / 37.).to_bits()).collect(),
        };
        let floating = Tensor::<3>::from_data(
            TensorData::new(floats.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>(), [length, 3, 2]),
            device,
        )
        .flip([0, 2]);
        let float_logical: Vec<_> = (0..floats.len())
            .map(|i| floats[(length - 1 - i / 6) * 6 + i / 2 % 3 * 2 + 1 - i % 2])
            .collect();
        for axis in 0..3 {
            check(integer.clone(), &logical, axis, true, policy)?;
            check(floating.clone(), &float_logical, axis, false, policy)?;
        }
    }
    let values = [7i32, -3, 0];
    let input = Tensor::<2, Int>::from_data(TensorData::new(values.to_vec(), [3, 1]), device).expand([3, 129]);
    let logical: Vec<_> = (0..3 * 129).map(|i| values[i / 129] as u32).collect();
    check(input, &logical, 1, true, policy)?;
    if include_empty {
        for axis in 0..3 {
            check(Tensor::<3, Int>::zeros([2, 0, 3], device), &[], axis, true, policy)?;
            check(Tensor::<3>::zeros([2, 0, 3], device), &[], axis, false, policy)?;
        }
    }
    gradients(device)
}
