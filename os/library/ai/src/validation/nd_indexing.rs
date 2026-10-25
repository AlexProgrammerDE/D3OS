//! Independent logical ND addressing, derivatives, and exact integer indices.
use super::scatter_updates::{float, integer};
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, IndexingUpdateOp, Int, Slice, Tensor},
};

const DIMS: [usize; 3] = [137, 3, 5];
const UPDATES: [IndexingUpdateOp; 5] = [
    IndexingUpdateOp::Add,
    IndexingUpdateOp::Assign,
    IndexingUpdateOp::Mul,
    IndexingUpdateOp::Min,
    IndexingUpdateOp::Max,
];

fn physical(i: usize) -> usize {
    let c = [i / 15, i / 5 % 3, i % 5];
    (c[1] * 5 + 4 - c[2]) * 137 + 136 - c[0]
}

fn tuples(count: usize, width: usize) -> Vec<i32> {
    let rows: usize = DIMS[..width].iter().product();
    let mut result = Vec::with_capacity(count * width);
    for i in 0..count {
        let mut linear = (i * 13 + 7) % rows;
        let mut coordinates = vec![0; width];
        for axis in (0..width).rev() {
            coordinates[axis] = (linear % DIMS[axis]) as i32;
            linear /= DIMS[axis];
        }
        result.extend(coordinates);
    }
    result
}

fn addresses(indices: &[i32], width: usize) -> Vec<usize> {
    let slice: usize = DIMS[width..].iter().product();
    let mut result = Vec::new();
    for tuple in indices.chunks_exact(width) {
        let row = tuple.iter().zip(DIMS).fold(0, |n, (&v, d)| n * d + v as usize);
        result.extend(row * slice..(row + 1) * slice);
    }
    result
}

fn index_tensor<const M: usize>(device: &Device, indices: &[i32], shape: [usize; M]) -> Tensor<M, Int> {
    let mut storage = vec![-99];
    storage.extend(indices.iter().rev());
    storage.push(-99);
    Tensor::<1, Int>::from_data(TensorData::new(storage, [indices.len() + 2]), device)
        .slice([Slice::new(1, Some((indices.len() + 1) as isize), 1)])
        .reshape(shape)
        .flip::<M>(core::array::from_fn(|i| i))
}

fn output_shape<const M: usize, const V: usize>(shape: [usize; M]) -> [usize; V] {
    let width = shape[M - 1];
    shape[..M - 1].iter().chain(&DIMS[width..]).copied().collect::<Vec<_>>().try_into().unwrap()
}

fn case<const M: usize, const V: usize>(device: &Device, shape: [usize; M]) -> Result<(), String> {
    let width = shape[M - 1];
    let count: usize = shape[..M - 1].iter().product();
    let ids = tuples(count, width);
    let at = addresses(&ids, width);
    let output_shape = output_shape::<M, V>(shape);
    let a: Vec<_> = (0..2055).map(|i| (i % 97 - 48) as f32 / 4.).collect();
    let ai: Vec<_> = (0..2055).map(|i| (i as u32).wrapping_mul(0x9e37_79b9) as i32).collect();
    let logical: Vec<_> = (0..2055).map(|i| a[physical(i)]).collect();
    let logical_i: Vec<_> = (0..2055).map(|i| ai[physical(i)]).collect();
    let data = Tensor::<3>::from_data(TensorData::new(a, [3, 5, 137]), device).permute([2, 0, 1]).flip([0, 2]);
    let data_i = Tensor::<3, Int>::from_data(TensorData::new(ai, [3, 5, 137]), device)
        .permute([2, 0, 1])
        .flip([0, 2]);
    let indices = index_tensor(device, &ids, shape);
    let result: Tensor<V> = data.clone().gather_nd(indices.clone());
    if result.dims() != output_shape
        || result.try_into_vec_as::<f32>().map_err(|e| format!("ND float gather: {e:?}"))? != at.iter().map(|&i| logical[i]).collect::<Vec<_>>()
    {
        return Err(format!("ND float gather M={M} K={width} count={count}"));
    }
    let result: Tensor<V, Int> = data_i.clone().gather_nd(indices.clone());
    if result.dims() != output_shape
        || result.try_into_vec_as::<i32>().map_err(|e| format!("ND integer gather: {e:?}"))? != at.iter().map(|&i| logical_i[i]).collect::<Vec<_>>()
    {
        return Err(format!("ND integer gather M={M} K={width} count={count}"));
    }
    let b: Vec<_> = (0..at.len()).map(|i| (i as i32 % 31 - 15) as f32 / 8.).collect();
    let bi: Vec<_> = (0..at.len()).map(|i| (i as u32).wrapping_mul(0x85eb_ca6b) as i32).collect();
    let values = Tensor::<1>::from_data(TensorData::new(b.iter().rev().copied().collect::<Vec<_>>(), [b.len()]), device)
        .reshape(output_shape)
        .flip::<V>(core::array::from_fn(|i| i));
    let values_i = Tensor::<1, Int>::from_data(TensorData::new(bi.iter().rev().copied().collect::<Vec<_>>(), [bi.len()]), device)
        .reshape(output_shape)
        .flip::<V>(core::array::from_fn(|i| i));
    for update in UPDATES {
        let mut expected = logical.clone();
        let mut expected_i = logical_i.clone();
        for (i, &destination) in at.iter().enumerate() {
            expected[destination] = float(expected[destination], b[i], update);
            expected_i[destination] = integer(expected_i[destination] as u32, bi[i] as u32, update) as i32;
        }
        if data
            .clone()
            .scatter_nd(indices.clone(), values.clone(), update)
            .try_into_vec_as::<f32>()
            .map_err(|e| format!("ND float scatter: {e:?}"))?
            != expected
            || data_i
                .clone()
                .scatter_nd(indices.clone(), values_i.clone(), update)
                .try_into_vec_as::<i32>()
                .map_err(|e| format!("ND integer scatter: {e:?}"))?
                != expected_i
        {
            return Err(format!("ND scatter M={M} K={width} count={count} op={update:?}"));
        }
    }
    if data.try_into_vec_as::<f32>().map_err(|e| format!("retained ND data: {e:?}"))? != logical
        || data_i.try_into_vec_as::<i32>().map_err(|e| format!("retained ND integer data: {e:?}"))? != logical_i
        || indices.try_into_vec_as::<i32>().map_err(|e| format!("retained ND indices: {e:?}"))? != ids
        || values.try_into_vec_as::<f32>().map_err(|e| format!("retained ND values: {e:?}"))? != b
    {
        return Err(String::from("ND operation changed shared input"));
    }
    Ok(())
}

fn gradients<const M: usize, const V: usize>(device: &Device, shape: [usize; M]) -> Result<(), String> {
    let device = device.clone().autodiff();
    let width = shape[M - 1];
    let count: usize = shape[..M - 1].iter().product();
    let ids = tuples(count, width);
    let at = addresses(&ids, width);
    let dims = output_shape::<M, V>(shape);
    let a: Vec<_> = (0..2055).map(|i| (i % 9 - 4) as f32).collect();
    let b: Vec<_> = (0..at.len()).map(|i| (i as i32 % 7 - 3) as f32).collect();
    let weights: Vec<_> = (0..2055).map(|i| (i % 7 + 1) as f32).collect();
    for update in UPDATES {
        let data = Tensor::<3>::from_data(TensorData::new(a.clone(), DIMS), &device).require_grad();
        let values = Tensor::<1>::from_data(TensorData::new(b.clone(), [b.len()]), &device)
            .reshape(dims)
            .require_grad();
        let indices = index_tensor(&device, &ids, shape);
        let mut da = weights.clone();
        let mut db = vec![0.; b.len()];
        for (i, &dest) in at.iter().enumerate() {
            let w = weights[dest];
            match update {
                IndexingUpdateOp::Add => db[i] = w,
                IndexingUpdateOp::Assign => {
                    da[dest] = 0.;
                    db[i] = w;
                }
                IndexingUpdateOp::Mul => {
                    da[dest] = w * b[i];
                    db[i] = w * a[dest];
                }
                IndexingUpdateOp::Min => {
                    da[dest] = if a[dest] <= b[i] { w } else { 0. };
                    db[i] = if b[i] <= a[dest] { w } else { 0. };
                }
                IndexingUpdateOp::Max => {
                    da[dest] = if a[dest] >= b[i] { w } else { 0. };
                    db[i] = if b[i] >= a[dest] { w } else { 0. };
                }
            }
        }
        let output = data.clone().scatter_nd(indices, values.clone(), update);
        let gradients = (output * Tensor::<3>::from_data(TensorData::new(weights.clone(), DIMS), &device))
            .sum()
            .backward();
        if data
            .grad(&gradients)
            .ok_or_else(|| String::from("missing ND data gradient"))?
            .try_into_vec_as::<f32>()
            .map_err(|e| format!("ND data gradient: {e:?}"))?
            != da
            || values
                .grad(&gradients)
                .ok_or_else(|| String::from("missing ND value gradient"))?
                .try_into_vec_as::<f32>()
                .map_err(|e| format!("ND value gradient: {e:?}"))?
                != db
        {
            return Err(format!("ND scatter gradient K={width} M={M} op={update:?}"));
        }
    }
    let data = Tensor::<3>::from_data(TensorData::new(a, DIMS), &device).require_grad();
    let output: Tensor<V> = data.clone().gather_nd(index_tensor(&device, &ids, shape));
    let w: Vec<_> = (0..at.len()).map(|i| (i % 7 + 1) as f32).collect();
    let mut expected = vec![0.; 2055];
    for (i, &dest) in at.iter().enumerate() {
        expected[dest] += w[i];
    }
    let weight = Tensor::<1>::from_data(TensorData::new(w.clone(), [w.len()]), &device).reshape(dims);
    let gradients = (output * weight).sum().backward();
    if data
        .grad(&gradients)
        .ok_or_else(|| String::from("missing ND gather gradient"))?
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("ND gather gradient: {e:?}"))?
        != expected
    {
        return Err(format!("ND gather gradient K={width} M={M}"));
    }
    Ok(())
}

fn duplicate_add(device: &Device) -> Result<(), String> {
    let device = device.clone().autodiff();
    let data = Tensor::<2>::from_floats([[2., 3.], [5., 7.]], &device).require_grad();
    let ids = Tensor::<2, Int>::from_data(TensorData::new(vec![0i32; 513 * 2], [513, 2]), &device);
    let output: Tensor<1> = data.clone().gather_nd(ids.clone());
    if output.clone().try_into_vec_as::<f32>().map_err(|e| format!("duplicate gather: {e:?}"))? != vec![2.; 513] {
        return Err(String::from("duplicate gather values"));
    }
    let gradients = output.sum().backward();
    let actual = data
        .grad(&gradients)
        .ok_or_else(|| String::from("missing duplicate gather gradient"))?
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("duplicate gather gradient: {e:?}"))?;
    if actual != [513., 0., 0., 0.] {
        return Err(format!("duplicate ND gather gradient {actual:?}, expected [513,0,0,0]"));
    }
    let plain = device.without_autodiff();
    let data = Tensor::<2, Int>::from_ints([[0, 3], [5, 7]], &plain);
    let ids = Tensor::<2, Int>::from_data(TensorData::new(vec![0i32; 513 * 2], [513, 2]), &plain);
    let values = Tensor::<1, Int>::from_data(TensorData::new(vec![1i32; 513], [513]), &plain);
    let actual = data
        .scatter_nd(ids, values, IndexingUpdateOp::Add)
        .try_into_vec_as::<i32>()
        .map_err(|e| format!("duplicate scatter add: {e:?}"))?;
    if actual != [513, 3, 5, 7] {
        return Err(format!("duplicate ND I32 scatter add {actual:?}"));
    }
    Ok(())
}

fn scalar_and_exact_indices(device: &Device, include_scalar: bool) -> Result<(), String> {
    if include_scalar {
        let data = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], device);
        let scalar: Tensor<0> = data.gather_nd(Tensor::<1, Int>::from_ints([1, 0], device));
        if (scalar + 2.).try_into_vec_as::<f32>().map_err(|e| format!("scalar arithmetic: {e:?}"))? != [5.] {
            return Err(String::from("scalar ND result arithmetic"));
        }
    }
    let data = Tensor::<1, Int>::arange(0..16_785_409, device).reshape([4097, 4097]);
    let result: Tensor<1, Int> = data.gather_nd(Tensor::<2, Int>::from_ints([[4096, 4095], [4096, 4096]], device));
    if result.try_into_vec_as::<i32>().map_err(|e| format!("exact ND indices: {e:?}"))? != [16_785_407, 16_785_408] {
        return Err(String::from("ND indices rounded through FP32"));
    }
    Ok(())
}

fn empty_slices(device: &Device) -> Result<(), String> {
    let data = Tensor::<3>::from_data(TensorData::new(Vec::<f32>::new(), [2, 0, 3]), device);
    let ids = Tensor::<2, Int>::from_ints([[0], [1]], device);
    let result: Tensor<3> = data.clone().gather_nd(ids.clone());
    if result.dims() != [2, 0, 3] || !result.try_into_vec_as::<f32>().map_err(|e| format!("empty ND gather: {e:?}"))?.is_empty() {
        return Err(String::from("empty ND gather shape"));
    }
    for update in UPDATES {
        let result = data.clone().scatter_nd(ids.clone(), data.clone(), update);
        if result.dims() != [2, 0, 3] || !result.try_into_vec_as::<f32>().map_err(|e| format!("empty ND scatter: {e:?}"))?.is_empty() {
            return Err(String::from("empty ND scatter shape"));
        }
    }
    Ok(())
}

pub fn run(device: &Device, duplicates: bool, include_empty: bool, include_scalar: bool) -> Result<(), String> {
    for length in [1, 127, 128, 129] {
        case::<2, 3>(device, [length, 1])?;
        case::<2, 2>(device, [length, 2])?;
        case::<2, 1>(device, [length, 3])?;
    }
    case::<3, 4>(device, [1, 129, 1])?;
    case::<3, 3>(device, [1, 129, 2])?;
    case::<3, 2>(device, [1, 129, 3])?;
    case::<1, 2>(device, [1])?;
    case::<1, 1>(device, [2])?;
    if include_scalar {
        case::<1, 0>(device, [3])?;
    }
    gradients::<2, 3>(device, [3, 1])?;
    gradients::<2, 2>(device, [3, 2])?;
    gradients::<2, 1>(device, [3, 3])?;

    scalar_and_exact_indices(device, include_scalar)?;
    if include_empty {
        empty_slices(device)?;
    }
    if duplicates {
        duplicate_add(device)?;
    }
    Ok(())
}

/// Separate the pinned autodiff scalar-shape diagnostic from backend execution.
pub fn scalar_gradients(device: &Device) -> Result<(), String> {
    gradients::<1, 0>(device, [3])
}
