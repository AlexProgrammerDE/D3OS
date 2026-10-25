//! Independent vector-product references, including both input derivatives.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, Tensor},
};

fn read<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>, String> {
    tensor.try_into_vec_as::<f32>().map_err(|e| format!("cross read: {e:?}"))
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<(), String> {
    if actual.len() != expected.len() || actual.iter().zip(expected).any(|(&a, &b)| a != b && !(a.is_nan() && b.is_nan())) {
        return Err(String::from("cross numerical reference mismatch"));
    }
    Ok(())
}

fn view(values: &[f32], shape: [usize; 3], order: [usize; 3], flip: [bool; 3]) -> Vec<f32> {
    let dims = order.map(|axis| shape[axis]);
    let mut output = Vec::with_capacity(values.len());
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let mut source = [0; 3];
                for (axis, coordinate) in [i, j, k].into_iter().enumerate() {
                    source[order[axis]] = if flip[axis] { dims[axis] - 1 - coordinate } else { coordinate };
                }
                output.push(values[(source[0] * shape[1] + source[1]) * shape[2] + source[2]]);
            }
        }
    }
    output
}

fn reference(a: &[f32], b: &[f32], shape: [usize; 3], b_shape: [usize; 3], axis: usize, weights: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let suffix: usize = shape[axis + 1..].iter().product();
    let count: usize = shape.iter().product();
    let mut out = vec![0.; count];
    let mut da = vec![0.; a.len()];
    let mut db = vec![0.; b.len()];
    for (i, output) in out.iter_mut().enumerate() {
        let col = i / suffix % 3;
        let base = i - col * suffix;
        let j = base + (col + 1) % 3 * suffix;
        let k = base + (col + 2) % 3 * suffix;
        let b_address = |mut linear: usize| {
            let mut coords = [0; 3];
            for d in (0..3).rev() {
                coords[d] = (linear % shape[d]) % b_shape[d];
                linear /= shape[d];
            }
            (coords[0] * b_shape[1] + coords[1]) * b_shape[2] + coords[2]
        };
        let bj = b_address(j);
        let bk = b_address(k);
        *output = a[j] * b[bk] - a[k] * b[bj];
        if let Some(&w) = weights.get(i) {
            da[j] += w * b[bk];
            da[k] -= w * b[bj];
            db[bk] += w * a[j];
            db[bj] -= w * a[k];
        }
    }
    (out, da, db)
}

pub fn run(device: &Device, include_empty: bool) -> Result<(), String> {
    for length in [1, 43, 129] {
        let shape = [2, 3, length];
        let a: Vec<_> = (0..6 * length).map(|i| (i as i32 % 17 - 8) as f32).collect();
        for axis in 0..3 {
            let mut order = [0, 1, 2];
            order.swap(axis, 1);
            let dims = order.map(|d| shape[d]);
            let a_view = view(&a, shape, order, [true, false, true]);
            let a_tensor = Tensor::<3>::from_data(TensorData::new(a.clone(), shape), device).permute(order).flip([0, 2]);
            for broadcast in [false, true] {
                let b_shape = if broadcast { [1, 3, 1] } else { shape };
                let b: Vec<_> = (0..b_shape.iter().product()).map(|i: usize| (i as i32 % 13 - 6) as f32).collect();
                let b_dims = order.map(|d| b_shape[d]);
                let b_view = view(&b, b_shape, order, [false, true, false]);
                let b_tensor = Tensor::<3>::from_data(TensorData::new(b, b_shape), device).permute(order).flip([1]);
                let expected = reference(&a_view, &b_view, dims, b_dims, axis, &[]).0;
                let output = a_tensor.clone().cross(b_tensor.clone(), axis);
                if output.dims() != dims {
                    return Err(String::from("cross broadcast shape"));
                }
                compare(&read(output)?, &expected)?;
                compare(&read(b_tensor)?, &b_view)?;
            }
            compare(&read(a_tensor)?, &a_view)?;
        }
    }
    let auto = device.clone().autodiff();
    for axis in 0..3 {
        let mut dims = [2, 3, 5];
        dims.swap(axis, 1);
        let a: Vec<_> = (0..30).map(|i| (i % 11 - 5) as f32).collect();
        let b: Vec<_> = (0..30).map(|i| (i % 7 - 3) as f32).collect();
        let weights: Vec<_> = (0..30).map(|i| (i % 5 - 2) as f32).collect();
        let (_, da, db) = reference(&a, &b, dims, dims, axis, &weights);
        let a = Tensor::<3>::from_data(TensorData::new(a, dims), &auto).require_grad();
        let b = Tensor::<3>::from_data(TensorData::new(b, dims), &auto).require_grad();
        let output = a.clone().cross(b.clone(), axis);
        let gradients = (output * Tensor::<3>::from_data(TensorData::new(weights, dims), &auto)).sum().backward();
        compare(&read(a.grad(&gradients).ok_or_else(|| String::from("missing left cross gradient"))?)?, &da)?;
        compare(&read(b.grad(&gradients).ok_or_else(|| String::from("missing right cross gradient"))?)?, &db)?;
    }
    for axis in 0..if include_empty { 3 } else { 0 } {
        let mut shape = [2, 3, 0];
        shape.swap(axis, 1);
        let a = Tensor::<3>::from_data(TensorData::new(Vec::<f32>::new(), shape), device);
        let output = a.clone().cross(a, axis);
        if output.dims() != shape || !read(output)?.is_empty() {
            return Err(String::from("empty cross output"));
        }
    }
    let a = Tensor::<1>::from_data(TensorData::new(vec![f32::INFINITY, -0., 1.], [3]), device);
    let b = Tensor::<1>::from_floats([0., 1., 2.], device);
    compare(&read(a.cross(b, 0))?, &[-1., f32::NEG_INFINITY, f32::INFINITY])?;
    Ok(())
}
