//! Fused state updates consume Burn storage and preserve any surviving aliases.
use super::{dense, output, projection, weights};
use crate::{
    NativeTensor,
    backend::{BackendTensor, Native},
};
use alloc::vec::Vec;
use burn_backend::{DType, Shape, Slice, TensorMetadata};
use d3os_compute::{Error, Result, element_count, nvidia::Nvidia};
use d3os_nvidia::compute::Kernel;
use d3os_tensor_file::matrix::MatrixFormat;

fn check(tensor: &NativeTensor, shape: &[usize]) -> Result<()> {
    if tensor.shape().as_slice() != shape || tensor.dtype() != DType::F32 {
        return Err(Error::ShapeMismatch);
    }
    Ok(())
}
fn writable(tensor: &NativeTensor, unique: bool) -> Result<NativeTensor> {
    if unique { dense(tensor, DType::F32) } else { tensor.copy() }
}
fn dispatch(kernel: Kernel, tensors: &[(&NativeTensor, usize)], parameters: &[usize]) -> Result<()> {
    let pointers: Vec<_> = tensors.iter().map(|(t, offset)| (&*t.buffer, t.layout.offset() + offset)).collect();
    Nvidia.dispatch_words(kernel, &pointers, parameters)
}
fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.
}

fn attention(
    input: [&NativeTensor; 5], cache: [&NativeTensor; 2], unique: [bool; 2], geometry: [usize; 4], position: usize, theta: f32, epsilon: f32,
) -> Result<[NativeTensor; 3]> {
    let [heads, kv_heads, dim, rotary] = geometry;
    let q_width = element_count(&[heads, dim])?;
    let kv_width = element_count(&[kv_heads, dim])?;
    let qg_width = element_count(&[2, q_width])?;
    let [rows, _] = *input[0].shape().as_slice() else {
        return Err(Error::ShapeMismatch);
    };
    let [capacity, _, _] = *cache[0].shape().as_slice() else {
        return Err(Error::ShapeMismatch);
    };
    element_count(&[rows, q_width])?;
    if !heads.is_multiple_of(kv_heads) || rotary == 0 || !rotary.is_multiple_of(2) || rotary > dim || !positive(theta) || !positive(epsilon) {
        return Err(Error::InvalidParameter);
    }
    check(input[0], &[rows, qg_width])?;
    check(input[1], &[rows, kv_width])?;
    check(input[2], &[rows, kv_width])?;
    check(input[3], &[dim])?;
    check(input[4], &[dim])?;
    check(cache[0], &[capacity, kv_heads, dim])?;
    check(cache[1], &[capacity, kv_heads, dim])?;
    let positions = position.checked_add(rows).filter(|&end| end <= capacity).ok_or(Error::InvalidParameter)?;
    let input = input.map(|t| dense(t, DType::F32)).into_iter().collect::<Result<Vec<_>>>()?;
    let keys = writable(cache[0], unique[0])?;
    let values = writable(cache[1], unique[1])?;
    let parameters = |heads, stride| [heads, dim, rotary, position, theta.to_bits() as usize, epsilon.to_bits() as usize, stride, rows];
    let query = output(
        [rows, q_width],
        Kernel::AttentionPrepareBatch,
        &[(&input[0], 0), (&input[3], 0)],
        &parameters(heads, 2),
    )?;
    dispatch(
        Kernel::AttentionPrepareBatch,
        &[(&input[1], 0), (&input[4], 0), (&keys, position * kv_width)],
        &parameters(kv_heads, 1),
    )?;
    dispatch(Kernel::Copy, &[(&input[2], 0), (&values, position * kv_width)], &[rows * kv_width])?;
    let result = if dim <= 256 {
        if rows == 1 && heads <= 16 && positions >= 512 {
            let partials = output(
                [rows * heads * 16, dim + 2],
                Kernel::AttentionFusedBatch,
                &[(&query, 0), (&keys, 0), (&values, 0), (&input[0], 0)],
                &[rows, heads, kv_heads, dim, position, 16],
            )?;
            output(
                [rows, q_width],
                Kernel::AttentionOutput,
                &[(&partials, 0), (&input[0], 0)],
                &[rows * heads, dim, 16],
            )?
        } else {
            output(
                [rows, q_width],
                Kernel::AttentionFusedBatch,
                &[(&query, 0), (&keys, 0), (&values, 0), (&input[0], 0)],
                &[rows, heads, kv_heads, dim, position],
            )?
        }
    } else {
        let scores = NativeTensor::empty(Shape::new([heads, positions]), DType::F32)?;
        let result = NativeTensor::empty(Shape::new([rows, q_width]), DType::F32)?;
        for row in 0..rows {
            let visible = position + row + 1;
            dispatch(
                Kernel::AttentionScores,
                &[(&query, row * q_width), (&keys, 0), (&scores, 0)],
                &[heads, kv_heads, dim, visible],
            )?;
            dispatch(Kernel::Softmax, &[(&scores, 0), (&scores, 0)], &[heads, visible])?;
            dispatch(
                Kernel::AttentionOutput,
                &[(&scores, 0), (&values, 0), (&input[0], row * 2 * q_width), (&result, row * q_width)],
                &[heads, kv_heads, dim, visible],
            )?;
        }
        result
    };
    Ok([result, keys, values])
}

fn delta(input: [&NativeTensor; 8], cache: [&NativeTensor; 2], unique: [bool; 2], geometry: [usize; 5], epsilon: f32) -> Result<[NativeTensor; 3]> {
    let [key_heads, heads, key_dim, value_dim, kernel] = geometry;
    let key_width = element_count(&[key_heads, key_dim])?;
    let value_width = element_count(&[heads, value_dim])?;
    let channels = key_width.checked_mul(2).and_then(|v| v.checked_add(value_width)).ok_or(Error::SizeOverflow)?;
    element_count(&[channels, kernel])?;
    element_count(&[heads, key_dim, value_dim])?;
    if !heads.is_multiple_of(key_heads) || !positive(epsilon) {
        return Err(Error::InvalidParameter);
    }
    let [rows, _] = *input[0].shape().as_slice() else {
        return Err(Error::ShapeMismatch);
    };
    element_count(&[rows, channels])?;
    for (tensor, shape) in input.iter().zip([
        &[rows, channels][..],
        &[rows, value_width],
        &[rows, heads],
        &[rows, heads],
        &[channels, 1, kernel],
        &[heads],
        &[heads],
        &[value_dim],
    ]) {
        check(tensor, shape)?;
    }
    check(cache[0], &[channels, kernel])?;
    check(cache[1], &[heads, key_dim, value_dim])?;
    if key_heads != heads && (key_dim > 128 || value_dim > 128 || kernel > 8) {
        return Err(Error::DeviceUnavailable);
    }
    let input = input.map(|t| dense(t, DType::F32)).into_iter().collect::<Result<Vec<_>>>()?;
    let convolution = writable(cache[0], unique[0])?;
    let recurrent = writable(cache[1], unique[1])?;
    let result = if key_dim <= 128 && value_dim <= 128 && kernel <= 8 {
        if rows == 1 && key_heads == heads {
            output(
                [rows, value_width],
                Kernel::DeltaFused,
                &[
                    (&input[0], 0),
                    (&input[1], 0),
                    (&input[2], 0),
                    (&input[3], 0),
                    (&input[4], 0),
                    (&input[5], 0),
                    (&input[6], 0),
                    (&input[7], 0),
                    (&convolution, 0),
                    (&recurrent, 0),
                ],
                &[heads, key_dim, value_dim, kernel, epsilon.to_bits() as usize],
            )?
        } else {
            let mixed = output(
                [rows, channels],
                Kernel::DeltaConvolution,
                &[(&input[0], 0), (&input[4], 0), (&convolution, 0)],
                &[channels, kernel, rows],
            )?;
            let update = [heads, key_dim, value_dim, rows, key_heads];
            let raw = output(
                [rows, value_width],
                Kernel::DeltaUpdate,
                &[(&mixed, 0), (&input[2], 0), (&input[3], 0), (&input[5], 0), (&input[6], 0), (&recurrent, 0)],
                &update[..if key_heads == heads { 4 } else { 5 }],
            )?;
            output(
                [rows, value_width],
                Kernel::DeltaOutput,
                &[(&raw, 0), (&input[7], 0), (&input[1], 0)],
                &[rows * heads, value_dim, epsilon.to_bits() as usize],
            )?
        }
    } else {
        let scales = NativeTensor::empty(Shape::new([heads, 4]), DType::F32)?;
        let result = NativeTensor::empty(Shape::new([rows, value_width]), DType::F32)?;
        for row in 0..rows {
            let mixed = output(
                [1, channels],
                Kernel::DeltaConvolution,
                &[(&input[0], row * channels), (&input[4], 0), (&convolution, 0)],
                &[channels, kernel],
            )?;
            dispatch(
                Kernel::DeltaPrepare,
                &[
                    (&mixed, 0),
                    (&input[2], row * heads),
                    (&input[3], row * heads),
                    (&input[5], 0),
                    (&input[6], 0),
                    (&scales, 0),
                ],
                &[heads, key_dim, value_dim],
            )?;
            let raw = output(
                [heads, value_dim],
                Kernel::DeltaUpdate,
                &[(&mixed, 0), (&scales, 0), (&recurrent, 0)],
                &[heads, key_dim, value_dim],
            )?;
            dispatch(
                Kernel::DeltaOutput,
                &[(&raw, 0), (&input[7], 0), (&input[1], row * value_width), (&result, row * value_width)],
                &[heads, value_dim, epsilon.to_bits() as usize],
            )?;
        }
        result
    };
    Ok([result, convolution, recurrent])
}

impl Native {
    /// Inputs: query/gate, key, value, query norm, key norm. Cache: keys, values.
    /// Geometry: query heads, key/value heads, head width, rotary width.
    pub fn qwen_attention(
        input: [BackendTensor; 5], cache: [BackendTensor; 2], geometry: [usize; 4], position: usize, theta: f32, epsilon: f32,
    ) -> [BackendTensor; 3] {
        let rows = input[0].shape().first().copied().unwrap_or(0);
        let width = geometry[0].checked_mul(geometry[2]).unwrap_or(0);
        let unique = cache.each_ref().map(TensorMetadata::can_mut);
        let inputs: Vec<_> = input.iter().chain(&cache).collect();
        BackendTensor::outputs(
            "stateful attention",
            [Shape::new([rows, width]), cache[0].shape(), cache[1].shape()],
            [DType::F32; 3],
            &inputs,
            |v| attention([v[0], v[1], v[2], v[3], v[4]], [v[5], v[6]], unique, geometry, position, theta, epsilon),
        )
    }
    /// Inputs: qkv, gate, beta, alpha, convolution weights, log decay, time bias, norm.
    /// Cache: convolution history, recurrent matrix. Geometry: key heads, heads, key width, value width, kernel width.
    pub fn qwen_delta(input: [BackendTensor; 8], cache: [BackendTensor; 2], geometry: [usize; 5], epsilon: f32) -> [BackendTensor; 3] {
        let rows = input[0].shape().first().copied().unwrap_or(0);
        let values = geometry[1].checked_mul(geometry[3]).unwrap_or(0);
        let unique = cache.each_ref().map(TensorMetadata::can_mut);
        let inputs: Vec<_> = input.iter().chain(&cache).collect();
        BackendTensor::outputs(
            "stateful delta",
            [Shape::new([rows, values]), cache[0].shape(), cache[1].shape()],
            [DType::F32; 3],
            &inputs,
            |v| delta([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]], [v[8], v[9]], unique, geometry, epsilon),
        )
    }
    /// Project qkv, gate, beta, and alpha, then update the recurrent mixer.
    /// Scalar parameters: convolution weights, log decay, time bias, norm.
    pub fn qwen_project_delta(
        input: BackendTensor, packed: [BackendTensor; 4], parameters: [BackendTensor; 4], cache: [BackendTensor; 2], geometry: [usize; 5], epsilon: f32,
        formats: [MatrixFormat; 4],
    ) -> [BackendTensor; 3] {
        let rows = input.shape().first().copied().unwrap_or(0);
        let [key_heads, heads, key_dim, value_dim, kernel] = geometry;
        let values = heads.checked_mul(value_dim).unwrap_or(0);
        let unique = cache.each_ref().map(TensorMetadata::can_mut);
        let inputs: Vec<_> = core::iter::once(&input).chain(&packed).chain(&parameters).chain(&cache).collect();
        BackendTensor::outputs(
            "projected stateful delta",
            [Shape::new([rows, values]), cache[0].shape(), cache[1].shape()],
            [DType::F32; 3],
            &inputs,
            |v| {
                let [_, width] = *v[0].shape().as_slice() else {
                    return Err(Error::ShapeMismatch);
                };
                let channels = element_count(&[2, key_heads, key_dim])?
                    .checked_add(element_count(&[heads, value_dim])?)
                    .ok_or(Error::SizeOverflow)?;
                let projected = if rows == 1
                    && key_heads == heads
                    && width.is_multiple_of(2)
                    && channels.is_multiple_of(4)
                    && values.is_multiple_of(4)
                    && heads.is_multiple_of(4)
                    && key_dim <= 128
                    && value_dim <= 128
                    && kernel <= 8
                    && formats.iter().all(|&f| f == MatrixFormat::Bf16)
                {
                    let hidden = dense(v[0], DType::F32)?;
                    let matrices = [channels, values, heads, heads].map(|out| [out, width]);
                    let weights = matrices
                        .iter()
                        .enumerate()
                        .map(|(i, &shape)| weights(v[i + 1], shape, formats[i]))
                        .collect::<Result<Vec<_>>>()?;
                    let total = channels
                        .checked_add(values)
                        .and_then(|n| heads.checked_mul(2).and_then(|h| n.checked_add(h)))
                        .ok_or(Error::SizeOverflow)?;
                    let all = output(
                        [1, total],
                        Kernel::GroupedProjectBf16,
                        &[(&hidden, 0), (&weights[0], 0), (&weights[1], 0), (&weights[2], 0), (&weights[3], 0)],
                        &[width, channels, values, heads],
                    )?;
                    let mut offset = 0;
                    [channels, values, heads, heads]
                        .map(|length| {
                            let view = all
                                .clone()
                                .slice(&[Slice::full(), Slice::new(offset as isize, Some((offset + length) as isize), 1)]);
                            offset += length;
                            view
                        })
                        .into_iter()
                        .collect::<Result<Vec<_>>>()?
                } else {
                    [channels, values, heads, heads]
                        .iter()
                        .enumerate()
                        .map(|(i, &out)| projection(v[0], v[i + 1], [out, width], formats[i]))
                        .collect::<Result<Vec<_>>>()?
                };
                delta(
                    [&projected[0], &projected[1], &projected[2], &projected[3], v[5], v[6], v[7], v[8]],
                    [v[9], v[10]],
                    unique,
                    geometry,
                    epsilon,
                )
            },
        )
    }
}
