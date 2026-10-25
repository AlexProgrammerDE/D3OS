//! Raw dtype storage, resident view materialization, fills, and reloads.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{BoolStore, DType, Dispatch, TensorData, TensorMetadata, ops::BoolTensorOps, tensor::BoolTensor},
    tensor::{Device, Float, Int, Slice, Tensor, TensorCreationOptions, kind::Basic},
};

pub const DTYPES: [DType; 15] = [
    DType::F64,
    DType::F32,
    DType::Flex32,
    DType::F16,
    DType::BF16,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
    DType::Bool(BoolStore::U8),
    DType::Bool(BoolStore::U32),
];

fn payload(dtype: DType, count: usize) -> Vec<u8> {
    let size = dtype.size();
    (0..count)
        .flat_map(|i| {
            let bits = if dtype.is_bool() {
                u64::from(i % 3 != 0)
            } else {
                match i % 12 {
                    0 => 0,
                    1 => 1,
                    2 => u64::MAX,
                    3 => 0x8000_0000_8000_8000,
                    4 => 0x7ff0_0000_7fc0_1234,
                    5 => 0xfff0_0000_ff80_0001,
                    6 => 0x7ff0_0000_7f80_0000,
                    7 => 0xfff0_0000_ff80_0000,
                    8 => 0x7fef_ffff_7f7f_ffff,
                    9 => 0x8000_0000_0000_0001,
                    _ => (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                }
            };
            bits.to_le_bytes()[..size].to_vec()
        })
        .collect()
}

fn elements(raw: &[u8], size: usize, positions: impl Iterator<Item = usize>) -> Vec<u8> {
    positions.flat_map(|i| raw[i * size..(i + 1) * size].iter().copied()).collect()
}

fn check<const D: usize, K: Basic>(tensor: Tensor<D, K>, dtype: DType, shape: [usize; D], expected: &[u8]) -> Result<TensorData, String> {
    if tensor.dtype() != dtype || tensor.dims() != shape {
        return Err(format!("storage metadata {dtype:?} actual={:?} shape={:?}", tensor.dtype(), tensor.dims()));
    }
    let data = tensor.try_into_data().map_err(|e| format!("storage read {dtype:?}: {e:?}"))?;
    if data.dtype() != dtype || data.as_bytes() != expected {
        return Err(format!(
            "storage bytes {dtype:?} read_dtype={:?} bytes={} expected={}",
            data.dtype(),
            data.as_bytes().len(),
            expected.len()
        ));
    }
    Ok(data)
}

fn views<K: Basic>(device: &Device, dtype: DType, length: usize) -> Result<(), String> {
    let raw = payload(dtype, 3 * length);
    let size = dtype.size();
    let options = TensorCreationOptions::new(device.clone()).with_dtype(dtype);
    let source = Tensor::<2, K>::from_data(TensorData::from_bytes_vec(raw.clone(), [3, length], dtype), options.clone());
    let logical = elements(
        &raw,
        size,
        (0..length).flat_map(|row| (0..3).map(move |col| (2 - col) * length + length - 1 - row)),
    );
    let view = source.clone().permute([1, 0]).flip([0, 1]);
    let record = check(view.clone().reshape([3 * length]), dtype, [3 * length], &logical)?;
    check(Tensor::<1, K>::from_data(record, options.clone()), dtype, [3 * length], &logical)?;
    let expected: Vec<_> = (0..length)
        .step_by(2)
        .flat_map(|r| logical[r * 3 * size..(r + 1) * 3 * size].to_vec())
        .collect();
    let step = view.slice([Slice::new(0, Some(length as isize), 2), Slice::full()]);
    check(step.reshape([3 * length.div_ceil(2)]), dtype, [3 * length.div_ceil(2)], &expected)?;
    let scalar = source
        .clone()
        .slice([Slice::new(1, Some(2), 1), Slice::new((length - 1) as isize, Some(length as isize), 1)]);
    let value = &raw[(2 * length - 1) * size..2 * length * size];
    check(scalar.expand([129, 3]).reshape([387]), dtype, [387], &value.repeat(387))?;
    if length >= 3 {
        let window: Tensor<3, K> = source.clone().unfold(1, 3, 1);
        let expected = elements(
            &raw,
            size,
            (0..3).flat_map(|row| (0..length - 2).flat_map(move |at| (0..3).map(move |k| row * length + at + k))),
        );
        check(window.reshape([9 * (length - 2)]), dtype, [9 * (length - 2)], &expected)?;
    }
    check(source, dtype, [3, length], &raw)?;
    let empty = Tensor::<2, K>::from_data(TensorData::from_bytes_vec(Vec::new(), [0, length], dtype), options);
    check(empty.permute([1, 0]).reshape([-1]), dtype, [0], &[])?;
    Ok(())
}

// High-level boolean creation normalizes to the device default width. Use the
// public backend API to exercise both explicitly declared physical widths.
fn check_bool<const D: usize>(tensor: BoolTensor<Dispatch>, dtype: DType, shape: [usize; D], expected: &[u8]) -> Result<TensorData, String> {
    if tensor.dtype() != dtype || tensor.shape().as_slice() != shape {
        return Err(format!("boolean storage metadata {dtype:?}: {:?} {:?}", tensor.dtype(), tensor.shape()));
    }
    let data = burn::tensor::read_sync(Dispatch::bool_into_data(tensor)).map_err(|e| format!("boolean storage read: {e:?}"))?;
    if data.dtype() != dtype || data.as_bytes() != expected {
        return Err(format!("boolean storage bytes {dtype:?}"));
    }
    Ok(data)
}

fn bool_views(device: &Device, dtype: DType, length: usize) -> Result<(), String> {
    let raw = payload(dtype, 3 * length);
    let size = dtype.size();
    let source = Dispatch::bool_from_data(TensorData::from_bytes_vec(raw.clone(), [3, length], dtype), device.as_dispatch());
    let view = Dispatch::bool_flip(Dispatch::bool_permute(source.clone(), &[1, 0]), &[0, 1]);
    let logical = elements(&raw, size, (0..length).flat_map(|r| (0..3).map(move |c| (2 - c) * length + length - 1 - r)));
    let record = check_bool(Dispatch::bool_reshape(view.clone(), [3 * length].into()), dtype, [3 * length], &logical)?;
    check_bool(Dispatch::bool_from_data(record, device.as_dispatch()), dtype, [3 * length], &logical)?;
    let step = Dispatch::bool_slice(view, &[Slice::new(0, Some(length as isize), 2), Slice::full()]);
    let expected: Vec<_> = (0..length)
        .step_by(2)
        .flat_map(|r| logical[r * 3 * size..(r + 1) * 3 * size].to_vec())
        .collect();
    check_bool(
        Dispatch::bool_reshape(step, [3 * length.div_ceil(2)].into()),
        dtype,
        [3 * length.div_ceil(2)],
        &expected,
    )?;
    let scalar = Dispatch::bool_slice(
        source.clone(),
        &[Slice::new(1, Some(2), 1), Slice::new((length - 1) as isize, Some(length as isize), 1)],
    );
    let broadcast = Dispatch::bool_expand(scalar, [129, 3].into());
    check_bool(
        Dispatch::bool_reshape(broadcast, [387].into()),
        dtype,
        [387],
        &raw[(2 * length - 1) * size..2 * length * size].repeat(387),
    )?;
    if length >= 3 {
        let window = Dispatch::bool_unfold(source.clone(), 1, 3, 1);
        let expected = elements(
            &raw,
            size,
            (0..3).flat_map(|r| (0..length - 2).flat_map(move |at| (0..3).map(move |k| r * length + at + k))),
        );
        check_bool(Dispatch::bool_reshape(window, [9 * (length - 2)].into()), dtype, [9 * (length - 2)], &expected)?;
    }
    check_bool(source, dtype, [3, length], &raw)?;
    let empty = Dispatch::bool_from_data(TensorData::from_bytes_vec(Vec::new(), [0, length], dtype), device.as_dispatch());
    check_bool(Dispatch::bool_reshape(Dispatch::bool_permute(empty, &[1, 0]), [0].into()), dtype, [0], &[])?;
    Ok(())
}

fn fills(device: &Device, dtype: DType) -> Result<(), String> {
    let options = TensorCreationOptions::new(device.clone()).with_dtype(dtype);
    let size = dtype.size();
    match dtype {
        DType::F64 | DType::F32 | DType::Flex32 | DType::F16 | DType::BF16 => {
            let bits = match dtype {
                DType::F64 => 0.5f64.to_bits(),
                DType::F16 => 0x3800,
                DType::BF16 => 0x3f00,
                _ => u64::from(0.5f32.to_bits()),
            };
            check(
                Tensor::<1>::full([129], 0.5f64, options.clone()),
                dtype,
                [129],
                &bits.to_le_bytes()[..size].repeat(129),
            )?;
            let bits = 1u64 << (size * 8 - 1);
            check(
                Tensor::<1>::full([3], -0.0f64, options.clone()),
                dtype,
                [3],
                &bits.to_le_bytes()[..size].repeat(3),
            )?;
            let one = match dtype {
                DType::F64 => 1f64.to_bits(),
                DType::F16 => 0x3c00,
                DType::BF16 => 0x3f80,
                _ => u64::from(1f32.to_bits()),
            };
            check(Tensor::<1>::ones([3], options), dtype, [3], &one.to_le_bytes()[..size].repeat(3))?;
        }
        DType::Bool(_) => {
            check_bool(
                Dispatch::bool_ones([129].into(), device.as_dispatch(), dtype.into()),
                dtype,
                [129],
                &1u32.to_le_bytes()[..size].repeat(129),
            )?;
            check_bool(
                Dispatch::bool_zeros([3].into(), device.as_dispatch(), dtype.into()),
                dtype,
                [3],
                &vec![0; 3 * size],
            )?;
        }
        DType::U64 => {
            let v = 0xfedc_ba98_7654_3210u64;
            check(Tensor::<1, Int>::full([129], v, options), dtype, [129], &v.to_le_bytes().repeat(129))?;
        }
        DType::I64 => {
            let v = i64::MIN + 19;
            check(Tensor::<1, Int>::full([129], v, options), dtype, [129], &v.to_le_bytes().repeat(129))?;
        }
        DType::I8 | DType::I16 | DType::I32 => {
            let v = -13i32;
            check(Tensor::<1, Int>::full([129], v, options), dtype, [129], &v.to_le_bytes()[..size].repeat(129))?;
        }
        _ => {
            let v = 253u32;
            check(Tensor::<1, Int>::full([129], v, options), dtype, [129], &v.to_le_bytes()[..size].repeat(129))?;
        }
    }
    Ok(())
}

pub fn run(device: &Device, include_flex32: bool) -> Result<(), String> {
    for dtype in DTYPES {
        if dtype == DType::Flex32 && !include_flex32 {
            continue;
        }
        for length in [1, 3, 127, 128, 129, 137] {
            match dtype {
                DType::F64 | DType::F32 | DType::Flex32 | DType::F16 | DType::BF16 => views::<Float>(device, dtype, length),
                DType::Bool(_) => bool_views(device, dtype, length),
                _ => views::<Int>(device, dtype, length),
            }
            .map_err(|e| format!("{dtype:?} views length={length}: {e}"))?;
        }
        fills(device, dtype).map_err(|e| format!("{dtype:?} fill: {e}"))?;
    }
    for dtype in [DType::F16, DType::BF16] {
        let raw: Vec<_> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let options = TensorCreationOptions::new(device.clone()).with_dtype(dtype);
        let source = Tensor::<2>::from_data(TensorData::from_bytes_vec(raw.clone(), [256, 256], dtype), options);
        let expected: Vec<_> = (0..256)
            .flat_map(|row| (0..256).flat_map(move |col| ((col * 256 + row) as u16).to_le_bytes()))
            .collect();
        check(source.permute([1, 0]).reshape([65536]), dtype, [65536], &expected)?;
    }
    Ok(())
}

/// Isolate directly tagged Flex32 uploads for backend diagnostics.
pub fn flex32_direct(device: &Device) -> Result<(), String> {
    let raw = payload(DType::Flex32, 3);
    let options = TensorCreationOptions::new(device.clone()).with_dtype(DType::Flex32);
    check(
        Tensor::<1>::from_data(TensorData::from_bytes_vec(raw.clone(), [3], DType::Flex32), options),
        DType::Flex32,
        [3],
        &raw,
    )?;
    Ok(())
}
