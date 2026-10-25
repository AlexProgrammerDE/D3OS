//! Independent logical-order references for dynamic GPU selection.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{DType, TensorData},
    tensor::{
        Bool, Device, Int, Tensor,
        quantization::{QuantScheme, QuantStore, QuantizationParameters},
    },
};

fn words<const D: usize>(tensor: Tensor<D>) -> Result<Vec<u32>, String> {
    let data = tensor.try_into_data().map_err(|e| format!("compaction read: {e:?}"))?;
    Ok(data.as_bytes().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
}

async fn coordinates<const D: usize>(mask: Tensor<D, Bool>, logical: &[bool]) -> Result<(), String> {
    let dims = mask.dims();
    let mut expected = Vec::new();
    for (i, &selected) in logical.iter().enumerate() {
        if selected {
            let mut linear = i;
            let mut row = [0i32; D];
            for axis in (0..D).rev() {
                row[axis] = (linear % dims[axis]) as i32;
                linear /= dims[axis];
            }
            expected.extend(row);
        }
    }
    let count = expected.len() / D;
    let actual = mask.clone().argwhere_async().await;
    if actual.dims() != [count, D] || actual.try_into_vec_as::<i32>().map_err(|e| format!("argwhere: {e:?}"))? != expected {
        return Err(format!("argwhere logical order rank={D} shape={dims:?}"));
    }
    let nonzero = mask.clone().nonzero_async().await;
    if nonzero.len() != if count == 0 { 0 } else { D } {
        return Err(String::from("nonzero axis count"));
    }
    for (axis, tensor) in nonzero.into_iter().enumerate() {
        let want: Vec<_> = expected.chunks_exact(D).map(|row| row[axis]).collect();
        if tensor.dims() != [count] || tensor.try_into_vec_as::<i32>().map_err(|e| format!("nonzero: {e:?}"))? != want {
            return Err(String::from("nonzero axis coordinates"));
        }
    }
    if mask.try_into_vec_as::<bool>().map_err(|e| format!("mask retention: {e:?}"))? != logical {
        return Err(String::from("coordinate extraction mutated its shared mask"));
    }
    Ok(())
}

fn source_index(i: usize, length: usize) -> usize {
    (1 - i % 2) * length * 3 + (length - 1 - i / 6) * 3 + i / 2 % 3
}

fn selected(i: usize, pattern: usize) -> bool {
    match pattern {
        0 => false,
        1 => true,
        2 => i.is_multiple_of(2),
        _ => i.is_multiple_of(7) || i % 129 == 128,
    }
}

pub async fn run(device: &Device, include_empty: bool, include_compact_q8: bool) -> Result<(), String> {
    for length in [1, 127, 128, 129, 257, 531] {
        let shape = [2, length, 3];
        let raw: Vec<u32> = (0..6 * length)
            .map(|i| {
                if i % 3 == 0 {
                    [0, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234, 0xff80_0001, 1, 0x8000_0001][i / 3 % 8]
                } else {
                    (i as u32).wrapping_mul(0x9e37_79b9)
                }
            })
            .collect();
        let logical: Vec<_> = (0..raw.len()).map(|i| raw[source_index(i, length)]).collect();
        let floats = Tensor::<3>::from_data(TensorData::new(raw.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(), shape), device)
            .permute([1, 2, 0])
            .flip([0, 2]);
        let integers = Tensor::<3, Int>::from_data(TensorData::new(raw.iter().map(|&v| v as i32).collect::<Vec<_>>(), shape), device)
            .permute([1, 2, 0])
            .flip([0, 2]);
        let bool_data: Vec<_> = (0..raw.len()).map(|i| i % 3 == 1).collect();
        let booleans = Tensor::<3, Bool>::from_data(TensorData::new(bool_data.clone(), shape), device)
            .permute([1, 2, 0])
            .flip([0, 2]);
        for pattern in 0..5 {
            let mask = if pattern == 4 {
                Tensor::<3, Bool>::from_data(
                    TensorData::new((0..length).map(|i| i.is_multiple_of(5)).collect::<Vec<_>>(), [1, length, 1]),
                    device,
                )
                .expand(shape)
            } else {
                Tensor::<3, Bool>::from_data(TensorData::new((0..raw.len()).map(|i| selected(i, pattern)).collect::<Vec<_>>(), shape), device)
            }
            .permute([1, 2, 0])
            .flip([0, 2]);
            let wanted_mask: Vec<_> = (0..raw.len())
                .map(|i| {
                    let source = source_index(i, length);
                    if pattern == 4 {
                        (source / 3 % length).is_multiple_of(5)
                    } else {
                        selected(source, pattern)
                    }
                })
                .collect();
            coordinates(mask.clone(), &wanted_mask).await?;
            let want: Vec<_> = logical.iter().zip(&wanted_mask).filter_map(|(&value, &keep)| keep.then_some(value)).collect();
            let output = floats.clone().mask_select_async(mask.clone()).await;
            if output.dims() != [want.len()] || words(output)? != want {
                return Err(format!("float mask_select bits length={length} pattern={pattern}"));
            }
            let output = integers.clone().mask_select_async(mask.clone()).await;
            if output.dims() != [want.len()]
                || output.try_into_vec_as::<i32>().map_err(|e| format!("integer mask_select: {e:?}"))? != want.iter().map(|&v| v as i32).collect::<Vec<_>>()
            {
                return Err(String::from("integer mask_select precision"));
            }
            let output = booleans.clone().mask_select_async(mask).await;
            let want: Vec<_> = wanted_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &keep)| keep.then_some(bool_data[source_index(i, length)]))
                .collect();
            if output.dims() != [want.len()] || output.try_into_vec_as::<bool>().map_err(|e| format!("bool mask_select: {e:?}"))? != want {
                return Err(String::from("boolean mask_select"));
            }
        }
        if words(floats)? != logical
            || integers.try_into_vec_as::<i32>().map_err(|e| format!("shared integers: {e:?}"))? != logical.iter().map(|&v| v as i32).collect::<Vec<_>>()
        {
            return Err(String::from("mask selection mutated shared values"));
        }
    }
    coordinates(
        Tensor::<1, Bool>::from_bool([false, true, true, false, true], device),
        &[false, true, true, false, true],
    )
    .await?;
    coordinates(
        Tensor::<8, Bool>::from_data(TensorData::new(vec![true, false, true, true], [1, 2, 1, 1, 2, 1, 1, 1]), device),
        &[true, false, true, true],
    )
    .await?;
    let auto = device.clone().autodiff();
    for pattern in 0..4 {
        let length = 129;
        let values: Vec<_> = (0..6 * length).map(|i| (i as i32 % 17 - 8) as f32).collect();
        let source = Tensor::<3>::from_data(TensorData::new(values.clone(), [2, length, 3]), &auto).require_grad();
        let mask_values: Vec<_> = (0..values.len()).map(|i| selected(i, pattern)).collect();
        let mask = Tensor::<3, Bool>::from_data(TensorData::new(mask_values.clone(), [2, length, 3]), &auto)
            .permute([1, 2, 0])
            .flip([0, 2]);
        let mut expected = vec![0f32; values.len()];
        let mut weight = 0;
        for i in 0..values.len() {
            let original = source_index(i, length);
            if mask_values[original] {
                weight += 1;
                expected[original] = weight as f32;
            }
        }
        let output = source.clone().permute([1, 2, 0]).flip([0, 2]).mask_select_async(mask).await;
        let weights = Tensor::<1>::from_data(TensorData::new((1..=weight).map(|i| i as f32).collect::<Vec<_>>(), [weight]), &auto);
        let gradients = (output * weights).sum().backward();
        let actual = source.grad(&gradients).ok_or_else(|| String::from("missing compaction gradient"))?;
        if actual.try_into_vec_as::<f32>().map_err(|e| format!("compaction gradient: {e:?}"))? != expected {
            return Err(format!("mask_select gradient pattern={pattern}"));
        }
    }
    if include_empty {
        let mask = Tensor::<3, Bool>::from_data(TensorData::new(Vec::<bool>::new(), [2, 0, 3]), device);
        coordinates(mask.clone(), &[]).await?;
        let floats = Tensor::<3>::from_data(TensorData::new(Vec::<f32>::new(), [2, 0, 3]), device);
        let integers = Tensor::<3, Int>::from_data(TensorData::new(Vec::<i32>::new(), [2, 0, 3]), device);
        if !words(floats.mask_select_async(mask.clone()).await)?.is_empty()
            || !integers
                .mask_select_async(mask.clone())
                .await
                .try_into_vec_as::<i32>()
                .map_err(|e| format!("empty integer compaction: {e:?}"))?
                .is_empty()
            || !mask
                .clone()
                .mask_select_async(mask)
                .await
                .try_into_vec_as::<bool>()
                .map_err(|e| format!("empty bool compaction: {e:?}"))?
                .is_empty()
        {
            return Err(String::from("empty mask selection"));
        }
    }
    for compact in [false, true].into_iter().take(if include_compact_q8 { 2 } else { 1 }) {
        let (values, mask, shape, expected, want_coordinates) = if compact {
            (
                vec![-2.0f32, 0., 2., 2., 0., -2.],
                vec![true, false, true, false, true, false],
                [2, 3],
                vec![-2.0f32, 2., 0.],
                vec![0, 0, 0, 2, 1, 1],
            )
        } else {
            (
                vec![-2.0f32, 0., 2., 1., 2., 0., -2., -1.],
                vec![true, false, true, false, true, true, false, false],
                [2, 4],
                vec![-2.0f32, 2., 2., 0.],
                vec![0, 0, 0, 2, 1, 0, 1, 1],
            )
        };
        let original = values.clone();
        let scheme = if compact {
            QuantScheme::default().with_store(QuantStore::Native)
        } else {
            QuantScheme::default()
        };
        let q = Tensor::<2>::from_data(TensorData::new(values, shape), device).quantize(
            &scheme,
            QuantizationParameters {
                scales: Tensor::<1>::from_floats([1.], device),
                global: None,
            },
        );
        let mask = Tensor::<2, Bool>::from_data(TensorData::new(mask, shape), device);
        let selected = q.clone().mask_select(mask.clone());
        let dtype = selected.dtype();
        let actual = selected
            .clone()
            .dequantize()
            .try_into_vec_as::<f32>()
            .map_err(|e| format!("quantized compaction compact={compact}: {e:?}"))?;
        // q_select may dequantize and requantize. For this Q8 corpus, a fresh
        // min/max calibration has step 2 * max_abs / (127 - -128). Half that
        // step bounds rounding/clipping error; allow four FP32 rounding ULPs.
        let max_abs = expected.iter().map(|&v| f64::from(v).abs()).fold(0f64, f64::max);
        let tolerance = max_abs / 255. + 4. * f64::from(f32::EPSILON) * max_abs;
        if !matches!(dtype, DType::QFloat(_))
            || selected.dims() != [expected.len()]
            || actual.len() != expected.len()
            || actual
                .iter()
                .zip(&expected)
                .any(|(&a, &b)| !a.is_finite() || (f64::from(a) - f64::from(b)).abs() > tolerance)
        {
            let packed = selected.try_into_data().map_err(|e| format!("quantized compaction diagnostic: {e:?}"))?;
            return Err(format!(
                "Q8 mask_select compact={compact} dtype={dtype:?} actual={actual:?} expected={expected:?} bytes={:?}",
                packed.as_bytes()
            ));
        }
        if q.dequantize().try_into_vec_as::<f32>().map_err(|e| format!("shared Q8 source: {e:?}"))? != original {
            return Err(String::from("quantized mask selection mutated its shared source"));
        }
        if mask.argwhere().try_into_vec_as::<i32>().map_err(|e| format!("sync argwhere: {e:?}"))? != want_coordinates {
            return Err(String::from("synchronous argwhere"));
        }
    }
    Ok(())
}
