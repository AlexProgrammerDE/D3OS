//! Shared public-API integer quantization conformance for native and Linux GPUs.
use alloc::{format, string::String, vec::Vec};
use burn::{
    backend::{DType, TensorData},
    tensor::{
        Device, Tensor,
        quantization::{QuantScheme, QuantStore, QuantValue, QuantizationParameters},
    },
};

fn check<const D: usize>(tensor: Tensor<D>, expected: &[f32]) -> Result<(), String> {
    let actual = tensor.dequantize().try_into_vec_as::<f32>().map_err(|e| format!("quantized read: {e:?}"))?;
    if actual.len() != expected.len() || actual.iter().zip(expected).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return Err(format!("quantized values: {actual:?} != {expected:?}"));
    }
    Ok(())
}

pub fn packed_reference(values: &[i32], bits: usize) -> Vec<u8> {
    values
        .chunks(32 / bits)
        .flat_map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u32, |word, (i, &value)| word | ((value as u32 & ((1 << bits) - 1)) << (i * bits)))
                .to_le_bytes()
        })
        .collect()
}

/// Check integer rounding, packed bytes, views, serialization, and calibration
/// using one set of independent references on each explicitly selected device.
pub fn integer(device: &Device) -> Result<(), String> {
    let input = [-9., -8., -7.5, -7., -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 6.5, 7., 7.5, 8., 9., 0.];
    let rounded = [-9i32, -8, -8, -7, -2, -2, 0, 0, 2, 2, 6, 7, 8, 8, 9, 0];
    let values: Vec<_> = input.into_iter().cycle().take(32).collect();
    let source = Tensor::<2>::from_data(TensorData::new(values.clone(), [2, 16]), device);
    for value in [
        QuantValue::Q8S,
        QuantValue::Q8F,
        QuantValue::Q4S,
        QuantValue::Q4F,
        QuantValue::Q2S,
        QuantValue::Q2F,
    ] {
        let scheme = QuantScheme::default().with_value(value);
        let (low, high) = value.range();
        let expected: Vec<_> = rounded.into_iter().cycle().take(32).map(|n| n.clamp(low as i32, high as i32)).collect();
        let floats: Vec<_> = expected.iter().map(|&n| n as f32).collect();
        let scales = Tensor::<1>::from_floats([1.], device);
        let q = source.clone().quantize(&scheme, QuantizationParameters { scales, global: None });
        check(q.clone(), &floats)?;
        let transposed: Vec<_> = (0..16).flat_map(|col| [floats[col], floats[16 + col]]).collect();
        check(q.clone().transpose(), &transposed)?;
        let outer_scheme = scheme.with_store(QuantStore::PackedU32(1));
        let outer = source.clone().transpose().quantize(
            &outer_scheme,
            QuantizationParameters {
                scales: Tensor::<1>::from_floats([1.], device),
                global: None,
            },
        );
        check(outer, &transposed)?;

        // Burn CUDA cannot materialize a reshape of packed sub-byte values.
        // Native-only checks retain that additional capability separately.
        if value.size_bits() == 8 {
            check(q.clone().reshape([1, 32]), &floats)?;
        }
        let mut bytes = packed_reference(&expected, value.size_bits());
        bytes.extend_from_slice(&1f32.to_le_bytes());
        let data = q.clone().try_into_data().map_err(|e| format!("quantized save: {e:?}"))?;
        if data.as_bytes() != bytes || data.dtype() != DType::QFloat(scheme) {
            return Err(format!("quantized packed serialization mismatch: {value:?}"));
        }
        check(Tensor::<2>::from_data(data, device), &floats)?;
        let data = q.transpose().try_into_data().map_err(|e| format!("quantized view save: {e:?}"))?;
        check(Tensor::<2>::from_data(data, device), &transposed)?;
        if value.is_symmetric() {
            let endpoints: Vec<_> = (0..32).map(|i| if i % 2 == 0 { high } else { -high }).collect();
            let x = Tensor::<2>::from_data(TensorData::new(endpoints.clone(), [2, 16]), device);
            check(x.quantize_dynamic(&scheme), &endpoints)?;
            check(Tensor::<2>::zeros([2, 16], device).quantize_dynamic(&scheme), &[0.; 32])?;
        }
    }
    check(source, &values)?;

    Ok(())
}
