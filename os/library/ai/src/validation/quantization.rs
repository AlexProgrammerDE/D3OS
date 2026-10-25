//! Shared public-API quantization conformance for native and Linux GPUs.
use alloc::{format, string::String, vec::Vec};
use burn::{
    backend::{DType, TensorData},
    tensor::{
        Device, Tensor,
        quantization::{QuantScheme, QuantStore, QuantValue, QuantizationParameters, ScaleDtype},
    },
};

fn check<const D: usize>(tensor: Tensor<D>, expected: &[f32]) -> Result<(), String> {
    let actual = tensor.dequantize().try_into_vec_as::<f32>().map_err(|e| format!("quantized read: {e:?}"))?;
    if actual.len() != expected.len() {
        return Err(format!("quantized length: {} != {}", actual.len(), expected.len()));
    }
    if let Some((i, (a, b))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
    {
        return Err(format!("quantized value[{i}]: {a:?} (0x{:08x}) != {b:?} (0x{:08x})", a.to_bits(), b.to_bits()));
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

// Independent enumeration of the format's representable values. The encoder
// searches this table rather than duplicating the GPU bit-rounding algorithm.
fn minifloat_value(format: QuantValue, code: u32) -> f32 {
    let (mantissa_bits, exponent_bits, bias) = match format {
        QuantValue::E4M3 => (3, 4, 7),
        QuantValue::E5M2 => (2, 5, 15),
        QuantValue::E2M1 => (1, 2, 1),
        _ => unreachable!(),
    };
    let magnitude = code & ((1 << (mantissa_bits + exponent_bits)) - 1);
    let mantissa = magnitude & ((1 << mantissa_bits) - 1);
    let exponent = magnitude >> mantissa_bits;
    let sign = (code >> (mantissa_bits + exponent_bits)) << 31;
    let value = if format == QuantValue::E4M3 && magnitude == 127 {
        f32::NAN
    } else if format == QuantValue::E5M2 && exponent == 31 {
        if mantissa == 0 { f32::INFINITY } else { f32::NAN }
    } else if exponent == 0 {
        let step = f32::from_bits((127 + 1 - bias - mantissa_bits) << 23);
        mantissa as f32 * step
    } else {
        let unit = f32::from_bits((127 + exponent - bias) << 23);
        (1.0 + mantissa as f32 / (1 << mantissa_bits) as f32) * unit
    };
    f32::from_bits(value.to_bits() | sign)
}

fn minifloat_code(format: QuantValue, value: f32) -> i32 {
    let sign = (value.to_bits() >> 31) << (format.size_bits() - 1);
    let maximum = match format {
        QuantValue::E4M3 => 126,
        QuantValue::E5M2 => 123,
        QuantValue::E2M1 => 7,
        _ => unreachable!(),
    };
    let magnitude = f32::from_bits(value.to_bits() & 0x7fff_ffff);
    // The public API clamps before encoding. Linux CUDA's minimum/maximum
    // operations map either NaN sign to the positive saturation endpoint.
    if magnitude.is_nan() {
        return maximum as i32;
    }
    let code = if magnitude >= minifloat_value(format, maximum) {
        maximum
    } else {
        let mut best = 0;
        let mut distance = magnitude;
        for code in 1..=maximum {
            let difference = (magnitude - minifloat_value(format, code)).abs();
            if difference < distance || (difference == distance && code % 2 == 0) {
                best = code;
                distance = difference;
            }
        }
        best
    };
    (code | sign) as i32
}

fn minifloat_bytes(codes: &[i32], scheme: QuantScheme) -> Vec<u8> {
    let mut bytes = packed_reference(codes, scheme.value.size_bits());
    if matches!(scheme.store, QuantStore::Native | QuantStore::PackedNative(_)) {
        bytes.truncate((codes.len() * scheme.value.size_bits()).div_ceil(8));
    }
    bytes.extend_from_slice(&1f32.to_le_bytes());
    bytes
}

fn saved_bytes<const D: usize>(tensor: Tensor<D>, expected: &[u8]) -> Result<TensorData, String> {
    let data = tensor.try_into_data().map_err(|e| format!("minifloat save: {e:?}"))?;
    if data.as_bytes() != expected {
        let mismatch = data.as_bytes().iter().zip(expected).position(|(a, b)| a != b);
        return Err(format!(
            "minifloat bytes {:?}: index={mismatch:?} actual={:?} expected={expected:?}",
            data.dtype(),
            data.as_bytes()
        ));
    }
    Ok(data)
}

/// Exhaust every encoded value and each finite rounding boundary, including
/// its two adjacent F32 values, on either public Burn backend.
pub fn minifloat(device: &Device) -> Result<(), String> {
    for format in [QuantValue::E4M3, QuantValue::E5M2] {
        minifloat_format(device, format, &[QuantStore::PackedU32(0), QuantStore::Native], true, true)?;
    }
    // Pinned Linux rejects native FP4 quantization and returns incorrect data
    // in a rectangular FP4 block case. Native checks retain both separately.
    minifloat_format(device, QuantValue::E2M1, &[QuantStore::PackedU32(0)], false, false)
}

pub fn minifloat_native_additional(device: &Device) -> Result<(), String> {
    minifloat_format(device, QuantValue::E2M1, &[QuantStore::PackedU32(0), QuantStore::PackedNative(0)], true, true)
}

fn minifloat_format(device: &Device, format: QuantValue, stores: &[QuantStore], blocks: bool, compact_tails: bool) -> Result<(), String> {
    let count = 1 << format.size_bits();
    let codes: Vec<_> = (0..count).map(|c| c as i32).collect();
    let decoded: Vec<_> = (0..count).map(|c| minifloat_value(format, c as u32)).collect();
    let mut inputs = decoded.clone();
    let maximum = if format == QuantValue::E4M3 {
        126
    } else if format == QuantValue::E5M2 {
        123
    } else {
        7
    };
    for code in 0..maximum {
        let midpoint = (minifloat_value(format, code) + minifloat_value(format, code + 1)) * 0.5;
        for bits in [midpoint.to_bits() - 1, midpoint.to_bits(), midpoint.to_bits() + 1] {
            inputs.extend([f32::from_bits(bits), -f32::from_bits(bits)]);
        }
    }
    inputs.extend([f32::MAX, -f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN, 0.0, -0.0]);
    inputs.resize(inputs.len().next_multiple_of(16), 0.0);
    let rounded: Vec<_> = inputs.iter().map(|&x| minifloat_code(format, x)).collect();
    let expected: Vec<_> = rounded.iter().map(|&c| minifloat_value(format, c as u32)).collect();
    let compact = if format == QuantValue::E2M1 {
        QuantStore::PackedNative(0)
    } else {
        QuantStore::Native
    };
    for &store in stores {
        let scheme = QuantScheme::default().with_value(format).with_store(store);
        let bytes = minifloat_bytes(&codes, scheme);
        let q = Tensor::<2>::from_data(TensorData::from_bytes_vec(bytes.clone(), [2, count / 2], DType::QFloat(scheme)), device);
        check(q.clone(), &decoded)?;
        let data = saved_bytes(q.clone(), &bytes)?;
        check(Tensor::<2>::from_data(data, device), &decoded)?;
        let transposed: Vec<_> = (0..count / 2).flat_map(|i| [decoded[i], decoded[count / 2 + i]]).collect();
        check(q.clone().transpose(), &transposed)?;
        let data = q.transpose().try_into_data().map_err(|e| format!("minifloat transposed save: {e:?}"))?;
        check(Tensor::<2>::from_data(data, device), &transposed)?;

        let q = Tensor::<2>::from_data(TensorData::new(inputs.clone(), [2, inputs.len() / 2]), device).quantize(
            &scheme,
            QuantizationParameters {
                scales: Tensor::<1>::from_floats([1.], device),
                global: None,
            },
        );
        check(q.clone(), &expected).map_err(|e| format!("{format:?} {store:?} rounding: {e}"))?;
        saved_bytes(q, &minifloat_bytes(&rounded, scheme))?;
        let maximum = format.range().1;
        let endpoints: Vec<_> = (0..32).map(|i| if i % 2 == 0 { maximum } else { -maximum }).collect();
        check(
            Tensor::<2>::from_data(TensorData::new(endpoints.clone(), [2, 16]), device).quantize_dynamic(&scheme),
            &endpoints,
        )?;
        check(Tensor::<2>::zeros([2, 16], device).quantize_dynamic(&scheme), &[0.; 32])?;

        if blocks {
            let scheme = scheme.per_block([2, 8], ScaleDtype::F32);
            let scales = [0.5, 1., 2., 4.];
            let block_values: Vec<_> = (0..64).map(|i| minifloat_value(format, (i % 7) as u32) * scales[(i % 32) / 8]).collect();
            let q = Tensor::<2>::from_data(TensorData::new(block_values.clone(), [2, 32]), device).quantize(
                &scheme,
                QuantizationParameters {
                    scales: Tensor::<1>::from_floats(scales, device),
                    global: None,
                },
            );
            check(q.clone(), &block_values).map_err(|e| format!("{format:?} {store:?} block quantization: {e}"))?;
            let transposed: Vec<_> = (0..32).flat_map(|i| [block_values[i], block_values[32 + i]]).collect();
            let data = q.transpose().try_into_data().map_err(|e| format!("minifloat block view save: {e:?}"))?;
            check(Tensor::<2>::from_data(data, device), &transposed).map_err(|e| format!("{format:?} {store:?} block view reload: {e}"))?;
        }
    }

    // Small native representations exercise scales following a partial
    // word, including a single FP4 byte followed by an unaligned F32 scale.
    if compact_tails {
        let scheme = QuantScheme::default().with_value(format).with_store(compact);
        let q = Tensor::<1>::from_floats([0.5, -1.5], device).quantize(
            &scheme,
            QuantizationParameters {
                scales: Tensor::<1>::from_floats([1.], device),
                global: None,
            },
        );
        check(q.clone(), &[0.5, -1.5])?;
        let codes = [minifloat_code(format, 0.5), minifloat_code(format, -1.5)];
        let data = saved_bytes(q, &minifloat_bytes(&codes, scheme))?;
        check(Tensor::<1>::from_data(data, device), &[0.5, -1.5])?;
    }
    Ok(())
}

fn scale_value(dtype: ScaleDtype, code: u32) -> f32 {
    match dtype {
        ScaleDtype::F32 => f32::from_bits(code),
        ScaleDtype::F16 => {
            let exponent = code >> 10;
            let mantissa = code & 1023;
            if exponent == 0 {
                mantissa as f32 / 16_777_216.0
            } else {
                (1.0 + mantissa as f32 / 1024.0) * f32::from_bits((exponent + 112) << 23)
            }
        }
        ScaleDtype::BF16 => f32::from_bits(code << 16),
        ScaleDtype::UE8M0 => {
            if code == 0 {
                f32::from_bits(0x0040_0000)
            } else {
                f32::from_bits(code << 23)
            }
        }
        ScaleDtype::UE4M3 => minifloat_value(QuantValue::E4M3, code),
    }
}

// Search the representable scale table instead of reproducing the GPU's
// upward bit bias. All inputs here are non-negative and finite.
fn scale_code_up(dtype: ScaleDtype, input: f32) -> u32 {
    let mut high = match dtype {
        ScaleDtype::F32 => return input.to_bits(),
        ScaleDtype::F16 => 0x7bff,
        ScaleDtype::BF16 => 0x7f7f,
        ScaleDtype::UE8M0 => 254,
        ScaleDtype::UE4M3 => 126,
    };
    let mut low = 0;
    while low < high {
        let middle = low + (high - low) / 2;
        if scale_value(dtype, middle) < input {
            low = middle + 1
        } else {
            high = middle
        }
    }
    low
}

fn scale_bytes(dtype: ScaleDtype, codes: &[u32]) -> Vec<u8> {
    let size = match dtype {
        ScaleDtype::F32 => 4,
        ScaleDtype::F16 | ScaleDtype::BF16 => 2,
        _ => 1,
    };
    codes.iter().flat_map(|code| code.to_le_bytes().into_iter().take(size)).collect()
}

fn scale_case(device: &Device, dtype: ScaleDtype, format: QuantValue, global: Option<f32>, matrix: bool) -> Result<(), String> {
    let inputs = [0.0, 0.00001, 0.5, 1.00001, 1.001, 1.1, 65520.0];
    let codes: Vec<_> = inputs.iter().map(|&s| scale_code_up(dtype, s)).collect();
    let scales: Vec<_> = codes.iter().map(|&code| scale_value(dtype, code)).collect();
    let global_value = global.unwrap_or(1.0);
    let values: Vec<_> = scales
        .iter()
        .flat_map(|&s| (0..16).map(move |i| if i % 2 == 0 { s * global_value } else { -s * global_value }))
        .collect();
    let mut scheme = QuantScheme::default().with_value(format).per_block([1, 16], dtype);
    if global.is_some() {
        scheme = scheme.per_tensor(ScaleDtype::F32);
    }
    let params = QuantizationParameters {
        scales: Tensor::<1>::from_floats(inputs, device),
        global: global.map(|v| Tensor::<1>::from_floats([v], device)),
    };
    let q = if matrix {
        Tensor::<2>::from_data(TensorData::new(values.clone(), [7, 16]), device).quantize(&scheme, params)
    } else {
        Tensor::<1>::from_data(TensorData::new(values.clone(), [112]), device)
            .quantize(&scheme, params)
            .reshape([7, 16])
    };
    // A zero scale creates a NaN quotient and clamps to the positive
    // endpoint, so reconstruction produces positive zero for either input sign.
    let expected: Vec<_> = values.iter().map(|&v| if v == 0.0 { 0.0 } else { v }).collect();
    check(q.clone(), &expected).map_err(|e| format!("scale {dtype:?} {format:?} global={global:?}: {e}"))?;
    let fields: Vec<_> = scales
        .iter()
        .flat_map(|&s| {
            (0..16).map(move |i| {
                if s == 0.0 {
                    if format.is_symmetric() {
                        format.range().1 as i32
                    } else {
                        minifloat_code(format, f32::NAN)
                    }
                } else if format.is_symmetric() {
                    if i % 2 == 0 { 1 } else { -1 }
                } else {
                    minifloat_code(format, if i % 2 == 0 { 1.0 } else { -1.0 })
                }
            })
        })
        .collect();
    let mut bytes = packed_reference(&fields, format.size_bits());
    bytes.extend(scale_bytes(dtype, &codes));
    if let Some(global) = global {
        bytes.extend(global.to_le_bytes());
    }
    let data = saved_bytes(q.clone(), &bytes).map_err(|e| format!("scale {dtype:?} {format:?} global={global:?}: {e}"))?;
    check(Tensor::<2>::from_data(data, device), &expected)?;
    if matrix {
        let reversed: Vec<_> = expected.iter().copied().rev().collect();
        let flipped = q.clone().flip([0, 1]);
        check(flipped.clone(), &reversed)?;
        let data = flipped.try_into_data().map_err(|e| format!("flipped scale save: {e:?}"))?;
        check(Tensor::<2>::from_data(data, device), &reversed)?;
    }
    let transposed: Vec<_> = (0..16)
        .flat_map(|col| (0..7).map(move |row| (row, col)))
        .map(|(r, c)| expected[r * 16 + c])
        .collect();
    let data = q.transpose().try_into_data().map_err(|e| format!("scale view save: {e:?}"))?;
    check(Tensor::<2>::from_data(data, device), &transposed)?;
    Ok(())
}

fn scale_record(device: &Device, dtype: ScaleDtype, global: Option<f32>) -> Result<(), String> {
    let codes = match dtype {
        ScaleDtype::F32 => [0, 1, 0x3f80_0000, 0x7f7f_ffff, 0x7f80_0000, 0x7fc0_0001, 0xffb1_2345],
        ScaleDtype::F16 => [0, 1, 0x3c00, 0x7bff, 0x7c00, 0x7d01, 0xfe55],
        ScaleDtype::BF16 => [0, 1, 0x3f80, 0x7f7f, 0x7f80, 0x7fc1, 0xffa5],
        ScaleDtype::UE8M0 => [0, 1, 127, 253, 254, 255, 128],
        ScaleDtype::UE4M3 => [0, 1, 0x38, 126, 127, 255, 0x80],
    };
    let mut scheme = QuantScheme::default().with_value(QuantValue::Q8S).per_block([16], dtype);
    if global.is_some() {
        scheme = scheme.per_tensor(ScaleDtype::F32);
    }
    let mut bytes = packed_reference(&[0; 112], 8);
    bytes.extend(scale_bytes(dtype, &codes));
    if let Some(global) = global {
        bytes.extend(global.to_le_bytes());
    }
    let q = Tensor::<1>::from_data(TensorData::from_bytes_vec(bytes.clone(), [112], DType::QFloat(scheme)), device);
    saved_bytes(q.clone(), &bytes).map_err(|e| format!("raw scale record {dtype:?}: {e}"))?;
    saved_bytes(q.reshape([7, 16]).transpose(), &bytes).map_err(|e| format!("raw scale view record {dtype:?}: {e}"))?;
    Ok(())
}

fn scale_dynamic(device: &Device, dtype: ScaleDtype, format: QuantValue) -> Result<(), String> {
    let inner = if dtype == ScaleDtype::F16 {
        [1.0, 2., 3., 4., 64., 128., 65504.]
    } else {
        [0.5, 1., 1.5, 2., 16., 32., 448.]
    };
    let maximum = format.range().1;
    let values: Vec<_> = inner
        .iter()
        .flat_map(|&s| (0..16).map(move |i| (if i % 2 == 0 { maximum } else { -maximum }) * (s * 0.25)))
        .collect();
    let scheme = QuantScheme::default().with_value(format).per_block([16], dtype).per_tensor(ScaleDtype::F32);
    let q = Tensor::<1>::from_data(TensorData::new(values.clone(), [112]), device).quantize_dynamic(&scheme);
    check(q.clone(), &values).map_err(|e| format!("dynamic scales {dtype:?} {format:?}: {e}"))?;
    let fields: Vec<_> = (0..112)
        .map(|i| {
            let value = if i % 2 == 0 { maximum } else { -maximum };
            if format.is_symmetric() { value as i32 } else { minifloat_code(format, value) }
        })
        .collect();
    let mut bytes = packed_reference(&fields, format.size_bits());
    bytes.extend(scale_bytes(dtype, &inner.map(|s| scale_code_up(dtype, s))));
    bytes.extend(0.25f32.to_le_bytes());
    let data = saved_bytes(q, &bytes).map_err(|e| format!("dynamic scale bytes {dtype:?} {format:?}: {e}"))?;
    check(Tensor::<1>::from_data(data, device), &values)?;
    check(Tensor::<1>::zeros([112], device).quantize_dynamic(&scheme), &[0.; 112])?;
    Ok(())
}

/// Public scale storage and reconstruction, with independent code tables.
pub fn scales(device: &Device) -> Result<(), String> {
    for format in [QuantValue::Q8S, QuantValue::E4M3] {
        for dtype in [ScaleDtype::F32, ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE8M0, ScaleDtype::UE4M3] {
            scale_case(device, dtype, format, None, false)?;
        }
        for dtype in [ScaleDtype::F16, ScaleDtype::UE4M3] {
            scale_case(device, dtype, format, Some(0.25), false)?;
            scale_dynamic(device, dtype, format)?;
        }
    }
    for dtype in [ScaleDtype::F32, ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE8M0, ScaleDtype::UE4M3] {
        scale_record(device, dtype, None)?;
    }
    for dtype in [ScaleDtype::F16, ScaleDtype::UE4M3] {
        scale_record(device, dtype, Some(0.25))?;
    }

    Ok(())
}

/// Cases beyond the pinned CUDA backend's multidimensional scale mapping.
pub fn scales_native_additional(device: &Device) -> Result<(), String> {
    for format in [QuantValue::Q4S, QuantValue::E2M1] {
        for dtype in [ScaleDtype::F32, ScaleDtype::F16, ScaleDtype::BF16, ScaleDtype::UE8M0, ScaleDtype::UE4M3] {
            scale_case(device, dtype, format, None, true)?;
        }
        for dtype in [ScaleDtype::F16, ScaleDtype::UE4M3] {
            scale_case(device, dtype, format, Some(0.25), true)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minifloat_reference_preserves_finite_codes_and_known_rounding_ties() {
        for format in [QuantValue::E4M3, QuantValue::E5M2, QuantValue::E2M1] {
            for code in 0..1 << format.size_bits() {
                let value = minifloat_value(format, code);
                if value.is_finite() {
                    assert_eq!(minifloat_code(format, value), code as i32);
                }
            }
        }
        for (input, code) in [(0.25, 0), (0.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5.0, 6)] {
            assert_eq!(minifloat_code(QuantValue::E2M1, input), code);
            assert_eq!(minifloat_code(QuantValue::E2M1, -input), code | 8);
        }
    }

    #[test]
    fn scale_reference_ceilings_match_known_boundaries() {
        for (dtype, code) in [
            (ScaleDtype::F16, 0x3c01),
            (ScaleDtype::BF16, 0x3f81),
            (ScaleDtype::UE4M3, 0x39),
            (ScaleDtype::UE8M0, 128),
        ] {
            assert_eq!(scale_code_up(dtype, 1.00001), code);
            assert_eq!(scale_code_up(dtype, scale_value(dtype, code)), code);
        }
        assert_eq!(scale_code_up(ScaleDtype::F16, 1e-10), 1);
        assert_eq!(scale_code_up(ScaleDtype::BF16, f32::from_bits(1)), 1);
        assert_eq!(scale_code_up(ScaleDtype::UE8M0, 0.0), 0);
        assert_eq!(scale_code_up(ScaleDtype::F16, f32::MAX), 0x7bff);
        assert_eq!(scale_code_up(ScaleDtype::UE4M3, f32::MAX), 126);
    }
}
