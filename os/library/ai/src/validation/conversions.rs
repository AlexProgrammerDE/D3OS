//! Ordinary numeric conversions with an independent binary rounding reference.
use super::storage::DTYPES;
use alloc::{format, string::String, vec::Vec};
use burn::{
    backend::{
        DType, Dispatch, TensorData, TensorMetadata,
        ops::{BoolTensorOps, FloatTensorOps, IntTensorOps},
    },
    tensor::{Device, FloatDType, Int, IntDType, Tensor, TensorCreationOptions},
};

fn float_format(dtype: DType) -> Option<(u32, u32)> {
    match dtype {
        DType::F64 => Some((11, 52)),
        DType::F32 | DType::Flex32 => Some((8, 23)),
        DType::F16 => Some((5, 10)),
        DType::BF16 => Some((8, 7)),
        _ => None,
    }
}
fn bits(bytes: &[u8]) -> u64 {
    let mut raw = [0; 8];
    raw[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(raw)
}
fn round_shift(value: u64, shift: i32) -> u64 {
    if shift <= 0 {
        return value << -shift;
    }
    if shift > 64 {
        return 0;
    }
    if shift == 64 {
        return u64::from(value > (1u64 << 63));
    }
    let upper = value >> shift;
    let remainder = value & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    upper + u64::from(remainder > half || (remainder == half && upper & 1 != 0))
}

/// Encode an exact integer times a power of two, rounding only once.
fn binary_float(negative: bool, significand: u64, exponent: i32, dtype: DType) -> u64 {
    let (e, m) = float_format(dtype).unwrap();
    let sign = u64::from(negative) << (e + m);
    if significand == 0 {
        return sign;
    }
    let bias = (1i32 << (e - 1)) - 1;
    let top = (63 - significand.leading_zeros()) as i32 + exponent;
    let mut effective = top.max(1 - bias);
    let mut rounded = round_shift(significand, effective - m as i32 - exponent);
    if rounded >= (1u64 << (m + 1)) {
        rounded >>= 1;
        effective += 1;
    }
    let encoded_exp = if rounded < (1u64 << m) { 0 } else { effective + bias };
    if encoded_exp >= (1i32 << e) - 1 {
        return sign | (((1u64 << e) - 1) << m);
    }
    sign | ((encoded_exp as u64) << m) | (rounded & ((1u64 << m) - 1))
}
fn float_to_float(raw: u64, source: DType, destination: DType) -> u64 {
    let (e, m) = float_format(source).unwrap();
    let negative = raw >> (e + m) != 0;
    let exponent = (raw >> m) & ((1u64 << e) - 1);
    let fraction = raw & ((1u64 << m) - 1);
    if exponent == (1u64 << e) - 1 {
        let (de, dm) = float_format(destination).unwrap();
        return (u64::from(negative) << (de + dm)) | (((1u64 << de) - 1) << dm) | if fraction != 0 { 1u64 << (dm - 1) } else { 0 };
    }
    let bias = (1i32 << (e - 1)) - 1;
    let significand = fraction | if exponent != 0 { 1u64 << m } else { 0 };
    binary_float(negative, significand, exponent.max(1) as i32 - bias - m as i32, destination)
}
fn signed(dtype: DType) -> bool {
    matches!(dtype, DType::I64 | DType::I32 | DType::I16 | DType::I8)
}
fn integer(raw: u64, dtype: DType) -> (bool, u64) {
    if signed(dtype) {
        let value = ((raw << (64 - dtype.size() * 8)) as i64) >> (64 - dtype.size() * 8);
        (value < 0, value.unsigned_abs())
    } else {
        (false, if dtype.is_bool() { u64::from(raw != 0) } else { raw })
    }
}
fn as_float(raw: u64, dtype: DType) -> f64 {
    f64::from_bits(float_to_float(raw, dtype, DType::F64))
}

pub fn reference(source: DType, destination: DType, raw: u64) -> u64 {
    if source == destination || (matches!(source, DType::F32 | DType::Flex32) && matches!(destination, DType::F32 | DType::Flex32)) {
        return raw;
    }
    if destination.is_bool() {
        return u64::from(if source.is_float() { as_float(raw, source) != 0.0 } else { raw != 0 });
    }
    if destination.is_float() {
        if source.is_float() {
            return float_to_float(raw, source, destination);
        }
        let (negative, magnitude) = integer(raw, source);
        return binary_float(negative, magnitude, 0, destination);
    }
    if source.is_float() {
        let value = as_float(raw, source);
        return match destination {
            DType::I64 => value as i64 as u64,
            DType::I32 => value as i32 as u32 as u64,
            DType::I16 => value as i16 as u16 as u64,
            DType::I8 => value as i8 as u8 as u64,
            DType::U64 => value as u64,
            DType::U32 => value as u32 as u64,
            DType::U16 => value as u16 as u64,
            DType::U8 => value as u8 as u64,
            _ => unreachable!(),
        };
    }
    let (negative, magnitude) = integer(raw, source);
    if negative { magnitude.wrapping_neg() } else { magnitude }
}
fn portable(raw: u64, source: DType, destination: DType) -> bool {
    if !source.is_float() || destination.is_float() || destination.is_bool() {
        return true;
    }
    let value = as_float(raw, source);
    let width = destination.size() * 8;
    let limit = libm::scalbn(1.0, (width - usize::from(signed(destination))) as i32);
    value.is_finite() && value >= if signed(destination) { -limit } else { 0.0 } && value < limit
}

pub fn input(source: DType, destination: DType, count: usize, portable_only: bool) -> Vec<u8> {
    const INTEGERS: [u64; 18] = [
        0,
        1,
        2,
        u64::MAX,
        i64::MIN as u64,
        i64::MAX as u64,
        16777217,
        16777219,
        9007199254740993,
        0x8000_0080_0000_0001,
        0x8080_0000_0000_0001,
        65519,
        65520,
        127,
        128,
        255,
        256,
        (-127i64) as u64,
    ];
    let floats = [
        0.0f64,
        -0.0,
        1.0,
        -1.0,
        127.75,
        -128.75,
        65519.0,
        65520.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::from_bits(1),
        1.0000000596046448,
        f64::from_bits(1.0000000596046448f64.to_bits() + 1),
        1.00390625,
        f64::from_bits(1.00390625f64.to_bits() + 1),
        -0.75,
        0.75,
    ];
    let mask = if source.size() == 8 { u64::MAX } else { (1u64 << (source.size() * 8)) - 1 };
    (0usize..)
        .filter_map(|i| {
            let raw = if source.is_bool() {
                match i % 3 {
                    0 => 0,
                    1 => 1,
                    _ => mask,
                }
            } else if source.is_float() && i % 2 == 0 {
                float_to_float(floats[(i / 2) % floats.len()].to_bits(), DType::F64, source)
            } else if i % 3 != 2 {
                INTEGERS[i % INTEGERS.len()] & mask
            } else {
                (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) & mask
            };
            if portable_only && !portable(raw, source, destination) {
                None
            } else {
                Some(raw)
            }
        })
        .take(count)
        .flat_map(|raw| raw.to_le_bytes()[..source.size()].to_vec())
        .collect()
}

pub fn expected(source: DType, destination: DType, input: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(source.size())
        .flat_map(|v| reference(source, destination, bits(v)).to_le_bytes()[..destination.size()].to_vec())
        .collect()
}
fn nan(raw: u64, dtype: DType) -> bool {
    float_format(dtype).is_some_and(|(e, m)| (raw >> m) & ((1u64 << e) - 1) == (1u64 << e) - 1 && raw & ((1u64 << m) - 1) != 0)
}
pub fn verify(source: DType, destination: DType, actual: &[u8], expected: &[u8]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("cast {source:?}->{destination:?} byte length {} != {}", actual.len(), expected.len()));
    }
    for (i, (a, e)) in actual
        .chunks_exact(destination.size())
        .zip(expected.chunks_exact(destination.size()))
        .enumerate()
    {
        let (a, e) = (bits(a), bits(e));
        let identity = source == destination || (matches!(source, DType::F32 | DType::Flex32) && matches!(destination, DType::F32 | DType::Flex32));
        if a != e && !(!identity && nan(e, destination) && nan(a, destination)) {
            return Err(format!("cast {source:?}->{destination:?} index={i} actual={a:#x} expected={e:#x}"));
        }
    }
    Ok(())
}

pub fn view_bytes(raw: &[u8], dtype: DType, rows: usize, columns: usize) -> Vec<u8> {
    (0..columns)
        .flat_map(|row| {
            (0..rows).flat_map(move |column| {
                let at = ((rows - 1 - column) * columns + columns - 1 - row) * dtype.size();
                raw[at..at + dtype.size()].iter().copied()
            })
        })
        .collect()
}

fn convert(data: TensorData, source: DType, destination: DType, device: &Device) -> Result<TensorData, String> {
    let options = TensorCreationOptions::new(device.clone()).with_dtype(source);
    let result = if source.is_float() {
        let tensor = Tensor::<2>::from_data(data, options).permute([1, 0]).flip([0, 1]);
        if destination.is_float() {
            tensor.cast(FloatDType::from(destination)).try_into_data()
        } else {
            tensor.cast(IntDType::from(destination)).try_into_data()
        }
    } else if !source.is_bool() {
        let tensor = Tensor::<2, Int>::from_data(data, options).permute([1, 0]).flip([0, 1]);
        if destination.is_float() {
            tensor.cast(FloatDType::from(destination)).try_into_data()
        } else {
            tensor.cast(IntDType::from(destination)).try_into_data()
        }
    } else {
        let tensor = Dispatch::bool_from_data(data, device.as_dispatch());
        let tensor = Dispatch::bool_flip(Dispatch::bool_permute(tensor, &[1, 0]), &[0, 1]);
        let tensor = if destination.is_float() {
            Dispatch::bool_into_float(tensor, destination.into())
        } else {
            Dispatch::bool_into_int(tensor, destination.into())
        };
        if tensor.dtype() != destination {
            return Err(format!("boolean cast metadata {source:?}->{destination:?}: {:?}", tensor.dtype()));
        }
        return if destination.is_float() {
            burn::tensor::read_sync(Dispatch::float_into_data(tensor))
        } else {
            burn::tensor::read_sync(Dispatch::int_into_data(tensor))
        }
        .map_err(|e| format!("boolean cast read: {e:?}"));
    };
    result.map_err(|e| format!("public cast read: {e:?}"))
}

pub fn run(device: &Device, include_flex32: bool) -> Result<usize, String> {
    let mut pairs = 0;
    for source in DTYPES {
        if !include_flex32 && source == DType::Flex32 {
            continue;
        }
        for destination in DTYPES.into_iter().filter(|t| !t.is_bool() && (include_flex32 || *t != DType::Flex32)) {
            let raw = input(source, destination, 411, true);
            let logical = view_bytes(&raw, source, 3, 137);
            let actual = convert(TensorData::from_bytes_vec(raw, [3, 137], source), source, destination, device)?;
            if actual.dtype() != destination || actual.shape().as_slice() != [137, 3] {
                return Err(format!(
                    "cast output metadata {source:?}->{destination:?}: {:?} {:?}",
                    actual.dtype(),
                    actual.shape()
                ));
            }
            verify(source, destination, actual.as_bytes(), &expected(source, destination, &logical))?;
            let empty = convert(TensorData::from_bytes_vec(Vec::new(), [0, 3], source), source, destination, device)?;
            if empty.dtype() != destination || empty.shape().as_slice() != [3, 0] || !empty.as_bytes().is_empty() {
                return Err(format!("empty cast {source:?}->{destination:?}"));
            }
            pairs += 1;
        }
    }
    for source in [DType::F16, DType::BF16] {
        let raw: Vec<_> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let logical = view_bytes(&raw, source, 256, 256);
        for destination in DTYPES.into_iter().filter(|t| t.is_float() && (include_flex32 || *t != DType::Flex32)) {
            let actual = convert(TensorData::from_bytes_vec(raw.clone(), [256, 256], source), source, destination, device)?;
            if actual.dtype() != destination || actual.shape().as_slice() != [256, 256] {
                return Err(format!("exhaustive cast metadata {source:?}->{destination:?}"));
            }
            verify(source, destination, actual.as_bytes(), &expected(source, destination, &logical))?;
        }
    }
    bf16_midpoints(device)?;
    gradients(device, include_flex32)?;
    Ok(pairs)
}

fn bf16_boundary_case(source: DType, raw: Vec<u8>, device: &Device) -> Result<(), String> {
    let columns = raw.len() / source.size() / 3;
    let logical = view_bytes(&raw, source, 3, columns);
    let actual = convert(TensorData::from_bytes_vec(raw, [3, columns], source), source, DType::BF16, device)?;
    if actual.dtype() != DType::BF16 || actual.shape().as_slice() != [columns, 3] {
        return Err(format!("BF16 boundary metadata {source:?}"));
    }
    verify(source, DType::BF16, actual.as_bytes(), &expected(source, DType::BF16, &logical))
}

fn bf16_midpoints(device: &Device) -> Result<(), String> {
    let mut raw = Vec::new();
    for code in 0u64..0x7f80 {
        let low = f64::from_bits(float_to_float(code, DType::BF16, DType::F64));
        let midpoint = if code == 0x7f7f {
            low + libm::scalbn(1.0, 119)
        } else {
            let high = f64::from_bits(float_to_float(code + 1, DType::BF16, DType::F64));
            (low + high) * 0.5
        };
        let middle = midpoint.to_bits();
        for bits in [middle - 1, middle, middle + 1] {
            raw.extend_from_slice(&bits.to_le_bytes());
            raw.extend_from_slice(&(bits ^ (1u64 << 63)).to_le_bytes());
        }
    }
    bf16_boundary_case(DType::F64, raw, device)?;
    for source in [DType::I64, DType::U64, DType::I32, DType::U32, DType::I16, DType::U16] {
        let mut raw = Vec::new();
        let width = source.size() * 8 - usize::from(signed(source));
        for exponent in 8..width {
            for significant in 128u64..256 {
                let midpoint = (2 * significant + 1) << (exponent - 8);
                for value in [midpoint - 1, midpoint, midpoint + 1] {
                    raw.extend_from_slice(&value.to_le_bytes()[..source.size()]);
                    if signed(source) {
                        raw.extend_from_slice(&value.wrapping_neg().to_le_bytes()[..source.size()]);
                    }
                }
            }
        }
        bf16_boundary_case(source, raw, device)?;
    }
    Ok(())
}

fn gradients(device: &Device, include_flex32: bool) -> Result<(), String> {
    let auto = device.clone().autodiff();
    for dtype in [DType::F64, DType::F16, DType::BF16, DType::Flex32] {
        if dtype == DType::Flex32 && !include_flex32 {
            continue;
        }
        for source_dtype in [DType::F32, dtype] {
            let options = TensorCreationOptions::new(auto.clone()).with_dtype(source_dtype);
            // Autodiff inherits a host `full` implementation whose Flex32 data
            // is tagged F32. Start from explicitly tagged data for cast checks.
            let value = reference(DType::F64, source_dtype, 0.25f64.to_bits()).to_le_bytes();
            let data = TensorData::from_bytes_vec(value[..source_dtype.size()].repeat(411), [3, 137], source_dtype);
            let source = Tensor::<2>::from_data(data, options).require_grad();
            if source.dtype() != source_dtype {
                return Err(format!("cast gradient source metadata {source_dtype:?}: {:?}", source.dtype()));
            }
            let target = if source_dtype == DType::F32 { dtype } else { DType::F32 };
            let output = source.clone().permute([1, 0]).flip([0, 1]).cast(FloatDType::from(target)).cast(FloatDType::F32);
            let gradients = output.sum().backward();
            let gradient = source
                .grad(&gradients)
                .ok_or_else(|| format!("missing cast gradient {source_dtype:?}->{target:?}"))?;
            if gradient.dtype() != source_dtype || gradient.dims() != [3, 137] {
                return Err(format!(
                    "cast gradient metadata {source_dtype:?}->{target:?}: {:?} {:?}",
                    gradient.dtype(),
                    gradient.dims()
                ));
            }
            let data = gradient.try_into_data().map_err(|e| format!("cast gradient read: {e:?}"))?;
            let one = reference(DType::F32, source_dtype, 1f32.to_bits() as u64).to_le_bytes();
            verify(source_dtype, source_dtype, data.as_bytes(), &one[..source_dtype.size()].repeat(411))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_reference_rounds_once_and_keeps_ties_even() {
        assert_eq!(reference(DType::U64, DType::F32, 16777217), 16777216f32.to_bits() as u64);
        assert_eq!(reference(DType::U64, DType::F32, 16777219), 16777220f32.to_bits() as u64);
        assert_eq!(reference(DType::U64, DType::F32, 0x8000_0080_0000_0001), 0x5f00_0001);
        assert_eq!(reference(DType::U64, DType::BF16, 0x8080_0000_0000_0001), 0x5f01);
        assert_eq!(reference(DType::F64, DType::BF16, 1.00390625f64.to_bits()), 0x3f80);
        assert_eq!(reference(DType::F64, DType::BF16, 1.00390625f64.to_bits() + 1), 0x3f81);
        assert_eq!(reference(DType::F64, DType::BF16, libm::scalbn(1.0, -134).to_bits()), 0);
        assert_eq!(reference(DType::F64, DType::BF16, libm::scalbn(1.0, -134).to_bits() + 1), 1);
        assert_eq!(reference(DType::F64, DType::F32, 1.0000000596046448f64.to_bits()), 1f32.to_bits() as u64);
        assert_eq!(
            reference(DType::F64, DType::F32, 1.0000000596046448f64.to_bits() + 1),
            (1f32.to_bits() + 1) as u64
        );
        assert_eq!(reference(DType::F64, DType::F16, 65520f64.to_bits()), 0x7c00);
        assert_eq!(reference(DType::F64, DType::F16, libm::scalbn(1.0, -25).to_bits()), 0);
        assert_eq!(reference(DType::F64, DType::F16, libm::scalbn(1.0, -25).to_bits() + 1), 1);
    }
}
