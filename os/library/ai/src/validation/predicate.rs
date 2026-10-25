//! Public predicates checked against host numeric types and exact boolean bytes.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{
        BoolDType, DType, Dispatch, Scalar, TensorData, TensorMetadata, bf16, f16,
        ops::{BoolTensorOps, FloatTensorOps, IntTensorOps},
        tensor::{BoolTensor, FloatTensor, IntTensor},
    },
    tensor::{Device, Slice, read_sync},
};

pub const NUMERIC_DTYPES: [DType; 13] = [
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
];
const BOOL_DTYPES: [BoolDType; 2] = [BoolDType::U8, BoolDType::U32];

#[derive(Clone, Copy, Debug)]
pub enum Comparison {
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
}
impl Comparison {
    pub const ALL: [Self; 6] = [Self::Equal, Self::NotEqual, Self::Greater, Self::GreaterEqual, Self::Less, Self::LessEqual];
    fn apply<T: PartialOrd>(self, lhs: T, rhs: T) -> bool {
        match self {
            Self::Equal => lhs == rhs,
            Self::NotEqual => lhs != rhs,
            Self::Greater => lhs > rhs,
            Self::GreaterEqual => lhs >= rhs,
            Self::Less => lhs < rhs,
            Self::LessEqual => lhs <= rhs,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Classification {
    Nan,
    Infinite,
    Finite,
}
impl Classification {
    pub const ALL: [Self; 3] = [Self::Nan, Self::Infinite, Self::Finite];
}

// Widen only after decoding the host storage type. Every F16/BF16/F32 value
// has an exact F64 representation; integers never pass through floating point.
fn host_value(dtype: DType, bits: u64) -> Scalar {
    match dtype {
        DType::F64 => f64::from_bits(bits).into(),
        DType::F32 | DType::Flex32 => f64::from(f32::from_bits(bits as u32)).into(),
        DType::F16 => f16::from_bits(bits as u16).to_f64().into(),
        DType::BF16 => bf16::from_bits(bits as u16).to_f64().into(),
        DType::I64 => (bits as i64).into(),
        DType::I32 => (bits as i32).into(),
        DType::I16 => (bits as i16).into(),
        DType::I8 => (bits as i8).into(),
        DType::U64 => bits.into(),
        DType::U32 => (bits as u32).into(),
        DType::U16 => (bits as u16).into(),
        DType::U8 => (bits as u8).into(),
        _ => panic!("numeric predicate reference requires a numeric dtype"),
    }
}

/// Compare decoded host values, including unordered IEEE comparisons.
pub fn compare_reference(dtype: DType, lhs: u64, rhs: u64, op: Comparison) -> bool {
    match (host_value(dtype, lhs), host_value(dtype, rhs)) {
        (Scalar::Float(lhs), Scalar::Float(rhs)) => op.apply(lhs, rhs),
        (Scalar::Int(lhs), Scalar::Int(rhs)) => op.apply(lhs, rhs),
        (Scalar::UInt(lhs), Scalar::UInt(rhs)) => op.apply(lhs, rhs),
        _ => unreachable!(),
    }
}

/// Classify a decoded host float without using the device's predicate encoding.
pub fn classify_reference(dtype: DType, bits: u64, op: Classification) -> bool {
    let Scalar::Float(value) = host_value(dtype, bits) else {
        panic!("classification requires a float")
    };
    match op {
        Classification::Nan => value.is_nan(),
        Classification::Infinite => value.is_infinite(),
        Classification::Finite => value.is_finite(),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub include_flex32: bool,
    /// Exclude NaNs and infinities, retaining zeros and subnormals.
    pub finite_only: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            include_flex32: true,
            finite_only: false,
        }
    }
}

#[derive(Default, Debug)]
pub struct Summary {
    pub numeric_dtypes: usize,
    pub comparison_cases: usize,
    pub scalar_cases: usize,
    pub classification_cases: usize,
    pub boolean_cases: usize,
    pub source_checks: usize,
    /// Distinct half storage patterns tested, summed across F16 and BF16.
    pub half_patterns: usize,
    pub values: usize,
}
impl Summary {
    pub fn cases(&self) -> usize {
        self.comparison_cases + self.scalar_cases + self.classification_cases + self.boolean_cases
    }
}

fn bytes(dtype: DType, values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()[..dtype.size()].to_vec()).collect()
}

fn edges(dtype: DType, finite: bool) -> Vec<u64> {
    let mut values = match dtype {
        DType::F64 => vec![
            0,
            1 << 63,
            1,
            (1 << 63) | 1,
            0x000f_ffff_ffff_ffff,
            0x800f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x8010_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x3fef_ffff_ffff_ffff,
            0x3ff0_0000_0000_0001,
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x7ff0_0000_0000_0001,
            0xfff8_0000_0000_1234,
        ],
        DType::F32 | DType::Flex32 => vec![
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x3f7f_ffff,
            0x3f80_0001,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0x7f80_0001,
            0xffc0_1234,
        ],
        DType::F16 => vec![
            0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x3bff, 0x3c01, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7c01, 0xfe34,
        ],
        DType::BF16 => vec![
            0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x3f7f, 0x3f81, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0x7f81, 0xffd3,
        ],
        _ => {
            let width = dtype.size() * 8;
            let sign = 1u64 << (width - 1);
            let max = u64::MAX >> (64 - width);
            let mut v = vec![0, 1, 2, max, max - 1, sign, sign + 1, sign - 1, sign - 2, 127, 128, 255];
            if width == 64 {
                v.extend_from_slice(&[
                    (1 << 53) - 1,
                    1 << 53,
                    (1 << 53) + 1,
                    (1 << 53) + 2,
                    (-(1i64 << 53) - 1) as u64,
                    (-(1i64 << 53)) as u64,
                    0xfedc_ba98_7654_3210,
                    0x0123_4567_89ab_cdef,
                ]);
            }
            v
        }
    };
    if finite && dtype.is_float() {
        values.retain(|&v| classify_reference(dtype, v, Classification::Finite));
    }
    values
}

fn payload(dtype: DType, count: usize, finite: bool, salt: usize) -> Vec<u64> {
    let edge = edges(dtype, finite);
    (0..count)
        .map(|i| {
            let random = ((i + salt) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left((i % 64) as u32);
            if i % 3 != 2 || (finite && dtype.is_float() && !classify_reference(dtype, random, Classification::Finite)) {
                edge[(i + salt) % edge.len()]
            } else {
                random
            }
        })
        .collect()
}

#[derive(Clone)]
enum NumericTensor {
    Float(FloatTensor<Dispatch>),
    Int(IntTensor<Dispatch>),
}
impl NumericTensor {
    fn new(device: &Device, dtype: DType, shape: &[usize], values: &[u64]) -> Self {
        let data = TensorData::from_bytes_vec(bytes(dtype, values), shape.to_vec(), dtype);
        if dtype.is_float() {
            Self::Float(Dispatch::float_from_data(data, device.as_dispatch()))
        } else {
            Self::Int(Dispatch::int_from_data(data, device.as_dispatch()))
        }
    }
    fn slice(self, slices: &[Slice]) -> Self {
        match self {
            Self::Float(t) => Self::Float(Dispatch::float_slice(t, slices)),
            Self::Int(t) => Self::Int(Dispatch::int_slice(t, slices)),
        }
    }
    fn flip(self, axes: &[usize]) -> Self {
        match self {
            Self::Float(t) => Self::Float(Dispatch::float_flip(t, axes)),
            Self::Int(t) => Self::Int(Dispatch::int_flip(t, axes)),
        }
    }
    fn permute(self, axes: &[usize]) -> Self {
        match self {
            Self::Float(t) => Self::Float(Dispatch::float_permute(t, axes)),
            Self::Int(t) => Self::Int(Dispatch::int_permute(t, axes)),
        }
    }
    fn compare(&self, rhs: &Self, op: Comparison, out: BoolDType) -> BoolTensor<Dispatch> {
        match (self, rhs) {
            (Self::Float(a), Self::Float(b)) => match op {
                Comparison::Equal => Dispatch::float_equal(a.clone(), b.clone(), out),
                Comparison::NotEqual => Dispatch::float_not_equal(a.clone(), b.clone(), out),
                Comparison::Greater => Dispatch::float_greater(a.clone(), b.clone(), out),
                Comparison::GreaterEqual => Dispatch::float_greater_equal(a.clone(), b.clone(), out),
                Comparison::Less => Dispatch::float_lower(a.clone(), b.clone(), out),
                Comparison::LessEqual => Dispatch::float_lower_equal(a.clone(), b.clone(), out),
            },
            (Self::Int(a), Self::Int(b)) => match op {
                Comparison::Equal => Dispatch::int_equal(a.clone(), b.clone(), out),
                Comparison::NotEqual => Dispatch::int_not_equal(a.clone(), b.clone(), out),
                Comparison::Greater => Dispatch::int_greater(a.clone(), b.clone(), out),
                Comparison::GreaterEqual => Dispatch::int_greater_equal(a.clone(), b.clone(), out),
                Comparison::Less => Dispatch::int_lower(a.clone(), b.clone(), out),
                Comparison::LessEqual => Dispatch::int_lower_equal(a.clone(), b.clone(), out),
            },
            _ => unreachable!(),
        }
    }
    fn scalar(&self, rhs: Scalar, op: Comparison, out: BoolDType) -> BoolTensor<Dispatch> {
        match self {
            Self::Float(a) => match op {
                Comparison::Equal => Dispatch::float_equal_elem(a.clone(), rhs, out),
                Comparison::NotEqual => Dispatch::float_not_equal_elem(a.clone(), rhs, out),
                Comparison::Greater => Dispatch::float_greater_elem(a.clone(), rhs, out),
                Comparison::GreaterEqual => Dispatch::float_greater_equal_elem(a.clone(), rhs, out),
                Comparison::Less => Dispatch::float_lower_elem(a.clone(), rhs, out),
                Comparison::LessEqual => Dispatch::float_lower_equal_elem(a.clone(), rhs, out),
            },
            Self::Int(a) => match op {
                Comparison::Equal => Dispatch::int_equal_elem(a.clone(), rhs, out),
                Comparison::NotEqual => Dispatch::int_not_equal_elem(a.clone(), rhs, out),
                Comparison::Greater => Dispatch::int_greater_elem(a.clone(), rhs, out),
                Comparison::GreaterEqual => Dispatch::int_greater_equal_elem(a.clone(), rhs, out),
                Comparison::Less => Dispatch::int_lower_elem(a.clone(), rhs, out),
                Comparison::LessEqual => Dispatch::int_lower_equal_elem(a.clone(), rhs, out),
            },
        }
    }
    fn classify(&self, op: Classification, out: BoolDType) -> BoolTensor<Dispatch> {
        let Self::Float(t) = self else { unreachable!() };
        match op {
            Classification::Nan => Dispatch::float_is_nan(t.clone(), out),
            Classification::Infinite => Dispatch::float_is_inf(t.clone(), out),
            // This is the public is_finite definition, with explicit storage.
            Classification::Finite => Dispatch::bool_and(
                Dispatch::bool_not(Dispatch::float_is_nan(t.clone(), out)),
                Dispatch::bool_not(Dispatch::float_is_inf(t.clone(), out)),
            ),
        }
    }
    fn preserve(self, dtype: DType, shape: &[usize], values: &[u64], summary: &mut Summary) -> Result<(), String> {
        let (actual_dtype, actual_shape) = match &self {
            Self::Float(t) => (t.dtype(), t.shape()),
            Self::Int(t) => (t.dtype(), t.shape()),
        };
        if actual_dtype != dtype || actual_shape.as_slice() != shape {
            return Err(format!("source metadata {dtype:?}"));
        }
        let data = match self {
            Self::Float(t) => read_sync(Dispatch::float_into_data(t)),
            Self::Int(t) => read_sync(Dispatch::int_into_data(t)),
        }
        .map_err(|e| format!("source read {dtype:?}: {e:?}"))?;
        if data.dtype() != dtype || data.as_bytes() != bytes(dtype, values) {
            return Err(format!("source modified {dtype:?}"));
        }
        summary.source_checks += 1;
        Ok(())
    }
}

fn check_bytes(tensor: BoolTensor<Dispatch>, out: BoolDType, shape: &[usize], expected: &[bool]) -> Result<(), String> {
    let dtype = DType::from(out);
    if tensor.dtype() != dtype || tensor.shape().as_slice() != shape {
        return Err(format!(
            "output metadata: {:?} {:?}, expected {dtype:?} {shape:?}",
            tensor.dtype(),
            tensor.shape()
        ));
    }
    let data = read_sync(Dispatch::bool_into_data(tensor)).map_err(|e| format!("predicate read: {e:?}"))?;
    let raw = bytes(dtype, &expected.iter().map(|&v| u64::from(v)).collect::<Vec<_>>());
    if data.dtype() != dtype || data.as_bytes().len() != raw.len() {
        return Err(format!("output storage {dtype:?}"));
    }
    if let Some(index) = data.as_bytes().iter().zip(&raw).position(|(a, b)| a != b) {
        return Err(format!(
            "output {dtype:?} element={} byte={index}: {} != {}",
            index / dtype.size(),
            data.as_bytes()[index],
            raw[index]
        ));
    }
    Ok(())
}

fn check(tensor: BoolTensor<Dispatch>, out: BoolDType, shape: &[usize], expected: &[bool], summary: &mut Summary) -> Result<(), String> {
    check_bytes(tensor, out, shape, expected)?;
    summary.values += expected.len();
    Ok(())
}

struct Pair {
    lhs: NumericTensor,
    rhs: NumericTensor,
    left: Vec<u64>,
    right: Vec<u64>,
    shape: Vec<usize>,
}
impl Pair {
    fn check(&self, dtype: DType, scalars: &[u64], summary: &mut Summary) -> Result<(), String> {
        for out in BOOL_DTYPES {
            for op in Comparison::ALL {
                let expected: Vec<_> = self.left.iter().zip(&self.right).map(|(&a, &b)| compare_reference(dtype, a, b, op)).collect();
                check(self.lhs.compare(&self.rhs, op, out), out, &self.shape, &expected, summary).map_err(|e| format!("binary {op:?} {out:?}: {e}"))?;
                summary.comparison_cases += 1;
                for &scalar in scalars {
                    let expected: Vec<_> = self.left.iter().map(|&a| compare_reference(dtype, a, scalar, op)).collect();
                    check(self.lhs.scalar(host_value(dtype, scalar), op, out), out, &self.shape, &expected, summary)
                        .map_err(|e| format!("scalar {op:?} {out:?} bits={scalar:#x}: {e}"))?;
                    summary.scalar_cases += 1;
                }
            }
            if dtype.is_float() {
                for op in Classification::ALL {
                    let expected: Vec<_> = self.left.iter().map(|&v| classify_reference(dtype, v, op)).collect();
                    check(self.lhs.classify(op, out), out, &self.shape, &expected, summary).map_err(|e| format!("classification {op:?} {out:?}: {e}"))?;
                    summary.classification_cases += 1;
                }
            }
        }
        Ok(())
    }
}

fn numeric(device: &Device, dtype: DType, options: &Options, summary: &mut Summary) -> Result<(), String> {
    let edge = edges(dtype, options.finite_only);
    // Cartesian edges ensure adjacent 64-bit integers are compared directly.
    let left: Vec<_> = edge.iter().flat_map(|&v| core::iter::repeat_n(v, edge.len())).collect();
    let right = edge.repeat(edge.len());
    let shape = vec![left.len()];
    let pair = Pair {
        lhs: NumericTensor::new(device, dtype, &shape, &left),
        rhs: NumericTensor::new(device, dtype, &shape, &right),
        left,
        right,
        shape,
    };
    // Every edge also serves as an exact, explicitly typed scalar.
    pair.check(dtype, &edge, summary).map_err(|e| format!("edge pairs: {e}"))?;
    pair.lhs.preserve(dtype, &pair.shape, &pair.left, summary)?;
    pair.rhs.preserve(dtype, &pair.shape, &pair.right, summary)?;

    for length in [0, 1, 127, 128, 129, 257] {
        let a = payload(dtype, length + 2, options.finite_only, 0);
        let b = payload(dtype, length + 3, options.finite_only, 7);
        let source = NumericTensor::new(device, dtype, &[length + 2], &a);
        let rhs = NumericTensor::new(device, dtype, &[length + 3], &b);
        let pair = Pair {
            lhs: source.clone().slice(&[Slice::new(1, Some((length + 1) as isize), 1)]).flip(&[0]),
            rhs: rhs.clone().slice(&[Slice::new(2, Some((length + 2) as isize), 1)]),
            left: a[1..length + 1].iter().rev().copied().collect(),
            right: b[2..length + 2].to_vec(),
            shape: vec![length],
        };
        pair.check(dtype, &[edge[0], edge[3]], summary)
            .map_err(|e| format!("tail length={length}: {e}"))?;
        source.preserve(dtype, &[length + 2], &a, summary)?;
        rhs.preserve(dtype, &[length + 3], &b, summary)?;
    }

    for length in [0, 129] {
        let a = payload(dtype, 3 * (length + 2), options.finite_only, 3);
        let b = [edge[1], edge[3], edge[edge.len() - 1]];
        let source = NumericTensor::new(device, dtype, &[3, length + 2], &a);
        let rhs = NumericTensor::new(device, dtype, &[1, 3], &b);
        let pair = Pair {
            lhs: source
                .clone()
                .slice(&[Slice::full(), Slice::new(1, Some((length + 1) as isize), 1)])
                .permute(&[1, 0])
                .flip(&[0, 1]),
            rhs: rhs.clone().flip(&[1]),
            left: (0..length)
                .flat_map(|r| (0..3).map(move |c| (2 - c) * (length + 2) + length - r))
                .map(|i| a[i])
                .collect(),
            right: b.into_iter().rev().collect::<Vec<_>>().repeat(length),
            shape: vec![length, 3],
        };
        pair.check(dtype, &[edge[0], edge[3]], summary)
            .map_err(|e| format!("transposed offset broadcast length={length}: {e}"))?;
        source.preserve(dtype, &[3, length + 2], &a, summary)?;
        rhs.preserve(dtype, &[1, 3], &b, summary)?;
    }
    Ok(())
}

fn half_exhaustive(device: &Device, dtype: DType, options: &Options, summary: &mut Summary) -> Result<(), String> {
    let raw: Vec<_> = (0..=u16::MAX)
        .map(u64::from)
        .filter(|&v| !options.finite_only || classify_reference(dtype, v, Classification::Finite))
        .collect();
    let left: Vec<_> = raw.iter().rev().copied().collect();
    let right: Vec<_> = left
        .iter()
        .map(|&v| {
            let next = u64::from((v as u16).wrapping_add(1));
            if options.finite_only && !classify_reference(dtype, next, Classification::Finite) {
                v
            } else {
                next
            }
        })
        .collect();
    let shape = vec![raw.len()];
    let source = NumericTensor::new(device, dtype, &shape, &raw);
    let pair = Pair {
        lhs: source.clone().flip(&[0]),
        rhs: NumericTensor::new(device, dtype, &shape, &right),
        left,
        right,
        shape,
    };
    pair.check(dtype, &[0, 0x8000], summary)?;
    source.preserve(dtype, &pair.shape, &raw, summary)?;
    pair.rhs.preserve(dtype, &pair.shape, &pair.right, summary)?;
    summary.half_patterns += raw.len();
    Ok(())
}

fn bool_source(device: &Device, dtype: DType, shape: &[usize], values: &[u64]) -> BoolTensor<Dispatch> {
    Dispatch::bool_from_data(TensorData::from_bytes_vec(bytes(dtype, values), shape.to_vec(), dtype), device.as_dispatch())
}

struct BoolPair {
    lhs: BoolTensor<Dispatch>,
    rhs: BoolTensor<Dispatch>,
    left: Vec<bool>,
    right: Vec<bool>,
    shape: Vec<usize>,
}
impl BoolPair {
    fn check(&self, out: BoolDType, summary: &mut Summary) -> Result<(), String> {
        for operation in 0..6 {
            let output = match operation {
                0 => Dispatch::bool_equal(self.lhs.clone(), self.rhs.clone()),
                1 => Dispatch::bool_not_equal(self.lhs.clone(), self.rhs.clone()),
                2 => Dispatch::bool_and(self.lhs.clone(), self.rhs.clone()),
                3 => Dispatch::bool_or(self.lhs.clone(), self.rhs.clone()),
                4 => Dispatch::bool_xor(self.lhs.clone(), self.rhs.clone()),
                _ => Dispatch::bool_not(self.lhs.clone()),
            };
            let expected: Vec<_> = self
                .left
                .iter()
                .zip(&self.right)
                .map(|(&a, &b)| match operation {
                    0 => a == b,
                    1 => a != b,
                    2 => a && b,
                    3 => a || b,
                    4 => a ^ b,
                    _ => !a,
                })
                .collect();
            check(output, out, &self.shape, &expected, summary).map_err(|e| format!("boolean operation={operation}: {e}"))?;
            summary.boolean_cases += 1;
        }
        for scalar in [false, true] {
            for negate in [false, true] {
                let output = if negate {
                    Dispatch::bool_not_equal_elem(self.lhs.clone(), scalar.into())
                } else {
                    Dispatch::bool_equal_elem(self.lhs.clone(), scalar.into())
                };
                let expected: Vec<_> = self.left.iter().map(|&a| if negate { a != scalar } else { a == scalar }).collect();
                check(output, out, &self.shape, &expected, summary)?;
                summary.boolean_cases += 1;
            }
        }
        Ok(())
    }
}

fn preserve_bool(tensor: BoolTensor<Dispatch>, out: BoolDType, shape: &[usize], values: &[u64], summary: &mut Summary) -> Result<(), String> {
    check_bytes(tensor, out, shape, &values.iter().map(|&v| v != 0).collect::<Vec<_>>())?;
    summary.source_checks += 1;
    Ok(())
}

fn boolean(device: &Device, out: BoolDType, summary: &mut Summary) -> Result<(), String> {
    let dtype = DType::from(out);
    for length in [0, 1, 127, 128, 129, 257] {
        let a: Vec<_> = (0..length + 2).map(|i| u64::from(i % 4 < 2)).collect();
        let b: Vec<_> = (0..length + 3).map(|i| (i % 2) as u64).collect();
        let source = bool_source(device, dtype, &[length + 2], &a);
        let rhs = bool_source(device, dtype, &[length + 3], &b);
        let pair = BoolPair {
            lhs: Dispatch::bool_flip(Dispatch::bool_slice(source.clone(), &[Slice::new(1, Some((length + 1) as isize), 1)]), &[0]),
            rhs: Dispatch::bool_slice(rhs.clone(), &[Slice::new(2, Some((length + 2) as isize), 1)]),
            left: a[1..length + 1].iter().rev().map(|&v| v != 0).collect(),
            right: b[2..length + 2].iter().map(|&v| v != 0).collect(),
            shape: vec![length],
        };
        pair.check(out, summary).map_err(|e| format!("boolean {out:?} tail length={length}: {e}"))?;
        preserve_bool(source, out, &[length + 2], &a, summary)?;
        preserve_bool(rhs, out, &[length + 3], &b, summary)?;
    }
    for length in [0, 129] {
        let a: Vec<_> = (0..3 * (length + 2)).map(|i| u64::from(i % 4 < 2)).collect();
        let b = [0, 1, 1];
        let source = bool_source(device, dtype, &[3, length + 2], &a);
        let rhs = bool_source(device, dtype, &[1, 3], &b);
        let view = Dispatch::bool_flip(
            Dispatch::bool_permute(
                Dispatch::bool_slice(source.clone(), &[Slice::full(), Slice::new(1, Some((length + 1) as isize), 1)]),
                &[1, 0],
            ),
            &[0, 1],
        );
        let pair = BoolPair {
            lhs: view,
            rhs: Dispatch::bool_flip(rhs.clone(), &[1]),
            left: (0..length)
                .flat_map(|r| (0..3).map(move |c| (2 - c) * (length + 2) + length - r))
                .map(|i| a[i] != 0)
                .collect(),
            right: b.into_iter().rev().map(|v| v != 0).collect::<Vec<_>>().repeat(length),
            shape: vec![length, 3],
        };
        pair.check(out, summary)?;
        preserve_bool(source, out, &[3, length + 2], &a, summary)?;
        preserve_bool(rhs, out, &[1, 3], &b, summary)?;
    }
    Ok(())
}

/// Run the same strict byte checks on native D3OS and Linux Dispatch devices.
pub fn run(device: &Device, options: &Options) -> Result<Summary, String> {
    let mut summary = Summary::default();
    for dtype in NUMERIC_DTYPES {
        if dtype == DType::Flex32 && !options.include_flex32 {
            continue;
        }
        numeric(device, dtype, options, &mut summary).map_err(|e| format!("{dtype:?}: {e}"))?;
        summary.numeric_dtypes += 1;
    }
    for dtype in [DType::F16, DType::BF16] {
        half_exhaustive(device, dtype, options, &mut summary).map_err(|e| format!("exhaustive {dtype:?}: {e}"))?;
    }
    for out in BOOL_DTYPES {
        boolean(device, out, &mut summary)?;
    }
    Ok(summary)
}
