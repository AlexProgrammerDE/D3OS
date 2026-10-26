//! Exact IEEE step reference and bounded forward floating reduction checks.
//!
//! Native serial arithmetic and stored global stages define this policy.
//! Linux results are observations of its selected graph, not an oracle.
//! Physical padding, allocation and residency require the separate raw harness.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{
        DType, Dispatch, Shape, TensorData, TensorMetadata,
        ops::{BoolTensorOps, FloatTensorOps, IntTensorOps},
        tensor::FloatTensor,
    },
    tensor::{Device, Slice, Tensor},
};
use core::cmp::Ordering;

pub const CORPUS_VERSION: u32 = 1;
pub const DTYPES: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Sum,
    Mean,
    Product,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Axis(usize),
    Global,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arithmetic {
    Add,
    Multiply,
    Divide,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub dtype: Option<DType>,
    pub skip_rank_zero: bool,
    pub include_empty: bool,
    pub native_error_extensions: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            dtype: None,
            skip_rank_zero: false,
            include_empty: true,
            native_error_extensions: false,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Values {
    pub shape: Vec<usize>,
    pub values: Vec<u64>,
}
impl Values {
    pub fn new(shape: &[usize], values: Vec<u64>) -> Self {
        assert_eq!(elements(shape), Some(values.len()));
        Self { shape: shape.to_vec(), values }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceError {
    Dtype,
    Shape,
    Axis,
    Count,
    LegacyOrderUnspecified,
}
fn fields(dtype: DType) -> (u32, u32, i32) {
    match dtype {
        DType::F64 => (11, 52, 1023),
        DType::F32 | DType::Flex32 => (8, 23, 127),
        DType::F16 => (5, 10, 15),
        DType::BF16 => (8, 7, 127),
        _ => panic!("floating dtype required"),
    }
}
pub fn mask(dtype: DType) -> u64 {
    u64::MAX >> (64 - dtype.size() * 8)
}
pub fn pack(dtype: DType, values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes().into_iter().take(dtype.size())).collect()
}
pub fn unpack(dtype: DType, bytes: &[u8]) -> Vec<u64> {
    assert!(bytes.len().is_multiple_of(dtype.size()));
    bytes
        .chunks_exact(dtype.size())
        .map(|b| {
            let mut a = [0; 8];
            a[..b.len()].copy_from_slice(b);
            u64::from_le_bytes(a)
        })
        .collect()
}
pub fn canonical_nan(dtype: DType) -> u64 {
    let (e, f, _) = fields(dtype);
    (((1u64 << e) - 1) << f) | (1u64 << (f - 1))
}
fn infinity(dtype: DType, negative: bool) -> u64 {
    let (e, f, _) = fields(dtype);
    (((1u64 << e) - 1) << f) | ((negative as u64) << (e + f))
}
fn zero(dtype: DType, negative: bool) -> u64 {
    (negative as u64) << (dtype.size() * 8 - 1)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Number {
    Nan,
    Infinity(bool),
    Finite { negative: bool, significand: u64, exponent: i32 },
}
fn decode(dtype: DType, bits: u64) -> Number {
    let (e, f, bias) = fields(dtype);
    let negative = bits & zero(dtype, true) != 0;
    let fraction = bits & ((1u64 << f) - 1);
    let exponent = (bits >> f) & ((1u64 << e) - 1);
    if exponent == (1u64 << e) - 1 {
        return if fraction == 0 { Number::Infinity(negative) } else { Number::Nan };
    }
    Number::Finite {
        negative,
        significand: fraction | if exponent == 0 { 0 } else { 1u64 << f },
        exponent: (if exponent == 0 { 1 - bias } else { exponent as i32 - bias }) - f as i32,
    }
}
/// Small arbitrary precision unsigned magnitude. Exact addition can span the
/// full F64 exponent range; a u128 sticky approximation would lose cancellation.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Natural(Vec<u64>);
impl Natural {
    fn from_u128(n: u128) -> Self {
        let mut a = Self(vec![n as u64, (n >> 64) as u64]);
        a.trim();
        a
    }
    fn trim(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }
    fn bits(&self) -> i32 {
        self.0.last().map_or(0, |v| (self.0.len() as i32 - 1) * 64 + 64 - v.leading_zeros() as i32)
    }
    fn cmp(&self, b: &Self) -> Ordering {
        self.0.len().cmp(&b.0.len()).then_with(|| self.0.iter().rev().cmp(b.0.iter().rev()))
    }
    fn shl(&self, n: usize) -> Self {
        if self.0.is_empty() {
            return self.clone();
        }
        let words = n / 64;
        let shift = n % 64;
        let mut a = vec![0; self.0.len() + words + usize::from(shift != 0)];
        for (i, &v) in self.0.iter().enumerate() {
            a[i + words] |= v << shift;
            if shift != 0 {
                a[i + words + 1] |= v >> (64 - shift);
            }
        }
        let mut a = Self(a);
        a.trim();
        a
    }
    fn add(&self, b: &Self) -> Self {
        let mut a = Vec::new();
        let mut carry = 0u128;
        for i in 0..self.0.len().max(b.0.len()) {
            let v = u128::from(*self.0.get(i).unwrap_or(&0)) + u128::from(*b.0.get(i).unwrap_or(&0)) + carry;
            a.push(v as u64);
            carry = v >> 64;
        }
        if carry != 0 {
            a.push(carry as u64);
        }
        Self(a)
    }
    fn sub(&self, b: &Self) -> Self {
        assert!(self.cmp(b) != Ordering::Less);
        let mut a = self.clone();
        let mut borrow = false;
        for (i, v) in a.0.iter_mut().enumerate() {
            let (x, b1) = v.overflowing_sub(*b.0.get(i).unwrap_or(&0));
            let (y, b2) = x.overflowing_sub(borrow as u64);
            *v = y;
            borrow = b1 || b2;
        }
        assert!(!borrow);
        a.trim();
        a
    }
    fn mul(&self, b: u64) -> Self {
        let mut a = Vec::new();
        let mut carry = 0u128;
        for &v in &self.0 {
            let n = u128::from(v) * u128::from(b) + carry;
            a.push(n as u64);
            carry = n >> 64;
        }
        if carry != 0 {
            a.push(carry as u64);
        }
        let mut a = Self(a);
        a.trim();
        a
    }
}
/// Round n/d * 2^exponent directly to storage, using exact quotient/remainder.
fn round_ratio(dtype: DType, negative: bool, n: Natural, d: Natural, exponent: i32) -> u64 {
    if n.bits() == 0 {
        return zero(dtype, negative);
    }
    let (eb, fb, bias) = fields(dtype);
    let mut k = n.bits() - d.bits();
    let cmp = if k >= 0 { n.cmp(&d.shl(k as usize)) } else { n.shl((-k) as usize).cmp(&d) };
    if cmp == Ordering::Less {
        k -= 1;
    }
    k += exponent;
    if k > bias {
        return infinity(dtype, negative);
    }
    let quantum = k.max(1 - bias) - fb as i32;
    let shift = exponent - quantum;
    let (n, d) = if shift >= 0 {
        (n.shl(shift as usize), d)
    } else {
        (n, d.shl((-shift) as usize))
    };
    // The scaled quotient has at most fb+1 bits, even for gradual underflow.
    let mut q = 0u64;
    for bit in (0..=fb).rev() {
        let trial = q | (1u64 << bit);
        if d.mul(trial).cmp(&n) != Ordering::Greater {
            q = trial;
        }
    }
    let remainder = n.sub(&d.mul(q));
    let half = remainder.shl(1).cmp(&d);
    q += u64::from(half == Ordering::Greater || (half == Ordering::Equal && q & 1 != 0));
    let mut k = k.max(1 - bias);
    if q == 1u64 << (fb + 1) {
        q >>= 1;
        k += 1;
    }
    if k > bias {
        return infinity(dtype, negative);
    }
    let (stored_exp, fraction) = if q < 1u64 << fb { (0, q) } else { ((k + bias) as u64, q - (1u64 << fb)) };
    ((negative as u64) << (eb + fb)) | (stored_exp << fb) | fraction
}
pub fn integer_bits(dtype: DType, count: u64) -> u64 {
    round_ratio(dtype, false, Natural::from_u128(u128::from(count)), Natural::from_u128(1), 0)
}
pub fn cast_bits(src: DType, dst: DType, bits: u64) -> u64 {
    match decode(src, bits) {
        Number::Nan => canonical_nan(dst),
        Number::Infinity(s) => infinity(dst, s),
        Number::Finite {
            negative,
            significand,
            exponent,
        } => round_ratio(dst, negative, Natural::from_u128(u128::from(significand)), Natural::from_u128(1), exponent),
    }
}
/// One IEEE round-to-nearest-even operation. No host floating arithmetic.
pub fn ieee_binary(dtype: DType, a: u64, b: u64, operation: Arithmetic) -> u64 {
    use Number::*;
    let (a, b) = (decode(dtype, a), decode(dtype, b));
    if a == Nan || b == Nan {
        return canonical_nan(dtype);
    }
    let sign = |n: Number| match n {
        Infinity(s) | Finite { negative: s, .. } => s,
        Nan => false,
    };
    let is_zero = |n: Number| matches!(n, Finite { significand: 0, .. });
    let negative = sign(a) ^ sign(b);
    match operation {
        Arithmetic::Add => {
            match (a, b) {
                (Infinity(sa), Infinity(sb)) if sa != sb => return canonical_nan(dtype),
                (Infinity(sa), _) | (_, Infinity(sa)) => return infinity(dtype, sa),
                _ => {}
            }
            let (
                Finite {
                    negative: sa,
                    significand: ma,
                    exponent: ea,
                },
                Finite {
                    negative: sb,
                    significand: mb,
                    exponent: eb,
                },
            ) = (a, b)
            else {
                unreachable!()
            };
            let e = ea.min(eb);
            let a = Natural::from_u128(u128::from(ma)).shl((ea - e) as usize);
            let b = Natural::from_u128(u128::from(mb)).shl((eb - e) as usize);
            let (negative, n) = if sa == sb {
                (sa, a.add(&b))
            } else {
                match a.cmp(&b) {
                    Ordering::Less => (sb, b.sub(&a)),
                    Ordering::Greater => (sa, a.sub(&b)),
                    Ordering::Equal => (false, Natural::from_u128(0)),
                }
            };
            round_ratio(dtype, negative, n, Natural::from_u128(1), e)
        }
        Arithmetic::Multiply => {
            if (matches!(a, Infinity(_)) && is_zero(b)) || (matches!(b, Infinity(_)) && is_zero(a)) {
                return canonical_nan(dtype);
            }
            if matches!(a, Infinity(_)) || matches!(b, Infinity(_)) {
                return infinity(dtype, negative);
            }
            let (
                Finite {
                    significand: ma, exponent: ea, ..
                },
                Finite {
                    significand: mb, exponent: eb, ..
                },
            ) = (a, b)
            else {
                unreachable!()
            };
            round_ratio(
                dtype,
                negative,
                Natural::from_u128(u128::from(ma) * u128::from(mb)),
                Natural::from_u128(1),
                ea + eb,
            )
        }
        Arithmetic::Divide => {
            if (matches!(a, Infinity(_)) && matches!(b, Infinity(_))) || (is_zero(a) && is_zero(b)) {
                return canonical_nan(dtype);
            }
            if matches!(a, Infinity(_)) || is_zero(b) {
                return infinity(dtype, negative);
            }
            if matches!(b, Infinity(_)) {
                return zero(dtype, negative);
            }
            let (
                Finite {
                    significand: ma, exponent: ea, ..
                },
                Finite {
                    significand: mb, exponent: eb, ..
                },
            ) = (a, b)
            else {
                unreachable!()
            };
            round_ratio(dtype, negative, Natural::from_u128(u128::from(ma)), Natural::from_u128(u128::from(mb)), ea - eb)
        }
    }
}
fn elements(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        Some(0)
    } else {
        shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
    }
}
fn coordinates(shape: &[usize], mut i: usize) -> Vec<usize> {
    let mut c = vec![0; shape.len()];
    for a in (0..shape.len()).rev() {
        c[a] = i % shape[a];
        i /= shape[a];
    }
    c
}
fn offset(shape: &[usize], c: &[usize]) -> usize {
    shape.iter().zip(c).fold(0, |n, (&d, &c)| n * d + c)
}
fn accumulator(dtype: DType) -> DType {
    if dtype == DType::F64 { DType::F64 } else { DType::F32 }
}
/// Canonical public scalar construction converts F64 through F32 for half.
/// Counts enter directly as integers first; raw axis Mean never uses this cast.
pub fn scalar_count(dtype: DType, count: u64) -> u64 {
    if dtype == DType::F64 {
        integer_bits(dtype, count)
    } else {
        cast_bits(DType::F32, dtype, integer_bits(DType::F32, count))
    }
}
pub fn axis_reference(dtype: DType, values: &[u64], operation: Operation) -> u64 {
    let acc = accumulator(dtype);
    let mut n = integer_bits(acc, u64::from(operation == Operation::Product));
    for &v in values {
        n = ieee_binary(
            acc,
            n,
            cast_bits(dtype, acc, v),
            if operation == Operation::Product {
                Arithmetic::Multiply
            } else {
                Arithmetic::Add
            },
        );
    }
    if operation == Operation::Mean {
        n = ieee_binary(acc, n, integer_bits(acc, values.len() as u64), Arithmetic::Divide);
    }
    cast_bits(acc, dtype, n)
}
/// Exact Native schedule. F32 callers must supply an all-order witness; the
/// cooperative legacy reduction has no specified tree and is not serial.
pub fn reference(dtype: DType, input: &Values, operation: Operation, scope: Scope) -> Result<Values, ReferenceError> {
    if !DTYPES.contains(&dtype) {
        return Err(ReferenceError::Dtype);
    }
    if input.shape.len() > 8 || elements(&input.shape) != Some(input.values.len()) {
        return Err(ReferenceError::Shape);
    }
    if input.shape.iter().any(|&d| d > i32::MAX as usize) || input.values.len() > i32::MAX as usize {
        return Err(ReferenceError::Count);
    }
    if let Scope::Axis(axis) = scope {
        let Some(&length) = input.shape.get(axis) else {
            return Err(ReferenceError::Axis);
        };
        if dtype == DType::F32 && !legacy_witness(input, operation) {
            return Err(ReferenceError::LegacyOrderUnspecified);
        }
        let mut shape = input.shape.clone();
        shape[axis] = 1;
        let count = elements(&shape).ok_or(ReferenceError::Shape)?;
        if count > i32::MAX as usize {
            return Err(ReferenceError::Count);
        }
        let mut output = Vec::with_capacity(count);
        for i in 0..count {
            let mut c = coordinates(&shape, i);
            let row = (0..length)
                .map(|j| {
                    c[axis] = j;
                    input.values[offset(&input.shape, &c)]
                })
                .collect::<Vec<_>>();
            output.push(axis_reference(dtype, &row, operation));
        }
        return Ok(Values::new(&shape, output));
    }
    if input.shape.is_empty() {
        return Ok(Values::new(&[1], input.values.clone()));
    }
    if input.values.is_empty() {
        return Ok(Values::new(
            &[1],
            vec![if operation == Operation::Mean {
                canonical_nan(dtype)
            } else {
                integer_bits(dtype, u64::from(operation == Operation::Product))
            }],
        ));
    }
    if dtype == DType::F32 && !legacy_witness(input, operation) {
        return Err(ReferenceError::LegacyOrderUnspecified);
    }
    let mut axes = (0..input.shape.len()).collect::<Vec<_>>();
    if dtype == DType::F32 {
        axes.reverse();
    } else {
        axes.sort_by_key(|&a| input.shape[a]);
    }
    let mut result = input.clone();
    let stage = if dtype == DType::F32 && operation == Operation::Mean {
        Operation::Sum
    } else {
        operation
    };
    for axis in axes {
        result = reference(dtype, &result, stage, Scope::Axis(axis))?;
    }
    if dtype == DType::F32 && operation == Operation::Mean {
        // Legacy F32 first casts the count to F32. The new four use their
        // canonical storage count, including direct integer-to-F64 conversion.
        result.values[0] = ieee_binary(dtype, result.values[0], scalar_count(dtype, input.values.len() as u64), Arithmetic::Divide);
    }
    result.shape = vec![1];
    Ok(result)
}
/// Sufficient all-order proof for the finite legacy corpus. Sum intermediates
/// are exact nonnegative integers bounded by 2^24. Products use only +/-1 or
/// positive zero. This deliberately rejects cancellation/order counterexamples.
fn legacy_witness(input: &Values, operation: Operation) -> bool {
    if input.values.is_empty() {
        return true;
    }
    if operation == Operation::Product {
        return input.values.iter().all(|&b| b == 0 || b == 0x3f800000 || b == 0xbf800000);
    }
    let mut total = 0u64;
    for &b in &input.values {
        let Number::Finite {
            negative: false,
            significand,
            exponent,
        } = decode(DType::F32, b)
        else {
            return false;
        };
        let value = if exponent >= 0 {
            if exponent <= 24 && significand <= ((1u64 << 24) >> exponent) {
                Some(significand << exponent)
            } else {
                None
            }
        } else {
            let shift = (-exponent) as u32;
            if shift >= 64 {
                if significand == 0 { Some(0) } else { None }
            } else if significand & ((1u64 << shift) - 1) == 0 {
                Some(significand >> shift)
            } else {
                None
            }
        };
        let Some(value) = value else {
            return false;
        };
        let Some(sum) = total.checked_add(value) else {
            return false;
        };
        total = sum;
        if total > 1 << 24 {
            return false;
        }
    }
    true
}

#[derive(Clone, Debug)]
pub enum View {
    Permute(Vec<usize>),
    Flip(Vec<usize>),
    Slice { starts: Vec<usize>, steps: Vec<usize>, shape: Vec<usize> },
    Expand(Vec<usize>),
}
#[derive(Clone, Debug)]
pub struct Input {
    pub root: Values,
    pub views: Vec<View>,
}
impl Input {
    pub fn logical(&self) -> Values {
        self.views.iter().fold(self.root.clone(), |input, view| {
            let shape = match view {
                View::Permute(axes) => axes.iter().map(|&a| input.shape[a]).collect(),
                View::Flip(_) => input.shape.clone(),
                View::Slice { shape, .. } | View::Expand(shape) => shape.clone(),
            };
            let values = (0..elements(&shape).expect("bounded view shape"))
                .map(|i| {
                    let coords = coordinates(&shape, i);
                    let source = match view {
                        View::Permute(axes) => {
                            let mut source = vec![0; axes.len()];
                            for (&axis, &coord) in axes.iter().zip(&coords) {
                                source[axis] = coord;
                            }
                            source
                        }
                        View::Flip(axes) => coords
                            .iter()
                            .enumerate()
                            .map(|(a, &c)| if axes.contains(&a) { input.shape[a] - 1 - c } else { c })
                            .collect(),
                        View::Slice { starts, steps, .. } => coords.iter().enumerate().map(|(a, &c)| starts[a] + c * steps[a]).collect(),
                        View::Expand(_) => coords.iter().enumerate().map(|(a, &c)| if input.shape[a] == 1 { 0 } else { c }).collect(),
                    };
                    input.values[offset(&input.shape, &source)]
                })
                .collect();
            Values::new(&shape, values)
        })
    }
    fn build(&self, dtype: DType, device: &Device) -> (FloatTensor<Dispatch>, FloatTensor<Dispatch>) {
        let source = Dispatch::float_from_data(
            TensorData::from_bytes_vec(pack(dtype, &self.root.values), self.root.shape.clone(), dtype),
            device.as_dispatch(),
        );
        let mut input = source.clone();
        for view in &self.views {
            input = match view {
                View::Permute(axes) => Dispatch::float_permute(input, axes),
                View::Flip(axes) => Dispatch::float_flip(input, axes),
                View::Slice { starts, steps, shape } => {
                    let slices: Vec<_> = starts
                        .iter()
                        .zip(steps)
                        .zip(shape)
                        .map(|((&start, &step), &len)| {
                            Slice::new(
                                start as isize,
                                Some((start + (len.saturating_sub(1)) * step + usize::from(len > 0)) as isize),
                                step as isize,
                            )
                        })
                        .collect();
                    Dispatch::float_slice(input, &slices)
                }
                View::Expand(shape) => Dispatch::float_expand(input, Shape::from(shape.clone())),
            };
        }
        (source, input)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Reduce(Operation, Scope),
    Dims(Operation, Vec<usize>),
    Multiply(u64),
    SubtractSource,
    Square,
    Cast(DType),
    IsNan,
    Minimum(usize),
    CumulativeSum(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    NativeSerial,
    LegacyAllOrderExact,
    ExistingExactSelection,
    IntegerRegression,
    SymbolicOnly,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    OneHot65536,
    Ones65536,
    HalfCounts,
    F64Count,
    F32Counts,
    EmptyHuge,
    ScalarMidpoint,
}
#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub label: &'static str,
    pub dtype: DType,
    pub input: Input,
    pub steps: Vec<Step>,
    pub evidence: Evidence,
    pub model: Option<Model>,
    /// Chains queue all dependencies and download only the final logical value.
    pub chain: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected {
    pub dtype: DType,
    pub output: Values,
}
impl Case {
    pub fn expected(&self) -> Result<Expected, ReferenceError> {
        if let Some(model) = self.model {
            return Ok(Expected {
                dtype: self.dtype,
                output: model_expected(model, self.dtype),
            });
        }
        let original = self.input.logical();
        if matches!(self.dtype, DType::I32 | DType::I64) {
            let Step::Reduce(op, scope) = self.steps[0] else { unreachable!() };
            let operation = match op {
                Operation::Sum => super::integer_reduction::Operation::Sum,
                Operation::Mean => super::integer_reduction::Operation::Mean,
                Operation::Product => super::integer_reduction::Operation::Product,
            };
            let scope = match scope {
                Scope::Axis(a) => super::integer_reduction::Scope::Axis(a),
                Scope::Global => super::integer_reduction::Scope::Global,
            };
            let input = super::integer_reduction::Values::new(&original.shape, original.values);
            let out = super::integer_reduction::reference(self.dtype, &input, operation, scope)
                .map_err(|_| ReferenceError::Shape)?
                .output;
            return Ok(Expected {
                dtype: self.dtype,
                output: Values::new(&out.shape, out.values),
            });
        }
        let mut result = Expected {
            dtype: self.dtype,
            output: original.clone(),
        };
        for step in &self.steps {
            let dtype = result.dtype;
            let input = &result.output;
            result.output = match step {
                Step::Reduce(op, scope) => reference(dtype, input, *op, *scope)?,
                Step::Dims(op, axes) => {
                    let mut out = input.clone();
                    for &axis in axes {
                        out = reference(dtype, &out, *op, Scope::Axis(axis))?;
                    }
                    out
                }
                Step::Multiply(scalar) => Values::new(
                    &input.shape,
                    input.values.iter().map(|&b| ieee_binary(dtype, b, *scalar, Arithmetic::Multiply)).collect(),
                ),
                Step::Square => Values::new(
                    &input.shape,
                    input.values.iter().map(|&b| ieee_binary(dtype, b, b, Arithmetic::Multiply)).collect(),
                ),
                Step::SubtractSource => {
                    let shape = original.shape.clone();
                    let values = original
                        .values
                        .iter()
                        .enumerate()
                        .map(|(i, &a)| {
                            let c = coordinates(&shape, i);
                            let c = c.iter().enumerate().map(|(a, &c)| if input.shape[a] == 1 { 0 } else { c }).collect::<Vec<_>>();
                            let b = input.values[offset(&input.shape, &c)] ^ zero(dtype, true);
                            ieee_binary(dtype, a, b, Arithmetic::Add)
                        })
                        .collect();
                    Values::new(&shape, values)
                }
                Step::Cast(dst) => {
                    result.dtype = *dst;
                    Values::new(&input.shape, input.values.iter().map(|&b| cast_bits(dtype, *dst, b)).collect())
                }
                Step::IsNan => Values::new(
                    &input.shape,
                    input
                        .values
                        .iter()
                        .map(|&b| integer_bits(dtype, u64::from(decode(dtype, b) == Number::Nan)))
                        .collect(),
                ),
                Step::Minimum(axis) => {
                    let input = super::float_extrema::Values::new(&input.shape, input.values.clone());
                    let out = super::float_extrema::reference(
                        dtype,
                        DType::I32,
                        &input,
                        super::float_extrema::Operation::Min,
                        super::float_extrema::Scope::Axis(*axis),
                    )
                    .map_err(|_| ReferenceError::Axis)?
                    .output;
                    Values::new(&out.shape, out.values)
                }
                Step::CumulativeSum(axis) => {
                    if *axis >= input.shape.len() {
                        return Err(ReferenceError::Axis);
                    }
                    if dtype == DType::F32 && !legacy_witness(input, Operation::Sum) {
                        return Err(ReferenceError::LegacyOrderUnspecified);
                    }
                    let values = (0..input.values.len())
                        .map(|i| {
                            let mut c = coordinates(&input.shape, i);
                            let end = c[*axis];
                            let row = (0..=end)
                                .map(|j| {
                                    c[*axis] = j;
                                    input.values[offset(&input.shape, &c)]
                                })
                                .collect::<Vec<_>>();
                            axis_reference(dtype, &row, Operation::Sum)
                        })
                        .collect();
                    Values::new(&input.shape, values)
                }
            };
        }
        Ok(result)
    }
}
fn dense(shape: &[usize], values: Vec<u64>) -> Input {
    Input {
        root: Values::new(shape, values),
        views: vec![],
    }
}
fn numbers(dtype: DType, values: &[i64]) -> Vec<u64> {
    values.iter().map(|&v| integer_bits(dtype, v.unsigned_abs()) | zero(dtype, v < 0)).collect()
}
fn fixture(dtype: DType, shape: &[usize], values: &[i64]) -> Input {
    dense(shape, numbers(dtype, values))
}
fn model_expected(model: Model, dtype: DType) -> Values {
    let bits = match model {
        Model::OneHot65536 => vec![cast_bits(
            DType::F32,
            dtype,
            ieee_binary(DType::F32, 0x3f800000, integer_bits(DType::F32, 65536), Arithmetic::Divide),
        )],
        Model::Ones65536 => vec![cast_bits(
            DType::F32,
            dtype,
            ieee_binary(DType::F32, integer_bits(DType::F32, 65536), integer_bits(DType::F32, 65536), Arithmetic::Divide),
        )],
        Model::HalfCounts => [65504, 65505, 65519, 65520, 65536].into_iter().map(|n| scalar_count(dtype, n)).collect(),
        Model::F64Count => vec![integer_bits(dtype, 16777217), cast_bits(DType::F32, dtype, integer_bits(DType::F32, 16777217))],
        Model::F32Counts => [16777219, 2147483647].into_iter().map(|n| integer_bits(dtype, n)).collect(),
        Model::EmptyHuge => {
            reference(
                dtype,
                &Values::new(&[0, i32::MAX as usize, i32::MAX as usize], vec![]),
                Operation::Sum,
                Scope::Global,
            )
            .expect("zero-aware empty shortcut")
            .values
        }
        Model::ScalarMidpoint => {
            // 1 + 2^-11 + 2^-52: direct F64->F16 rounds upward, while
            // canonical scalar construction through F32 lands on an even tie.
            let source = 0x3ff0020000000001;
            vec![
                cast_bits(DType::F32, dtype, cast_bits(DType::F64, DType::F32, source)),
                cast_bits(DType::F64, dtype, source),
            ]
        }
    };
    Values::new(&[bits.len()], bits)
}
fn add(cases: &mut Vec<Case>, n: usize, label: &'static str, dtype: DType, input: Input, steps: Vec<Step>) {
    cases.push(Case {
        id: format!("P{n:02}"),
        label,
        dtype,
        input,
        steps,
        evidence: if dtype == DType::F32 {
            Evidence::LegacyAllOrderExact
        } else {
            Evidence::NativeSerial
        },
        model: None,
        chain: false,
    });
}
/// Exactly 72 standalone rows and eight chains before selection. Seven rows
/// are symbolic CPU checks, never large reductions or device observations.
pub fn corpus(options: &Options) -> Vec<Case> {
    use Operation::*;
    use Scope::*;
    use Step::*;
    let mut c = Vec::new();
    let mut n = 1;
    for dtype in [DType::F64, DType::Flex32, DType::F16, DType::BF16] {
        for op in [Sum, Product, Mean] {
            for scope in [Axis(1), Global] {
                add(
                    &mut c,
                    n,
                    "core-axis-global",
                    dtype,
                    fixture(dtype, &[2, 3], &[1, 2, 3, 4, 5, 6]),
                    vec![Reduce(op, scope)],
                );
                n += 1;
            }
        }
    }
    add(
        &mut c,
        25,
        "stable-original-axis-tie-mean",
        DType::F16,
        fixture(DType::F16, &[2, 2], &[2048, 1, -2048, 0]),
        vec![Reduce(Mean, Global)],
    );
    add(
        &mut c,
        26,
        "caller-reverse-mean-dims",
        DType::F16,
        fixture(DType::F16, &[2, 2], &[2048, 1, -2048, 0]),
        vec![Dims(Mean, vec![1, 0])],
    );
    let mut input = fixture(DType::BF16, &[2, 2], &[256, 1, -256, 0]);
    input.views.push(View::Permute(vec![1, 0]));
    add(&mut c, 27, "caller-sum-dims-transpose", DType::BF16, input, vec![Dims(Sum, vec![1, 0])]);
    add(
        &mut c,
        28,
        "empty-dims-source-bits",
        DType::F64,
        dense(&[2], vec![0x8000000000000000, 0x7ff0000000000001]),
        vec![Dims(Sum, vec![])],
    );
    add(
        &mut c,
        29,
        "repeated-singleton-mean",
        DType::Flex32,
        fixture(DType::Flex32, &[2, 1], &[3, 5]),
        vec![Dims(Mean, vec![1, 1])],
    );
    let mut input = fixture(DType::F64, &[2, 3], &[-1, 2, 3, 4, -5, 6]);
    input.views = vec![View::Permute(vec![1, 0]), View::Flip(vec![0])];
    add(
        &mut c,
        30,
        "caller-product-signed-transpose",
        DType::F64,
        input,
        vec![Dims(Product, vec![1, 0])],
    );
    let mut input = fixture(DType::F16, &[2, 3], &[-1, 2, -3, 4, -5, 6]);
    input.views = vec![View::Flip(vec![0, 1])];
    add(&mut c, 31, "signed-stride-product", DType::F16, input, vec![Reduce(Product, Axis(1))]);
    let mut input = fixture(DType::BF16, &[1, 3], &[2, 4, 6]);
    input.views = vec![View::Expand(vec![3, 3])];
    add(&mut c, 32, "broadcast-logical-multiplicity", DType::BF16, input, vec![Reduce(Mean, Axis(0))]);
    let mut input = fixture(DType::F64, &[2, 1, 1, 1, 1, 1, 1, 3], &[1, 2, 3, 4, 5, 6]);
    input.views = vec![View::Expand(vec![2, 1, 1, 1, 1, 1, 2, 3]), View::Flip(vec![0, 7])];
    add(&mut c, 33, "rank-eight-signed-broadcast", DType::F64, input, vec![Reduce(Mean, Axis(7))]);
    // Public slicing cannot manufacture an arbitrary overlapping stride. This
    // expand view is overlapping read-only storage with logical multiplicity.
    let mut input = fixture(DType::Flex32, &[1, 3], &[1, 2, 3]);
    input.views = vec![View::Expand(vec![2, 3])];
    add(&mut c, 34, "overlapping-read-only-expand", DType::Flex32, input, vec![Reduce(Sum, Axis(0))]);
    let mut input = fixture(DType::F16, &[4, 4], &(0..16).collect::<Vec<_>>());
    input.views = vec![View::Slice {
        starts: vec![1, 1],
        steps: vec![1, 1],
        shape: vec![3, 2],
    }];
    add(&mut c, 35, "offset-odd-output", DType::F16, input, vec![Reduce(Mean, Axis(1))]);
    add(
        &mut c,
        36,
        "packed-logical-output-129",
        DType::BF16,
        fixture(DType::BF16, &[129, 2], &vec![1; 258]),
        vec![Reduce(Sum, Axis(1))],
    );
    add(
        &mut c,
        37,
        "unequal-original-axis-order",
        DType::F64,
        fixture(DType::F64, &[3, 2], &[1, 2, 3, 4, 5, 6]),
        vec![Reduce(Product, Global)],
    );
    add(
        &mut c,
        38,
        "stable-singleton-global-mean",
        DType::Flex32,
        fixture(DType::Flex32, &[2, 3, 1], &[1, 2, 3, 4, 5, 6]),
        vec![Reduce(Mean, Global)],
    );
    add(
        &mut c,
        39,
        "stored-global-overflow",
        DType::F16,
        fixture(DType::F16, &[2, 2], &[65504, 65504, -65504, -65504]),
        vec![Reduce(Sum, Global)],
    );
    add(
        &mut c,
        40,
        "subnormal-global-product",
        DType::BF16,
        dense(&[2, 1], vec![1, integer_bits(DType::BF16, 1)]),
        vec![Reduce(Product, Global)],
    );
    add(
        &mut c,
        41,
        "rank-zero-nan-payload-relabel",
        DType::F64,
        dense(&[], vec![0xfff0000000000001]),
        vec![Reduce(Sum, Global)],
    );
    add(
        &mut c,
        42,
        "rank-zero-negative-zero-relabel",
        DType::Flex32,
        dense(&[], vec![0x80000000]),
        vec![Reduce(Product, Global)],
    );
    for (id, dtype, shape, op, scope) in [
        (43, DType::F64, vec![2, 0], Sum, Axis(1)),
        (44, DType::F64, vec![2, 0], Mean, Axis(1)),
        (45, DType::BF16, vec![3, 0], Product, Axis(1)),
        (46, DType::F16, vec![0, 3], Mean, Axis(1)),
        (47, DType::F64, vec![0, 0], Product, Axis(1)),
        (48, DType::Flex32, vec![0, 0], Sum, Axis(1)),
        (49, DType::F16, vec![0, 3], Sum, Global),
        (50, DType::BF16, vec![0, 0], Mean, Global),
        (51, DType::F64, vec![0, 3], Product, Global),
    ] {
        add(
            &mut c,
            id,
            "empty-identities-or-N0-metadata",
            dtype,
            dense(&shape, vec![]),
            vec![Reduce(op, scope)],
        );
    }
    add(
        &mut c,
        52,
        "rank-zero-global-mean-bits",
        DType::F64,
        dense(&[], vec![0x8000000000000000]),
        vec![Reduce(Mean, Global)],
    );
    for (id, dtype, model, label) in [
        (53, DType::F16, Model::OneHot65536, "symbolic-one-hot-axis-mean"),
        (54, DType::F16, Model::Ones65536, "symbolic-ones-axis-mean"),
        (55, DType::F16, Model::HalfCounts, "symbolic-half-scalar-count-boundary"),
        (56, DType::F64, Model::F64Count, "symbolic-raw-versus-legacy-count"),
        (57, DType::Flex32, Model::F32Counts, "symbolic-integer-count-rounding"),
        (59, DType::F64, Model::EmptyHuge, "symbolic-zero-aware-empty-shortcut"),
        (60, DType::F16, Model::ScalarMidpoint, "symbolic-canonical-scalar-double-rounding"),
    ] {
        add(&mut c, id, label, dtype, fixture(dtype, &[1], &[1]), vec![]);
        let row = c.last_mut().unwrap();
        row.model = Some(model);
        row.evidence = Evidence::SymbolicOnly;
    }
    add(
        &mut c,
        58,
        "invalid-axis-before-empty",
        DType::BF16,
        dense(&[0], vec![]),
        vec![Reduce(Mean, Axis(1))],
    );
    add(
        &mut c,
        61,
        "legacy-F32-axis129-all-order",
        DType::F32,
        fixture(DType::F32, &[129], &vec![1; 129]),
        vec![Reduce(Sum, Axis(0))],
    );
    add(
        &mut c,
        62,
        "legacy-F32-reverse-global-product",
        DType::F32,
        fixture(DType::F32, &[2, 3], &[-1, 1, 1, 1, -1, 1]),
        vec![Reduce(Product, Global)],
    );
    add(
        &mut c,
        63,
        "legacy-F32-stored-sum-divide",
        DType::F32,
        fixture(DType::F32, &[2, 3], &[1, 2, 3, 4, 5, 6]),
        vec![Reduce(Mean, Global)],
    );
    add(
        &mut c,
        64,
        "legacy-F32-axis-mean-all-order",
        DType::F32,
        fixture(DType::F32, &[2, 3], &[1, 2, 3, 4, 5, 6]),
        vec![Reduce(Mean, Axis(1))],
    );
    add(
        &mut c,
        65,
        "legacy-I32-axis",
        DType::I32,
        dense(&[2, 3], vec![1, 2, 3, 4, 5, 6]),
        vec![Reduce(Sum, Axis(1))],
    );
    add(
        &mut c,
        66,
        "legacy-I32-global",
        DType::I32,
        dense(&[2, 3], vec![1, 2, 3, 4, 5, 6]),
        vec![Reduce(Product, Global)],
    );
    add(
        &mut c,
        67,
        "legacy-F32-empty-alias-raw-owned",
        DType::F32,
        dense(&[0, 3], vec![]),
        vec![Reduce(Sum, Axis(1))],
    );
    add(
        &mut c,
        68,
        "legacy-I32-empty-alias-raw-owned",
        DType::I32,
        dense(&[0, 3], vec![]),
        vec![Reduce(Sum, Axis(1))],
    );
    add(
        &mut c,
        69,
        "legacy-first-NaN-and-zero-selection",
        DType::F32,
        dense(&[2, 3], vec![0xff800001, 0x7fc00002, 0x3f800000, 0x80000000, 0, 0x3f800000]),
        vec![Minimum(1)],
    );
    add(
        &mut c,
        70,
        "existing-F64-adjacent-extrema",
        DType::F64,
        dense(&[2], vec![0x3ff0000000000001, 0x3ff0000000000000]),
        vec![Minimum(0)],
    );
    add(
        &mut c,
        71,
        "existing-I64-typed-sum",
        DType::I64,
        dense(&[2], vec![1, 2]),
        vec![Reduce(Sum, Axis(0))],
    );
    add(
        &mut c,
        72,
        "legacy-exact-cumsum",
        DType::F32,
        fixture(DType::F32, &[4], &[1, 2, 3, 4]),
        vec![CumulativeSum(0)],
    );
    for row in &mut c {
        if matches!(row.dtype, DType::I32 | DType::I64) {
            row.evidence = Evidence::IntegerRegression;
        }
        if matches!(row.steps.first(), Some(Minimum(_))) {
            row.evidence = Evidence::ExistingExactSelection;
        }
    }
    c.sort_by(|a, b| a.id.cmp(&b.id));
    let mut chain = |id, label, dtype, input, steps| {
        c.push(Case {
            id: format!("C{id:02}"),
            label,
            dtype,
            input,
            steps,
            evidence: Evidence::NativeSerial,
            model: None,
            chain: true,
        });
    };
    chain(
        1,
        "sum-scale-mean-final-only",
        DType::F64,
        fixture(DType::F64, &[2, 3], &[1, 2, 3, 4, 5, 6]),
        vec![Reduce(Sum, Axis(1)), Multiply(integer_bits(DType::F64, 2)), Reduce(Mean, Axis(0))],
    );
    chain(
        2,
        "mean-center-square-sum-final-only",
        DType::F16,
        fixture(DType::F16, &[2, 3], &[1, 2, 3, 4, 5, 6]),
        vec![Reduce(Mean, Axis(1)), SubtractSource, Square, Reduce(Sum, Global)],
    );
    chain(
        3,
        "signed-zero-product-consumer",
        DType::BF16,
        dense(&[2], vec![0x8000, 0x3f80]),
        vec![Reduce(Product, Axis(0)), Multiply(integer_bits(DType::BF16, 1))],
    );
    chain(
        4,
        "caller-sum-dims-scale-global-mean",
        DType::Flex32,
        fixture(DType::Flex32, &[2, 2], &[1, 2, 3, 4]),
        vec![Dims(Sum, vec![1]), Multiply(integer_bits(DType::Flex32, 2)), Reduce(Mean, Global)],
    );
    chain(
        5,
        "global-mean-lifetime-consumer-raw-scope",
        DType::F64,
        fixture(DType::F64, &[2, 3], &[1, 2, 3, 4, 5, 6]),
        vec![Reduce(Mean, Global), Multiply(integer_bits(DType::F64, 2))],
    );
    chain(
        6,
        "logical-half-cast-no-padding-claim",
        DType::F16,
        fixture(DType::F16, &[129, 2], &vec![1; 258]),
        vec![Reduce(Sum, Axis(1)), Cast(DType::F64), Multiply(integer_bits(DType::F64, 3))],
    );
    chain(
        7,
        "empty-axis-NaN-consumer",
        DType::BF16,
        dense(&[3, 0], vec![]),
        vec![Reduce(Mean, Axis(1)), IsNan, Reduce(Sum, Global)],
    );
    chain(
        8,
        "second-stage-error-and-recovery",
        DType::F64,
        fixture(DType::F64, &[2, 3], &[1, 2, 3, 4, 5, 6]),
        vec![Reduce(Sum, Axis(1)), Reduce(Mean, Axis(2))],
    );
    // Add rank 3..7 coverage without multiplying cases or allocating large roots.
    // Singleton axes preserve core values and expose the public rank dispatcher.
    for (id, rank) in [("P07", 3), ("P09", 4), ("P13", 5), ("P15", 6), ("P19", 7)] {
        let row = c.iter_mut().find(|r| r.id == id).unwrap();
        let mut shape = vec![1; rank - 2];
        shape.extend([2, 3]);
        row.input.root.shape = shape;
        row.steps = vec![Reduce(
            match id {
                "P09" | "P15" => Product,
                _ => Sum,
            },
            Axis(rank - 1),
        )];
    }
    // Serial cancellation, overflow and widened half accumulation witnesses.
    for (id, values) in [
        (
            "P01",
            vec![
                0x4340000000000000,
                0x3ff0000000000000,
                0xc340000000000000,
                0x3ff0000000000000,
                0x3ff0000000000000,
                0x3ff0000000000000,
            ],
        ),
        (
            "P03",
            vec![
                0x7fefffffffffffff,
                0x4000000000000000,
                0x3fe0000000000000,
                0x8000000000000000,
                0x3ff0000000000000,
                0x3ff0000000000000,
            ],
        ),
        ("P05", vec![0x7fefffffffffffff, 0x7fefffffffffffff, 0, 1, 0, 0]),
        ("P07", vec![0x4b800000, 0x3f800000, 0xcb800000, 0x3f800000, 0x40000000, 0x40400000]),
        ("P09", vec![1, 0x3f000000, 0x40000000, 0x7f7fffff, 0x40000000, 0x3f000000]),
        ("P13", vec![0x3c00, 0x1000, 0x1000, 0x7bff, 0x7bff, 0]),
        ("P19", vec![0x3f80, 0x3b80, 0x3b80, 1, 1, 0]),
        (
            "P37",
            vec![
                0x7fefffffffffffff,
                0x3ff0000000000000,
                0x4000000000000000,
                0x3fe0000000000000,
                0x3fe0000000000000,
                0x3ff0000000000000,
            ],
        ),
    ] {
        c.iter_mut().find(|c| c.id == id).unwrap().input.root.values = values;
    }
    c.retain(|case| {
        let selection = match case.dtype {
            DType::I32 => DType::F32,
            DType::I64 => DType::F64,
            d => d,
        };
        options.dtype.is_none_or(|d| d == selection)
            && (!options.skip_rank_zero || !case.input.root.shape.is_empty())
            && (options.include_empty || !case.input.logical().shape.contains(&0))
    });
    c
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionScope {
    #[default]
    LinuxDefault,
    NativeErrorExtensions,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Checks {
    pub input: usize,
    pub source: usize,
    pub alias: usize,
    pub output: usize,
    pub error: usize,
    pub recovery: usize,
    pub model: usize,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub options: Options,
    pub execution_scope: ExecutionScope,
    pub cases: usize,
    pub standalone_cases: usize,
    pub chain_cases: usize,
    pub device_cases: usize,
    pub model_cases: usize,
    pub diagnostic_cases: usize,
    pub steps: usize,
    /// Only actually downloaded final output values; no symbolic aggregates.
    pub values: usize,
    pub checks: Checks,
}
pub fn expected_summary(options: &Options) -> Summary {
    let mut s = Summary {
        options: *options,
        execution_scope: if options.native_error_extensions {
            ExecutionScope::NativeErrorExtensions
        } else {
            ExecutionScope::LinuxDefault
        },
        ..Summary::default()
    };
    for case in corpus(options) {
        s.cases += 1;
        s.chain_cases += usize::from(case.chain);
        s.standalone_cases += usize::from(!case.chain);
        if case.model.is_some() {
            s.model_cases += 1;
            s.checks.model += 1;
            continue;
        }
        let expected = case.expected();
        if let Ok(expected) = &expected {
            s.device_cases += 1;
            s.steps += case.steps.len();
            s.values += expected.output.values.len();
            s.checks.output += 1;
        } else {
            s.diagnostic_cases += 1;
            if options.native_error_extensions {
                s.checks.error += 1;
                s.checks.recovery += 1;
            }
        }
        if expected.is_ok() || options.native_error_extensions {
            s.checks.input += 1;
            s.checks.source += 1;
            s.checks.alias += 1;
        }
    }
    s
}
fn checks_json(c: &Checks) -> String {
    format!(
        "{{\"input\":{},\"source\":{},\"alias\":{},\"output\":{},\"error\":{},\"recovery\":{},\"model\":{}}}",
        c.input, c.source, c.alias, c.output, c.error, c.recovery, c.model
    )
}
pub fn summary_json(s: &Summary) -> String {
    let dtype = s.options.dtype.map_or_else(|| String::from("null"), |d| format!("\"{d:?}\""));
    format!(
        concat!(
            "{{\"corpus\":{},\"options\":{{\"dtype\":{},\"skip_rank_zero\":{},\"include_empty\":{},\"native_error_extensions\":{}}},",
            "\"executionScope\":\"{:?}\",\"cases\":{},\"standalone_cases\":{},\"chain_cases\":{},\"device_cases\":{},\"model_cases\":{},\"diagnostic_cases\":{},\"steps\":{},\"values\":{},\"checks\":{},",
            "\"reference\":\"exact_integer_ieee_rne_steps\",\"new_four_global_mean\":\"stable_original_axis_length_axis_mean_stored_each_stage\",\"F32\":\"legacy_all_order_witnesses_only_stored_sum_scalar_divide_global_mean\",",
            "\"linux\":\"selected_graph_observations_not_general_equivalence\",\"physical_padding_alias_allocation_residency\":\"separate_raw_contract\"}}"
        ),
        CORPUS_VERSION,
        dtype,
        s.options.skip_rank_zero,
        s.options.include_empty,
        s.options.native_error_extensions,
        s.execution_scope,
        s.cases,
        s.standalone_cases,
        s.chain_cases,
        s.device_cases,
        s.model_cases,
        s.diagnostic_cases,
        s.steps,
        s.values,
        checks_json(&s.checks)
    )
}
/// Expected plan and observed checks stay separate even after complete success.
#[derive(Clone, Debug)]
pub struct Report {
    pub expected: Summary,
    pub attempted: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped_diagnostics: usize,
    pub observed_errors: usize,
    pub model_passed: usize,
    pub checks: Checks,
    pub failures: Vec<Failure>,
}
#[derive(Clone, Debug)]
pub struct Failure {
    pub case_id: String,
    pub message: String,
}
impl Report {
    pub fn is_success(&self) -> bool {
        self.failed == 0
            && self.failures.is_empty()
            && self.attempted == self.passed
            && self.attempted == self.expected.device_cases + self.observed_errors
            && self.observed_errors + self.skipped_diagnostics == self.expected.diagnostic_cases
            && self.model_passed == self.expected.model_cases
            && self.checks == self.expected.checks
    }
    fn failure(&mut self, id: &str, message: String) {
        self.failed += 1;
        if self.failures.len() < 24 {
            self.failures.push(Failure {
                case_id: String::from(id),
                message,
            });
        }
    }
}
pub fn success_footer(options: &Options, report: &Report) -> Result<String, String> {
    if options.dtype.is_some_and(|d| !DTYPES.contains(&d)) {
        return Err(String::from("unsupported floating dtype"));
    }
    if report.expected != expected_summary(options) {
        return Err(String::from("report plan differs from selected options and execution scope"));
    }
    let errors = if options.native_error_extensions {
        report.expected.diagnostic_cases
    } else {
        0
    };
    if report.observed_errors != errors || !report.is_success() {
        return Err(format!(
            "floating reduction conformance failed: failed={} attempted={} first={:?}",
            report.failed,
            report.attempted,
            report.failures.first()
        ));
    }
    Ok(format!(
        "BURN_FLOAT_REDUCTION_OK corpus={CORPUS_VERSION} totals={} attempted={} passed={} model_passed={} skipped_diagnostics={} observed_errors={} reference=exact_integer_ieee_rne_steps new_four=serial_axis_stable_original_length_global_stored_each_stage F32=legacy_all_order_exact linux=selected_graph_observations_not_general_equivalence physical_padding_alias_allocations_residency=separate_raw_contract scope=forward_sum_product_mean",
        summary_json(&report.expected),
        report.attempted,
        report.passed,
        report.model_passed,
        report.skipped_diagnostics,
        report.observed_errors
    ))
}
fn apply(input: FloatTensor<Dispatch>, step: &Step, source: &FloatTensor<Dispatch>) -> FloatTensor<Dispatch> {
    match step {
        Step::Reduce(Operation::Sum, Scope::Axis(a)) => Dispatch::float_sum_dim(input, *a),
        Step::Reduce(Operation::Mean, Scope::Axis(a)) => Dispatch::float_mean_dim(input, *a),
        Step::Reduce(Operation::Product, Scope::Axis(a)) => Dispatch::float_prod_dim(input, *a),
        Step::Reduce(Operation::Sum, Scope::Global) => Dispatch::float_sum(input),
        Step::Reduce(Operation::Mean, Scope::Global) => Dispatch::float_mean(input),
        Step::Reduce(Operation::Product, Scope::Global) => Dispatch::float_prod(input),
        Step::Dims(op, axes) => axes.iter().fold(input, |t, &a| apply(t, &Step::Reduce(*op, Scope::Axis(a)), source)),
        Step::Multiply(bits) => {
            // Chain scalar constants are exactly represented small integers.
            // Decode only for the API value, never for reference arithmetic.
            let dtype = input.dtype();
            let scalar = f64::from_bits(cast_bits(dtype, DType::F64, *bits));
            Dispatch::float_mul_scalar(input, scalar.into())
        }
        Step::SubtractSource => Dispatch::float_sub(source.clone(), input),
        Step::Square => Dispatch::float_mul(input.clone(), input),
        Step::Cast(dtype) => Dispatch::float_cast(input, (*dtype).into()),
        Step::IsNan => {
            let dtype = input.dtype();
            Dispatch::bool_into_float(Dispatch::float_is_nan(input, burn::backend::BoolDType::U32), dtype.into())
        }
        Step::Minimum(a) => Dispatch::float_min_dim(input, *a),
        Step::CumulativeSum(a) => Dispatch::float_cumsum(input, *a),
    }
}
fn apply_public<const D: usize>(input: FloatTensor<Dispatch>, step: &Step, source: &FloatTensor<Dispatch>) -> FloatTensor<Dispatch> {
    let t = Tensor::<D>::from_dispatch(input);
    match step {
        Step::Reduce(Operation::Sum, Scope::Axis(a)) => t.sum_dim(*a).into_dispatch(),
        Step::Reduce(Operation::Mean, Scope::Axis(a)) => t.mean_dim(*a).into_dispatch(),
        Step::Reduce(Operation::Product, Scope::Axis(a)) => t.prod_dim(*a).into_dispatch(),
        Step::Reduce(Operation::Sum, Scope::Global) => t.sum().into_dispatch(),
        Step::Reduce(Operation::Mean, Scope::Global) => t.mean().into_dispatch(),
        Step::Reduce(Operation::Product, Scope::Global) => t.prod().into_dispatch(),
        Step::Dims(Operation::Sum, axes) => t.sum_dims(axes).into_dispatch(),
        Step::Dims(Operation::Mean, axes) => t.mean_dims(axes).into_dispatch(),
        Step::Dims(Operation::Product, axes) => t.prod_dims(axes).into_dispatch(),
        Step::Minimum(a) => t.min_dim(*a).into_dispatch(),
        Step::CumulativeSum(a) => t.cumsum(*a).into_dispatch(),
        _ => apply(t.into_dispatch(), step, source),
    }
}
fn public_by_rank(input: FloatTensor<Dispatch>, step: &Step, source: &FloatTensor<Dispatch>) -> FloatTensor<Dispatch> {
    match input.shape().num_dims() {
        0 => apply(input, step, source), // Primitive scalar relabel, no Tensor<0> claim.
        1 => apply_public::<1>(input, step, source),
        2 => apply_public::<2>(input, step, source),
        3 => apply_public::<3>(input, step, source),
        4 => apply_public::<4>(input, step, source),
        5 => apply_public::<5>(input, step, source),
        6 => apply_public::<6>(input, step, source),
        7 => apply_public::<7>(input, step, source),
        8 => apply_public::<8>(input, step, source),
        _ => unreachable!("corpus ranks are bounded"),
    }
}
fn validate_data(data: TensorData, dtype: DType, expected: &Values) -> Result<(), String> {
    if data.dtype() != dtype || data.shape().as_slice() != expected.shape {
        return Err(String::from("read metadata changed"));
    }
    let bytes = pack(dtype, &expected.values);
    if bytes.len() != data.as_bytes().len() {
        return Err(format!("logical byte count actual={} expected={}", data.as_bytes().len(), bytes.len()));
    }
    if let Some(at) = data.as_bytes().iter().zip(&bytes).position(|(a, b)| a != b) {
        let i = at / dtype.size();
        let actual = unpack(dtype, data.as_bytes())[i];
        return Err(format!("element={i} actual={actual:#x} expected={:#x}", expected.values[i]));
    }
    Ok(())
}
async fn check(t: FloatTensor<Dispatch>, dtype: DType, expected: &Values) -> Result<(), String> {
    if t.dtype() != dtype || t.shape().as_slice() != expected.shape {
        return Err(format!(
            "metadata actual={:?}/{:?} expected={dtype:?}/{:?}",
            t.dtype(),
            t.shape(),
            expected.shape
        ));
    }
    let data = Dispatch::float_into_data(t).await.map_err(|e| format!("read: {e:?}"))?;
    validate_data(data, dtype, expected)
}
fn counted(result: Result<(), String>, n: &mut usize) -> Result<(), String> {
    if result.is_ok() {
        *n += 1;
    }
    result
}
async fn run_integer(device: &Device, case: &Case, expected: &Expected, checks: &mut Checks) -> Result<(), String> {
    let source = Dispatch::int_from_data(
        TensorData::from_bytes_vec(pack(case.dtype, &case.input.root.values), case.input.root.shape.clone(), case.dtype),
        device.as_dispatch(),
    );
    let alias = source.clone();
    let initial = counted(
        Dispatch::int_into_data(source.clone())
            .await
            .map_err(|e| format!("input: {e:?}"))
            .and_then(|d| validate_data(d, case.dtype, &case.input.root)),
        &mut checks.input,
    );
    let Step::Reduce(op, scope) = case.steps[0] else { unreachable!() };
    let output = match (op, scope) {
        (Operation::Sum, Scope::Axis(a)) => Dispatch::int_sum_dim(source.clone(), a),
        (Operation::Sum, Scope::Global) => Dispatch::int_sum(source.clone()),
        (Operation::Product, Scope::Global) => Dispatch::int_prod(source.clone()),
        _ => unreachable!(),
    };
    let output = counted(
        Dispatch::int_into_data(output)
            .await
            .map_err(|e| format!("output: {e:?}"))
            .and_then(|d| validate_data(d, expected.dtype, &expected.output)),
        &mut checks.output,
    );
    let src = counted(
        Dispatch::int_into_data(source)
            .await
            .map_err(|e| format!("source: {e:?}"))
            .and_then(|d| validate_data(d, case.dtype, &case.input.root)),
        &mut checks.source,
    );
    let alias = counted(
        Dispatch::int_into_data(alias)
            .await
            .map_err(|e| format!("alias: {e:?}"))
            .and_then(|d| validate_data(d, case.dtype, &case.input.root)),
        &mut checks.alias,
    );
    initial.and(output).and(src).and(alias)
}
async fn run_case(device: &Device, case: &Case, expected: &Expected, checks: &mut Checks) -> Result<(), String> {
    if matches!(case.dtype, DType::I32 | DType::I64) {
        return run_integer(device, case, expected, checks).await;
    }
    let (source, input) = case.input.build(case.dtype, device);
    let alias = input.clone();
    let original = input.clone();
    let initial = counted(check(input.clone(), case.dtype, &case.input.logical()).await, &mut checks.input);
    let mut current = input;
    #[cfg(not(target_os = "none"))]
    let mut dispatch = Ok(());
    #[cfg(target_os = "none")]
    let dispatch = Ok(());
    for step in &case.steps {
        current = public_by_rank(current, step, &original);
        // Flush Linux fusion between explicit chain steps so its tensors have
        // stored dtype boundaries. This submits work without downloading data.
        // Native retains queued composition; the raw harness tests recording.
        #[cfg(not(target_os = "none"))]
        if case.chain {
            dispatch = dispatch.and(device.flush().map_err(|e| format!("chain flush: {e:?}")));
        }
    }
    let output = counted(check(current, expected.dtype, &expected.output).await, &mut checks.output);
    let src = counted(check(source, case.dtype, &case.input.root).await, &mut checks.source);
    let alias = counted(check(alias, case.dtype, &case.input.logical()).await, &mut checks.alias);
    initial.and(dispatch).and(output).and(src).and(alias)
}
async fn expected_error(device: &Device, case: &Case, checks: &mut Checks) -> Result<(), String> {
    let (source, input) = case.input.build(case.dtype, device);
    let alias = input.clone();
    let original = input.clone();
    let initial = counted(check(input.clone(), case.dtype, &case.input.logical()).await, &mut checks.input);
    let mut current = input;
    let mut shape = case.input.logical().shape;
    for step in &case.steps {
        if let Step::Reduce(_, Scope::Axis(axis)) = step
            && let Some(d) = shape.get_mut(*axis)
        {
            *d = 1;
        }
        current = apply(current, step, &original); // Bypass frontend axis assertions.
    }
    let metadata = current.dtype() == case.dtype && current.shape().as_slice() == shape;
    let failed = Dispatch::float_into_data(current).await.is_err();
    let error = counted(
        if metadata && failed {
            Ok(())
        } else {
            Err(String::from("deferred error lost dtype/shape or returned data"))
        },
        &mut checks.error,
    );
    let src = counted(check(source, case.dtype, &case.input.root).await, &mut checks.source);
    let alias = counted(check(alias, case.dtype, &case.input.logical()).await, &mut checks.alias);
    let valid = fixture(case.dtype, &[2], &[1, 2]);
    let (_, t) = valid.build(case.dtype, device);
    let recovery = counted(
        check(Dispatch::float_sum_dim(t, 0), case.dtype, &Values::new(&[1], vec![integer_bits(case.dtype, 3)])).await,
        &mut checks.recovery,
    );
    initial.and(error).and(src).and(alias).and(recovery)
}
fn check_model(case: &Case) -> Result<(), String> {
    let Some(model) = case.model else { unreachable!() };
    let anchors = match model {
        Model::OneHot65536 => vec![0x0100],
        Model::Ones65536 => vec![0x3c00],
        Model::HalfCounts => vec![0x7bff, 0x7bff, 0x7bff, 0x7c00, 0x7c00],
        Model::F64Count => vec![0x4170000010000000, 0x4170000000000000],
        Model::F32Counts => vec![0x4b800002, 0x4f000000],
        Model::EmptyHuge => vec![0],
        Model::ScalarMidpoint => vec![0x3c00, 0x3c01],
    };
    if model_expected(model, case.dtype).values == anchors {
        Ok(())
    } else {
        Err(String::from("symbolic reference differs from independent IEEE anchors"))
    }
}
/// Isolated Linux diagnostics are deliberate invalid API calls. Returning or
/// panicking here cannot count as shared conformance success.
pub async fn linux_error_diagnostic(device: &Device, options: &Options, id: &str) -> Result<(), String> {
    let case = corpus(options)
        .into_iter()
        .find(|c| c.id == id)
        .ok_or_else(|| format!("unknown diagnostic {id}"))?;
    if case.expected().is_ok() {
        return Err(String::from("invalid API row required"));
    }
    let (_, input) = case.input.build(case.dtype, device);
    let original = input.clone();
    let mut current = input;
    for step in &case.steps {
        current = apply(current, step, &original);
    }
    let result = Dispatch::float_into_data(current).await;
    Err(format!("isolated Linux diagnostic returned: {result:?}"))
}
/// Run only selected finite fixtures. Model rows never initialize tensors and
/// never increment attempted/passed device counts. Native errors require opt-in.
pub async fn run(device: &Device, options: &Options) -> Report {
    let mut report = Report {
        expected: expected_summary(options),
        attempted: 0,
        passed: 0,
        failed: 0,
        skipped_diagnostics: 0,
        observed_errors: 0,
        model_passed: 0,
        checks: Checks::default(),
        failures: vec![],
    };
    #[cfg(not(target_os = "none"))]
    if options.native_error_extensions {
        report.failure("options", String::from("Native error extensions cannot run on Linux"));
        return report;
    }
    for case in corpus(options) {
        if case.model.is_some() {
            match counted(check_model(&case), &mut report.checks.model) {
                Ok(()) => report.model_passed += 1,
                Err(e) => report.failure(&case.id, e),
            }
            continue;
        }
        let expected = case.expected();
        if expected.is_err() && !options.native_error_extensions {
            report.skipped_diagnostics += 1;
            continue;
        }
        report.attempted += 1;
        let result = match expected {
            Ok(expected) => run_case(device, &case, &expected, &mut report.checks).await,
            Err(_) => {
                let result = expected_error(device, &case, &mut report.checks).await;
                if result.is_ok() {
                    report.observed_errors += 1;
                }
                result
            }
        };
        match result {
            Ok(()) => report.passed += 1,
            Err(e) => report.failure(
                &case.id,
                format!(
                    "evidence={:?} native_contract_mismatch Linux_selected_graph_not_general_equivalence: {e}",
                    case.evidence
                ),
            ),
        }
    }
    report
}
