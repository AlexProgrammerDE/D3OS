//! Bounded typed float conformance with checked-in exact rational expectations.
//!
//! The generator uses integers/Fraction and an independent IEEE RNE encoder.
//! Runtime code only indexes those fixtures. Arithmetic NaNs use classification;
//! finite values, infinities and signed zeros require identical stored bits.
//! Source and alias checks always require exact bytes, including NaN payloads.
//! Public poison allocations do not certify physical padding or adjacent guards.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{Backend as BurnBackend, DType, Dispatch, Scalar, TensorData, TensorMetadata, tensor::FloatTensor},
    tensor::{Device, Slice, Tensor, TensorCreationOptions},
};

pub const DTYPES: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
pub const POOL_LENGTH: usize = 32;
pub const CORPUS_VERSION: u32 = 1;
const LENGTHS: [usize; 6] = [0, 1, 127, 128, 129, 257];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Recip,
    Sign,
}
impl Operation {
    pub const ALL: [Self; 8] = [Self::Add, Self::Sub, Self::Mul, Self::Div, Self::Neg, Self::Abs, Self::Recip, Self::Sign];
    pub const fn unary(self) -> bool {
        (self as usize) >= Self::Neg as usize
    }
}

#[derive(Clone, Copy, Debug, Default)]
/// The default selects all five dtypes and all eight operations.
pub struct Options {
    /// Exclude the separate Flex32 group.
    pub no_flex32: bool,
    /// Exclude rank-zero cases without a replacement reshape.
    pub skip_rank_zero: bool,
    /// Require finite logical operands and finite expected results.
    /// Scalar cases without suitable values are omitted. Retained poison can
    /// contain NaNs because it does not supply arithmetic operands.
    pub finite: bool,
    pub dtype: Option<DType>,
    /// Mixed-operation chains are omitted when one operation is selected.
    pub operation: Option<Operation>,
}

#[derive(Clone, Copy, Debug)]
pub struct ScalarFixture {
    pub scalar: Scalar,
    pub bits: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct ChainStep {
    pub operation: Operation,
    pub rhs: u64,
    pub expected: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct ChainFixture {
    pub initial: u64,
    /// Single-rounding a*b+c oracle for the FMA witness only. The runner always
    /// checks the separate stored destinations in `steps`, never this value.
    pub fused: Option<u64>,
    pub steps: &'static [ChainStep],
}
struct Fixture {
    offset: usize,
    scalar_count: usize,
}
include!("float_arithmetic_fixtures.rs");

fn format_index(dtype: DType) -> usize {
    match dtype {
        DType::F64 => 0,
        DType::F32 | DType::Flex32 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        _ => panic!("float arithmetic fixture dtype required"),
    }
}
fn word(dtype: DType, index: usize) -> u64 {
    let size = dtype.size();
    let start = FIXTURES[format_index(dtype)].offset + index * size;
    let mut bytes = [0; 8];
    bytes[..size].copy_from_slice(&FIXTURE_BYTES[start..start + size]);
    u64::from_le_bytes(bytes)
}
fn scalar_start() -> usize {
    POOL_LENGTH + 4 * POOL_LENGTH * POOL_LENGTH + 4 * POOL_LENGTH
}
/// The immutable 32-value pool. Flex32 deliberately shares F32 value bits.
pub fn pool(dtype: DType) -> Vec<u64> {
    (0..POOL_LENGTH).map(|i| word(dtype, i)).collect()
}
/// Direct backend scalars and independently generated staged normalized bits.
/// Float values are exact F64 bit patterns; Int/UInt pass through RNE F64,
/// then F32, then half/BF16 as required by the pinned InputScalar contract.
pub fn scalar_fixtures(dtype: DType) -> Vec<ScalarFixture> {
    SCALARS[format_index(dtype)]
        .iter()
        .enumerate()
        .map(|(i, &scalar)| ScalarFixture {
            scalar,
            bits: word(dtype, scalar_start() + i),
        })
        .collect()
}
pub fn chain_fixtures(dtype: DType) -> &'static [ChainFixture] {
    CHAINS[format_index(dtype)]
}
/// Look up independently generated bits within the fixture domain.
///
/// Supports every pool/pool binary pair, every pool unary input, and every
/// pool/direct-scalar pair. Returns None outside that finite domain. Unary rhs
/// is ignored. Native raw tests may reuse these bits without production helpers.
pub fn expected_bits(dtype: DType, operation: Operation, lhs: u64, rhs: u64) -> Option<u64> {
    if !DTYPES.contains(&dtype) {
        return None;
    }
    let a = (0..POOL_LENGTH).find(|&i| word(dtype, i) == lhs)?;
    let op = operation as usize;
    if operation.unary() {
        return Some(word(dtype, POOL_LENGTH + 4 * POOL_LENGTH * POOL_LENGTH + (op - 4) * POOL_LENGTH + a));
    }
    if let Some(b) = (0..POOL_LENGTH).find(|&i| word(dtype, i) == rhs) {
        return Some(word(dtype, POOL_LENGTH + (op * POOL_LENGTH + a) * POOL_LENGTH + b));
    }
    let scalar_count = FIXTURES[format_index(dtype)].scalar_count;
    let scalar = (0..scalar_count).find(|&i| word(dtype, scalar_start() + i) == rhs)?;
    Some(word(dtype, scalar_start() + scalar_count + (scalar * 4 + op) * POOL_LENGTH + a))
}
pub fn is_nan(dtype: DType, bits: u64) -> bool {
    let (exponent, fraction) = match dtype {
        DType::F64 => (0x7ff0_0000_0000_0000, 0x000f_ffff_ffff_ffff),
        DType::F32 | DType::Flex32 => (0x7f80_0000, 0x007f_ffff),
        DType::F16 => (0x7c00, 0x03ff),
        DType::BF16 => (0x7f80, 0x007f),
        _ => return false,
    };
    bits & exponent == exponent && bits & fraction != 0
}
pub fn is_finite(dtype: DType, bits: u64) -> bool {
    let exponent = match dtype {
        DType::F64 => 0x7ff0_0000_0000_0000,
        DType::F32 | DType::Flex32 => 0x7f80_0000,
        DType::F16 => 0x7c00,
        DType::BF16 => 0x7f80,
        _ => return false,
    };
    bits & exponent != exponent
}
pub fn pack(dtype: DType, values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()[..dtype.size()].to_vec()).collect()
}
pub fn unpack(dtype: DType, bytes: &[u8]) -> Vec<u64> {
    assert!(bytes.len().is_multiple_of(dtype.size()));
    bytes
        .chunks_exact(dtype.size())
        .map(|chunk| {
            let mut word = [0; 8];
            word[..dtype.size()].copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Values {
    pub shape: Vec<usize>,
    pub values: Vec<u64>,
}
fn coordinates(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    result
}
fn offset(shape: &[usize], coordinates: &[usize]) -> usize {
    assert_eq!(shape.len(), coordinates.len());
    shape.iter().zip(coordinates).fold(0, |n, (&size, &at)| {
        assert!(at < size);
        n * size + at
    })
}
impl Values {
    pub fn new(shape: &[usize], values: Vec<u64>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self { shape: shape.to_vec(), values }
    }
    fn mapped(&self, shape: &[usize], map: impl Fn(&[usize]) -> Vec<usize>) -> Self {
        Self::new(
            shape,
            (0..shape.iter().product())
                .map(|i| self.values[offset(&self.shape, &map(&coordinates(shape, i)))])
                .collect(),
        )
    }
    pub fn slice(&self, starts: &[usize], steps: &[usize], shape: &[usize]) -> Self {
        self.mapped(shape, |c| c.iter().enumerate().map(|(axis, &i)| starts[axis] + steps[axis] * i).collect())
    }
    pub fn permute(&self, axes: &[usize]) -> Self {
        self.mapped(&axes.iter().map(|&axis| self.shape[axis]).collect::<Vec<_>>(), |c| {
            let mut original = vec![0; axes.len()];
            for (&axis, &i) in axes.iter().zip(c) {
                original[axis] = i;
            }
            original
        })
    }
    pub fn flip(&self, axes: &[usize]) -> Self {
        self.mapped(&self.shape, |c| {
            c.iter()
                .enumerate()
                .map(|(axis, &i)| if axes.contains(&axis) { self.shape[axis] - 1 - i } else { i })
                .collect()
        })
    }
    pub fn expand(&self, shape: &[usize]) -> Self {
        assert_eq!(broadcast_shape(&self.shape, shape), shape);
        Self::new(shape, (0..shape.iter().product()).map(|i| self.broadcast_at(shape, i)).collect())
    }
    fn broadcast_at(&self, output: &[usize], linear: usize) -> u64 {
        let c = coordinates(output, linear);
        let skip = output.len() - self.shape.len();
        let own: Vec<_> = self
            .shape
            .iter()
            .enumerate()
            .map(|(axis, &size)| if size == 1 { 0 } else { c[skip + axis] })
            .collect();
        self.values[offset(&self.shape, &own)]
    }
}
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    let rank = lhs.len().max(rhs.len());
    (0..rank)
        .map(|axis| {
            let a = if axis + lhs.len() >= rank { lhs[axis + lhs.len() - rank] } else { 1 };
            let b = if axis + rhs.len() >= rank { rhs[axis + rhs.len() - rank] } else { 1 };
            assert!(a == b || a == 1 || b == 1, "incompatible float fixture broadcast");
            if a == 1 { b } else { a }
        })
        .collect()
}

#[derive(Clone, Debug)]
enum View {
    Slice { starts: Vec<usize>, steps: Vec<usize>, shape: Vec<usize> },
    Permute(Vec<usize>),
    Flip(Vec<usize>),
    Expand(Vec<usize>),
}
#[derive(Clone, Debug)]
struct Input {
    dtype: DType,
    root: Values,
    views: Vec<View>,
}
impl Input {
    fn shaped(dtype: DType, shape: &[usize], values: Vec<u64>) -> Self {
        Self {
            dtype,
            root: Values::new(shape, values),
            views: vec![],
        }
    }
    fn flat(dtype: DType, values: Vec<u64>) -> Self {
        Self::shaped(dtype, &[values.len()], values)
    }
    fn logical(&self) -> Values {
        self.views.iter().fold(self.root.clone(), |data, view| match view {
            View::Slice { starts, steps, shape } => data.slice(starts, steps, shape),
            View::Permute(axes) => data.permute(axes),
            View::Flip(axes) => data.flip(axes),
            View::Expand(shape) => data.expand(shape),
        })
    }
    fn build<B: BurnBackend>(&self, device: &B::Device) -> (FloatTensor<B>, FloatTensor<B>) {
        let source = B::float_from_data(
            TensorData::from_bytes_vec(pack(self.dtype, &self.root.values), self.root.shape.clone(), self.dtype),
            device,
        );
        let mut tensor = source.clone();
        for view in &self.views {
            tensor = match view {
                View::Slice { starts, steps, shape } => {
                    let slices: Vec<_> = starts
                        .iter()
                        .zip(steps)
                        .zip(shape)
                        .map(|((&start, &step), &length)| Slice::new(start as isize, Some((start + step * length) as isize), step as isize))
                        .collect();
                    B::float_slice(tensor, &slices)
                }
                View::Permute(axes) => B::float_permute(tensor, axes),
                View::Flip(axes) => B::float_flip(tensor, axes),
                View::Expand(shape) => B::float_expand(tensor, shape.clone().into()),
            };
        }
        (source, tensor)
    }
}
#[derive(Clone, Debug)]
enum Right {
    Tensor(Input),
    Scalar(ScalarFixture),
    None,
}
#[derive(Clone, Debug)]
struct Case {
    label: &'static str,
    operation: Operation,
    lhs: Input,
    rhs: Right,
    high_level: bool,
}
impl Case {
    fn expected(&self) -> Values {
        let lhs = self.lhs.logical();
        let right = if let Right::Tensor(input) = &self.rhs { Some(input.logical()) } else { None };
        let shape = right.as_ref().map_or_else(|| lhs.shape.clone(), |r| broadcast_shape(&lhs.shape, &r.shape));
        let values = (0..shape.iter().product())
            .map(|i| {
                let a = lhs.broadcast_at(&shape, i);
                let b = match &self.rhs {
                    Right::Tensor(_) => right.as_ref().unwrap().broadcast_at(&shape, i),
                    Right::Scalar(scalar) => scalar.bits,
                    Right::None => 0,
                };
                expected_bits(self.lhs.dtype, self.operation, a, b).expect("fixture covers public case")
            })
            .collect();
        Values::new(&shape, values)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    /// Primitive entry calls, including every chain step and high-level call.
    pub operations: [usize; 8],
    pub tensor_cases: usize,
    pub scalar_cases: usize,
    pub unary_cases: usize,
    pub high_level_cases: usize,
    pub chain_cases: usize,
    pub chain_steps: usize,
    pub source_checks: usize,
    pub alias_checks: usize,
    pub poison_checks: usize,
    /// Result values, including every independently checked chain intermediate.
    pub values: usize,
    pub exact_values: usize,
    pub nan_values: usize,
}
impl Summary {
    pub fn cases(&self) -> usize {
        self.tensor_cases + self.scalar_cases + self.unary_cases + self.high_level_cases + self.chain_cases
    }
    fn results(&mut self, dtype: DType, values: &[u64]) {
        let nans = values.iter().filter(|&&x| is_nan(dtype, x)).count();
        self.values += values.len();
        self.nan_values += nans;
        self.exact_values += values.len() - nans;
    }
    fn record(&mut self, case: &Case) {
        self.operations[case.operation as usize] += 1;
        if case.high_level {
            self.high_level_cases += 1;
        } else {
            match case.rhs {
                Right::Tensor(_) => self.tensor_cases += 1,
                Right::Scalar(_) => self.scalar_cases += 1,
                Right::None => self.unary_cases += 1,
            }
        }
        let sources = 1 + usize::from(matches!(case.rhs, Right::Tensor(_)));
        self.source_checks += sources;
        self.alias_checks += sources;
        self.poison_checks += 1;
        self.results(case.lhs.dtype, &case.expected().values);
    }
    fn chain(&mut self, dtype: DType, chain: &ChainFixture) {
        self.chain_cases += 1;
        self.chain_steps += chain.steps.len();
        self.source_checks += 1 + chain.steps.iter().filter(|s| !s.operation.unary()).count();
        self.alias_checks += 1;
        self.poison_checks += 1;
        for step in chain.steps {
            self.operations[step.operation as usize] += 1;
            self.results(dtype, &[step.expected]);
        }
    }
}

// Exactly 64 reviewed pairs. All pool values occur in each tensor vector;
// directed pairs cover cancellation, midpoint parity, overflow, subnormals,
// invalid arithmetic, signed zero, both noncommutative operand orders and FMA.
const PAIRS: [(usize, usize); 64] = {
    let mut pairs = [(0, 0); 64];
    let mut i = 0;
    while i < 32 {
        pairs[i] = (i, (i * 7 + 3) % 32);
        i += 1;
    }
    let directed = [
        (2, 10),
        (8, 10),
        (2, 11),
        (3, 10),
        (2, 3),
        (3, 2),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (22, 22),
        (22, 23),
        (0, 22),
        (18, 12),
        (16, 12),
        (12, 4),
        (14, 4),
        (20, 4),
        (20, 2),
        (2, 6),
        (6, 2),
        (28, 29),
        (29, 28),
        (2, 22),
        (3, 23),
        (24, 2),
        (26, 3),
        (20, 20),
        (12, 20),
    ];
    i = 0;
    while i < 32 {
        pairs[32 + i] = directed[i];
        i += 1;
    }
    pairs
};

fn selected(dtype: DType, options: &Options) -> bool {
    options.dtype.is_none_or(|only| only == dtype) && !(options.no_flex32 && dtype == DType::Flex32)
}
fn selected_operation(operation: Operation, options: &Options) -> bool {
    options.operation.is_none_or(|only| only == operation)
}
fn finite_pair(dtype: DType, operation: Operation, a: u64, b: u64) -> bool {
    is_finite(dtype, a) && (operation.unary() || is_finite(dtype, b)) && is_finite(dtype, expected_bits(dtype, operation, a, b).expect("finite fixture lookup"))
}
fn flat_case(dtype: DType, operation: Operation, length: usize, scalar: Option<ScalarFixture>, finite: bool, label: &'static str) -> Option<Case> {
    let values = pool(dtype);
    let pairs: Vec<_> = PAIRS
        .iter()
        .map(|&(a, b)| (values[a], scalar.map_or(values[b], |s| s.bits)))
        .filter(|&(a, b)| !finite || finite_pair(dtype, operation, a, b))
        .collect();
    if pairs.is_empty() {
        return None;
    }
    let lhs = Input::flat(dtype, (0..length).map(|i| pairs[i % pairs.len()].0).collect());
    let rhs = if operation.unary() {
        Right::None
    } else if let Some(scalar) = scalar {
        Right::Scalar(scalar)
    } else {
        Right::Tensor(Input::flat(dtype, (0..length).map(|i| pairs[i % pairs.len()].1).collect()))
    };
    Some(Case {
        label,
        operation,
        lhs,
        rhs,
        high_level: false,
    })
}
fn geometry(dtype: DType, operation: Operation, scalar: bool, geometry: usize, finite: bool) -> Case {
    let p = pool(dtype);
    // Geometry isolates addressing with safe finite operands in both modes;
    // IEEE boundaries use the flat vectors rather than accidental view slots.
    let value = |i: usize, right: bool| p[2 + (i * if right { 5 } else { 1 } + usize::from(right)) % 6];
    let (lhs, rhs, label) = match geometry {
        0 => (Input::shaped(dtype, &[], vec![p[6]]), Input::shaped(dtype, &[], vec![p[4]]), "rank zero"),
        1 => {
            let strided = |right: bool| {
                let length = 17;
                let step = if right { 3 } else { 2 };
                let start = if right { 3 } else { 1 };
                let mut root = vec![p[24]; start + step * length + 5];
                for i in 0..length {
                    root[start + step * i] = value(i, right);
                }
                let mut input = Input::flat(dtype, root);
                input.views.push(View::Slice {
                    starts: vec![start],
                    steps: vec![step],
                    shape: vec![length],
                });
                input
            };
            (strided(false), strided(true), "odd offset and asymmetric positive steps")
        }
        2 => {
            let mut lhs = Input::shaped(dtype, &[3, 5], (0..15).map(|i| value(i, false)).collect());
            lhs.views.push(View::Permute(vec![1, 0]));
            let mut rhs = Input::shaped(dtype, &[1, 3], (0..3).map(|i| value(i, true)).collect());
            rhs.views.push(View::Flip(vec![1]));
            (lhs, rhs, "permuted distinct operands and rhs only broadcast")
        }
        3 => {
            let mut lhs = Input::shaped(dtype, &[1, 7], (0..7).map(|i| value(i, false)).collect());
            lhs.views.push(View::Flip(vec![1]));
            let mut rhs = Input::shaped(dtype, &[3, 7], (0..21).map(|i| value(i, true)).collect());
            rhs.views.push(View::Flip(vec![0, 1]));
            (lhs, rhs, "negative strides and lhs only broadcast")
        }
        4 => {
            let mut lhs = Input::shaped(dtype, &[2, 1, 3], (0..6).map(|i| value(i, false)).collect());
            let rhs = Input::shaped(dtype, &[1, 4, 1], (0..4).map(|i| value(i, true)).collect());
            if scalar || operation.unary() {
                lhs.views.push(View::Expand(vec![2, 4, 3]));
            }
            (lhs, rhs, "both side broadcast or zero stride expanded reader")
        }
        5 => {
            let shape = [1, 2, 1, 2, 1, 2, 1, 3];
            let mut lhs = Input::shaped(dtype, &[1, 2, 1, 1, 1, 2, 1, 3], (0..12).map(|i| value(i, false)).collect());
            lhs.views.push(View::Flip(vec![7]));
            lhs.views.push(View::Expand(shape.to_vec()));
            let mut rhs = Input::shaped(dtype, &[1, 1, 1, 2, 1, 1, 1, 3], (0..6).map(|i| value(i, true)).collect());
            rhs.views.push(View::Flip(vec![3]));
            rhs.views.push(View::Expand(shape.to_vec()));
            (lhs, rhs, "rank eight reversed singleton expansions")
        }
        _ => unreachable!(),
    };
    let rhs = if operation.unary() {
        Right::None
    } else if scalar {
        Right::Scalar(scalar_fixtures(dtype)[6])
    } else {
        Right::Tensor(rhs)
    };
    let case = Case {
        label,
        operation,
        lhs,
        rhs,
        high_level: false,
    };
    assert!(!finite || case.expected().values.iter().all(|&v| is_finite(dtype, v)));
    case
}
fn cases(dtype: DType, options: &Options) -> Vec<Case> {
    let mut result = Vec::new();
    let scalars = scalar_fixtures(dtype);
    for operation in Operation::ALL {
        if !selected_operation(operation, options) {
            continue;
        }
        for length in LENGTHS {
            if let Some(case) = flat_case(dtype, operation, length, None, options.finite, "bounded tensor or unary pool") {
                result.push(case);
            }
        }
        if !operation.unary() {
            // Eight base scalar specifications; three IEEE scalars and all
            // precision witnesses add bounded cases. Float(1) needs one Add.
            for (i, &scalar) in scalars.iter().enumerate() {
                if i == 11 && operation != Operation::Add {
                    continue;
                }
                if let Some(case) = flat_case(dtype, operation, 257, Some(scalar), options.finite, "direct backend staged Scalar") {
                    result.push(case);
                }
            }
        }
        for view in 0..6 {
            if options.skip_rank_zero && view == 0 {
                continue;
            }
            result.push(geometry(dtype, operation, false, view, options.finite));
            if !operation.unary() {
                result.push(geometry(dtype, operation, true, view, options.finite));
            }
        }
    }
    if dtype == DType::F64 {
        for (operation, witnesses) in [
            (Operation::Add, &[12, 13, 14, 15][..]),
            (Operation::Sub, &[13, 15][..]),
            (Operation::Mul, &[13, 14, 15][..]),
            (Operation::Div, &[13, 14, 15][..]),
        ] {
            if selected_operation(operation, options) {
                for &i in witnesses {
                    if let Some(mut case) = flat_case(dtype, operation, 32, Some(scalars[i]), options.finite, "high level F64 scalar precision") {
                        case.high_level = true;
                        result.push(case);
                    }
                }
            }
        }
    }
    result
}
fn selected_chains(dtype: DType, options: &Options) -> impl Iterator<Item = &'static ChainFixture> {
    // An --op diagnostic executes only that operation, so mixed chains are
    // explicitly omitted rather than silently calling an unselected operation.
    chain_fixtures(dtype).iter().filter(move |chain| {
        options.operation.is_none()
            && (!options.finite || (is_finite(dtype, chain.initial) && chain.steps.iter().all(|s| is_finite(dtype, s.rhs) && is_finite(dtype, s.expected))))
    })
}
/// Enumerate the actual selected corpus without constructing any device.
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if selected(dtype, options) {
            summary.dtypes += 1;
            for case in cases(dtype, options) {
                summary.record(&case);
            }
            for chain in selected_chains(dtype, options) {
                summary.chain(dtype, chain);
            }
        }
    }
    summary
}

fn apply_tensor<B: BurnBackend>(op: Operation, lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B> {
    match op {
        Operation::Add => B::float_add(lhs, rhs),
        Operation::Sub => B::float_sub(lhs, rhs),
        Operation::Mul => B::float_mul(lhs, rhs),
        Operation::Div => B::float_div(lhs, rhs),
        _ => unreachable!(),
    }
}
fn apply_scalar<B: BurnBackend>(op: Operation, lhs: FloatTensor<B>, rhs: Scalar) -> FloatTensor<B> {
    match op {
        Operation::Add => B::float_add_scalar(lhs, rhs),
        Operation::Sub => B::float_sub_scalar(lhs, rhs),
        Operation::Mul => B::float_mul_scalar(lhs, rhs),
        Operation::Div => B::float_div_scalar(lhs, rhs),
        _ => unreachable!(),
    }
}
fn apply_unary<B: BurnBackend>(op: Operation, lhs: FloatTensor<B>) -> FloatTensor<B> {
    match op {
        Operation::Neg => B::float_neg(lhs),
        Operation::Abs => B::float_abs(lhs),
        Operation::Recip => B::float_recip(lhs),
        Operation::Sign => B::float_sign(lhs),
        _ => unreachable!(),
    }
}
async fn check<B: BurnBackend>(tensor: FloatTensor<B>, dtype: DType, expected: &Values, raw: bool) -> Result<(), String> {
    if tensor.dtype() != dtype || tensor.shape().as_slice() != expected.shape {
        return Err(format!(
            "metadata {:?} {:?}, expected {dtype:?} {:?}",
            tensor.dtype(),
            tensor.shape(),
            expected.shape
        ));
    }
    let data = B::float_into_data(tensor).await.map_err(|e| format!("read: {e:?}"))?;
    if data.dtype() != dtype || data.shape().as_slice() != expected.shape || data.as_bytes().len() != expected.values.len() * dtype.size() {
        return Err(format!(
            "read storage metadata {:?} {:?} bytes={}",
            data.dtype(),
            data.shape(),
            data.as_bytes().len()
        ));
    }
    for (i, (actual, &wanted)) in unpack(dtype, data.as_bytes()).into_iter().zip(&expected.values).enumerate() {
        if actual != wanted && (raw || !is_nan(dtype, wanted) || !is_nan(dtype, actual)) {
            return Err(format!(
                "element={i} actual={actual:#018x} expected={wanted:#018x} nan_classification={}",
                !raw && is_nan(dtype, wanted)
            ));
        }
    }
    Ok(())
}
fn poison(dtype: DType) -> Input {
    let p = pool(dtype);
    Input::flat(dtype, (0..37).map(|i| p[24 + i % 4]).collect())
}
async fn run_case(device: &Device, case: &Case, summary: &mut Summary) -> Result<(), String> {
    let dtype = case.lhs.dtype;
    let expected = case.expected();
    let (source, lhs) = if case.high_level {
        // An explicit dtype prevents the default F32 creation path from
        // destroying F64 witnesses before the scalar method runs.
        let tensor = Tensor::<1>::from_data(
            TensorData::from_bytes_vec(pack(dtype, &case.lhs.root.values), case.lhs.root.shape.clone(), dtype),
            TensorCreationOptions::new(device.clone()).with_dtype(dtype),
        );
        (tensor.clone().into_dispatch(), tensor.into_dispatch())
    } else {
        case.lhs.build::<Dispatch>(device.as_dispatch())
    };
    let alias = lhs.clone();
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build::<Dispatch>(device.as_dispatch());
    let mut retained_rhs = None;
    let output = if case.high_level {
        let Right::Scalar(scalar) = case.rhs else { unreachable!() };
        let Scalar::Float(value) = scalar.scalar else { unreachable!() };
        let tensor = Tensor::<1>::from_dispatch(lhs);
        let result = match case.operation {
            Operation::Add => tensor.add_scalar(value),
            Operation::Sub => tensor.sub_scalar(value),
            Operation::Mul => tensor.mul_scalar(value),
            Operation::Div => tensor.div_scalar(value),
            _ => unreachable!(),
        };
        result.into_dispatch()
    } else {
        match &case.rhs {
            Right::Tensor(rhs) => {
                let (source, tensor) = rhs.build::<Dispatch>(device.as_dispatch());
                retained_rhs = Some((source, tensor.clone(), rhs));
                apply_tensor::<Dispatch>(case.operation, lhs, tensor)
            }
            Right::Scalar(scalar) => apply_scalar::<Dispatch>(case.operation, lhs, scalar.scalar),
            Right::None => apply_unary::<Dispatch>(case.operation, lhs),
        }
    };
    check::<Dispatch>(output, dtype, &expected, false).await?;
    check::<Dispatch>(source, dtype, &case.lhs.root, true)
        .await
        .map_err(|e| format!("lhs source preservation: {e}"))?;
    check::<Dispatch>(alias, dtype, &case.lhs.logical(), true)
        .await
        .map_err(|e| format!("lhs alias preservation: {e}"))?;
    if let Some((source, alias, rhs)) = retained_rhs {
        check::<Dispatch>(source, dtype, &rhs.root, true)
            .await
            .map_err(|e| format!("rhs source preservation: {e}"))?;
        check::<Dispatch>(alias, dtype, &rhs.logical(), true)
            .await
            .map_err(|e| format!("rhs alias preservation: {e}"))?;
    }
    check::<Dispatch>(retained_poison, dtype, &poison.root, true)
        .await
        .map_err(|e| format!("retained poison preservation: {e}"))?;
    summary.record(case);
    Ok(())
}
async fn run_chain(device: &Device, dtype: DType, chain: &ChainFixture, summary: &mut Summary) -> Result<(), String> {
    let input = Input::flat(dtype, vec![chain.initial]);
    let (source, mut current) = input.build::<Dispatch>(device.as_dispatch());
    let alias = current.clone();
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build::<Dispatch>(device.as_dispatch());
    let mut operands = Vec::new();
    for (i, step) in chain.steps.iter().enumerate() {
        current = if step.operation.unary() {
            apply_unary::<Dispatch>(step.operation, current)
        } else {
            let rhs = Input::flat(dtype, vec![step.rhs]);
            let (retained, tensor) = rhs.build::<Dispatch>(device.as_dispatch());
            operands.push((retained, rhs));
            apply_tensor::<Dispatch>(step.operation, current, tensor)
        };
        // Read every public destination before constructing the next operation.
        // This pins materialization and prevents cross-primitive FMA contraction.
        check::<Dispatch>(current.clone(), dtype, &Values::new(&[1], vec![step.expected]), false)
            .await
            .map_err(|e| format!("chain step {i} {:?}: {e}", step.operation))?;
    }
    drop(current);
    check::<Dispatch>(source, dtype, &input.root, true).await?;
    check::<Dispatch>(alias, dtype, &input.root, true).await?;
    for (source, operand) in operands {
        check::<Dispatch>(source, dtype, &operand.root, true).await?;
    }
    check::<Dispatch>(retained_poison, dtype, &poison.root, true).await?;
    summary.chain(dtype, chain);
    Ok(())
}
pub async fn run(device: &Device, options: &Options) -> Result<Summary, String> {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if !selected(dtype, options) {
            continue;
        }
        summary.dtypes += 1;
        for (i, case) in cases(dtype, options).into_iter().enumerate() {
            run_case(device, &case, &mut summary)
                .await
                .map_err(|e| format!("float case={i} {dtype:?} {:?} {}: {e}", case.operation, case.label))?;
        }
        for (i, chain) in selected_chains(dtype, options).enumerate() {
            run_chain(device, dtype, chain, &mut summary)
                .await
                .map_err(|e| format!("float chain={i} {dtype:?}: {e}"))?;
        }
    }
    let expected = expected_summary(options);
    if summary != expected {
        return Err(format!("float corpus totals {summary:?}, expected {expected:?}"));
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_bounded_and_every_selected_expectation_is_in_the_fixture_domain() {
        let full = expected_summary(&Options::default());
        assert_eq!(full.cases(), 900);
        assert_eq!(full.chain_cases, 15);
        assert_eq!(full.high_level_cases, 12);
        assert_eq!(full.chain_steps, 40);
        assert_eq!(full.exact_values + full.nan_values, full.values);
        assert_eq!(full.operations.iter().sum::<usize>(), full.cases() - full.chain_cases + full.chain_steps);
        for finite in [false, true] {
            let options = Options { finite, ..Options::default() };
            for dtype in DTYPES {
                for case in cases(dtype, &options) {
                    let expected = case.expected();
                    assert!(expected.values.len() <= 257);
                    if finite {
                        assert!(expected.values.iter().all(|&x| is_finite(dtype, x)));
                        assert!(case.lhs.logical().values.iter().all(|&x| is_finite(dtype, x)));
                        match &case.rhs {
                            Right::Tensor(rhs) => assert!(rhs.logical().values.iter().all(|&x| is_finite(dtype, x))),
                            Right::Scalar(rhs) => assert!(is_finite(dtype, rhs.bits)),
                            Right::None => {}
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn diagnostic_filters_remove_real_work_and_do_not_execute_other_operations() {
        let full = expected_summary(&Options::default());
        let finite = expected_summary(&Options {
            finite: true,
            ..Options::default()
        });
        assert!(finite.cases() < full.cases());
        assert_eq!(finite.nan_values, 0);
        assert!(finite.values < full.values);
        let no_rank_zero = expected_summary(&Options {
            skip_rank_zero: true,
            ..Options::default()
        });
        assert_eq!(full.cases() - no_rank_zero.cases(), 60);
        for dtype in DTYPES {
            assert!(
                cases(
                    dtype,
                    &Options {
                        skip_rank_zero: true,
                        ..Options::default()
                    }
                )
                .iter()
                .all(|c| !c.lhs.logical().shape.is_empty())
            );
        }
        let no_flex = expected_summary(&Options {
            no_flex32: true,
            ..Options::default()
        });
        let flex = expected_summary(&Options {
            dtype: Some(DType::Flex32),
            ..Options::default()
        });
        assert_eq!(full.cases() - no_flex.cases(), flex.cases());
        assert_eq!(full.values - no_flex.values, flex.values);
        assert_eq!(no_flex.dtypes, 4);
        for operation in Operation::ALL {
            let s = expected_summary(&Options {
                operation: Some(operation),
                ..Options::default()
            });
            assert_eq!(s.chain_cases, 0);
            assert_eq!(s.operations[operation as usize], s.cases());
            assert!(s.operations.iter().enumerate().all(|(i, &n)| i == operation as usize || n == 0));
        }
    }

    #[test]
    fn views_cover_broadcast_direction_rank_and_poisoned_source_slots() {
        for dtype in DTYPES {
            let p = pool(dtype);
            let strided = geometry(dtype, Operation::Div, false, 1, false);
            let rhs = match &strided.rhs {
                Right::Tensor(rhs) => rhs,
                _ => unreachable!(),
            };
            assert_eq!(strided.lhs.logical().shape, [17]);
            assert_eq!(rhs.logical().shape, [17]);
            assert_eq!(strided.lhs.root.values[0], p[24]);
            assert_eq!(strided.lhs.root.values[1], p[2]);
            assert_eq!(strided.lhs.root.values[2], p[24]);
            for (g, left, right, output) in [
                (0, vec![], vec![], vec![]),
                (2, vec![5, 3], vec![1, 3], vec![5, 3]),
                (3, vec![1, 7], vec![3, 7], vec![3, 7]),
                (4, vec![2, 1, 3], vec![1, 4, 1], vec![2, 4, 3]),
                (5, vec![1, 2, 1, 2, 1, 2, 1, 3], vec![1, 2, 1, 2, 1, 2, 1, 3], vec![1, 2, 1, 2, 1, 2, 1, 3]),
            ] {
                let case = geometry(dtype, Operation::Sub, false, g, false);
                assert_eq!(case.lhs.logical().shape, left);
                let Right::Tensor(rhs) = &case.rhs else { unreachable!() };
                assert_eq!(rhs.logical().shape, right);
                assert_eq!(case.expected().shape, output);
            }
        }
    }
}
