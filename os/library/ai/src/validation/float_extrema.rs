//! Exact stored-bit extrema policy and bounded public conformance corpus.
//!
//! Native chooses the first logical NaN, retaining every source bit, including
//! signaling payloads. Numeric ties, including opposite zeros, keep the first
//! coordinate. Linux source guarantees are weaker for payloads, zero signs and
//! global traversal. Strict failures label those differences without relaxing
//! comparisons. Allocation, padding and residency counters require raw checks.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{
        DType, Dispatch, Shape, TensorData, TensorMetadata,
        ops::{FloatTensorOps, IntTensorOps},
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{Device, Slice, Tensor},
};
use core::cmp::Ordering;

pub const CORPUS_VERSION: u32 = 1;
pub const DTYPES: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
pub const INDEX_DTYPES: [DType; 8] = [DType::I64, DType::I32, DType::I16, DType::I8, DType::U64, DType::U32, DType::U16, DType::U8];
pub const LENGTHS: [usize; 5] = [1, 127, 128, 129, 257];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Min,
    Max,
    ArgMin,
    ArgMax,
    MinWithIndices,
    MaxWithIndices,
}
impl Operation {
    pub const AXIS: [Self; 4] = [Self::Min, Self::Max, Self::ArgMin, Self::ArgMax];
    pub const GLOBAL: [Self; 2] = [Self::Min, Self::Max];
    pub const PAIRS: [Self; 2] = [Self::MinWithIndices, Self::MaxWithIndices];
    pub fn is_pair(self) -> bool {
        matches!(self, Self::MinWithIndices | Self::MaxWithIndices)
    }
    pub fn is_arg(self) -> bool {
        matches!(self, Self::ArgMin | Self::ArgMax)
    }
    pub fn is_min(self) -> bool {
        matches!(self, Self::Min | Self::ArgMin | Self::MinWithIndices)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Axis(usize),
    Global,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub dtype: Option<DType>,
    pub skip_rank_zero: bool,
    pub include_empty: bool,
    /// Opt into Native deferred-error execution. Linux only describes this plan.
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
    /// Original IEEE or integer storage bits, never a widened float surrogate.
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
    IndexDtype,
    Shape,
    Axis,
    GlobalIndex,
    EmptyAxis,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected {
    pub output: Values,
    pub indices: Option<Values>,
    /// Original coordinates, before low-bit index projection.
    pub logical_indices: Vec<usize>,
}

pub fn width(dtype: DType) -> u32 {
    assert!(DTYPES.contains(&dtype) || INDEX_DTYPES.contains(&dtype));
    (dtype.size() * 8) as u32
}
pub fn mask(dtype: DType) -> u64 {
    u64::MAX >> (64 - width(dtype))
}
pub fn pack(dtype: DType, values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes().into_iter().take(dtype.size())).collect()
}
pub fn unpack(dtype: DType, bytes: &[u8]) -> Vec<u64> {
    assert!(bytes.len().is_multiple_of(dtype.size()));
    bytes
        .chunks_exact(dtype.size())
        .map(|bytes| {
            let mut word = [0; 8];
            word[..bytes.len()].copy_from_slice(bytes);
            u64::from_le_bytes(word)
        })
        .collect()
}
pub fn project_index(dtype: DType, coordinate: usize) -> u64 {
    assert!(INDEX_DTYPES.contains(&dtype));
    coordinate as u64 & mask(dtype)
}
/// Interpret exposed signed indices without changing their stored bits.
pub fn decode_index(dtype: DType, bits: u64) -> i128 {
    assert!(INDEX_DTYPES.contains(&dtype));
    let bits = i128::from(bits & mask(dtype));
    if dtype.is_int() && bits & (1i128 << (width(dtype) - 1)) != 0 {
        bits - (1i128 << width(dtype))
    } else {
        bits
    }
}
fn fields(dtype: DType) -> (u32, u32, i32) {
    match dtype {
        DType::F64 => (11, 52, 1023),
        DType::F32 | DType::Flex32 => (8, 23, 127),
        DType::F16 => (5, 10, 15),
        DType::BF16 => (8, 7, 127),
        _ => panic!("IEEE dtype required"),
    }
}
/// A finite number is exactly (-1)^negative * significand * 2^exponent.
/// This decoder and comparator do not construct sortable bit keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoded {
    Nan { negative: bool, payload: u64 },
    Infinity { negative: bool },
    Finite { negative: bool, significand: u64, exponent: i32 },
}
pub fn decode(dtype: DType, bits: u64) -> Decoded {
    let (eb, fb, bias) = fields(dtype);
    let negative = bits & (1u64 << (eb + fb)) != 0;
    let fraction = bits & ((1u64 << fb) - 1);
    let exponent = (bits >> fb) & ((1u64 << eb) - 1);
    if exponent == (1u64 << eb) - 1 {
        if fraction == 0 {
            Decoded::Infinity { negative }
        } else {
            Decoded::Nan { negative, payload: fraction }
        }
    } else {
        Decoded::Finite {
            negative,
            significand: fraction | if exponent == 0 { 0 } else { 1u64 << fb },
            exponent: if exponent == 0 {
                1 - bias - fb as i32
            } else {
                exponent as i32 - bias - fb as i32
            },
        }
    }
}
pub fn is_nan(dtype: DType, bits: u64) -> bool {
    matches!(decode(dtype, bits), Decoded::Nan { .. })
}
/// Exact numeric ordering. NaNs have no numeric order. Both zeros compare equal.
pub fn compare(dtype: DType, a: u64, b: u64) -> Option<Ordering> {
    use Decoded::{Finite, Infinity, Nan};
    let (a, b) = (decode(dtype, a), decode(dtype, b));
    Some(match (a, b) {
        (Nan { .. }, _) | (_, Nan { .. }) => return None,
        (Infinity { negative: a }, Infinity { negative: b }) => b.cmp(&a),
        (Infinity { negative }, _) => {
            if negative {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (_, Infinity { negative }) => {
            if negative {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (
            Finite {
                negative: an,
                significand: a,
                exponent: ae,
            },
            Finite {
                negative: bn,
                significand: b,
                exponent: be,
            },
        ) => {
            if a == 0 && b == 0 {
                return Some(Ordering::Equal);
            }
            if an != bn {
                return Some(if an { Ordering::Less } else { Ordering::Greater });
            }
            // Different leading binary exponents decide order before alignment.
            // Equal leading exponents need at most 52 alignment bits, fitting u128.
            let magnitude = if a == 0 {
                Ordering::Less
            } else if b == 0 {
                Ordering::Greater
            } else {
                let al = 64 - a.leading_zeros();
                let bl = 64 - b.leading_zeros();
                let leading = (ae + al as i32).cmp(&(be + bl as i32));
                if leading != Ordering::Equal {
                    leading
                } else {
                    let base = ae.min(be);
                    (u128::from(a) << (ae - base)).cmp(&(u128::from(b) << (be - base)))
                }
            };
            if an { magnitude.reverse() } else { magnitude }
        }
    })
}
fn elements(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        Some(0)
    } else {
        shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
    }
}
fn coordinates(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    coords
}
fn offset(shape: &[usize], coords: &[usize]) -> usize {
    shape.iter().zip(coords).fold(0, |n, (&d, &c)| n * d + c)
}

pub fn reference(dtype: DType, index_dtype: DType, input: &Values, op: Operation, scope: Scope) -> Result<Expected, ReferenceError> {
    if !DTYPES.contains(&dtype) {
        return Err(ReferenceError::Dtype);
    }
    if !INDEX_DTYPES.contains(&index_dtype) {
        return Err(ReferenceError::IndexDtype);
    }
    if input.shape.len() > 8
        || input.shape.iter().any(|&d| d > i32::MAX as usize)
        || input.values.len() > i32::MAX as usize
        || elements(&input.shape) != Some(input.values.len())
        || input.values.iter().any(|&v| v & !mask(dtype) != 0)
    {
        return Err(ReferenceError::Shape);
    }
    let (shape, length) = match scope {
        Scope::Axis(axis) => {
            let Some(&length) = input.shape.get(axis) else {
                return Err(ReferenceError::Axis);
            };
            let mut shape = input.shape.clone();
            shape[axis] = 1;
            (shape, length)
        }
        Scope::Global => {
            if op.is_arg() || op.is_pair() {
                return Err(ReferenceError::GlobalIndex);
            }
            (vec![1], input.values.len())
        }
    };
    // Validate before the outer-empty shortcut, including [0,0].
    if length == 0 {
        return Err(ReferenceError::EmptyAxis);
    }
    let count = elements(&shape).filter(|&n| n <= i32::MAX as usize).ok_or(ReferenceError::Shape)?;
    let mut output = Vec::with_capacity(count);
    let mut indices = Vec::new();
    let mut logical_indices = Vec::new();
    for row in 0..count {
        let mut coords = coordinates(&shape, row);
        let mut source_at = |i| match scope {
            Scope::Global => i,
            Scope::Axis(axis) => {
                coords[axis] = i;
                offset(&input.shape, &coords)
            }
        };
        let mut winner = 0;
        let mut bits = input.values[source_at(0)];
        for i in 1..length {
            let candidate = input.values[source_at(i)];
            let replace = if is_nan(dtype, bits) {
                false
            } else if is_nan(dtype, candidate) {
                true
            } else {
                compare(dtype, candidate, bits) == Some(if op.is_min() { Ordering::Less } else { Ordering::Greater })
            };
            if replace {
                winner = i;
                bits = candidate;
            }
        }
        logical_indices.push(winner);
        let projected = project_index(index_dtype, winner);
        output.push(if op.is_arg() { projected } else { bits });
        if op.is_pair() {
            indices.push(projected);
        }
    }
    Ok(Expected {
        output: Values::new(&shape, output),
        indices: op.is_pair().then(|| Values::new(&shape, indices)),
        logical_indices,
    })
}
/// Logical view recipes are public so raw and public runners can reuse IDs,
/// bytes and oracle rows without interpreting backend storage.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub operation: Operation,
    pub scope: Scope,
    pub index_dtype: DType,
}
#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub dtype: DType,
    pub input: Input,
    pub steps: Vec<Step>,
    /// Tensor<D> frontend for value operations and default-I32 index operations.
    /// Other requested index dtypes use the public backend trait directly.
    pub public_api: bool,
}
impl Case {
    pub fn expected(&self) -> Result<Vec<Expected>, ReferenceError> {
        let mut input = self.input.logical();
        let mut outputs = Vec::new();
        for step in &self.steps {
            let result = reference(self.dtype, step.index_dtype, &input, step.operation, step.scope)?;
            input = result.output.clone();
            outputs.push(result);
            // An arg result is integer storage and is never a float chain input.
            if step.operation.is_arg() && self.steps.len() > 1 {
                return Err(ReferenceError::Dtype);
            }
        }
        Ok(outputs)
    }
}
fn dense(shape: &[usize], values: Vec<u64>) -> Input {
    Input {
        root: Values::new(shape, values),
        views: vec![],
    }
}
fn add(cases: &mut Vec<Case>, dtype: DType, label: &str, input: Input, operation: Operation, scope: Scope, index_dtype: DType) {
    let public_api = !input.logical().shape.is_empty() && (!(operation.is_arg() || operation.is_pair()) || index_dtype == DType::I32);
    cases.push(Case {
        id: format!("{dtype:?}/{label}/{operation:?}/{scope:?}/{index_dtype:?}"),
        dtype,
        input,
        steps: vec![Step { operation, scope, index_dtype }],
        public_api,
    });
}
/// Deterministic edge encodings. No host floating conversion is used.
pub fn pool(dtype: DType) -> Vec<u64> {
    let (eb, fb, bias) = fields(dtype);
    let sign = 1u64 << (eb + fb);
    let infinity = ((1u64 << eb) - 1) << fb;
    let one = (bias as u64) << fb;
    vec![
        0,
        sign,
        1,
        sign | 1,
        (1 << fb) - 1,
        1 << fb,
        one,
        one + 1,
        one + 2,
        sign | one,
        infinity - 1,
        sign | (infinity - 1),
        infinity,
        sign | infinity,
        infinity | 1,
        sign | infinity | (1 << (fb - 1)) | 3,
    ]
}
// Five independent rows keep numerical witnesses observable even beside NaNs.
fn fixture(dtype: DType, length: usize, seed: usize) -> Input {
    let p = pool(dtype);
    let mut values = vec![p[6]; 5 * length];
    for row in 0..5 {
        let witnesses: &[u64] = match row {
            0 => &p[6..9],
            1 => &p[..2],
            2 => &p[2..6],
            3 => &p[10..14],
            _ => &p[14..16],
        };
        for i in 0..length {
            values[row * length + i] = witnesses[(i + seed) % witnesses.len()];
        }
    }
    dense(&[5, length], values)
}
// Rotate layouts; preserve five independent rows and the reduction length.
fn layout(mut input: Input, rotation: usize) -> (Input, usize) {
    match rotation % 4 {
        0 => {
            input.views.push(View::Flip(vec![1]));
            (input, 1)
        }
        1 => {
            input.views.push(View::Permute(vec![1, 0]));
            (input, 0)
        }
        2 => {
            let length = input.root.shape[1];
            let mut values = vec![input.root.values[0]; 5 * (length * 2 + 1)];
            for row in 0..5 {
                for i in 0..length {
                    values[row * (length * 2 + 1) + 1 + i * 2] = input.root.values[row * length + i];
                }
            }
            (
                Input {
                    root: Values::new(&[5, length * 2 + 1], values),
                    views: vec![View::Slice {
                        starts: vec![0, 1],
                        steps: vec![1, 2],
                        shape: vec![5, length],
                    }],
                },
                1,
            )
        }
        _ => (input, 1),
    }
}
/// At most 126 standalone API rows plus eight resident chains. Layout and
/// numeric stress rotate rather than crossing every format and shape.
pub fn corpus(options: &Options) -> Vec<Case> {
    let mut cases = Vec::new();
    for (f, dtype) in DTYPES.into_iter().enumerate() {
        for (i, operation) in [Operation::Min, Operation::Max, Operation::Min, Operation::Max].into_iter().enumerate() {
            let mut input = fixture(dtype, LENGTHS[(f + i + 1) % 5], i);
            if i == 3 {
                // A numerical global row complements the NaN-bearing global
                // call without multiplying the number of public calls.
                let one = pool(dtype)[6];
                let start = 4 * input.root.shape[1];
                input.root.values[start..].fill(one);
            }
            let (input, axis) = layout(input, f + i);
            add(
                &mut cases,
                dtype,
                &format!("value-{i}"),
                input,
                operation,
                if i < 2 { Scope::Axis(axis) } else { Scope::Global },
                DType::I32,
            );
        }
        for i in 0..4 {
            let (input, axis) = layout(fixture(dtype, LENGTHS[(f + i) % 5], i), f + i);
            add(
                &mut cases,
                dtype,
                &format!("arg-{i}"),
                input,
                if i % 2 == 0 { Operation::ArgMin } else { Operation::ArgMax },
                Scope::Axis(axis),
                INDEX_DTYPES[(f * 3 + i) % 8],
            );
        }
        for i in 0..10 {
            let index_dtype = if i < 8 { INDEX_DTYPES[i] } else { DType::I32 };
            let operation = Operation::PAIRS[(f + i) % 2];
            let (input, axis) = layout(fixture(dtype, LENGTHS[(f + i) % 5], i), f + i);
            add(&mut cases, dtype, &format!("pair-{i}"), input, operation, Scope::Axis(axis), index_dtype);
        }
        // Four empty/error rows per format. Rejection precedes empty output.
        for (label, shape, axis, operation) in [
            ("outer-empty", [0, 3], 1, Operation::PAIRS[f % 2]),
            ("empty-axis", [2, 0], 1, Operation::Min),
            ("both-empty", [0, 0], 1, Operation::MaxWithIndices),
            ("invalid-axis-empty", [0, 3], 2, Operation::ArgMin),
        ] {
            add(
                &mut cases,
                dtype,
                label,
                dense(&shape, vec![]),
                operation,
                Scope::Axis(axis),
                INDEX_DTYPES[(f + 3) % 8],
            );
        }
        add(
            &mut cases,
            dtype,
            "rank-zero-global",
            dense(&[], vec![pool(dtype)[14]]),
            Operation::GLOBAL[f % 2],
            Scope::Global,
            DType::I32,
        );
    }
    add(
        &mut cases,
        DType::F32,
        "rank-zero-axis",
        dense(&[], vec![0]),
        Operation::MaxWithIndices,
        Scope::Axis(0),
        DType::U64,
    );
    for (dtype, operation) in [(DType::F64, Operation::Min), (DType::F32, Operation::Max)] {
        add(&mut cases, dtype, "global-empty", dense(&[0], vec![]), operation, Scope::Global, DType::I32);
    }
    for (i, (winner, index_dtype)) in [(129, DType::I8), (256, DType::U8), (32768, DType::I16), (65536, DType::U16)]
        .into_iter()
        .enumerate()
    {
        let dtype = DTYPES[i];
        let p = pool(dtype);
        let mut values = vec![p[6]; winner + 1];
        values[winner] = p[8];
        add(
            &mut cases,
            dtype,
            &format!("narrow-winner-{winner}"),
            dense(&[winner + 1], values),
            Operation::MaxWithIndices,
            Scope::Axis(0),
            index_dtype,
        );
    }
    for (i, dtype) in [DType::F64, DType::F32].into_iter().enumerate() {
        let p = pool(dtype);
        let mut values = vec![p[6]; 131];
        values[1] = if i == 0 { p[14] } else { p[8] };
        values[129] = if i == 0 { p[15] } else { p[8] };
        let input = Input {
            root: Values::new(&[131], values),
            views: vec![View::Flip(vec![0])],
        };
        add(&mut cases, dtype, "reversed-1-129", input, Operation::PAIRS[i], Scope::Axis(0), DType::I32);
    }
    add(
        &mut cases,
        DType::F16,
        "all-nan",
        dense(&[3], pool(DType::F16)[14..16].iter().copied().chain([pool(DType::F16)[14]]).collect()),
        Operation::MinWithIndices,
        Scope::Axis(0),
        DType::U64,
    );
    add(
        &mut cases,
        DType::BF16,
        "all-infinity",
        dense(&[129], vec![pool(DType::BF16)[12]; 129]),
        Operation::MaxWithIndices,
        Scope::Axis(0),
        DType::I32,
    );
    for i in 0..8 {
        let dtype = DTYPES[i % 5];
        let rank = i + 1;
        let mut shape = vec![1; rank];
        shape[rank - 1] = 3;
        let mut input = dense(&shape, pool(dtype)[6..9].to_vec());
        let mut expanded = shape.clone();
        expanded[0] = if rank == 1 { 3 } else { 2 };
        input.views.push(View::Expand(expanded));
        input.views.push(View::Flip(vec![rank - 1]));
        if rank > 1 {
            input.views.push(View::Permute((0..rank).rev().collect()));
        }
        let axis = 0; // permutation moves the length-three axis to zero
        cases.push(Case {
            id: format!("{dtype:?}/resident-rank-{rank}"),
            dtype,
            input,
            public_api: true,
            steps: vec![
                Step {
                    operation: Operation::MinWithIndices,
                    scope: Scope::Axis(axis),
                    index_dtype: DType::I32,
                },
                Step {
                    operation: Operation::MaxWithIndices,
                    scope: Scope::Axis(rank - 1),
                    index_dtype: DType::I32,
                },
                Step {
                    operation: Operation::Min,
                    scope: Scope::Global,
                    index_dtype: DType::I32,
                },
            ],
        });
    }
    cases.retain(|case| {
        options.dtype.is_none_or(|d| d == case.dtype)
            && (!options.skip_rank_zero || !case.input.root.shape.is_empty())
            && (options.include_empty || !case.input.root.shape.contains(&0))
    });
    cases
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionScope {
    #[default]
    LinuxDefault,
    NativeErrorExtensions,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub options: Options,
    pub scope: ExecutionScope,
    pub dtypes: usize,
    pub cases: usize,
    pub valid_cases: usize,
    pub diagnostic_cases: usize,
    /// Standalone API rows, including diagnostics. Resident chains are separate.
    pub standalone_cases: usize,
    pub steps: usize,
    /// Valid cases routed through Tensor<D> rather than only backend traits.
    pub public_cases: usize,
    pub pair_steps: usize,
    pub chain_cases: usize,
    pub chain_steps: usize,
    pub values: usize,
    pub checks: Checks,
}
/// Logical read/check counts only. No GPU, allocation, padding or kernel counts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Checks {
    pub input: usize,
    pub source: usize,
    pub alias: usize,
    pub output: usize,
    pub chain_index: usize,
    pub recovery: usize,
}
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary {
        options: *options,
        scope: if options.native_error_extensions {
            ExecutionScope::NativeErrorExtensions
        } else {
            ExecutionScope::LinuxDefault
        },
        dtypes: DTYPES.iter().filter(|&&d| options.dtype.is_none_or(|selected| selected == d)).count(),
        ..Summary::default()
    };
    for case in corpus(options) {
        summary.cases += 1;
        summary.standalone_cases += usize::from(case.steps.len() == 1);
        match case.expected() {
            Ok(outputs) => {
                summary.valid_cases += 1;
                summary.public_cases += usize::from(case.public_api);
                summary.steps += outputs.len();
                summary.pair_steps += outputs.iter().filter(|o| o.indices.is_some()).count();
                summary.values += outputs
                    .iter()
                    .map(|o| o.output.values.len() + o.indices.as_ref().map_or(0, |i| i.values.len()))
                    .sum::<usize>();
                summary.checks.output += outputs.len() + outputs.iter().filter(|o| o.indices.is_some()).count();
                if outputs.len() > 1 {
                    summary.chain_cases += 1;
                    summary.chain_steps += outputs.len();
                    summary.checks.chain_index += outputs.iter().filter(|o| o.indices.is_some()).count();
                }
            }
            Err(_) => {
                summary.diagnostic_cases += 1;
                if options.native_error_extensions {
                    summary.checks.recovery += 1;
                }
            }
        }
        if case.expected().is_ok() || options.native_error_extensions {
            summary.checks.input += 1;
            summary.checks.source += 1;
            summary.checks.alias += 1;
        }
    }
    summary
}
fn checks_json(c: &Checks) -> String {
    format!(
        "{{\"input\":{},\"source\":{},\"alias\":{},\"output\":{},\"chain_index\":{},\"recovery\":{}}}",
        c.input, c.source, c.alias, c.output, c.chain_index, c.recovery
    )
}
pub fn summary_json(s: &Summary) -> String {
    let dtype = s.options.dtype.map_or_else(|| String::from("null"), |d| format!("\"{d:?}\""));
    format!(
        concat!(
            "{{\"corpus\":{},\"options\":{{\"dtype\":{},\"skip_rank_zero\":{},\"include_empty\":{},\"native_error_extensions\":{}}},",
            "\"scope\":\"{:?}\",\"bit_policy\":\"strict_native_source_bits\",\"dtypes\":{},\"cases\":{},\"valid_cases\":{},\"diagnostic_cases\":{},",
            "\"standalone_cases\":{},\"steps\":{},\"public_cases\":{},\"pair_steps\":{},\"chain_cases\":{},\"chain_steps\":{},\"values\":{},\"checks\":{}}}"
        ),
        CORPUS_VERSION,
        dtype,
        s.options.skip_rank_zero,
        s.options.include_empty,
        s.options.native_error_extensions,
        s.scope,
        s.dtypes,
        s.cases,
        s.valid_cases,
        s.diagnostic_cases,
        s.standalone_cases,
        s.steps,
        s.public_cases,
        s.pair_steps,
        s.chain_cases,
        s.chain_steps,
        s.values,
        checks_json(&s.checks)
    )
}
#[derive(Clone, Debug)]
pub struct Failure {
    pub case_id: String,
    pub message: String,
}
#[derive(Clone, Debug)]
pub struct Report {
    /// Pure plan. Never inferred from observed successes.
    pub expected: Summary,
    pub attempted: usize,
    pub passed: usize,
    pub skipped_diagnostics: usize,
    pub observed_errors: usize,
    pub failed: usize,
    /// Incremented only after each actual check succeeds.
    pub checks: Checks,
    /// First 24 failures. `failed` retains the full case count.
    pub failures: Vec<Failure>,
}
impl Report {
    pub fn is_success(&self) -> bool {
        self.failed == 0
            && self.failures.is_empty()
            && self.attempted == self.passed
            && self.attempted + self.skipped_diagnostics == self.expected.cases
            && self.observed_errors + self.skipped_diagnostics == self.expected.diagnostic_cases
            && self.passed == self.expected.valid_cases + self.observed_errors
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
        return Err(String::from("report plan does not match selected options and scope"));
    }
    let errors = if options.native_error_extensions {
        report.expected.diagnostic_cases
    } else {
        0
    };
    if report.observed_errors != errors || !report.is_success() {
        return Err(format!(
            "float extrema failed: {}/{} cases; first={:?}",
            report.failed,
            report.attempted,
            report.failures.first()
        ));
    }
    Ok(format!(
        concat!(
            "BURN_FLOAT_EXTREMA_OK corpus={} totals={} attempted={} passed={} skipped_diagnostics={} observed_errors={} ",
            "reference=independent_ieee_significand_exponent bit_policy=strict_native_source_bits source_policy=burn_0.22.1 ",
            "native_extensions=first_nan_all_bits_first_zero index=low_bits_from_original_coordinate ",
            "linux_payload_zero_global_traversal=separate_strict_diagnostics signed_linux=qualification_pending ",
            "physical_padding_allocations_residency=raw_contract gpu_runtime_counters=external_raw_contract scope=float_extrema_forward_only"
        ),
        CORPUS_VERSION,
        summary_json(&report.expected),
        report.attempted,
        report.passed,
        report.skipped_diagnostics,
        report.observed_errors
    ))
}

// The public backend trait carries requested index dtypes without any cast or
// gather through a narrowed index. Value and integer results remain distinct.
#[derive(Clone)]
enum Output {
    Float(FloatTensor<Dispatch>),
    Int(IntTensor<Dispatch>),
}
fn apply(input: FloatTensor<Dispatch>, step: Step) -> (Output, Option<IntTensor<Dispatch>>) {
    let result = match (step.operation, step.scope) {
        (Operation::Min, Scope::Global) => Dispatch::float_min(input),
        (Operation::Max, Scope::Global) => Dispatch::float_max(input),
        (Operation::Min, Scope::Axis(a)) => Dispatch::float_min_dim(input, a),
        (Operation::Max, Scope::Axis(a)) => Dispatch::float_max_dim(input, a),
        (Operation::ArgMin, Scope::Axis(a)) => return (Output::Int(Dispatch::float_argmin(input, a, step.index_dtype.into())), None),
        (Operation::ArgMax, Scope::Axis(a)) => return (Output::Int(Dispatch::float_argmax(input, a, step.index_dtype.into())), None),
        (Operation::MinWithIndices, Scope::Axis(a)) => {
            let (v, i) = Dispatch::float_min_dim_with_indices(input, a, step.index_dtype.into());
            return (Output::Float(v), Some(i));
        }
        (Operation::MaxWithIndices, Scope::Axis(a)) => {
            let (v, i) = Dispatch::float_max_dim_with_indices(input, a, step.index_dtype.into());
            return (Output::Float(v), Some(i));
        }
        _ => unreachable!("no global arg API"),
    };
    (Output::Float(result), None)
}
fn apply_public<const D: usize>(input: FloatTensor<Dispatch>, step: Step) -> (Output, Option<IntTensor<Dispatch>>) {
    assert_eq!(input.shape().num_dims(), D);
    let input = Tensor::<D>::from_dispatch(input);
    macro_rules! output {
        ($t:expr) => {
            (Output::Float($t.into_dispatch()), None)
        };
    }
    match (step.operation, step.scope) {
        (Operation::Min, Scope::Global) => output!(input.min()),
        (Operation::Max, Scope::Global) => output!(input.max()),
        (Operation::Min, Scope::Axis(a)) => output!(input.min_dim(a)),
        (Operation::Max, Scope::Axis(a)) => output!(input.max_dim(a)),
        (Operation::ArgMin, Scope::Axis(a)) => (Output::Int(input.argmin(a).into_dispatch()), None),
        (Operation::ArgMax, Scope::Axis(a)) => (Output::Int(input.argmax(a).into_dispatch()), None),
        (Operation::MinWithIndices, Scope::Axis(a)) => {
            let (v, i) = input.min_dim_with_indices(a);
            (Output::Float(v.into_dispatch()), Some(i.into_dispatch()))
        }
        (Operation::MaxWithIndices, Scope::Axis(a)) => {
            let (v, i) = input.max_dim_with_indices(a);
            (Output::Float(v.into_dispatch()), Some(i.into_dispatch()))
        }
        _ => unreachable!("no global arg API"),
    }
}
fn public_by_rank(input: FloatTensor<Dispatch>, step: Step) -> (Output, Option<IntTensor<Dispatch>>) {
    match input.shape().num_dims() {
        1 => apply_public::<1>(input, step),
        2 => apply_public::<2>(input, step),
        3 => apply_public::<3>(input, step),
        4 => apply_public::<4>(input, step),
        5 => apply_public::<5>(input, step),
        6 => apply_public::<6>(input, step),
        7 => apply_public::<7>(input, step),
        8 => apply_public::<8>(input, step),
        _ => unreachable!("Tensor frontend ranks are 1..8"),
    }
}
fn difference(dtype: DType, actual: u64, expected: u64) -> &'static str {
    if !DTYPES.contains(&dtype) {
        "index_projection"
    } else if is_nan(dtype, actual) && is_nan(dtype, expected) {
        "nan_payload_or_signaling_source_bits"
    } else if compare(dtype, actual, expected) == Some(Ordering::Equal) {
        "zero_sign_source_bits"
    } else {
        "numeric_or_nan_winner"
    }
}
async fn check(output: Output, dtype: DType, expected: &Values) -> Result<(), String> {
    let (actual_dtype, shape) = match &output {
        Output::Float(t) => (t.dtype(), t.shape()),
        Output::Int(t) => (t.dtype(), t.shape()),
    };
    if actual_dtype != dtype || shape.as_slice() != expected.shape {
        return Err(format!("metadata {actual_dtype:?} {shape:?}, expected {dtype:?} {:?}", expected.shape));
    }
    let data = match output {
        Output::Float(t) => Dispatch::float_into_data(t).await,
        Output::Int(t) => Dispatch::int_into_data(t).await,
    }
    .map_err(|e| format!("read: {e:?}"))?;
    if data.dtype() != dtype || data.shape().as_slice() != expected.shape {
        return Err(String::from("read metadata changed"));
    }
    let expected_bytes = pack(dtype, &expected.values);
    if data.as_bytes().len() != expected_bytes.len() {
        return Err(format!("byte length {}, expected {}", data.as_bytes().len(), expected_bytes.len()));
    }
    if let Some(at) = data.as_bytes().iter().zip(&expected_bytes).position(|(a, b)| a != b) {
        let i = at / dtype.size();
        let actual = unpack(dtype, data.as_bytes())[i];
        let expected = expected.values[i];
        return Err(format!(
            "strict_native_bits category={} element={i} actual={actual:#x} expected={expected:#x}; Linux payload/zero/global traversal is separately qualified",
            difference(dtype, actual, expected)
        ));
    }
    Ok(())
}
fn counted(result: Result<(), String>, counter: &mut usize) -> Result<(), String> {
    if result.is_ok() {
        *counter += 1;
    }
    result
}
async fn run_case(device: &Device, case: &Case, expected: &[Expected], checks: &mut Checks) -> Result<(), String> {
    let (source, input) = case.input.build(case.dtype, device);
    let alias = input.clone();
    let initial = counted(check(Output::Float(input.clone()), case.dtype, &case.input.logical()).await, &mut checks.input);
    let mut current = input;
    let mut queued = Vec::new();
    let mut consumers = Vec::new();
    for (step, expected) in case.steps.iter().zip(expected) {
        // Tensor<D> uses the immutable device default integer dtype. Other
        // requested dtypes always take the explicitly typed backend route.
        if case.public_api && (step.operation.is_pair() || step.operation.is_arg()) && DType::from(device.settings().int_dtype) != step.index_dtype {
            return Err(String::from(
                "Tensor index fixture requires device default I32; requested typed APIs use backend traits",
            ));
        }
        let original = current.clone();
        let (output, indices) = if case.public_api {
            public_by_rank(current, *step)
        } else {
            apply(current, *step)
        };
        if case.steps.len() > 1
            && let (Some(indices), Scope::Axis(axis)) = (&indices, step.scope)
        {
            // Chains use representable I32 coordinates only. Narrow-winner
            // pairs never enter this consumer. Queue before the final read.
            consumers.push((Dispatch::float_gather(axis, original, indices.clone()), expected.output.clone()));
        }
        let next = match &output {
            Output::Float(t) => Some(t.clone()),
            Output::Int(_) => None,
        };
        queued.push((output, indices, *step, expected));
        if let Some(next) = next {
            current = next;
        } else {
            break;
        }
    }
    // Every chain dependency is queued before any result is downloaded. Raw
    // counters must still prove resident dispatch and deferred completion.
    let mut result = initial;
    for (output, indices, step, expected) in queued {
        let dtype = if step.operation.is_arg() { step.index_dtype } else { case.dtype };
        let value = counted(check(output, dtype, &expected.output).await, &mut checks.output);
        let index = if let Some(indices) = indices {
            counted(
                check(Output::Int(indices), step.index_dtype, expected.indices.as_ref().expect("pair oracle")).await,
                &mut checks.output,
            )
        } else {
            Ok(())
        };
        result = result.and(value).and(index);
    }
    for (consumer, expected) in consumers {
        let checked = counted(check(Output::Float(consumer), case.dtype, &expected).await, &mut checks.chain_index);
        result = result.and(checked);
    }
    let source = counted(check(Output::Float(source), case.dtype, &case.input.root).await, &mut checks.source);
    let alias = counted(check(Output::Float(alias), case.dtype, &case.input.logical()).await, &mut checks.alias);
    result.and(source).and(alias)
}
async fn expected_error(device: &Device, case: &Case, checks: &mut Checks) -> Result<(), String> {
    let (source, input) = case.input.build(case.dtype, device);
    let alias = input.clone();
    let initial = counted(check(Output::Float(input.clone()), case.dtype, &case.input.logical()).await, &mut checks.input);
    let step = case.steps[0];
    let mut shape = case.input.logical().shape;
    match step.scope {
        Scope::Global => shape = vec![1],
        Scope::Axis(axis) => {
            if let Some(size) = shape.get_mut(axis) {
                *size = 1;
            }
        }
    }
    // Deliberately bypass Tensor<D> assertions to exercise deferred errors.
    let (output, indices) = apply(input, step);
    let dtype = if step.operation.is_arg() { step.index_dtype } else { case.dtype };
    let (metadata, failed) = match output {
        Output::Float(t) => (t.dtype() == dtype && t.shape().as_slice() == shape, Dispatch::float_into_data(t).await.is_err()),
        Output::Int(t) => (t.dtype() == dtype && t.shape().as_slice() == shape, Dispatch::int_into_data(t).await.is_err()),
    };
    let index_failed = if let Some(t) = indices {
        let metadata = t.dtype() == step.index_dtype && t.shape().as_slice() == shape;
        let failed = Dispatch::int_into_data(t).await.is_err();
        metadata && failed
    } else {
        true
    };
    let rejected = if metadata && failed && index_failed {
        Ok(())
    } else {
        Err(String::from("invalid API did not preserve deferred error shape/dtype"))
    };
    // Invalid cases retain the same input/source/alias checks as valid cases.
    let source = counted(check(Output::Float(source), case.dtype, &case.input.root).await, &mut checks.source);
    let alias = counted(check(Output::Float(alias), case.dtype, &case.input.logical()).await, &mut checks.alias);
    let valid = dense(&[1], vec![pool(case.dtype)[6]]);
    let (_, input) = valid.build(case.dtype, device);
    let recovery = counted(
        check(Output::Float(Dispatch::float_min(input)), case.dtype, &valid.root).await,
        &mut checks.recovery,
    );
    initial.and(rejected).and(source).and(alias).and(recovery)
}
/// Run one invalid row in an isolated Linux process. A panic is a diagnostic,
/// never shared conformance success. Pure descriptions do not call this route.
pub async fn linux_error_diagnostic(device: &Device, options: &Options, id: &str) -> Result<(), String> {
    let case = corpus(options)
        .into_iter()
        .find(|c| c.id == id)
        .ok_or_else(|| format!("unknown diagnostic ID {id}"))?;
    if case.expected().is_ok() {
        return Err(String::from("diagnostic requires an invalid API row"));
    }
    let (_, input) = case.input.build(case.dtype, device);
    let (output, indices) = apply(input, case.steps[0]);
    let value = match output {
        Output::Float(t) => Dispatch::float_into_data(t).await,
        Output::Int(t) => Dispatch::int_into_data(t).await,
    };
    let index = if let Some(t) = indices {
        Some(Dispatch::int_into_data(t).await)
    } else {
        None
    };
    Err(format!("Linux diagnostic returned without panic: value={value:?} index={index:?}"))
}
/// No device is constructed by corpus, reference or expected_summary. Native
/// callers opt into recoverable errors. Linux skips invalid rows by default.
pub async fn run(device: &Device, options: &Options) -> Report {
    let mut report = Report {
        expected: expected_summary(options),
        attempted: 0,
        passed: 0,
        skipped_diagnostics: 0,
        observed_errors: 0,
        failed: 0,
        checks: Checks::default(),
        failures: vec![],
    };
    for case in corpus(options) {
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
            Err(error) => report.failure(&case.id, error),
        }
    }
    report
}
