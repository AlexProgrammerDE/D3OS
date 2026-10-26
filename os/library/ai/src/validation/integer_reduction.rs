//! Exact integer reduction policy, independent of backend kernels.
//!
//! Linux observations do not define this oracle. Signed wrapping/narrowing are
//! Native policy until selected Linux hardware checks qualify those cells.
//! Physical empty allocations and command residency belong to the raw contract.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{DType, Dispatch, Shape, TensorData, TensorMetadata, ops::IntTensorOps, tensor::IntTensor},
    tensor::{Device, Int, Slice, Tensor},
};

pub const CORPUS_VERSION: u32 = 1;
pub const DTYPES: [DType; 8] = [DType::I64, DType::I32, DType::I16, DType::I8, DType::U64, DType::U32, DType::U16, DType::U8];
pub const LENGTHS: [usize; 9] = [1, 2, 3, 127, 128, 129, 255, 256, 257];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Sum,
    Product,
    Mean,
    Min,
    Max,
    ArgMin,
    ArgMax,
    MinWithIndices,
    MaxWithIndices,
}
impl Operation {
    pub const AXIS: [Self; 7] = [Self::Sum, Self::Product, Self::Mean, Self::Min, Self::Max, Self::ArgMin, Self::ArgMax];
    pub const GLOBAL: [Self; 5] = [Self::Sum, Self::Product, Self::Mean, Self::Min, Self::Max];
    pub const PAIRS: [Self; 2] = [Self::MinWithIndices, Self::MaxWithIndices];
    pub fn is_pair(self) -> bool {
        matches!(self, Self::MinWithIndices | Self::MaxWithIndices)
    }
    fn is_arg(self) -> bool {
        matches!(self, Self::ArgMin | Self::ArgMax)
    }
    fn is_min(self) -> bool {
        matches!(self, Self::Min | Self::ArgMin | Self::MinWithIndices)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Axis(usize),
    Global,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub dtype: Option<DType>,
    pub skip_rank_zero: bool,
    pub include_empty: bool,
    /// Execute expected deferred errors only on Native. Linux may panic in
    /// these APIs, so its runner only describes these diagnostics.
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
    /// Stored element bits, little endian when packed. Never floating values.
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
    GlobalIndex,
    EmptyAxis,
    ZeroCount,
    CountNotRepresentable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected {
    pub output: Values,
    /// Only combined APIs have a second exposed tensor.
    pub indices: Option<Values>,
    /// Full coordinates before exposed-index narrowing, also for arg APIs.
    pub logical_indices: Option<Vec<usize>>,
}

pub fn width(dtype: DType) -> u32 {
    assert!(DTYPES.contains(&dtype));
    (dtype.size() * 8) as u32
}
pub fn mask(dtype: DType) -> u64 {
    u64::MAX >> (64 - width(dtype))
}
pub fn decode(dtype: DType, bits: u64) -> i128 {
    let bits = i128::from(bits & mask(dtype));
    if dtype.is_int() && bits & (1i128 << (width(dtype) - 1)) != 0 {
        bits - (1i128 << width(dtype))
    } else {
        bits
    }
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

/// Reduce exact stored values with explicit accumulator and store boundaries.
pub fn reference(dtype: DType, input: &Values, op: Operation, scope: Scope) -> Result<Expected, ReferenceError> {
    if !DTYPES.contains(&dtype) {
        return Err(ReferenceError::Dtype);
    }
    if input.shape.len() > 8 || elements(&input.shape) != Some(input.values.len()) {
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
    // Semantic rejection precedes the outer-empty shortcut, including [0,0].
    if length == 0 && !matches!(op, Operation::Sum | Operation::Product) {
        return Err(if scope == Scope::Global && op == Operation::Mean {
            ReferenceError::ZeroCount
        } else {
            ReferenceError::EmptyAxis
        });
    }
    if scope == Scope::Global && op == Operation::Mean {
        let max_count = if dtype.is_int() {
            (1u128 << (width(dtype) - 1)) - 1
        } else {
            u128::from(mask(dtype))
        };
        if length as u128 > max_count {
            return Err(ReferenceError::CountNotRepresentable);
        }
    }
    let count = elements(&shape).ok_or(ReferenceError::Shape)?;
    let mut output = Vec::with_capacity(count);
    let mut indices = Vec::new();
    let mut logical_indices = Vec::new();
    for row in 0..count {
        let mut coords = coordinates(&shape, row);
        let row_values: Vec<_> = (0..length)
            .map(|i| {
                let at = match scope {
                    Scope::Global => i,
                    Scope::Axis(axis) => {
                        coords[axis] = i;
                        offset(&input.shape, &coords)
                    }
                };
                input.values[at] & mask(dtype)
            })
            .collect();
        let result = if matches!(op, Operation::Sum | Operation::Product | Operation::Mean) {
            // Residues make modular arithmetic exact, including U64 products.
            // Sign extension must happen before entering the wider ring.
            let accumulator_width = width(dtype).max(32);
            let modulus = 1u128 << accumulator_width;
            let mut acc = u128::from(op == Operation::Product);
            for &bits in &row_values {
                let value = decode(dtype, bits).rem_euclid(modulus as i128) as u128;
                acc = if op == Operation::Product {
                    acc * value % modulus
                } else {
                    (acc + value) % modulus
                };
            }
            if op == Operation::Mean {
                let (bits, bits_width) = if scope == Scope::Global {
                    (acc & u128::from(mask(dtype)), width(dtype))
                } else {
                    (acc, accumulator_width)
                };
                let signed = if dtype.is_int() && bits & (1u128 << (bits_width - 1)) != 0 {
                    bits as i128 - (1i128 << bits_width)
                } else {
                    bits as i128
                };
                (signed / length as i128) as u64 & mask(dtype)
            } else {
                acc as u64 & mask(dtype)
            }
        } else {
            let mut winner = 0;
            for i in 1..length {
                let (candidate, best) = (decode(dtype, row_values[i]), decode(dtype, row_values[winner]));
                if if op.is_min() { candidate < best } else { candidate > best } {
                    winner = i;
                }
            }
            logical_indices.push(winner);
            let index = winner as u64 & mask(dtype);
            if op.is_pair() {
                indices.push(index);
            }
            if op.is_arg() { index } else { row_values[winner] }
        };
        output.push(result);
    }
    let winners = !matches!(op, Operation::Sum | Operation::Product | Operation::Mean);
    Ok(Expected {
        output: Values::new(&shape, output),
        indices: op.is_pair().then(|| Values::new(&shape, indices)),
        logical_indices: winners.then_some(logical_indices),
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
    fn build(&self, dtype: DType, device: &Device) -> (IntTensor<Dispatch>, IntTensor<Dispatch>) {
        let source = Dispatch::int_from_data(
            TensorData::from_bytes_vec(pack(dtype, &self.root.values), self.root.shape.clone(), dtype),
            device.as_dispatch(),
        );
        let mut input = source.clone();
        for view in &self.views {
            input = match view {
                View::Permute(axes) => Dispatch::int_permute(input, axes),
                View::Flip(axes) => Dispatch::int_flip(input, axes),
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
                    Dispatch::int_slice(input, &slices)
                }
                View::Expand(shape) => Dispatch::int_expand(input, Shape::from(shape.clone())),
            };
        }
        (source, input)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub operation: Operation,
    pub scope: Scope,
}
#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub dtype: DType,
    pub input: Input,
    pub steps: Vec<Step>,
    /// Uses a real Tensor<D, Int> with D equal to the input's logical rank.
    pub public_api: bool,
}
impl Case {
    pub fn expected(&self) -> Result<Vec<Expected>, ReferenceError> {
        let mut input = self.input.logical();
        let mut outputs = Vec::new();
        for step in &self.steps {
            let result = reference(self.dtype, &input, step.operation, step.scope)?;
            input = result.output.clone();
            outputs.push(result);
        }
        Ok(outputs)
    }
}
fn generated(dtype: DType, n: usize) -> Vec<u64> {
    let sign = 1u64 << (width(dtype) - 1);
    let edges = [0, 1, mask(dtype), sign, sign - 1, (1u64 << 53) + 1, u64::MAX - 1];
    (0..n).map(|i| edges[i % edges.len()] & mask(dtype)).collect()
}
fn add(cases: &mut Vec<Case>, dtype: DType, label: &str, input: Input, operation: Operation, scope: Scope, public_api: bool) {
    cases.push(Case {
        id: format!("{dtype:?}/{label}/{operation:?}/{scope:?}"),
        dtype,
        input,
        steps: vec![Step { operation, scope }],
        public_api,
    });
}
fn dense(shape: &[usize], values: Vec<u64>) -> Input {
    Input {
        root: Values::new(shape, values),
        views: vec![],
    }
}

/// Bounded, deterministic selections. Invalid APIs are diagnostic rows, never
/// silently removed from the description or executed on Linux by default.
pub fn corpus(options: &Options) -> Vec<Case> {
    let mut cases = Vec::new();
    for (format, dtype) in DTYPES.into_iter().enumerate() {
        if options.dtype.is_some_and(|d| d != dtype) {
            continue;
        }
        let input = dense(&[3, 3], generated(dtype, 9));
        for op in Operation::AXIS.into_iter().chain(Operation::PAIRS) {
            add(&mut cases, dtype, "canonical", input.clone(), op, Scope::Axis(1), true);
        }
        for op in Operation::GLOBAL {
            add(&mut cases, dtype, "canonical", input.clone(), op, Scope::Global, true);
        }
        // Rotate lengths across formats instead of crossing both domains.
        for (i, length) in LENGTHS.into_iter().enumerate() {
            if i % DTYPES.len() != format {
                continue;
            }
            let rank = 1 + (i + format) % 8;
            let axis = (i * 3 + format) % rank;
            let mut shape = vec![1; rank];
            shape[axis] = length;
            let mut input = dense(&shape, generated(dtype, length));
            match i % 4 {
                0 => input.views.push(View::Flip(vec![axis])),
                1 if rank > 1 => {
                    let mut axes: Vec<_> = (0..rank).collect();
                    axes.swap(0, rank - 1);
                    input.views.push(View::Permute(axes));
                }
                2 => {
                    let mut padded = shape.clone();
                    padded[axis] = length * 2 + 1;
                    input = dense(&padded, generated(dtype, length * 2 + 1));
                    let mut starts = vec![0; rank];
                    starts[axis] = 1;
                    let mut steps = vec![1; rank];
                    steps[axis] = 2;
                    input.views.push(View::Slice { starts, steps, shape });
                }
                _ => {}
            }
            let logical = input.logical();
            let actual_axis = logical.shape.iter().position(|&d| d == length).unwrap_or(axis);
            add(
                &mut cases,
                dtype,
                &format!("length-{length}-rank-{rank}"),
                input,
                Operation::AXIS[(i + format) % 7],
                Scope::Axis(actual_axis),
                true,
            );
        }
        // All nontrivial axes of each rank, with ranks/formats distributed once.
        let rank = format + 1;
        let shape = vec![2; rank];
        for axis in 0..rank {
            add(
                &mut cases,
                dtype,
                &format!("rank-{rank}-axis-{axis}"),
                dense(&shape, generated(dtype, 1 << rank)),
                Operation::AXIS[axis % 7],
                Scope::Axis(axis),
                axis == rank - 1,
            );
        }
        let input = Input {
            root: Values::new(&[1, 3], generated(dtype, 3)),
            views: vec![View::Expand(vec![5, 3]), View::Flip(vec![1])],
        };
        let op = [Operation::Sum, Operation::Mean, Operation::MaxWithIndices][format % 3];
        add(&mut cases, dtype, "stride-zero-broadcast", input, op, Scope::Axis(0), false);
        let input = Input {
            root: Values::new(&[3, 5], generated(dtype, 15)),
            views: vec![View::Permute(vec![1, 0]), View::Flip(vec![0, 1])],
        };
        if format % 2 == 0 {
            add(&mut cases, dtype, "transposed-signed", input, Operation::MinWithIndices, Scope::Axis(0), false);
        }
        // Winner coordinates cross both signed and unsigned byte boundaries.
        for op in Operation::PAIRS.into_iter().filter(|_| matches!(dtype, DType::I8 | DType::U8)) {
            let winning = if op.is_min() {
                if dtype.is_int() { 1u64 << (width(dtype) - 1) } else { 0 }
            } else if dtype.is_int() {
                mask(dtype) >> 1
            } else {
                mask(dtype)
            };
            let mut values = vec![1; 257];
            values[256] = winning;
            add(&mut cases, dtype, "winner-256", dense(&[257], values), op, Scope::Axis(0), true);
        }
        for op in [Operation::MinWithIndices, Operation::MaxWithIndices] {
            let value = if op.is_min() {
                if dtype.is_int() { 1u64 << (width(dtype) - 1) } else { 0 }
            } else if dtype.is_int() {
                mask(dtype) >> 1
            } else {
                mask(dtype)
            };
            add(&mut cases, dtype, "identity-equal-ties", dense(&[3], vec![value; 3]), op, Scope::Axis(0), false);
        }
        if !options.skip_rank_zero {
            for op in Operation::GLOBAL {
                add(&mut cases, dtype, "rank-zero", dense(&[], vec![mask(dtype)]), op, Scope::Global, false);
            }
            add(
                &mut cases,
                dtype,
                "rank-zero-axis-error",
                dense(&[], vec![1]),
                Operation::Sum,
                Scope::Axis(0),
                false,
            );
        }
        if options.include_empty {
            for op in [Operation::Sum, Operation::Product] {
                for (label, shape) in [("empty-axis", [2, 0]), ("both-empty", [0, 0])] {
                    add(&mut cases, dtype, label, dense(&shape, vec![]), op, Scope::Axis(1), false);
                }
                add(&mut cases, dtype, "global-empty", dense(&[0], vec![]), op, Scope::Global, false);
            }
            for op in [Operation::AXIS[format % 5], Operation::PAIRS[format % 2]] {
                add(&mut cases, dtype, "outer-empty", dense(&[0, 3], vec![]), op, Scope::Axis(1), false);
            }
            let op = if format % 2 == 0 { Operation::ArgMin } else { Operation::ArgMax };
            add(&mut cases, dtype, "outer-empty-public-index", dense(&[0, 3], vec![]), op, Scope::Axis(1), true);
            // Each rejecting operation appears once for both zero-axis shapes.
            for (i, op) in Operation::AXIS
                .into_iter()
                .chain(Operation::PAIRS)
                .filter(|op| !matches!(op, Operation::Sum | Operation::Product))
                .enumerate()
            {
                if i != format {
                    continue;
                }
                for (label, shape) in [("empty-axis-error", [2, 0]), ("both-empty-error", [0, 0])] {
                    add(&mut cases, dtype, label, dense(&shape, vec![]), op, Scope::Axis(1), false);
                }
            }
            add(&mut cases, dtype, "global-empty", dense(&[0], vec![]), Operation::Mean, Scope::Global, false);
            if format < 2 {
                add(
                    &mut cases,
                    dtype,
                    "global-empty",
                    dense(&[0], vec![]),
                    [Operation::Min, Operation::Max][format],
                    Scope::Global,
                    false,
                );
            }
            add(
                &mut cases,
                dtype,
                "invalid-axis-empty",
                dense(&[0], vec![]),
                Operation::Sum,
                Scope::Axis(1),
                false,
            );
        }
        // Caller-ordered materialized means distinguish staged truncation.
        if dtype.is_int() {
            for axes in [[0, 1], [1, 0]] {
                cases.push(Case {
                    id: format!("{dtype:?}/ordered-means-{axes:?}"),
                    dtype,
                    input: dense(&[2, 2], [-4i64, -4, -3, -1].into_iter().map(|v| v as u64 & mask(dtype)).collect()),
                    steps: axes
                        .into_iter()
                        .map(|a| Step {
                            operation: Operation::Mean,
                            scope: Scope::Axis(a),
                        })
                        .collect(),
                    public_api: false,
                });
            }
        }
        cases.push(Case {
            id: format!("{dtype:?}/materialized-chain"),
            dtype,
            input: dense(&[3, 3], vec![1; 9]),
            steps: vec![
                Step {
                    operation: Operation::Sum,
                    scope: Scope::Axis(1),
                },
                Step {
                    operation: Operation::Product,
                    scope: Scope::Axis(0),
                },
                Step {
                    operation: Operation::Mean,
                    scope: Scope::Global,
                },
            ],
            public_api: true,
        });
    }
    for (dtype, length) in [(DType::I8, 128), (DType::U8, 256)] {
        if options.dtype.is_none_or(|d| d == dtype) {
            for scope in [Scope::Axis(0), Scope::Global] {
                add(
                    &mut cases,
                    dtype,
                    "count-boundary",
                    dense(&[length], vec![1; length]),
                    Operation::Mean,
                    scope,
                    true,
                );
            }
        }
    }
    if options.dtype.is_none_or(|d| d == DType::I8) {
        for scope in [Scope::Axis(0), Scope::Global] {
            add(
                &mut cases,
                DType::I8,
                "widened-versus-stored",
                dense(&[2], vec![127, 127]),
                Operation::Mean,
                scope,
                true,
            );
        }
        for op in Operation::PAIRS {
            let mut values = vec![0; 130];
            values[129] = if op.is_min() { 128 } else { 127 };
            add(&mut cases, DType::I8, "signed-index-129", dense(&[130], values), op, Scope::Axis(0), true);
        }
    }
    cases
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    pub cases: usize,
    pub valid_cases: usize,
    pub diagnostic_cases: usize,
    pub steps: usize,
    pub public_cases: usize,
    pub pair_steps: usize,
    pub chain_cases: usize,
    pub chain_steps: usize,
    pub values: usize,
    pub source_checks: usize,
    pub alias_checks: usize,
}
pub fn summary_json(s: &Summary) -> String {
    format!(
        "{{\"corpus\":{CORPUS_VERSION},\"dtypes\":{},\"cases\":{},\"valid_cases\":{},\"diagnostic_cases\":{},\"steps\":{},\"public_cases\":{},\"pair_steps\":{},\"chain_cases\":{},\"chain_steps\":{},\"values\":{},\"source_checks\":{},\"alias_checks\":{}}}",
        s.dtypes,
        s.cases,
        s.valid_cases,
        s.diagnostic_cases,
        s.steps,
        s.public_cases,
        s.pair_steps,
        s.chain_cases,
        s.chain_steps,
        s.values,
        s.source_checks,
        s.alias_checks
    )
}
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary {
        dtypes: DTYPES.iter().filter(|&&d| options.dtype.is_none_or(|selected| selected == d)).count(),
        ..Summary::default()
    };
    for case in corpus(options) {
        summary.cases += 1;
        match case.expected() {
            Ok(outputs) => {
                summary.valid_cases += 1;
                summary.source_checks += 1;
                summary.alias_checks += 1;
                summary.public_cases += usize::from(case.public_api);
                summary.steps += outputs.len();
                summary.pair_steps += outputs.iter().filter(|o| o.indices.is_some()).count();
                summary.values += outputs
                    .iter()
                    .map(|o| o.output.values.len() + o.indices.as_ref().map_or(0, |i| i.values.len()))
                    .sum::<usize>();
                if outputs.len() > 1 {
                    summary.chain_cases += 1;
                    summary.chain_steps += outputs.len();
                }
            }
            Err(_) => {
                summary.diagnostic_cases += 1;
                if options.native_error_extensions {
                    summary.source_checks += 1;
                    summary.alias_checks += 1;
                }
            }
        }
    }
    summary
}

#[derive(Clone, Debug)]
pub struct Failure {
    pub case_id: String,
    pub message: String,
}
#[derive(Clone, Debug)]
pub struct Report {
    /// Pure plan, never populated from runtime success or failure.
    pub expected: Summary,
    pub attempted: usize,
    pub passed: usize,
    pub skipped_diagnostics: usize,
    pub observed_errors: usize,
    pub failed: usize,
    /// Bounded first failures. `failed` retains the complete count.
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

/// Common success record. The caller appends execution OS/backend/fusion labels.
pub fn success_footer(options: &Options, report: &Report) -> Result<String, String> {
    if options.dtype.is_some_and(|dtype| !DTYPES.contains(&dtype)) {
        return Err(String::from("unsupported integer dtype"));
    }
    if report.expected != expected_summary(options) {
        return Err(String::from("report plan does not match selected options"));
    }
    let expected_errors = if options.native_error_extensions {
        report.expected.diagnostic_cases
    } else {
        0
    };
    if report.observed_errors != expected_errors {
        return Err(String::from("observed diagnostics do not match selected error scope"));
    }
    if !report.is_success() {
        return Err(format!(
            "integer reduction failed: {}/{} cases; first={:?}",
            report.failed,
            report.attempted,
            report.failures.first()
        ));
    }
    Ok(format!(
        "BURN_INTEGER_REDUCTION_OK corpus={CORPUS_VERSION} totals={} attempted={} passed={} skipped_diagnostics={} observed_errors={} reference=independent_u128_i128 model_provenance=native_integer_policy_v2 source_policy=burn_0.22.1 layout_policy=logical_views_rank0_to8 accumulator=wrap32_or64 axis_mean=divide_then_narrow global_mean=stored_sum_then_checked_division index=input_dtype_low_bits ties=first_logical signed_linux=qualification_pending physical_empty=raw_contract scope=integer_forward_only dtype={:?} skip_rank_zero={} include_empty={} native_error_extensions={}",
        summary_json(&report.expected),
        report.attempted,
        report.passed,
        report.skipped_diagnostics,
        report.observed_errors,
        options.dtype,
        options.skip_rank_zero,
        options.include_empty,
        options.native_error_extensions
    ))
}

fn apply(input: IntTensor<Dispatch>, step: Step) -> (IntTensor<Dispatch>, Option<IntTensor<Dispatch>>) {
    let result = match (step.operation, step.scope) {
        (Operation::Sum, Scope::Global) => Dispatch::int_sum(input),
        (Operation::Product, Scope::Global) => Dispatch::int_prod(input),
        (Operation::Mean, Scope::Global) => Dispatch::int_mean(input),
        (Operation::Min, Scope::Global) => Dispatch::int_min(input),
        (Operation::Max, Scope::Global) => Dispatch::int_max(input),
        (Operation::Sum, Scope::Axis(a)) => Dispatch::int_sum_dim(input, a),
        (Operation::Product, Scope::Axis(a)) => Dispatch::int_prod_dim(input, a),
        (Operation::Mean, Scope::Axis(a)) => Dispatch::int_mean_dim(input, a),
        (Operation::Min, Scope::Axis(a)) => Dispatch::int_min_dim(input, a),
        (Operation::Max, Scope::Axis(a)) => Dispatch::int_max_dim(input, a),
        (Operation::ArgMin, Scope::Axis(a)) => Dispatch::int_argmin(input, a),
        (Operation::ArgMax, Scope::Axis(a)) => Dispatch::int_argmax(input, a),
        (Operation::MinWithIndices, Scope::Axis(a)) => {
            let (v, i) = Dispatch::int_min_dim_with_indices(input, a);
            return (v, Some(i));
        }
        (Operation::MaxWithIndices, Scope::Axis(a)) => {
            let (v, i) = Dispatch::int_max_dim_with_indices(input, a);
            return (v, Some(i));
        }
        _ => unreachable!("no global arg trait API"),
    };
    (result, None)
}
fn apply_public<const D: usize>(input: IntTensor<Dispatch>, step: Step) -> (IntTensor<Dispatch>, Option<IntTensor<Dispatch>>) {
    assert_eq!(input.shape().num_dims(), D);
    let input = Tensor::<D, Int>::from_dispatch(input);
    macro_rules! output {
        ($t:expr) => {
            ($t.into_dispatch(), None)
        };
    }
    match (step.operation, step.scope) {
        (Operation::Sum, Scope::Global) => output!(input.sum()),
        (Operation::Product, Scope::Global) => output!(input.prod()),
        (Operation::Mean, Scope::Global) => output!(input.mean()),
        (Operation::Min, Scope::Global) => output!(input.min()),
        (Operation::Max, Scope::Global) => output!(input.max()),
        (Operation::Sum, Scope::Axis(a)) => output!(input.sum_dim(a)),
        (Operation::Product, Scope::Axis(a)) => output!(input.prod_dim(a)),
        (Operation::Mean, Scope::Axis(a)) => output!(input.mean_dim(a)),
        (Operation::Min, Scope::Axis(a)) => output!(input.min_dim(a)),
        (Operation::Max, Scope::Axis(a)) => output!(input.max_dim(a)),
        (Operation::ArgMin, Scope::Axis(a)) => output!(input.argmin(a)),
        (Operation::ArgMax, Scope::Axis(a)) => output!(input.argmax(a)),
        (Operation::MinWithIndices, Scope::Axis(a)) => {
            let (v, i) = input.min_dim_with_indices(a);
            (v.into_dispatch(), Some(i.into_dispatch()))
        }
        (Operation::MaxWithIndices, Scope::Axis(a)) => {
            let (v, i) = input.max_dim_with_indices(a);
            (v.into_dispatch(), Some(i.into_dispatch()))
        }
        _ => unreachable!("no global arg trait API"),
    }
}
fn public_by_rank(input: IntTensor<Dispatch>, step: Step) -> (IntTensor<Dispatch>, Option<IntTensor<Dispatch>>) {
    match input.shape().num_dims() {
        1 => apply_public::<1>(input, step),
        2 => apply_public::<2>(input, step),
        3 => apply_public::<3>(input, step),
        4 => apply_public::<4>(input, step),
        5 => apply_public::<5>(input, step),
        6 => apply_public::<6>(input, step),
        7 => apply_public::<7>(input, step),
        8 => apply_public::<8>(input, step),
        _ => unreachable!("public ranks are explicitly 1..8"),
    }
}
async fn check(tensor: IntTensor<Dispatch>, dtype: DType, expected: &Values) -> Result<(), String> {
    if tensor.dtype() != dtype || tensor.shape().as_slice() != expected.shape {
        return Err(format!(
            "metadata {:?} {:?}, expected {dtype:?} {:?}",
            tensor.dtype(),
            tensor.shape(),
            expected.shape
        ));
    }
    let data = Dispatch::int_into_data(tensor).await.map_err(|e| format!("read: {e:?}"))?;
    if data.dtype() != dtype || data.shape().as_slice() != expected.shape {
        return Err(String::from("read metadata changed"));
    }
    let expected_bytes = pack(dtype, &expected.values);
    if data.as_bytes().len() != expected_bytes.len() {
        return Err(format!("byte length {}, expected {}", data.as_bytes().len(), expected_bytes.len()));
    }
    if let Some(at) = data.as_bytes().iter().zip(&expected_bytes).position(|(a, b)| a != b) {
        let index = at / dtype.size();
        return Err(format!(
            "element={index} actual={:#x} expected={:#x}",
            unpack(dtype, data.as_bytes())[index],
            expected.values[index] & mask(dtype)
        ));
    }
    Ok(())
}
async fn run_case(device: &Device, case: &Case, expected: &[Expected]) -> Result<(), String> {
    let (source, input) = case.input.build(case.dtype, device);
    let alias = input.clone();
    let mut current = input;
    let result = async {
        for (step, expected) in case.steps.iter().zip(expected) {
            let (output, indices) = if case.public_api {
                public_by_rank(current, *step)
            } else {
                apply(current, *step)
            };
            // Each step is read before the next operation. Fusion cannot hide a
            // materialization boundary or a wrong intermediate stored dtype.
            let values = check(output.clone(), case.dtype, &expected.output).await;
            let indices = if let Some(indices) = indices {
                check(indices, case.dtype, expected.indices.as_ref().expect("pair oracle")).await
            } else {
                Ok(())
            };
            values.and(indices)?;
            current = output;
        }
        Ok(())
    }
    .await;
    // Preserve evidence even when an output comparison failed.
    let source = check(source, case.dtype, &case.input.root).await.map_err(|e| format!("source: {e}"));
    let alias = check(alias, case.dtype, &case.input.logical()).await.map_err(|e| format!("alias: {e}"));
    result.and(source).and(alias)
}
async fn expected_error(device: &Device, case: &Case) -> Result<(), String> {
    let (source, input) = case.input.build(case.dtype, device);
    let alias = input.clone();
    let mut shape = case.input.logical().shape;
    match case.steps[0].scope {
        Scope::Global => shape = vec![1],
        Scope::Axis(axis) => {
            if let Some(size) = shape.get_mut(axis) {
                *size = 1;
            }
        }
    }
    // Bypass frontend assertions deliberately. Native must defer these errors
    // recoverably; Linux's panic policy is a separate isolated diagnostic.
    let (output, indices) = apply(input, case.steps[0]);
    let metadata = output.dtype() == case.dtype && output.shape().as_slice() == shape;
    let values_failed = Dispatch::int_into_data(output).await.is_err();
    let indices_failed = if let Some(indices) = indices {
        let metadata = indices.dtype() == case.dtype && indices.shape().as_slice() == shape;
        let failed = Dispatch::int_into_data(indices).await.is_err();
        metadata && failed
    } else {
        true
    };
    let rejected = if metadata && values_failed && indices_failed {
        Ok(())
    } else {
        Err(String::from("invalid API did not preserve deferred error shape/dtype"))
    };
    let source = check(source, case.dtype, &case.input.root).await;
    let alias = check(alias, case.dtype, &case.input.logical()).await;
    // Verify a subsequent valid command on the same device, not just readback.
    let valid = dense(&[1], vec![1]);
    let (_, input) = valid.build(case.dtype, device);
    let recovery = check(Dispatch::int_sum(input), case.dtype, &Values::new(&[1], vec![1])).await;
    rejected.and(source).and(alias).and(recovery)
}
/// Execute one intentionally invalid row in an isolated Linux process. A panic
/// is a Linux API observation, never a shared conformance success.
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
    let values = Dispatch::int_into_data(output).await;
    let index = if let Some(indices) = indices {
        Some(Dispatch::int_into_data(indices).await)
    } else {
        None
    };
    Err(format!("Linux diagnostic returned without panic: value={values:?} index={index:?}"))
}
/// Run the shared policy. No GPU/device initialization happens in corpus or
/// expected_summary. Native callers explicitly opt into recoverable errors.
pub async fn run(device: &Device, options: &Options) -> Report {
    let mut report = Report {
        expected: expected_summary(options),
        attempted: 0,
        passed: 0,
        skipped_diagnostics: 0,
        observed_errors: 0,
        failed: 0,
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
            Ok(expected) => run_case(device, &case, &expected).await,
            Err(_) => {
                let result = expected_error(device, &case).await;
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
