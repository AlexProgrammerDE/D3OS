//! Independent typed Log1p/Sinh/Cosh/Tan qualification against interval-confirmed MPFR fixtures.
//!
//! Numerical limits are project qualification targets, not vendor guarantees.
//! Half targets model neighboring ideal F32 results followed by destination RNE.
//! Forward qualification only; no autodiff or backward accuracy claim.
//! Retained sources and aliases always require their original bytes, including NaNs.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{DType, Dispatch, TensorData, TensorMetadata, ops::FloatTensorOps, tensor::FloatTensor},
    tensor::{Device, Slice, Tensor},
};

pub const CORPUS_VERSION: u32 = 1;
pub const DTYPES: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
pub const OPERATIONS: [Operation; 4] = [Operation::Log1p, Operation::Sinh, Operation::Cosh, Operation::Tan];
const LENGTHS: [usize; 10] = [0, 1, 2, 3, 127, 128, 129, 255, 256, 257];
const MAX_EXAMPLES: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Log1p,
    Sinh,
    Cosh,
    Tan,
}
impl Operation {
    pub const fn index(self) -> usize {
        match self {
            Self::Log1p => 0,
            Self::Sinh => 1,
            Self::Cosh => 2,
            Self::Tan => 3,
        }
    }
    pub const fn opcode(self) -> u32 {
        match self {
            Self::Log1p => 17,
            Self::Sinh => 30,
            Self::Cosh => 31,
            Self::Tan => 32,
        }
    }
    pub const fn label(self) -> &'static str {
        match self {
            Self::Log1p => "log1p",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tan => "tan",
        }
    }
    pub const fn budget(self, dtype: DType) -> u64 {
        match (dtype, self) {
            (_, Self::Log1p) | (DType::F64, Self::Cosh) => 1,
            (DType::F64, _) | (_, Self::Cosh) => 2,
            (_, Self::Sinh) => 3,
            (_, Self::Tan) => 4,
        }
    }
}
#[derive(Clone, Copy, Debug)]
struct ChainPlan {
    seed: u64,
    operations: [Operation; 3],
    finite: bool,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub no_flex32: bool,
    pub skip_rank_zero: bool,
    /// Retain finite inputs with finite ideal destination outputs.
    pub finite: bool,
    pub dtype: Option<DType>,
    pub op: Option<Operation>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    pub input: u64,
    pub ideal: u64,
    pub staged: u64,
    /// Inclusive ordered-code endpoints of the exact half neighbor image.
    pub allowed_low: u64,
    pub allowed_high: u64,
    pub intermediate_f32: u32,
}
impl Reference {
    pub const fn expected(self) -> u64 {
        self.staged
    }
}
include!("typed_unary_fixtures.rs");
fn format_index(dtype: DType) -> usize {
    DTYPES.iter().position(|&d| d == dtype).expect("typed unary float dtype")
}
pub fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F64 => "f64",
        DType::F32 => "f32",
        DType::Flex32 => "flex32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "unsupported",
    }
}
fn word(at: usize, width: usize) -> u64 {
    let mut bytes = [0; 8];
    bytes[..width].copy_from_slice(&FIXTURE_BYTES[at..at + width]);
    u64::from_le_bytes(bytes)
}
fn reference_at(dtype: DType, offset: usize, index: usize) -> Reference {
    let width = dtype.size();
    let at = offset + index * (width * 5 + 4);
    Reference {
        input: word(at, width),
        ideal: word(at + width, width),
        staged: word(at + 2 * width, width),
        allowed_low: word(at + 3 * width, width),
        allowed_high: word(at + 4 * width, width),
        intermediate_f32: word(at + 5 * width, 4) as u32,
    }
}
pub fn rows(dtype: DType, op: Operation) -> Vec<Reference> {
    let (offset, count) = RANGES[format_index(dtype)][op.index()];
    (0..count).map(|index| reference_at(dtype, offset, index)).collect()
}
/// Immutable fixture lookup. No production or host transcendental computes expectations.
pub fn expected_bits(dtype: DType, op: Operation, input: u64) -> Option<Reference> {
    let fi = DTYPES.iter().position(|&d| d == dtype)?;
    if is_nan(dtype, input) {
        let (_, inf, fraction) = masks(dtype);
        let nan = inf | ((fraction + 1) >> 1);
        return Some(Reference {
            input,
            ideal: nan,
            staged: nan,
            allowed_low: nan,
            allowed_high: nan,
            intermediate_f32: if matches!(dtype, DType::F16 | DType::BF16) { 0x7fc0_0000 } else { 0 },
        });
    }
    let (offset, count) = RANGES[fi][op.index()];
    let (mut low, mut high) = (0, count);
    while low < high {
        let mid = low + (high - low) / 2;
        let row = reference_at(dtype, offset, mid);
        match row.input.cmp(&input) {
            core::cmp::Ordering::Less => low = mid + 1,
            core::cmp::Ordering::Greater => high = mid,
            core::cmp::Ordering::Equal => return Some(row),
        }
    }
    None
}
fn selected(dtype: DType, op: Operation, options: &Options) -> bool {
    (!options.no_flex32 || dtype != DType::Flex32) && options.dtype.is_none_or(|d| dtype == d) && options.op.is_none_or(|o| o == op)
}
pub fn selected_rows(dtype: DType, op: Operation, options: &Options) -> Vec<Reference> {
    if !selected(dtype, op, options) {
        return vec![];
    }
    rows(dtype, op)
        .into_iter()
        .filter(|r| !options.finite || (is_finite(dtype, r.input) && is_finite(dtype, r.ideal)))
        .collect()
}
fn masks(dtype: DType) -> (u64, u64, u64) {
    match dtype {
        DType::F64 => (0x8000_0000_0000_0000, 0x7ff0_0000_0000_0000, 0x000f_ffff_ffff_ffff),
        DType::F32 | DType::Flex32 => (0x8000_0000, 0x7f80_0000, 0x007f_ffff),
        DType::F16 => (0x8000, 0x7c00, 0x03ff),
        DType::BF16 => (0x8000, 0x7f80, 0x007f),
        _ => panic!("typed unary float dtype"),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Class {
    Zero,
    Subnormal,
    Normal,
    Infinity,
    Nan,
}
pub fn classify(dtype: DType, bits: u64) -> Class {
    let (_, exponent, fraction) = masks(dtype);
    match (bits & exponent, bits & fraction) {
        (0, 0) => Class::Zero,
        (0, _) => Class::Subnormal,
        (e, 0) if e == exponent => Class::Infinity,
        (e, _) if e == exponent => Class::Nan,
        _ => Class::Normal,
    }
}
pub fn is_nan(dtype: DType, bits: u64) -> bool {
    classify(dtype, bits) == Class::Nan
}
pub fn is_finite(dtype: DType, bits: u64) -> bool {
    !matches!(classify(dtype, bits), Class::Nan | Class::Infinity)
}
fn nonzero_finite(class: Class) -> bool {
    matches!(class, Class::Subnormal | Class::Normal)
}
pub fn ordered_rank(dtype: DType, bits: u64) -> u64 {
    let (sign, _, _) = masks(dtype);
    if bits & sign == 0 { bits | sign } else { !bits & (sign | (sign - 1)) }
}
fn distance(dtype: DType, actual: u64, expected: u64) -> Option<u64> {
    (nonzero_finite(classify(dtype, actual)) && nonzero_finite(classify(dtype, expected)))
        .then(|| ordered_rank(dtype, actual).abs_diff(ordered_rank(dtype, expected)))
}
pub fn exact_special(dtype: DType, op: Operation, input: u64) -> bool {
    matches!(classify(dtype, input), Class::Nan | Class::Infinity | Class::Zero)
        || (op == Operation::Log1p && input & masks(dtype).0 != 0 && input & (masks(dtype).0 - 1) >= one(dtype))
}
fn one(dtype: DType) -> u64 {
    match dtype {
        DType::F64 => 0x3ff0_0000_0000_0000,
        DType::F16 => 0x3c00,
        DType::BF16 => 0x3f80,
        _ => 0x3f80_0000,
    }
}
fn selected_chains(dtype: DType, options: &Options) -> Vec<ChainPlan> {
    if (options.no_flex32 && dtype == DType::Flex32) || options.dtype.is_some_and(|d| d != dtype) {
        return vec![];
    }
    CHAIN_PLANS[format_index(dtype)]
        .iter()
        .copied()
        .filter(|plan| options.op.is_none_or(|op| plan.operations.contains(&op)) && (!options.finite || plan.finite))
        .collect()
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Comparison {
    pub passed: bool,
    pub classification_failure: bool,
    pub numerical_failure: bool,
    pub exact_special_failure: bool,
    pub envelope_member: bool,
    pub codomain_failure: bool,
    pub staged_ulp: Option<u64>,
    pub ideal_ulp: Option<u64>,
}
pub fn compare(dtype: DType, op: Operation, row: &Reference, actual: u64) -> Comparison {
    let expected = row.staged;
    let (wanted, observed) = (classify(dtype, expected), classify(dtype, actual));
    let (sign, _, _) = masks(dtype);
    let classification_failure = match wanted {
        Class::Nan => observed != Class::Nan,
        Class::Infinity | Class::Zero => actual != expected,
        Class::Subnormal | Class::Normal => !nonzero_finite(observed) || (actual ^ expected) & sign != 0,
    };
    let special = exact_special(dtype, op, row.input);
    let codomain_failure = op == Operation::Cosh && observed != Class::Nan && (actual & sign != 0 || (is_finite(dtype, actual) && actual < one(dtype)));
    let exact_special_failure = special && if wanted == Class::Nan { observed != Class::Nan } else { actual != expected };
    let staged_ulp = distance(dtype, actual, expected);
    let ideal_ulp = distance(dtype, actual, row.ideal);
    let envelope_member = if wanted == Class::Nan {
        observed == Class::Nan
    } else {
        let rank = ordered_rank(dtype, actual);
        !is_nan(dtype, actual) && rank >= ordered_rank(dtype, row.allowed_low) && rank <= ordered_rank(dtype, row.allowed_high)
    };
    let numerical_failure = if special {
        exact_special_failure
    } else if matches!(dtype, DType::F16 | DType::BF16) {
        !envelope_member
    } else if nonzero_finite(wanted) {
        staged_ulp.is_some_and(|steps| steps > op.budget(dtype))
    } else {
        false
    };
    Comparison {
        passed: !classification_failure && !numerical_failure && !exact_special_failure && !codomain_failure,
        classification_failure,
        numerical_failure,
        exact_special_failure,
        envelope_member,
        codomain_failure,
        staged_ulp,
        ideal_ulp,
    }
}
pub fn pack(dtype: DType, values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * dtype.size());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes()[..dtype.size()]);
    }
    bytes
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AccuracyCell {
    pub values: usize,
    pub failures: usize,
    pub classification_failures: usize,
    pub numerical_failures: usize,
    pub exact_special_failures: usize,
    pub codomain_failures: usize,
    pub unreferenced_results: usize,
    /// Missing-reference rows occupy the NaN row in both matrices for total
    /// accounting only. This vector distinguishes them from genuine NaN refs.
    /// They always fail and never contribute a finite-pair metric.
    pub unreferenced_observed_classes: [usize; 5],
    pub observed_classes: [usize; 5],
    pub passing_finite_pairs: usize,
    pub passing_max_ulp: Option<u64>,
    pub passing_histogram: [usize; 7],
    pub half_envelope_failures: usize,
    pub finite_pairs: usize,
    pub ideal_finite_pairs: usize,
    pub max_ulp: Option<u64>,
    pub max_ideal_ulp: Option<u64>,
    /// Bins: 0, 1, 2, 3..4, 5..8, 9..16, 17+ representable steps.
    pub histogram: [usize; 7],
    pub ideal_histogram: [usize; 7],
    /// Class order: zero, subnormal, normal, infinity, NaN. Expected then actual.
    pub transitions: [[usize; 5]; 5],
    /// Ideal reference class, then actual class. Direct ideal metrics are diagnostic.
    pub ideal_transitions: [[usize; 5]; 5],
    pub reference_class_disagreements: usize,
    pub reference_nonzero_finite_disagreements: usize,
    pub sign_mismatches: usize,
    pub nonzero_to_zero: usize,
    pub subnormal_preserved: usize,
    pub nan_payload_differences: usize,
    pub quiet_nan_outputs: usize,
    pub negative_nan_outputs: usize,
    pub worst_input_bits: Option<u64>,
    pub worst_actual_bits: Option<u64>,
    pub worst_ideal_input_bits: Option<u64>,
    pub worst_ideal_actual_bits: Option<u64>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Accuracy {
    pub failures: usize,
    /// dtype-major, then Log1p/Sinh/Cosh/Tan.
    pub cells: [AccuracyCell; 20],
}
fn histogram_bin(steps: u64) -> usize {
    match steps {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        _ => 6,
    }
}
impl Accuracy {
    pub fn record_unreferenced(&mut self, dtype: DType, op: Operation, actual: u64) {
        let cell = &mut self.cells[format_index(dtype) * 4 + op.index()];
        cell.values += 1;
        cell.failures += 1;
        cell.unreferenced_results += 1;
        let observed = classify(dtype, actual) as usize;
        cell.observed_classes[observed] += 1;
        cell.unreferenced_observed_classes[observed] += 1;
        cell.transitions[Class::Nan as usize][observed] += 1;
        cell.ideal_transitions[Class::Nan as usize][observed] += 1;
        self.failures += 1;
    }
    /// Check bounded counters and histograms independently of pass/fail status.
    pub fn is_coherent(&self) -> bool {
        self.failures == self.cells.iter().map(|cell| cell.failures).sum::<usize>()
            && self.cells.iter().all(|cell| {
                let maximum_matches = |maximum: Option<u64>, histogram: &[usize; 7], count: usize| {
                    histogram.iter().sum::<usize>() == count
                        && match maximum {
                            None => count == 0,
                            Some(value) => count > 0 && histogram.iter().rposition(|&n| n != 0) == Some(histogram_bin(value)),
                        }
                };
                cell.failures <= cell.values
                    && cell.observed_classes.iter().sum::<usize>() == cell.values
                    && cell.transitions.iter().flatten().sum::<usize>() == cell.values
                    && cell.ideal_transitions.iter().flatten().sum::<usize>() == cell.values
                    && (0..5).all(|column| {
                        cell.transitions.iter().map(|row| row[column]).sum::<usize>() == cell.observed_classes[column]
                            && cell.ideal_transitions.iter().map(|row| row[column]).sum::<usize>() == cell.observed_classes[column]
                    })
                    && cell.unreferenced_observed_classes.iter().sum::<usize>() == cell.unreferenced_results
                    && (0..5).all(|column| {
                        let missing = cell.unreferenced_observed_classes[column];
                        missing <= cell.observed_classes[column]
                            && missing <= cell.transitions[Class::Nan as usize][column]
                            && missing <= cell.ideal_transitions[Class::Nan as usize][column]
                    })
                    && cell.unreferenced_results <= cell.failures
                    && cell.reference_class_disagreements <= cell.values - cell.unreferenced_results
                    && cell.reference_nonzero_finite_disagreements <= cell.reference_class_disagreements
                    && cell.finite_pairs
                        == [Class::Subnormal, Class::Normal]
                            .into_iter()
                            .map(|r| {
                                [Class::Subnormal, Class::Normal]
                                    .into_iter()
                                    .map(|c| cell.transitions[r as usize][c as usize])
                                    .sum::<usize>()
                            })
                            .sum::<usize>()
                    && cell.ideal_finite_pairs
                        == [Class::Subnormal, Class::Normal]
                            .into_iter()
                            .map(|r| {
                                [Class::Subnormal, Class::Normal]
                                    .into_iter()
                                    .map(|c| cell.ideal_transitions[r as usize][c as usize])
                                    .sum::<usize>()
                            })
                            .sum::<usize>()
                    && cell.passing_finite_pairs <= cell.finite_pairs
                    && cell.passing_finite_pairs <= cell.values - cell.failures
                    && maximum_matches(cell.max_ulp, &cell.histogram, cell.finite_pairs)
                    && maximum_matches(cell.max_ideal_ulp, &cell.ideal_histogram, cell.ideal_finite_pairs)
                    && maximum_matches(cell.passing_max_ulp, &cell.passing_histogram, cell.passing_finite_pairs)
            })
    }

    pub fn record(&mut self, dtype: DType, op: Operation, row: &Reference, actual: u64) -> Comparison {
        let comparison = compare(dtype, op, row, actual);
        let cell = &mut self.cells[format_index(dtype) * 4 + op.index()];
        cell.values += 1;
        cell.observed_classes[classify(dtype, actual) as usize] += 1;
        cell.failures += usize::from(!comparison.passed);
        self.failures += usize::from(!comparison.passed);
        cell.classification_failures += usize::from(comparison.classification_failure);
        cell.numerical_failures += usize::from(comparison.numerical_failure);
        cell.exact_special_failures += usize::from(comparison.exact_special_failure);
        cell.codomain_failures += usize::from(comparison.codomain_failure);
        cell.half_envelope_failures += usize::from(matches!(dtype, DType::F16 | DType::BF16) && !comparison.envelope_member);
        let (expected, observed) = (classify(dtype, row.staged), classify(dtype, actual));
        cell.transitions[expected as usize][observed as usize] += 1;
        let ideal = classify(dtype, row.ideal);
        cell.ideal_transitions[ideal as usize][observed as usize] += 1;
        cell.reference_class_disagreements += usize::from(ideal != expected);
        cell.reference_nonzero_finite_disagreements += usize::from(nonzero_finite(ideal) != nonzero_finite(expected));
        cell.sign_mismatches += usize::from(expected != Class::Nan && observed != Class::Nan && (row.staged ^ actual) & masks(dtype).0 != 0);
        cell.nonzero_to_zero += usize::from(nonzero_finite(expected) && observed == Class::Zero);
        cell.subnormal_preserved += usize::from(expected == Class::Subnormal && observed == Class::Subnormal);
        if observed == Class::Nan {
            let (sign, _, fraction) = masks(dtype);
            cell.nan_payload_differences += usize::from(expected == Class::Nan && actual != row.staged);
            cell.quiet_nan_outputs += usize::from(actual & ((fraction + 1) >> 1) != 0);
            cell.negative_nan_outputs += usize::from(actual & sign != 0);
        }
        if let Some(steps) = comparison.staged_ulp {
            cell.finite_pairs += 1;
            if comparison.passed {
                cell.passing_finite_pairs += 1;
                cell.passing_histogram[histogram_bin(steps)] += 1;
                cell.passing_max_ulp = Some(cell.passing_max_ulp.unwrap_or(0).max(steps));
            }
            cell.histogram[histogram_bin(steps)] += 1;
            if cell.max_ulp.is_none_or(|maximum| steps > maximum) {
                cell.max_ulp = Some(steps);
                cell.worst_input_bits = Some(row.input);
                cell.worst_actual_bits = Some(actual);
            }
        }
        if let Some(steps) = comparison.ideal_ulp {
            cell.ideal_finite_pairs += 1;
            cell.ideal_histogram[histogram_bin(steps)] += 1;
            if cell.max_ideal_ulp.is_none_or(|maximum| steps > maximum) {
                cell.max_ideal_ulp = Some(steps);
                cell.worst_ideal_input_bits = Some(row.input);
                cell.worst_ideal_actual_bits = Some(actual);
            }
        }
        comparison
    }
}
#[derive(Clone, Debug, Default)]
pub struct Failures {
    pub count: usize,
    pub examples: Vec<String>,
}
impl Failures {
    pub fn record(&mut self, example: impl FnOnce() -> String) {
        self.count += 1;
        if self.examples.len() < MAX_EXAMPLES {
            self.examples.push(example());
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub options: Options,
    pub dtypes: usize,
    pub operations: usize,
    pub cases: usize,
    pub numerical_cases: usize,
    pub layout_cases: usize,
    pub public_cases: usize,
    pub chain_cases: usize,
    pub chain_steps: usize,
    /// Selected primitive/public fixture rows, counted once across both routes.
    /// Mixed chains can use other operations, tracked in expected_value_counts.
    pub tensor_rows: usize,
    pub selected_dtype_operation_counts: [[usize; 4]; 5],
    /// Pure planned outputs, including every operation within mixed chains.
    pub expected_value_counts: [[usize; 4]; 5],
    /// Mandatory primitive/public outputs only, ordered by dtype, operation,
    /// then reference class: zero, subnormal, normal, infinity, NaN.
    pub expected_staged_class_counts: [[[usize; 5]; 4]; 5],
    pub expected_ideal_class_counts: [[[usize; 5]; 4]; 5],
    /// Chain outputs by their actual operation. Their reference classes depend
    /// on preceding stored results and are deliberately not predicted here.
    pub chain_value_counts: [[usize; 4]; 5],
    pub values: usize,
    pub source_checks: usize,
    pub alias_checks: usize,
    pub poison_checks: usize,
    pub metadata_checks: usize,
    pub empty_cases: usize,
}
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub summary: Summary,
    pub accuracy: Accuracy,
    pub failures: Failures,
}
impl Report {
    pub fn is_success(&self) -> bool {
        self.failures.count == 0
            && self.accuracy.failures == 0
            && self.accuracy.is_coherent()
            && self.summary == expected_summary(&self.summary.options)
            && self
                .accuracy
                .cells
                .iter()
                .enumerate()
                .all(|(index, cell)| cell.values == self.summary.expected_value_counts[index / 4][index % 4])
    }
}

#[derive(Clone, Debug)]
struct Values {
    shape: Vec<usize>,
    bits: Vec<u64>,
}
impl Values {
    fn new(shape: &[usize], bits: Vec<u64>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), bits.len());
        Self { shape: shape.to_vec(), bits }
    }
    fn mapped(&self, shape: &[usize], map: impl Fn(&[usize]) -> Vec<usize>) -> Self {
        let bits = (0..shape.iter().product())
            .map(|mut index| {
                let mut coords = vec![0; shape.len()];
                for axis in (0..shape.len()).rev() {
                    coords[axis] = index % shape[axis];
                    index /= shape[axis];
                }
                let source = map(&coords);
                let at = self.shape.iter().zip(source).fold(0, |at, (&size, coord)| {
                    assert!(coord < size);
                    at * size + coord
                });
                self.bits[at]
            })
            .collect();
        Self::new(shape, bits)
    }
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
    fn shaped(dtype: DType, shape: &[usize], bits: Vec<u64>) -> Self {
        Self {
            dtype,
            root: Values::new(shape, bits),
            views: vec![],
        }
    }
    fn flat(dtype: DType, bits: Vec<u64>) -> Self {
        Self::shaped(dtype, &[bits.len()], bits)
    }
    fn logical(&self) -> Values {
        self.views.iter().fold(self.root.clone(), |values, view| match view {
            View::Slice { starts, steps, shape } => values.mapped(shape, |c| c.iter().enumerate().map(|(axis, &i)| starts[axis] + steps[axis] * i).collect()),
            View::Permute(axes) => values.mapped(&axes.iter().map(|&axis| values.shape[axis]).collect::<Vec<_>>(), |c| {
                let mut source = vec![0; axes.len()];
                for (&axis, &coord) in axes.iter().zip(c) {
                    source[axis] = coord;
                }
                source
            }),
            View::Flip(axes) => values.mapped(&values.shape, |c| {
                c.iter()
                    .enumerate()
                    .map(|(axis, &i)| if axes.contains(&axis) { values.shape[axis] - 1 - i } else { i })
                    .collect()
            }),
            View::Expand(shape) => {
                assert_eq!(shape.len(), values.shape.len());
                assert!(values.shape.iter().zip(shape).all(|(&source, &target)| source == target || source == 1));
                values.mapped(shape, |c| {
                    c.iter().zip(&values.shape).map(|(&coord, &size)| if size == 1 { 0 } else { coord }).collect()
                })
            }
        })
    }
    fn build(&self, device: &Device) -> (FloatTensor<Dispatch>, FloatTensor<Dispatch>) {
        let source = Dispatch::float_from_data(
            TensorData::from_bytes_vec(pack(self.dtype, &self.root.bits), self.root.shape.clone(), self.dtype),
            device.as_dispatch(),
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
                    Dispatch::float_slice(tensor, &slices)
                }
                View::Permute(axes) => Dispatch::float_permute(tensor, axes),
                View::Flip(axes) => Dispatch::float_flip(tensor, axes),
                View::Expand(shape) => Dispatch::float_expand(tensor, shape.clone().into()),
            };
        }
        (source, tensor)
    }
}
#[derive(Clone, Debug)]
struct Case {
    input: Input,
    op: Operation,
    public: bool,
    numerical: bool,
}
fn cases(dtype: DType, op: Operation, options: &Options, selected: &[Reference]) -> Vec<Case> {
    if selected.is_empty() {
        return vec![];
    }
    let values: Vec<_> = selected.iter().map(|r| r.input).collect();
    let value = |i: usize| values[(i * 4051 + i / 7) % values.len()];
    let mut inputs = Vec::new();
    for length in LENGTHS {
        inputs.push(Input::flat(dtype, (0..length).map(value).collect()));
        let mut offset = Input::flat(dtype, (0..length + 2).map(value).collect());
        offset.views.push(View::Slice {
            starts: vec![1],
            steps: vec![1],
            shape: vec![length],
        });
        inputs.push(offset);
        let mut reverse = Input::flat(dtype, (0..length).map(value).collect());
        reverse.views.push(View::Flip(vec![0]));
        inputs.push(reverse);
        let mut stepped = Input::flat(dtype, (0..length * 2 + 2).map(value).collect());
        stepped.views.push(View::Slice {
            starts: vec![1],
            steps: vec![2],
            shape: vec![length],
        });
        stepped.views.push(View::Flip(vec![0]));
        inputs.push(stepped);
    }
    if !options.skip_rank_zero {
        inputs.push(Input::shaped(dtype, &[], vec![value(0)]));
    }
    for rank in 2..=8 {
        let mut shape = vec![1; rank];
        shape[0] = 3;
        shape[rank - 1] = 7;
        let mut input = Input::shaped(dtype, &shape, (0..21).map(value).collect());
        let axes = (0..rank).rev().collect();
        input.views.push(View::Permute(axes));
        input.views.push(View::Flip(vec![0, rank - 1]));
        inputs.push(input);
        let mut shape = vec![1; rank];
        shape[0] = 2;
        shape[rank - 1] = 0;
        inputs.push(Input::shaped(dtype, &shape, vec![]));
    }
    for shape in [vec![4, 7], vec![0, 7]] {
        let mut expanded = Input::shaped(dtype, &[1, 7], (0..7).map(value).collect());
        expanded.views.push(View::Expand(shape));
        inputs.push(expanded);
    }
    let mut expanded = Input::shaped(dtype, &[1, 2, 1, 1, 1, 2, 1, 3], (0..12).map(value).collect());
    expanded.views.push(View::Expand(vec![1, 2, 1, 2, 1, 2, 1, 3]));
    expanded.views.push(View::Flip(vec![7]));
    inputs.push(expanded);
    let mut result = Vec::new();
    for public in [false, true] {
        result.push(Case {
            input: Input::flat(dtype, values.clone()),
            op,
            public,
            numerical: true,
        });
        result.extend(inputs.iter().cloned().map(|input| Case {
            input,
            op,
            public,
            numerical: false,
        }));
    }
    result
}
impl Summary {
    fn record(&mut self, case: &Case) {
        let logical = case.input.logical();
        self.cases += 1;
        self.numerical_cases += usize::from(case.numerical);
        self.layout_cases += usize::from(!case.numerical);
        self.public_cases += usize::from(case.public);
        self.values += logical.bits.len();
        let fi = format_index(case.input.dtype);
        let oi = case.op.index();
        self.expected_value_counts[fi][oi] += logical.bits.len();
        for input in logical.bits.iter().copied() {
            let row = expected_bits(case.input.dtype, case.op, input).expect("planned immutable reference");
            self.expected_staged_class_counts[fi][oi][classify(case.input.dtype, row.staged) as usize] += 1;
            self.expected_ideal_class_counts[fi][oi][classify(case.input.dtype, row.ideal) as usize] += 1;
        }
        self.empty_cases += usize::from(logical.bits.is_empty());
        self.source_checks += 1;
        self.alias_checks += 1;
        self.poison_checks += 1;
        self.metadata_checks += 4;
    }
    fn chain(&mut self, dtype: DType, operations: &[Operation]) {
        let steps = operations.len();
        for op in operations {
            self.expected_value_counts[format_index(dtype)][op.index()] += 1;
            self.chain_value_counts[format_index(dtype)][op.index()] += 1;
        }
        if steps == 0 {
            return;
        }
        self.chain_cases += 1;
        self.cases += 1;
        self.chain_steps += steps;
        self.values += steps;
        self.source_checks += 1;
        self.alias_checks += 1 + steps;
        self.poison_checks += 1;
        self.metadata_checks += 3 + steps * 2;
    }
}
/// Pure deterministic work description. Never creates a device or initializes a runtime.
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary {
        options: *options,
        ..Summary::default()
    };
    for (fi, dtype) in DTYPES.into_iter().enumerate() {
        let mut included = false;
        for op in OPERATIONS {
            let rows = selected_rows(dtype, op, options);
            if rows.is_empty() {
                continue;
            }
            included = true;
            summary.operations += 1;
            summary.tensor_rows += rows.len();
            summary.selected_dtype_operation_counts[fi][op.index()] = rows.len();
            for case in cases(dtype, op, options, &rows) {
                summary.record(&case);
            }
        }
        for plan in selected_chains(dtype, options) {
            summary.chain(dtype, &plan.operations);
        }
        summary.dtypes += usize::from(included);
    }
    summary
}
fn primitive(tensor: FloatTensor<Dispatch>, op: Operation) -> FloatTensor<Dispatch> {
    match op {
        Operation::Log1p => Dispatch::float_log1p(tensor),
        Operation::Sinh => Dispatch::float_sinh(tensor),
        Operation::Cosh => Dispatch::float_cosh(tensor),
        Operation::Tan => Dispatch::float_tan(tensor),
    }
}
fn public(tensor: FloatTensor<Dispatch>, op: Operation) -> FloatTensor<Dispatch> {
    fn apply<const D: usize>(tensor: FloatTensor<Dispatch>, op: Operation) -> FloatTensor<Dispatch> {
        let tensor = Tensor::<D>::from_dispatch(tensor);
        match op {
            Operation::Log1p => tensor.log1p(),
            Operation::Sinh => tensor.sinh(),
            Operation::Cosh => tensor.cosh(),
            Operation::Tan => tensor.tan(),
        }
        .into_dispatch()
    }
    match tensor.shape().num_dims() {
        0 => apply::<0>(tensor, op),
        1 => apply::<1>(tensor, op),
        2 => apply::<2>(tensor, op),
        3 => apply::<3>(tensor, op),
        4 => apply::<4>(tensor, op),
        5 => apply::<5>(tensor, op),
        6 => apply::<6>(tensor, op),
        7 => apply::<7>(tensor, op),
        8 => apply::<8>(tensor, op),
        _ => unreachable!("fixture rank"),
    }
}
fn poison(dtype: DType) -> Input {
    let (_, inf, fraction) = masks(dtype);
    Input::flat(dtype, (0..37).map(|i| inf | ((fraction + 1) >> 1) | (i + 1)).collect())
}
async fn read_bits(tensor: FloatTensor<Dispatch>, dtype: DType, shape: &[usize], label: &str, failures: &mut Failures) -> Option<Vec<u64>> {
    let expected_bytes = shape.iter().product::<usize>() * dtype.size();
    if tensor.dtype() != dtype || tensor.shape().as_slice() != shape {
        failures.record(|| {
            format!(
                "{label}: tensor metadata dtype={:?} shape={:?}, expected {dtype:?} {shape:?}",
                tensor.dtype(),
                tensor.shape()
            )
        });
    }
    let data = match Dispatch::float_into_data(tensor).await {
        Ok(data) => data,
        Err(error) => {
            failures.record(|| format!("{label}: read {error:?}"));
            return None;
        }
    };
    if data.dtype() != dtype || data.shape().as_slice() != shape || data.as_bytes().len() != expected_bytes {
        failures.record(|| {
            format!(
                "{label}: data metadata dtype={:?} shape={:?} bytes={}, expected {dtype:?} {shape:?} bytes={expected_bytes}",
                data.dtype(),
                data.shape(),
                data.as_bytes().len()
            )
        });
        return None;
    }
    Some(unpack(dtype, data.as_bytes()))
}
async fn preserved(tensor: FloatTensor<Dispatch>, dtype: DType, expected: &Values, label: &str, failures: &mut Failures) {
    if let Some(actual) = read_bits(tensor, dtype, &expected.shape, label, failures).await {
        for (index, (&actual, &expected)) in actual.iter().zip(&expected.bits).enumerate() {
            if actual != expected {
                failures.record(|| format!("{label}: byte preservation element={index} actual={actual:#018x} expected={expected:#018x}"));
            }
        }
    }
}
fn observe(report: &mut Report, dtype: DType, op: Operation, row: &Reference, actual: u64, label: &str, index: usize) {
    let result = report.accuracy.record(dtype, op, row, actual);
    if !result.passed {
        report.failures.record(|| format!("{label}: {dtype:?} {} element={index} input={:#018x} actual={actual:#018x} staged={:#018x} ideal={:#018x} class_failure={} numerical_failure={} exact_special_failure={} codomain_failure={} envelope_member={} staged_steps={:?} ideal_steps={:?}",
            op.label(), row.input, row.staged, row.ideal, result.classification_failure, result.numerical_failure,
            result.exact_special_failure, result.codomain_failure, result.envelope_member, result.staged_ulp, result.ideal_ulp));
    }
}
async fn run_case(device: &Device, case: &Case, report: &mut Report) {
    let dtype = case.input.dtype;
    let logical = case.input.logical();
    let (source, tensor) = case.input.build(device);
    let alias = tensor.clone();
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build(device);
    let output = if case.public { public(tensor, case.op) } else { primitive(tensor, case.op) };
    if let Some(actual) = read_bits(output, dtype, &logical.shape, "output", &mut report.failures).await {
        for (index, (&input, actual)) in logical.bits.iter().zip(actual).enumerate() {
            if let Some(row) = expected_bits(dtype, case.op, input) {
                observe(report, dtype, case.op, &row, actual, if case.public { "public" } else { "primitive" }, index);
            } else {
                report.accuracy.record_unreferenced(dtype, case.op, actual);
                report.failures.record(|| {
                    format!(
                        "unknown fixture input dtype={dtype:?} op={} input={input:#018x} actual={actual:#018x}",
                        case.op.label()
                    )
                });
            }
        }
    }
    preserved(source, dtype, &case.input.root, "source including view guards", &mut report.failures).await;
    preserved(alias, dtype, &logical, "source alias", &mut report.failures).await;
    preserved(retained_poison, dtype, &poison.root, "retained poison", &mut report.failures).await;
    report.summary.record(case);
}
async fn run_chain(device: &Device, dtype: DType, plan: &ChainPlan, report: &mut Report) {
    let seed = plan.seed;
    let input = Input::flat(dtype, vec![seed]);
    let (source, mut current) = input.build(device);
    let alias = current.clone();
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build(device);
    let mut intermediates = Vec::new();
    let mut stored_input = Some(seed);
    for (step, op) in plan.operations.into_iter().enumerate() {
        current = primitive(current, op);
        // Read the stored destination before issuing the next primitive. MPFR
        // references cover every legally accepted intermediate through depth 3.
        if let Some(actual) = read_bits(current.clone(), dtype, &[1], "materialized chain", &mut report.failures).await {
            if let Some(row) = stored_input.and_then(|bits| expected_bits(dtype, op, bits)) {
                observe(report, dtype, op, &row, actual[0], "materialized chain", step);
            } else {
                report.accuracy.record_unreferenced(dtype, op, actual[0]);
                report.failures.record(|| {
                    format!(
                        "materialized chain: {dtype:?} {} step={step} input={stored_input:?} outside checked-in closure actual={:#018x}",
                        op.label(),
                        actual[0]
                    )
                });
            }
            stored_input = Some(actual[0]);
            intermediates.push((current.clone(), Values::new(&[1], actual)));
        } else {
            stored_input = None;
        }
    }
    preserved(source, dtype, &input.root, "chain source", &mut report.failures).await;
    preserved(alias, dtype, &input.root, "chain source alias", &mut report.failures).await;
    preserved(retained_poison, dtype, &poison.root, "chain retained poison", &mut report.failures).await;
    for (tensor, bytes) in intermediates {
        preserved(tensor, dtype, &bytes, "retained materialized step", &mut report.failures).await;
    }
    report.summary.chain(dtype, &plan.operations);
}
/// Returns every observed counter even on qualification failure. Examples stay bounded.
pub async fn run(device: &Device, options: &Options) -> Result<Report, String> {
    if options.dtype.is_some_and(|dtype| !DTYPES.contains(&dtype)) {
        return Err(String::from("typed unary requires a supported float dtype"));
    }
    let mut report = Report {
        summary: Summary {
            options: *options,
            ..Summary::default()
        },
        ..Report::default()
    };
    for (fi, dtype) in DTYPES.into_iter().enumerate() {
        let mut included = false;
        for op in OPERATIONS {
            let rows = selected_rows(dtype, op, options);
            if rows.is_empty() {
                continue;
            }
            included = true;
            report.summary.operations += 1;
            report.summary.tensor_rows += rows.len();
            report.summary.selected_dtype_operation_counts[fi][op.index()] = rows.len();
            for case in cases(dtype, op, options, &rows) {
                run_case(device, &case, &mut report).await;
            }
        }
        for plan in selected_chains(dtype, options) {
            run_chain(device, dtype, &plan, &mut report).await;
        }
        report.summary.dtypes += usize::from(included);
    }
    let expected = expected_summary(options);
    if report.summary != expected {
        report
            .failures
            .record(|| format!("deterministic summary mismatch: actual={:?} expected={expected:?}", report.summary));
    }
    for (index, cell) in report.accuracy.cells.iter().enumerate() {
        let expected_values = expected.expected_value_counts[index / 4][index % 4];
        if cell.values != expected_values {
            report.failures.record(|| {
                format!(
                    "per-cell observed values dtype={} op={} actual={} expected={expected_values}",
                    dtype_label(DTYPES[index / 4]),
                    OPERATIONS[index % 4].label(),
                    cell.values
                )
            });
        }
    }
    let observed_values: usize = report.accuracy.cells.iter().map(|cell| cell.values).sum();
    if observed_values != report.summary.values {
        report
            .failures
            .record(|| format!("observed value count {observed_values}, expected {}", report.summary.values));
    }
    if !report.accuracy.is_coherent() {
        report.failures.record(|| String::from("incoherent observed accuracy counters"));
    }
    Ok(report)
}

fn option_number(value: Option<u64>) -> String {
    value.map_or_else(|| String::from("null"), |value| format!("{value}"))
}
fn option_bits(value: Option<u64>) -> String {
    value.map_or_else(|| String::from("null"), |value| format!("\"{value:016x}\""))
}
fn options_json(options: &Options) -> String {
    format!(
        "{{\"no_flex32\":{},\"skip_rank_zero\":{},\"finite\":{},\"dtype\":\"{}\",\"op\":\"{}\"}}",
        options.no_flex32,
        options.skip_rank_zero,
        options.finite,
        options.dtype.map_or("all", dtype_label),
        options.op.map_or("all", Operation::label)
    )
}
/// Stable compact JSON containing only deterministic work and identity fields.
pub fn summary_json(summary: &Summary) -> String {
    format!(
        concat!(
            "{{\"schema_version\":3,\"policy_sha256\":\"{}\",\"fixture_sha256\":\"{}\",\"options\":{},",
            "\"dtypes\":{},\"operations\":{},\"cases\":{},\"numerical_cases\":{},\"layout_cases\":{},\"public_cases\":{},",
            "\"chain_cases\":{},\"chain_steps\":{},\"tensor_rows\":{},\"selected_dtype_operation_counts\":{},\"expected_value_counts\":{},\"values\":{},",
            "\"expected_staged_class_counts\":{},\"expected_ideal_class_counts\":{},\"chain_value_counts\":{},",
            "\"source_checks\":{},\"alias_checks\":{},\"poison_checks\":{},\"metadata_checks\":{},\"empty_cases\":{}}}"
        ),
        POLICY_SHA256,
        FIXTURE_SHA256,
        options_json(&summary.options),
        summary.dtypes,
        summary.operations,
        summary.cases,
        summary.numerical_cases,
        summary.layout_cases,
        summary.public_cases,
        summary.chain_cases,
        summary.chain_steps,
        summary.tensor_rows,
        format!("{:?}", summary.selected_dtype_operation_counts).replace(' ', ""),
        format!("{:?}", summary.expected_value_counts).replace(' ', ""),
        summary.values,
        format!("{:?}", summary.expected_staged_class_counts).replace(' ', ""),
        format!("{:?}", summary.expected_ideal_class_counts).replace(' ', ""),
        format!("{:?}", summary.chain_value_counts).replace(' ', ""),
        summary.source_checks,
        summary.alias_checks,
        summary.poison_checks,
        summary.metadata_checks,
        summary.empty_cases
    )
}
/// Observed maxima are null until a finite nonzero pair has been measured.
pub fn accuracy_json(accuracy: &Accuracy) -> String {
    let cells: Vec<_> = accuracy
        .cells
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            let fields = [
                format!("\"dtype\":\"{}\"", dtype_label(DTYPES[index / 4])),
                format!("\"op\":\"{}\"", OPERATIONS[index % 4].label()),
                format!("\"values\":{}", cell.values),
                format!("\"failures\":{}", cell.failures),
                format!("\"classification_failures\":{}", cell.classification_failures),
                format!("\"numerical_failures\":{}", cell.numerical_failures),
                format!("\"exact_special_failures\":{}", cell.exact_special_failures),
                format!("\"half_envelope_failures\":{}", cell.half_envelope_failures),
                format!("\"codomain_failures\":{}", cell.codomain_failures),
                format!("\"unreferenced_results\":{}", cell.unreferenced_results),
                format!("\"reference_class_disagreements\":{}", cell.reference_class_disagreements),
                format!("\"reference_nonzero_finite_disagreements\":{}", cell.reference_nonzero_finite_disagreements),
                format!("\"passing_finite_pairs\":{}", cell.passing_finite_pairs),
                format!("\"finite_pairs\":{}", cell.finite_pairs),
                format!("\"ideal_finite_pairs\":{}", cell.ideal_finite_pairs),
                format!("\"sign_mismatches\":{}", cell.sign_mismatches),
                format!("\"nonzero_to_zero\":{}", cell.nonzero_to_zero),
                format!("\"subnormal_preserved\":{}", cell.subnormal_preserved),
                format!("\"nan_payload_differences\":{}", cell.nan_payload_differences),
                format!("\"quiet_nan_outputs\":{}", cell.quiet_nan_outputs),
                format!("\"negative_nan_outputs\":{}", cell.negative_nan_outputs),
                format!("\"observed_classes\":{}", format!("{:?}", cell.observed_classes).replace(' ', "")),
                format!(
                    "\"unreferenced_observed_classes\":{}",
                    format!("{:?}", cell.unreferenced_observed_classes).replace(' ', "")
                ),
                format!("\"passing_histogram\":{}", format!("{:?}", cell.passing_histogram).replace(' ', "")),
                format!("\"histogram\":{}", format!("{:?}", cell.histogram).replace(' ', "")),
                format!("\"ideal_histogram\":{}", format!("{:?}", cell.ideal_histogram).replace(' ', "")),
                format!("\"transitions\":{}", format!("{:?}", cell.transitions).replace(' ', "")),
                format!("\"ideal_transitions\":{}", format!("{:?}", cell.ideal_transitions).replace(' ', "")),
                format!("\"passing_max_ulp\":{}", option_number(cell.passing_max_ulp)),
                format!("\"max_ulp\":{}", option_number(cell.max_ulp)),
                format!("\"max_ideal_ulp\":{}", option_number(cell.max_ideal_ulp)),
                format!("\"worst_input_bits\":{}", option_bits(cell.worst_input_bits)),
                format!("\"worst_actual_bits\":{}", option_bits(cell.worst_actual_bits)),
                format!("\"worst_ideal_input_bits\":{}", option_bits(cell.worst_ideal_input_bits)),
                format!("\"worst_ideal_actual_bits\":{}", option_bits(cell.worst_ideal_actual_bits)),
            ];
            format!("{{{}}}", fields.join(","))
        })
        .collect();
    format!("{{\"failures\":{},\"cells\":[{}]}}", accuracy.failures, cells.join(","))
}
/// The caller prints this footer only after checking Report::is_success().
pub fn success_footer(options: &Options, report: &Report) -> String {
    format!(
        "totals={} accuracy={} policy_sha256={POLICY_SHA256} fixture_sha256={FIXTURE_SHA256} numerical_policy=project_qualification_targets staged_half=f32_neighbor_envelope no_ftz=strict preserved_inputs=exact_bytes chain=materialized_known_refs poison=retained_allocation scope=strict_native no_flex32={} skip_rank_zero={} finite={} dtype={} op={}",
        summary_json(&report.summary),
        accuracy_json(&report.accuracy),
        options.no_flex32,
        options.skip_rank_zero,
        options.finite,
        options.dtype.map_or("all", dtype_label),
        options.op.map_or("all", Operation::label)
    )
}
