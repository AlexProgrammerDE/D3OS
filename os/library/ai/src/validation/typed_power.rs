//! Stored-input Power conformance with separate mathematical and wrapper contracts.
//! Expectations come from directed MPFR fixtures, never backend arithmetic.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{
        Backend as BurnBackend, DType, Dispatch, Scalar, TensorData, TensorMetadata,
        ops::{FloatTensorOps, IntTensorOps},
        tensor::FloatTensor,
    },
    tensor::{Device, Slice, Tensor},
};

pub const CORPUS_VERSION: u32 = 1;
pub const DTYPES: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
pub const MAX_EXAMPLES: usize = 12;
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum Schedule {
    Mathematical,
    Ones,
    Identity,
    Square,
    Reciprocal,
    SquareReciprocal,
}
pub const SCHEDULES: [Schedule; 6] = [
    Schedule::Mathematical,
    Schedule::Ones,
    Schedule::Identity,
    Schedule::Square,
    Schedule::Reciprocal,
    Schedule::SquareReciprocal,
];
impl Schedule {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mathematical => "mathematical",
            Self::Ones => "ones",
            Self::Identity => "identity",
            Self::Square => "square",
            Self::Reciprocal => "reciprocal",
            Self::SquareReciprocal => "square_reciprocal",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum Route {
    Tensor,
    ScalarImpl,
    PowiScalarImpl,
    PowfScalar,
    PowiScalar,
    PublicPowfScalar,
    PublicPowiScalar,
    TensorPowi,
    PublicTensor,
}
pub const ROUTES: [Route; 9] = [
    Route::Tensor,
    Route::ScalarImpl,
    Route::PowiScalarImpl,
    Route::PowfScalar,
    Route::PowiScalar,
    Route::PublicPowfScalar,
    Route::PublicPowiScalar,
    Route::TensorPowi,
    Route::PublicTensor,
];
impl Route {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tensor => "tensor",
            Self::ScalarImpl => "scalar_impl",
            Self::PowiScalarImpl => "powi_scalar_impl",
            Self::PowfScalar => "powf_scalar",
            Self::PowiScalar => "powi_scalar",
            Self::PublicPowfScalar => "public_powf_scalar",
            Self::PublicPowiScalar => "public_powi_scalar",
            Self::TensorPowi => "tensor_powi",
            Self::PublicTensor => "public_tensor",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub no_flex32: bool,
    pub skip_rank_zero: bool,
    pub finite: bool,
    pub dtype: Option<DType>,
    /// Explicitly exclude inherited Linux panic routes and Native rejection probes.
    pub common_linux: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    pub lhs: u64,
    pub rhs: u64,
    pub ideal: u64,
    pub staged: u64,
    pub allowed_low: u64,
    pub allowed_high: u64,
    pub intermediate_f32: u32,
    pub exact_special: bool,
    pub schedule: Schedule,
}
impl Reference {
    pub const fn expected(self) -> u64 {
        self.staged
    }
}
#[derive(Clone, Debug)]
pub struct ScalarFixture {
    pub id: usize,
    pub scalar: Scalar,
    pub scalar_bits: u64,
    pub public_bits: u64,
    pub valid_powi: bool,
    pub linux_powf: bool,
    pub linux_powi: bool,
    pub public_linux_powf: bool,
    pub public_linux_powi: bool,
    pub inputs: Vec<u64>,
    pub primitive: Vec<Reference>,
    pub wrapper: Vec<Reference>,
    pub public_primitive: Vec<Reference>,
    pub public_wrapper: Vec<Reference>,
    pub public_powi_scalar: Scalar,
    pub public_powi_valid: bool,
    pub public_powi_bits: u64,
    pub public_powi_wrapper: Vec<Reference>,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Comparison {
    pub passed: bool,
    pub classification_failure: bool,
    pub numerical_failure: bool,
    pub exact_special_failure: bool,
    pub envelope_member: bool,
    pub normal_subnormal_transition: bool,
    pub staged_ulp: Option<u64>,
    pub ideal_ulp: Option<u64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Witness {
    pub lhs: u64,
    pub rhs: u64,
    pub schedule: Schedule,
    pub actual: u64,
    pub staged: u64,
    pub ideal: u64,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AccuracyCell {
    pub values: usize,
    pub failures: usize,
    pub classification_failures: usize,
    pub numerical_failures: usize,
    pub exact_special_failures: usize,
    pub unreferenced_results: usize,
    pub observed_classes: [usize; 5],
    pub unreferenced_observed_classes: [usize; 5],
    pub transitions: [[usize; 5]; 5],
    pub ideal_transitions: [[usize; 5]; 5],
    pub finite_pairs: usize,
    pub ideal_finite_pairs: usize,
    pub max_ulp: Option<u64>,
    pub max_ideal_ulp: Option<u64>,
    pub histogram: [usize; 7],
    pub ideal_histogram: [usize; 7],
    pub normal_subnormal_transitions: usize,
    pub sign_mismatches: usize,
    pub worst: Option<Witness>,
    pub worst_ideal: Option<Witness>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    pub dtype: DType,
    pub route: Route,
    pub schedule: Schedule,
    pub chain: bool,
    pub accuracy: AccuracyCell,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Accuracy {
    pub failures: usize,
    pub cells: Vec<Cell>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainFixture {
    pub initial: u64,
    pub rhs: [u64; 3],
    pub schedules: [Schedule; 3],
}
#[derive(Clone, Debug)]
pub struct IntegerFixture {
    pub exponent: i64,
    pub stored: u64,
    pub rows: Vec<Reference>,
}
#[derive(Clone, Copy)]
struct ScalarSpec {
    kind: u8,
    value: u64,
    valid_powi: bool,
    linux_powf: bool,
    linux_powi: bool,
    normalized: u64,
    public_powi_valid: bool,
    public_linux_powi: bool,
    public_linux_powf: bool,
}
include!("typed_power_fixtures.rs");
fn format_index(dtype: DType) -> usize {
    DTYPES.iter().position(|&d| d == dtype).expect("Power float dtype")
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
    let mut b = [0; 8];
    b[..width].copy_from_slice(&FIXTURE_BYTES[at..at + width]);
    u64::from_le_bytes(b)
}
fn reference_at(dtype: DType, index: usize) -> Reference {
    let width = dtype.size();
    let at = RANGES[format_index(dtype)].0 + index * (6 * width + 6);
    Reference {
        lhs: word(at, width),
        rhs: word(at + width, width),
        ideal: word(at + 2 * width, width),
        staged: word(at + 3 * width, width),
        allowed_low: word(at + 4 * width, width),
        allowed_high: word(at + 5 * width, width),
        intermediate_f32: word(at + 6 * width, 4) as u32,
        schedule: SCHEDULES[FIXTURE_BYTES[at + 6 * width + 4] as usize],
        exact_special: FIXTURE_BYTES[at + 6 * width + 5] != 0,
    }
}
pub fn rows(dtype: DType) -> Vec<Reference> {
    (0..RANGES[format_index(dtype)].1)
        .map(|i| reference_at(dtype, i))
        .take_while(|r| r.schedule == Schedule::Mathematical)
        .collect()
}
/// Immutable binary search. Unknown operands never trigger fresh numerical work.
pub fn expected_bits(dtype: DType, schedule: Schedule, lhs: u64, rhs: u64) -> Option<Reference> {
    let fi = DTYPES.iter().position(|&d| d == dtype)?;
    let (mut low, mut high) = (0, RANGES[fi].1);
    while low < high {
        let mid = low + (high - low) / 2;
        let row = reference_at(dtype, mid);
        match (row.schedule, row.lhs, row.rhs).cmp(&(schedule, lhs, rhs)) {
            core::cmp::Ordering::Less => low = mid + 1,
            core::cmp::Ordering::Greater => high = mid,
            core::cmp::Ordering::Equal => return Some(row),
        }
    }
    None
}
pub fn scalar_fixtures(dtype: DType) -> Vec<ScalarFixture> {
    let fi = format_index(dtype);
    SCALAR_SPECS
        .iter()
        .enumerate()
        .map(|(id, spec)| {
            let ScalarSpec {
                kind,
                value,
                valid_powi: valid,
                linux_powf: linux_f,
                linux_powi: linux_i,
                normalized,
                public_powi_valid: public_valid,
                public_linux_powi: public_linux_i,
                public_linux_powf: public_linux_f,
            } = *spec;
            let scalar = match kind {
                0 => Scalar::Float(f64::from_bits(value)),
                1 => Scalar::Int(value as i64),
                2 => Scalar::UInt(value),
                3 => Scalar::Bool(value != 0),
                _ => unreachable!(),
            };
            let (rhs, prhs, schedule, pschedule) = EFFECTIVE[fi][id];
            let get = |schedule, rhs| {
                SCALAR_INPUTS[fi]
                    .iter()
                    .map(|&lhs| expected_bits(dtype, schedule, lhs, rhs).expect("scalar Power fixture"))
                    .collect()
            };
            ScalarFixture {
                id,
                scalar,
                scalar_bits: rhs,
                public_bits: prhs,
                valid_powi: valid,
                linux_powf: linux_f,
                linux_powi: linux_i,
                public_linux_powf: public_linux_f,
                public_linux_powi: public_linux_i,
                inputs: SCALAR_INPUTS[fi].to_vec(),
                primitive: get(Schedule::Mathematical, rhs),
                wrapper: get(schedule, rhs),
                public_primitive: get(Schedule::Mathematical, prhs),
                public_wrapper: get(pschedule, prhs),
                public_powi_scalar: Scalar::Float(f64::from_bits(normalized)),
                public_powi_valid: public_valid,
                public_powi_bits: prhs,
                public_powi_wrapper: get(pschedule, prhs),
            }
        })
        .collect()
}
pub fn integer_fixtures(dtype: DType) -> Vec<IntegerFixture> {
    let fi = format_index(dtype);
    INTEGER_EXPONENTS
        .iter()
        .zip(INTEGER_STORED[fi])
        .map(|(&exponent, &stored)| IntegerFixture {
            exponent,
            stored,
            rows: SCALAR_INPUTS[fi]
                .iter()
                .map(|&lhs| expected_bits(dtype, Schedule::Mathematical, lhs, stored).expect("integer cast Power reference"))
                .collect(),
        })
        .collect()
}
pub fn chain_fixtures(dtype: DType) -> &'static [ChainFixture] {
    &CHAINS[format_index(dtype)]
}
fn masks(dtype: DType) -> (u64, u64, u64) {
    match dtype {
        DType::F64 => (1 << 63, 0x7ff0_0000_0000_0000, 0x000f_ffff_ffff_ffff),
        DType::F32 | DType::Flex32 => (1 << 31, 0x7f80_0000, 0x007f_ffff),
        DType::F16 => (1 << 15, 0x7c00, 0x03ff),
        DType::BF16 => (1 << 15, 0x7f80, 0x007f),
        _ => panic!("Power float dtype"),
    }
}
pub fn classify(dtype: DType, bits: u64) -> Class {
    let (_, exp, frac) = masks(dtype);
    match (bits & exp, bits & frac) {
        (0, 0) => Class::Zero,
        (0, _) => Class::Subnormal,
        (e, 0) if e == exp => Class::Infinity,
        (e, _) if e == exp => Class::Nan,
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
fn distance(dtype: DType, a: u64, b: u64) -> Option<u64> {
    (nonzero_finite(classify(dtype, a)) && nonzero_finite(classify(dtype, b))).then(|| ordered_rank(dtype, a).abs_diff(ordered_rank(dtype, b)))
}
pub fn compare(dtype: DType, row: &Reference, actual: u64) -> Comparison {
    let (wanted, observed) = (classify(dtype, row.staged), classify(dtype, actual));
    let classification_failure = match wanted {
        Class::Nan => observed != Class::Nan,
        Class::Zero | Class::Infinity => actual != row.staged,
        _ => !nonzero_finite(observed) || (actual ^ row.staged) & masks(dtype).0 != 0,
    };
    let exact_special_failure = if row.schedule == Schedule::Identity {
        actual != row.staged
    } else {
        row.exact_special && if wanted == Class::Nan { observed != Class::Nan } else { actual != row.staged }
    };
    let staged_ulp = distance(dtype, actual, row.staged);
    let ideal_ulp = distance(dtype, actual, row.ideal);
    let envelope_member = if wanted == Class::Nan {
        observed == Class::Nan
    } else {
        !is_nan(dtype, actual)
            && ordered_rank(dtype, actual) >= ordered_rank(dtype, row.allowed_low)
            && ordered_rank(dtype, actual) <= ordered_rank(dtype, row.allowed_high)
    };
    let numerical_failure = if row.schedule != Schedule::Mathematical || row.exact_special {
        exact_special_failure
    } else if matches!(dtype, DType::F16 | DType::BF16) {
        !envelope_member
    } else {
        staged_ulp.is_some_and(|n| n > if dtype == DType::F64 { 2 } else { 4 })
    };
    Comparison {
        passed: !classification_failure && !numerical_failure && !exact_special_failure,
        classification_failure,
        numerical_failure,
        exact_special_failure,
        envelope_member,
        normal_subnormal_transition: nonzero_finite(wanted) && nonzero_finite(observed) && wanted != observed,
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
        .map(|part| {
            let mut v = [0; 8];
            v[..part.len()].copy_from_slice(part);
            u64::from_le_bytes(v)
        })
        .collect()
}
fn histogram_bin(n: u64) -> usize {
    match n {
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
    fn cell(&mut self, dtype: DType, route: Route, schedule: Schedule, chain: bool) -> &mut AccuracyCell {
        let index = self
            .cells
            .iter()
            .position(|c| c.dtype == dtype && c.route == route && c.schedule == schedule && c.chain == chain)
            .unwrap_or_else(|| {
                self.cells.push(Cell {
                    dtype,
                    route,
                    schedule,
                    chain,
                    accuracy: AccuracyCell::default(),
                });
                self.cells.len() - 1
            });
        &mut self.cells[index].accuracy
    }
    pub fn record(&mut self, dtype: DType, route: Route, row: &Reference, actual: u64) -> Comparison {
        self.record_inner(dtype, route, row, actual, false)
    }
    pub fn record_chain(&mut self, dtype: DType, route: Route, row: &Reference, actual: u64) -> Comparison {
        self.record_inner(dtype, route, row, actual, true)
    }
    fn record_inner(&mut self, dtype: DType, route: Route, row: &Reference, actual: u64, chain: bool) -> Comparison {
        let result = compare(dtype, row, actual);
        self.failures += usize::from(!result.passed);
        let c = self.cell(dtype, route, row.schedule, chain);
        c.values += 1;
        c.failures += usize::from(!result.passed);
        c.classification_failures += usize::from(result.classification_failure);
        c.numerical_failures += usize::from(result.numerical_failure);
        c.exact_special_failures += usize::from(result.exact_special_failure);
        let observed = classify(dtype, actual) as usize;
        c.observed_classes[observed] += 1;
        c.transitions[classify(dtype, row.staged) as usize][observed] += 1;
        c.ideal_transitions[classify(dtype, row.ideal) as usize][observed] += 1;
        c.normal_subnormal_transitions += usize::from(result.normal_subnormal_transition);
        c.sign_mismatches += usize::from(!is_nan(dtype, actual) && !is_nan(dtype, row.staged) && (actual ^ row.staged) & masks(dtype).0 != 0);
        let witness = Witness {
            lhs: row.lhs,
            rhs: row.rhs,
            schedule: row.schedule,
            actual,
            staged: row.staged,
            ideal: row.ideal,
        };
        if let Some(n) = result.staged_ulp {
            c.finite_pairs += 1;
            c.histogram[histogram_bin(n)] += 1;
            if c.max_ulp.is_none_or(|old| n > old) {
                c.max_ulp = Some(n);
                c.worst = Some(witness);
            }
        }
        if let Some(n) = result.ideal_ulp {
            c.ideal_finite_pairs += 1;
            c.ideal_histogram[histogram_bin(n)] += 1;
            if c.max_ideal_ulp.is_none_or(|old| n > old) {
                c.max_ideal_ulp = Some(n);
                c.worst_ideal = Some(witness);
            }
        }
        result
    }
    pub fn record_unreferenced(&mut self, dtype: DType, route: Route, schedule: Schedule, _lhs: u64, _rhs: u64, actual: u64) {
        self.failures += 1;
        let c = self.cell(dtype, route, schedule, true);
        c.values += 1;
        c.failures += 1;
        c.unreferenced_results += 1;
        let observed = classify(dtype, actual) as usize;
        c.observed_classes[observed] += 1;
        c.unreferenced_observed_classes[observed] += 1;
        c.transitions[Class::Nan as usize][observed] += 1;
        c.ideal_transitions[Class::Nan as usize][observed] += 1;
    }
    pub fn is_coherent(&self) -> bool {
        self.failures == self.cells.iter().map(|c| c.accuracy.failures).sum::<usize>()
            && self.cells.iter().all(|cell| {
                let c = &cell.accuracy;
                let metrics = |hist: &[usize; 7], count, maximum: Option<u64>| {
                    hist.iter().sum::<usize>() == count
                        && match maximum {
                            None => count == 0,
                            Some(n) => count > 0 && hist.iter().rposition(|&v| v != 0) == Some(histogram_bin(n)),
                        }
                };
                let finite_matrix = |matrix: &[[usize; 5]; 5]| {
                    [Class::Subnormal, Class::Normal]
                        .iter()
                        .map(|&expected| {
                            [Class::Subnormal, Class::Normal]
                                .iter()
                                .map(|&actual| matrix[expected as usize][actual as usize])
                                .sum::<usize>()
                        })
                        .sum::<usize>()
                };
                let witness_matches = |witness: Option<Witness>, maximum: Option<u64>, ideal: bool| match (witness, maximum) {
                    (None, None) => true,
                    (Some(w), Some(n)) => {
                        w.schedule == cell.schedule
                            && distance(cell.dtype, w.actual, if ideal { w.ideal } else { w.staged }) == Some(n)
                            && expected_bits(cell.dtype, w.schedule, w.lhs, w.rhs).is_some_and(|row| row.staged == w.staged && row.ideal == w.ideal)
                    }
                    _ => false,
                };
                c.failures <= c.values
                    && c.classification_failures <= c.failures
                    && c.numerical_failures <= c.failures
                    && c.exact_special_failures <= c.failures
                    && c.finite_pairs == finite_matrix(&c.transitions)
                    && c.ideal_finite_pairs == finite_matrix(&c.ideal_transitions)
                    && witness_matches(c.worst, c.max_ulp, false)
                    && witness_matches(c.worst_ideal, c.max_ideal_ulp, true)
                    && c.unreferenced_observed_classes.iter().sum::<usize>() == c.unreferenced_results
                    && c.observed_classes.iter().sum::<usize>() == c.values
                    && c.transitions.iter().flatten().sum::<usize>() == c.values
                    && c.ideal_transitions.iter().flatten().sum::<usize>() == c.values
                    && metrics(&c.histogram, c.finite_pairs, c.max_ulp)
                    && metrics(&c.ideal_histogram, c.ideal_finite_pairs, c.max_ideal_ulp)
                    && (0..5).all(|i| {
                        c.transitions.iter().map(|r| r[i]).sum::<usize>() == c.observed_classes[i]
                            && c.ideal_transitions.iter().map(|r| r[i]).sum::<usize>() == c.observed_classes[i]
                    })
                    && c.unreferenced_results <= c.failures
            })
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceCell {
    pub dtype: DType,
    pub route: Route,
    pub schedule: Schedule,
    pub chain: bool,
    pub values: usize,
    pub reference_classes: [usize; 5],
    pub ideal_classes: [usize; 5],
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    pub cases: usize,
    pub values: usize,
    pub nonchain_values: usize,
    pub chain_values: usize,
    pub tensor_rows: usize,
    pub scalar_specs: usize,
    pub source_checks: usize,
    pub alias_checks: usize,
    pub poison_checks: usize,
    pub cast_checks: usize,
    pub rejected_cases: usize,
    pub extension_cases: usize,
    pub chain_cases: usize,
    pub chain_steps: usize,
    pub cells: Vec<ReferenceCell>,
}
impl Summary {
    fn value(&mut self, dtype: DType, route: Route, row: &Reference, chain: bool) {
        self.values += 1;
        if chain {
            self.chain_values += 1;
        } else {
            self.nonchain_values += 1;
        }
        let index = self
            .cells
            .iter()
            .position(|c| c.dtype == dtype && c.route == route && c.schedule == row.schedule && c.chain == chain)
            .unwrap_or_else(|| {
                self.cells.push(ReferenceCell {
                    dtype,
                    route,
                    schedule: row.schedule,
                    chain,
                    values: 0,
                    reference_classes: [0; 5],
                    ideal_classes: [0; 5],
                });
                self.cells.len() - 1
            });
        let c = &mut self.cells[index];
        c.values += 1;
        c.reference_classes[classify(dtype, row.staged) as usize] += 1;
        c.ideal_classes[classify(dtype, row.ideal) as usize] += 1;
    }
    fn case(&mut self, case: &Case) {
        self.cases += 1;
        self.tensor_rows += case.numerical_rows;
        self.scalar_specs += usize::from(case.scalar.is_some());
        let sources = if case.scalar.is_some() { 1 } else { 2 };
        self.source_checks += sources;
        self.alias_checks += sources;
        self.poison_checks += 1;
        self.cast_checks += usize::from(case.integer.is_some());
        self.rejected_cases += usize::from(case.invalid);
        self.extension_cases += usize::from(case.extension);
        if !case.invalid {
            for row in case.expected() {
                self.value(case.lhs.dtype, case.route, &row, false);
            }
        }
    }
    fn chain(&mut self, dtype: DType, chain: &ChainFixture) {
        self.cases += 1;
        self.chain_cases += 1;
        self.chain_steps += 3;
        self.source_checks += 2 + chain.schedules.iter().filter(|&&s| s == Schedule::Mathematical).count();
        self.alias_checks += 2 + chain.schedules.len() + chain.schedules.iter().filter(|&&s| s == Schedule::Mathematical).count();
        self.poison_checks += 1;
        let mut input = chain.initial;
        for (&rhs, &schedule) in chain.rhs.iter().zip(&chain.schedules) {
            let row = expected_bits(dtype, schedule, input, rhs).expect("planned chain reference");
            self.value(
                dtype,
                if schedule == Schedule::Mathematical {
                    Route::Tensor
                } else {
                    Route::PowfScalar
                },
                &row,
                true,
            );
            input = row.staged;
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub summary: Summary,
    pub accuracy: Accuracy,
    pub failures: Failures,
}
impl Report {
    /// Bind observed work to each planned route and schedule, including chains.
    pub fn observed_counts_match_plan(&self) -> bool {
        let same = |planned: &ReferenceCell, observed: &Cell| {
            planned.dtype == observed.dtype && planned.route == observed.route && planned.schedule == observed.schedule && planned.chain == observed.chain
        };
        self.accuracy.cells.iter().enumerate().all(|(i, observed)| {
            self.summary.cells.iter().any(|planned| same(planned, observed))
                && !self.accuracy.cells[..i].iter().any(|other| {
                    other.dtype == observed.dtype && other.route == observed.route && other.schedule == observed.schedule && other.chain == observed.chain
                })
        }) && self.summary.cells.iter().enumerate().all(|(i, planned)| {
            !self.summary.cells[..i]
                .iter()
                .any(|other| other.dtype == planned.dtype && other.route == planned.route && other.schedule == planned.schedule && other.chain == planned.chain)
                && self
                    .accuracy
                    .cells
                    .iter()
                    .find(|observed| same(planned, observed))
                    .map_or(0, |observed| observed.accuracy.values)
                    == planned.values
        }) && self.accuracy.cells.iter().map(|c| c.accuracy.values).sum::<usize>() == self.summary.values
    }
    pub fn is_success(&self) -> bool {
        self.failures.count == 0 && self.accuracy.failures == 0 && self.accuracy.is_coherent() && self.observed_counts_match_plan()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Values {
    pub shape: Vec<usize>,
    pub values: Vec<u64>,
}
fn coordinates(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut c = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        c[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    c
}
fn offset(shape: &[usize], c: &[usize]) -> usize {
    shape.iter().zip(c).fold(0, |n, (&d, &v)| n * d + v)
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
    fn slice(&self, starts: &[usize], steps: &[usize], shape: &[usize]) -> Self {
        self.mapped(shape, |c| c.iter().enumerate().map(|(i, &v)| starts[i] + v * steps[i]).collect())
    }
    fn permute(&self, axes: &[usize]) -> Self {
        self.mapped(&axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>(), |c| {
            let mut result = vec![0; axes.len()];
            for (&axis, &v) in axes.iter().zip(c) {
                result[axis] = v;
            }
            result
        })
    }
    fn flip(&self, axes: &[usize]) -> Self {
        self.mapped(&self.shape, |c| {
            c.iter()
                .enumerate()
                .map(|(i, &v)| if axes.contains(&i) { self.shape[i] - 1 - v } else { v })
                .collect()
        })
    }
    fn expand(&self, shape: &[usize]) -> Self {
        assert_eq!(broadcast_shape(&self.shape, shape), shape);
        Self::new(shape, (0..shape.iter().product()).map(|i| self.broadcast_at(shape, i)).collect())
    }
    fn broadcast_at(&self, shape: &[usize], index: usize) -> u64 {
        let c = coordinates(shape, index);
        let skip = shape.len() - self.shape.len();
        let own: Vec<_> = self.shape.iter().enumerate().map(|(i, &d)| if d == 1 { 0 } else { c[skip + i] }).collect();
        self.values[offset(&self.shape, &own)]
    }
}
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    let rank = lhs.len().max(rhs.len());
    (0..rank)
        .map(|axis| {
            let a = if axis + lhs.len() >= rank { lhs[axis + lhs.len() - rank] } else { 1 };
            let b = if axis + rhs.len() >= rank { rhs[axis + rhs.len() - rank] } else { 1 };
            assert!(a == b || a == 1 || b == 1);
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
struct Case {
    label: &'static str,
    lhs: Input,
    rhs: Input,
    route: Route,
    scalar: Option<Scalar>,
    schedule: Schedule,
    integer: Option<i64>,
    invalid: bool,
    extension: bool,
    public_powi_tensor: bool,
    same_source: bool,
    numerical_rows: usize,
}
impl Case {
    fn shape(&self) -> Vec<usize> {
        if self.scalar.is_some() || self.integer.is_some() {
            self.lhs.logical().shape
        } else {
            broadcast_shape(&self.lhs.logical().shape, &self.rhs.logical().shape)
        }
    }
    fn expected(&self) -> Vec<Reference> {
        if self.invalid {
            return vec![];
        }
        let a = self.lhs.logical();
        let b = self.rhs.logical();
        let shape = self.shape();
        (0..shape.iter().product())
            .map(|i| {
                let lhs = a.broadcast_at(&shape, i);
                let rhs = if self.scalar.is_some() || self.integer.is_some() {
                    b.values[0]
                } else {
                    b.broadcast_at(&shape, i)
                };
                expected_bits(self.lhs.dtype, self.schedule, lhs, rhs).expect("immutable Power geometry reference")
            })
            .collect()
    }
}
fn selected(dtype: DType, options: &Options) -> bool {
    (!options.no_flex32 || dtype != DType::Flex32) && options.dtype.is_none_or(|d| d == dtype)
}
fn admissible(dtype: DType, row: &Reference, finite: bool) -> bool {
    !finite || (is_finite(dtype, row.lhs) && is_finite(dtype, row.rhs) && is_finite(dtype, row.staged))
}
pub fn selected_rows(dtype: DType, options: &Options) -> Vec<Reference> {
    if !selected(dtype, options) {
        return vec![];
    }
    rows(dtype).into_iter().filter(|r| admissible(dtype, r, options.finite)).collect()
}
fn tensor_case(dtype: DType, rows: &[Reference], route: Route) -> Case {
    Case {
        label: "tensor values",
        lhs: Input::flat(dtype, rows.iter().map(|r| r.lhs).collect()),
        rhs: Input::flat(dtype, rows.iter().map(|r| r.rhs).collect()),
        route,
        scalar: None,
        schedule: Schedule::Mathematical,
        integer: None,
        invalid: false,
        extension: false,
        public_powi_tensor: false,
        same_source: false,
        numerical_rows: 0,
    }
}
fn scalar_case(dtype: DType, fixture: &ScalarFixture, route: Route, options: &Options) -> Option<Case> {
    let (rows, valid, linux) = match route {
        Route::ScalarImpl => (&fixture.primitive, true, true),
        Route::PowiScalarImpl => (&fixture.primitive, fixture.valid_powi, fixture.valid_powi),
        Route::PowfScalar => (&fixture.wrapper, true, fixture.linux_powf),
        Route::PowiScalar => (&fixture.wrapper, fixture.valid_powi, fixture.linux_powi && fixture.valid_powi),
        Route::PublicPowfScalar => (&fixture.public_wrapper, true, fixture.public_linux_powf),
        Route::PublicPowiScalar => (&fixture.public_powi_wrapper, fixture.public_powi_valid, fixture.public_linux_powi),
        _ => unreachable!(),
    };
    if options.common_linux && (!linux || !valid) {
        return None;
    }
    let rows: Vec<_> = rows.iter().copied().filter(|r| admissible(dtype, r, options.finite)).collect();
    if rows.is_empty() {
        return None;
    }
    Some(Case {
        label: "scalar original/public normalization",
        lhs: Input::flat(dtype, rows.iter().map(|r| r.lhs).collect()),
        rhs: Input::flat(dtype, vec![rows[0].rhs]),
        route,
        scalar: Some(fixture.scalar),
        schedule: rows[0].schedule,
        integer: None,
        invalid: !valid,
        extension: !linux,
        public_powi_tensor: false,
        same_source: false,
        numerical_rows: 0,
    })
}
fn view_input(dtype: DType, values: &[u64], geometry: usize, rhs: bool) -> Input {
    let at = |i: usize| values[i % values.len()];
    match geometry {
        0 => Input::shaped(dtype, &[], vec![at(0)]),
        1 => Input::shaped(dtype, &[2, 0, 7], vec![]),
        2 => {
            let mut a = Input::shaped(dtype, &[3, 7], (0..21).map(at).collect());
            a.views.push(View::Permute(vec![1, 0]));
            a.views.push(View::Flip(vec![0, 1]));
            a
        }
        3 => {
            let mut a = Input::shaped(dtype, &[5, 9], (0..45).map(at).collect());
            a.views.push(View::Slice {
                starts: vec![1, 1],
                steps: vec![2, 2],
                shape: vec![2, 3],
            });
            a
        }
        4 | 5 => {
            let mut a = Input::flat(dtype, (0..11).map(at).collect());
            a.views.push(View::Slice {
                starts: vec![1],
                steps: vec![if geometry == 4 { 1 } else { 2 }],
                shape: vec![5],
            });
            if geometry == 5 {
                a.views.push(View::Flip(vec![0]));
            }
            a
        }
        6 | 7 => {
            let short = rhs == (geometry == 7);
            Input::shaped(
                dtype,
                &[if short { 1 } else { 2 }, 3],
                (0..if short { 3 } else { 6 }).map(|i| at(i % 3)).collect(),
            )
        }
        8 => {
            let mut a = Input::shaped(dtype, &[1, 2, 1, 1, 1, 2, 1, 3], (0..12).map(at).collect());
            a.views.push(View::Expand(vec![1, 2, 1, 2, 1, 2, 1, 3]));
            a
        }
        _ => unreachable!(),
    }
}
fn cases(dtype: DType, options: &Options) -> Vec<Case> {
    let selected = selected_rows(dtype, options);
    let mut result = vec![];
    for chunk in selected.chunks(8192) {
        let mut case = tensor_case(dtype, chunk, Route::Tensor);
        case.numerical_rows = chunk.len();
        result.push(case);
    }
    // Both public float tensor APIs use stored floating exponents.
    let public: Vec<_> = selected.iter().step_by((selected.len() / 257).max(1)).take(257).copied().collect();
    for powi in [false, true] {
        let mut case = tensor_case(dtype, &public, Route::PublicTensor);
        case.public_powi_tensor = powi;
        result.push(case);
    }
    let scalars = scalar_fixtures(dtype);
    for fixture in &scalars {
        for route in [
            Route::ScalarImpl,
            Route::PowiScalarImpl,
            Route::PowfScalar,
            Route::PowiScalar,
            Route::PublicPowfScalar,
            Route::PublicPowiScalar,
        ] {
            if let Some(case) = scalar_case(dtype, fixture, route, options) {
                result.push(case);
            }
        }
    }
    for integer in integer_fixtures(dtype) {
        let rows: Vec<_> = integer.rows.into_iter().filter(|r| admissible(dtype, r, options.finite)).collect();
        if rows.is_empty() {
            continue;
        }
        let mut case = tensor_case(dtype, &rows, Route::TensorPowi);
        case.integer = Some(integer.exponent);
        case.rhs = Input::flat(dtype, vec![integer.stored]);
        result.push(case);
    }
    let geometry_pool: Vec<_> = scalars
        .iter()
        .find(|s| matches!(s.scalar, Scalar::Int(3)))
        .expect("cube scalar")
        .primitive
        .iter()
        .copied()
        .filter(|r| admissible(dtype, r, true))
        .collect();
    for length in [0, 1, 2, 3, 127, 128, 129, 255, 256, 257] {
        let rows: Vec<_> = (0..length).map(|i| geometry_pool[i % geometry_pool.len()]).collect();
        result.push(tensor_case(dtype, &rows, Route::Tensor));
    }
    for geometry in 0..9 {
        if geometry == 0 && options.skip_rank_zero {
            continue;
        }
        let values: Vec<_> = geometry_pool.iter().map(|r| r.lhs).collect();
        let mut case = tensor_case(dtype, &[], Route::Tensor);
        case.label = "binary view/broadcast";
        case.lhs = view_input(dtype, &values, geometry, false);
        case.rhs = view_input(dtype, &[geometry_pool[0].rhs], geometry, true);
        result.push(case);
        // Every shortcut also exercises view aliases and retained source bytes.
        for exponent in [0, 1, 2, -1, -2, 3] {
            let fixture = scalars
                .iter()
                .find(|s| matches!(s.scalar,Scalar::Int(n) if n==exponent))
                .expect("wrapper geometry scalar");
            let mut case = scalar_case(dtype, fixture, Route::PowfScalar, &Options { finite: true, ..*options }).expect("scalar geometry");
            case.lhs = view_input(dtype, &case.lhs.root.values, geometry, false);
            case.label = "wrapper alias view";
            result.push(case);
        }
    }
    // All ranks 1..8, with rank zero supplied by geometry 0 above.
    for rank in 1..=8 {
        let mut shape = vec![1; rank];
        shape[rank - 1] = 3;
        let mut case = tensor_case(dtype, &geometry_pool[..3], Route::Tensor);
        case.lhs = Input::shaped(dtype, &shape, geometry_pool[..3].iter().map(|r| r.lhs).collect());
        case.rhs = Input::shaped(dtype, &shape, geometry_pool[..3].iter().map(|r| r.rhs).collect());
        result.push(case);
    }
    // Both operands reference one retained allocation.
    let same = selected
        .iter()
        .find(|r| r.lhs == r.rhs && admissible(dtype, r, true))
        .expect("same-source Power pair");
    let mut case = tensor_case(dtype, &[*same], Route::Tensor);
    case.same_source = true;
    result.push(case);
    result
}
/// Pure work description. No device, observed errors, maxima, or histograms.
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if !selected(dtype, options) {
            continue;
        }
        summary.dtypes += 1;
        for case in cases(dtype, options) {
            summary.case(&case);
        }
        for chain in chain_fixtures(dtype) {
            summary.chain(dtype, chain);
        }
    }
    summary
}
fn cell_key(dtype: DType, route: Route, schedule: Schedule, chain: bool) -> String {
    format!(
        "\"dtype\":\"{}\",\"route\":\"{}\",\"schedule\":\"{}\",\"chain\":{chain}",
        dtype_label(dtype),
        route.label(),
        schedule.label()
    )
}
pub fn summary_json(s: &Summary) -> String {
    let cells: Vec<_> = s
        .cells
        .iter()
        .map(|c| {
            format!(
                "{{{},\"values\":{},\"reference_classes\":{:?},\"ideal_classes\":{:?}}}",
                cell_key(c.dtype, c.route, c.schedule, c.chain),
                c.values,
                c.reference_classes,
                c.ideal_classes
            )
        })
        .collect();
    format!(
        "{{\"dtypes\":{},\"cases\":{},\"values\":{},\"nonchain_values\":{},\"chain_values\":{},\"tensor_rows\":{},\"scalar_specs\":{},\"source_checks\":{},\"alias_checks\":{},\"poison_checks\":{},\"cast_checks\":{},\"rejected_cases\":{},\"extension_cases\":{},\"chain_cases\":{},\"chain_steps\":{},\"cells\":[{}]}}",
        s.dtypes,
        s.cases,
        s.values,
        s.nonchain_values,
        s.chain_values,
        s.tensor_rows,
        s.scalar_specs,
        s.source_checks,
        s.alias_checks,
        s.poison_checks,
        s.cast_checks,
        s.rejected_cases,
        s.extension_cases,
        s.chain_cases,
        s.chain_steps,
        cells.join(",")
    )
}
fn optional_number(n: Option<u64>) -> String {
    n.map_or_else(|| String::from("null"), |n| format!("{n}"))
}
fn witness_json(w: Option<Witness>) -> String {
    w.map_or_else(
        || String::from("null"),
        |w| {
            format!(
                "{{\"lhs\":\"0x{:016x}\",\"rhs\":\"0x{:016x}\",\"schedule\":\"{}\",\"actual\":\"0x{:016x}\",\"staged\":\"0x{:016x}\",\"ideal\":\"0x{:016x}\"}}",
                w.lhs,
                w.rhs,
                w.schedule.label(),
                w.actual,
                w.staged,
                w.ideal
            )
        },
    )
}
pub fn accuracy_json(accuracy: &Accuracy) -> String {
    let cells:Vec<_>=accuracy.cells.iter().map(|cell|{let c=&cell.accuracy;
        format!("{{{},\"accuracy\":{{\"values\":{},\"failures\":{},\"classification_failures\":{},\"numerical_failures\":{},\"exact_special_failures\":{},\"unreferenced_results\":{},\"observed_classes\":{:?},\"unreferenced_observed_classes\":{:?},\"transitions\":{:?},\"ideal_transitions\":{:?},\"finite_pairs\":{},\"ideal_finite_pairs\":{},\"max_ulp\":{},\"max_ideal_ulp\":{},\"histogram\":{:?},\"ideal_histogram\":{:?},\"normal_subnormal_transitions\":{},\"sign_mismatches\":{},\"worst\":{},\"worst_ideal\":{}}}}}",cell_key(cell.dtype,cell.route,cell.schedule,cell.chain),c.values,c.failures,c.classification_failures,c.numerical_failures,c.exact_special_failures,c.unreferenced_results,c.observed_classes,c.unreferenced_observed_classes,c.transitions,c.ideal_transitions,c.finite_pairs,c.ideal_finite_pairs,optional_number(c.max_ulp),optional_number(c.max_ideal_ulp),c.histogram,c.ideal_histogram,c.normal_subnormal_transitions,c.sign_mismatches,witness_json(c.worst),witness_json(c.worst_ideal))
    }).collect();
    format!("{{\"failures\":{},\"cells\":[{}]}}", accuracy.failures, cells.join(","))
}
pub fn success_footer(options: &Options, report: &Report) -> String {
    format!(
        "policy_sha256={POLICY_SHA256} fixture_sha256={FIXTURE_SHA256} qualification_passed={} totals={} accuracy={} reference=directed_mpfr_4.2.2_stored_inputs_f64_2steps_f32_4steps_half_f32_neighbors4 wrapper=stored_rne_schedule scalar=original_variants_public_float_f64 tensor_powi=int_into_float_then_pow nan=class_identity_alias_exact_bytes chain=materialized_each_primitive no_flex32={} skip_rank_zero={} finite={} common_linux={} dtype={}",
        report.is_success(),
        summary_json(&report.summary),
        accuracy_json(&report.accuracy),
        options.no_flex32,
        options.skip_rank_zero,
        options.finite,
        options.common_linux,
        options.dtype.map_or("all", dtype_label)
    )
}
fn check_data(data: &TensorData, dtype: DType, expected: &Values) -> Result<(), String> {
    if data.dtype() != dtype || data.shape().as_slice() != expected.shape || data.as_bytes() != pack(dtype, &expected.values) {
        return Err(format!(
            "retained storage differs dtype={:?} shape={:?} bytes={}",
            data.dtype(),
            data.shape(),
            data.as_bytes().len()
        ));
    }
    Ok(())
}
async fn preserve(tensor: FloatTensor<Dispatch>, dtype: DType, expected: &Values) -> Result<(), String> {
    if tensor.dtype() != dtype || tensor.shape().as_slice() != expected.shape {
        return Err(String::from("retained tensor metadata differs"));
    }
    let data = Dispatch::float_into_data(tensor).await.map_err(|e| format!("retained read: {e:?}"))?;
    check_data(&data, dtype, expected)
}
fn failure(report: &mut Report, label: &str, result: Result<(), String>) {
    if let Err(e) = result {
        report.failures.record(|| format!("{label}: {e}"));
    }
}
fn poison(dtype: DType) -> Input {
    let (_, inf, frac) = masks(dtype);
    Input::flat(dtype, vec![inf | ((frac + 1) >> 1) | 0x15; 37])
}
fn public_scalar(input: FloatTensor<Dispatch>, scalar: Scalar, powi: bool) -> FloatTensor<Dispatch> {
    let tensor = Tensor::<1>::from_dispatch(input);
    if powi {
        match scalar {
            Scalar::Float(v) => tensor.powi_scalar(v),
            Scalar::Int(v) => tensor.powi_scalar(v),
            Scalar::UInt(v) => tensor.powi_scalar(v),
            Scalar::Bool(v) => tensor.powi_scalar(v),
        }
    } else {
        match scalar {
            Scalar::Float(v) => tensor.powf_scalar(v),
            Scalar::Int(v) => tensor.powf_scalar(v),
            Scalar::UInt(v) => tensor.powf_scalar(v),
            Scalar::Bool(v) => tensor.powf_scalar(v),
        }
    }
    .into_dispatch()
}
async fn output(output: FloatTensor<Dispatch>, case: &Case, report: &mut Report) {
    let dtype = case.lhs.dtype;
    if output.dtype() != dtype || output.shape().as_slice() != case.shape() {
        report.failures.record(|| format!("{} output tensor metadata differs", case.label));
    }
    let data = Dispatch::float_into_data(output).await;
    if case.invalid {
        match data {
            Ok(_) => report
                .failures
                .record(|| format!("{:?} invalid powi scalar did not defer an error", case.scalar)),
            Err(error) if error.is_device_poisoned() => report.failures.record(|| format!("invalid powi scalar poisoned the device: {error:?}")),
            Err(_) => {}
        }
        return;
    }
    let data = match data {
        Ok(data) => data,
        Err(e) => {
            report.failures.record(|| format!("{} output read: {e:?}", case.label));
            return;
        }
    };
    let references = case.expected();
    if data.dtype() != dtype || data.shape().as_slice() != case.shape() || data.as_bytes().len() != references.len() * dtype.size() {
        report.failures.record(|| format!("{} output storage metadata differs", case.label));
    }
    if !data.as_bytes().len().is_multiple_of(dtype.size()) {
        return;
    }
    for (i, actual) in unpack(dtype, data.as_bytes()).into_iter().enumerate() {
        if let Some(row) = references.get(i) {
            let comparison = report.accuracy.record(dtype, case.route, row, actual);
            if !comparison.passed {
                report.failures.record(|| {
                    format!(
                        "Power {} route={} schedule={} index={i} lhs={:#018x} rhs={:#018x} actual={actual:#018x} staged={:#018x} ideal={:#018x}",
                        dtype_label(dtype),
                        case.route.label(),
                        row.schedule.label(),
                        row.lhs,
                        row.rhs,
                        row.staged,
                        row.ideal
                    )
                });
            }
        } else {
            report.accuracy.record_unreferenced(dtype, case.route, case.schedule, 0, 0, actual);
            report.failures.record(|| format!("{} extra unreferenced output index={i}", case.label));
        }
    }
}
async fn run_case(device: &Device, case: &Case, report: &mut Report) {
    let dtype = case.lhs.dtype;
    let (source_a, a) = case.lhs.build::<Dispatch>(device.as_dispatch());
    let alias_a = a.clone();
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build::<Dispatch>(device.as_dispatch());
    let mut retained_b = None;
    let mut retained_int = None;
    let result = if let Some(exponent) = case.integer {
        let data = TensorData::from_bytes_vec(exponent.to_le_bytes().to_vec(), vec![1], DType::I64);
        let b = Dispatch::int_from_data(data, device.as_dispatch());
        let alias = b.clone();
        let source = b.clone();
        let cast = Dispatch::int_into_float(b.clone(), dtype.into());
        failure(report, "integer exponent cast", preserve(cast, dtype, &case.rhs.root).await);
        retained_int = Some((source, alias, Values::new(&[1], vec![exponent as u64])));
        Dispatch::float_powi(a, b)
    } else if let Some(scalar) = case.scalar {
        match case.route {
            Route::ScalarImpl => Dispatch::float_powf_scalar_impl(a, scalar),
            Route::PowiScalarImpl => Dispatch::float_powi_scalar_impl(a, scalar),
            Route::PowfScalar => Dispatch::float_powf_scalar(a, scalar),
            Route::PowiScalar => Dispatch::float_powi_scalar(a, scalar),
            Route::PublicPowfScalar => public_scalar(a, scalar, false),
            Route::PublicPowiScalar => public_scalar(a, scalar, true),
            _ => unreachable!(),
        }
    } else {
        let (source, b) = if case.same_source {
            (source_a.clone(), a.clone())
        } else {
            case.rhs.build::<Dispatch>(device.as_dispatch())
        };
        retained_b = Some((source, b.clone()));
        if case.route == Route::PublicTensor {
            let left = Tensor::<1>::from_dispatch(a);
            let right = Tensor::<1>::from_dispatch(b);
            if case.public_powi_tensor { left.powi(right) } else { left.powf(right) }.into_dispatch()
        } else {
            Dispatch::float_powf(a, b)
        }
    };
    output(result, case, report).await;
    failure(report, "lhs source", preserve(source_a, dtype, &case.lhs.root).await);
    failure(report, "lhs alias", preserve(alias_a, dtype, &case.lhs.logical()).await);
    failure(report, "retained poison", preserve(retained_poison, dtype, &poison.root).await);
    if let Some((source, alias)) = retained_b {
        failure(report, "rhs source", preserve(source, dtype, &case.rhs.root).await);
        failure(report, "rhs alias", preserve(alias, dtype, &case.rhs.logical()).await);
    }
    if let Some((source, alias, wanted)) = retained_int {
        for tensor in [source, alias] {
            let result = Dispatch::int_into_data(tensor)
                .await
                .map_err(|e| format!("integer source read: {e:?}"))
                .and_then(|data| check_data(&data, DType::I64, &wanted));
            failure(report, "integer source/alias", result);
        }
    }
    report.summary.case(case);
}
async fn run_chain(device: &Device, dtype: DType, chain: &ChainFixture, report: &mut Report) {
    let input = Input::flat(dtype, vec![chain.initial]);
    let (source, mut current) = input.build::<Dispatch>(device.as_dispatch());
    let alias = current.clone();
    let poison = poison(dtype);
    let (retained_poison, poison_alias) = poison.build::<Dispatch>(device.as_dispatch());
    let mut actual_input = chain.initial;
    let mut retained = vec![];
    let mut divisors = vec![];
    for (step, (&rhs, &schedule)) in chain.rhs.iter().zip(&chain.schedules).enumerate() {
        let route = if schedule == Schedule::Mathematical {
            Route::Tensor
        } else {
            Route::PowfScalar
        };
        let row = expected_bits(dtype, schedule, actual_input, rhs);
        current = if schedule == Schedule::Mathematical {
            let input = Input::flat(dtype, vec![rhs]);
            let (source, b) = input.build::<Dispatch>(device.as_dispatch());
            divisors.push((source, b.clone(), input.root));
            Dispatch::float_powf(current, b)
        } else {
            let exponent = match schedule {
                Schedule::Ones => 0,
                Schedule::Identity => 1,
                Schedule::Square => 2,
                Schedule::Reciprocal => -1,
                Schedule::SquareReciprocal => -2,
                Schedule::Mathematical => unreachable!(),
            };
            Dispatch::float_powf_scalar(current, Scalar::Int(exponent))
        };
        // Materialization prevents Fusion from contracting stored schedules.
        match Dispatch::float_into_data(current.clone()).await {
            Ok(data) => {
                if data.dtype() != dtype || data.shape().as_slice() != [1] || data.as_bytes().len() != dtype.size() {
                    report.failures.record(|| format!("chain step={step} metadata differs"));
                }
                if data.as_bytes().len() == dtype.size() {
                    let actual = unpack(dtype, data.as_bytes())[0];
                    if let Some(row) = row {
                        if !report.accuracy.record_chain(dtype, route, &row, actual).passed {
                            report.failures.record(|| {
                                format!(
                                    "chain step={step} schedule={} lhs={actual_input:#x} rhs={rhs:#x} actual={actual:#x}",
                                    schedule.label()
                                )
                            });
                        }
                    } else {
                        report.accuracy.record_unreferenced(dtype, route, schedule, actual_input, rhs, actual);
                        report
                            .failures
                            .record(|| format!("chain step={step} missing immutable reference lhs={actual_input:#x} rhs={rhs:#x}"));
                    }
                    retained.push((current.clone(), Values::new(&[1], vec![actual])));
                    actual_input = actual;
                }
            }
            Err(e) => report.failures.record(|| format!("chain step={step} read: {e:?}")),
        }
    }
    drop(current);
    for (tensor, wanted) in retained {
        failure(report, "chain stored-stage alias", preserve(tensor, dtype, &wanted).await);
    }
    for (source, alias, wanted) in divisors {
        failure(report, "chain exponent source", preserve(source, dtype, &wanted).await);
        failure(report, "chain exponent alias", preserve(alias, dtype, &wanted).await);
    }
    for (label, tensor, wanted) in [
        ("chain source", source, &input.root),
        ("chain alias", alias, &input.root),
        ("chain poison", retained_poison, &poison.root),
        ("chain poison alias", poison_alias, &poison.root),
    ] {
        failure(report, label, preserve(tensor, dtype, wanted).await);
    }
    report.summary.chain(dtype, chain);
}
pub async fn run(device: &Device, options: &Options) -> Result<Report, String> {
    let mut report = Report::default();
    for dtype in DTYPES {
        if !selected(dtype, options) {
            continue;
        }
        report.summary.dtypes += 1;
        for case in cases(dtype, options) {
            run_case(device, &case, &mut report).await;
        }
        for chain in chain_fixtures(dtype) {
            run_chain(device, dtype, chain, &mut report).await;
        }
    }
    let planned = expected_summary(options);
    if report.summary != planned {
        report.failures.record(|| String::from("Power executed work differs from the pure plan"));
    }
    if !report.accuracy.is_coherent() {
        report.failures.record(|| String::from("Power observed accuracy accounting is incoherent"));
    }
    if !report.observed_counts_match_plan() {
        report
            .failures
            .record(|| String::from("Power observed cell keys or counts differ from the pure plan"));
    }
    Ok(report)
}
