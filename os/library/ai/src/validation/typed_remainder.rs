//! Typed tensor and scalar remainder against independent exact fixtures.
//!
//! Native finite values, signed zeros and infinities require exact bits.
//! Output NaNs require classification. Retained input bytes require exact bits.
//! Linux expression schedules are diagnostics, never this numerical oracle.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{Backend as BurnBackend, DType, Dispatch, Scalar, TensorData, TensorMetadata, ops::FloatTensorOps, tensor::FloatTensor},
    tensor::{Device, Slice, Tensor, TensorCreationOptions},
};

pub const DTYPES: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
pub const CORPUS_VERSION: u32 = 1;
const LENGTHS: [usize; 10] = [0, 1, 2, 3, 127, 128, 129, 255, 256, 257];

/// Default runs all five formats under the strict Native comparison policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub no_flex32: bool,
    pub skip_rank_zero: bool,
    /// Both effective operands and the expected output must be finite.
    pub finite: bool,
    pub dtype: Option<DType>,
    /// Diagnostic numerical group only. Excludes supplemental geometry/chains.
    pub group: Option<&'static str>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    pub lhs: u64,
    pub rhs: u64,
    pub expected: u64,
    pub group: &'static str,
}
#[derive(Clone, Debug)]
pub struct ScalarCase {
    pub scalar: Scalar,
    pub scalar_bits: u64,
    pub inputs: Vec<u64>,
    pub expected: Vec<u64>,
}
#[derive(Clone, Copy, Debug)]
pub struct ChainFixture {
    pub initial: u64,
    pub rhs: [u64; 3],
    pub expected: [u64; 3],
}
include!("typed_remainder_fixtures.rs");
fn format_index(dtype: DType) -> usize {
    DTYPES.iter().position(|&d| d == dtype).expect("typed remainder float dtype")
}
fn read_word(start: usize) -> u64 {
    u64::from_le_bytes(FIXTURE_BYTES[start..start + 8].try_into().expect("fixture word"))
}
pub fn pairs(dtype: DType) -> Vec<Pair> {
    let (offset, count) = PAIR_RANGES[format_index(dtype)];
    (0..count)
        .map(|i| {
            let at = offset + i * 32;
            Pair {
                lhs: read_word(at),
                rhs: read_word(at + 8),
                expected: read_word(at + 16),
                group: GROUPS[read_word(at + 24) as usize],
            }
        })
        .collect()
}
pub fn scalar_fixtures(dtype: DType) -> Vec<ScalarCase> {
    let (offset, count) = SCALAR_RANGES[format_index(dtype)];
    (0..count)
        .map(|i| {
            let at = offset + i * 23 * 8;
            let value = read_word(at + 8);
            let scalar = match read_word(at) {
                0 => Scalar::Float(f64::from_bits(value)),
                1 => Scalar::Int(value as i64),
                2 => Scalar::UInt(value),
                3 => Scalar::Bool(value != 0),
                _ => unreachable!("fixture scalar kind"),
            };
            ScalarCase {
                scalar,
                scalar_bits: read_word(at + 16),
                inputs: (0..10).map(|j| read_word(at + (3 + j) * 8)).collect(),
                expected: (0..10).map(|j| read_word(at + (13 + j) * 8)).collect(),
            }
        })
        .collect()
}
/// Half scalar storage witnesses supplement the 221 reference specifications.
pub fn storage_scalar_fixtures(dtype: DType) -> Vec<ScalarCase> {
    STORAGE_CASES
        .iter()
        .filter(|r| DTYPES[r.0] == dtype)
        .map(|&(_, source, scalar_bits, lhs, expected)| ScalarCase {
            scalar: Scalar::Float(f64::from_bits(source)),
            scalar_bits,
            inputs: vec![lhs],
            expected: vec![expected],
        })
        .collect()
}
pub fn chain_fixtures(dtype: DType) -> &'static [ChainFixture] {
    &CHAINS[format_index(dtype)]
}
/// Immutable table lookup. Runtime never calculates expected remainder.
pub fn expected_bits(dtype: DType, lhs: u64, rhs: u64) -> Option<u64> {
    if !DTYPES.contains(&dtype) {
        return None;
    }
    if let Some(row) = pairs(dtype).iter().find(|p| p.lhs == lhs && p.rhs == rhs) {
        return Some(row.expected);
    }
    for row in scalar_fixtures(dtype).into_iter().chain(storage_scalar_fixtures(dtype)) {
        if row.scalar_bits == rhs
            && let Some(i) = row.inputs.iter().position(|&v| v == lhs)
        {
            return Some(row.expected[i]);
        }
    }
    for chain in chain_fixtures(dtype) {
        let mut input = chain.initial;
        for (&divisor, &expected) in chain.rhs.iter().zip(&chain.expected) {
            if lhs == input && rhs == divisor {
                return Some(expected);
            }
            input = expected;
        }
    }
    None
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
struct Case {
    label: &'static str,
    lhs: Input,
    rhs: Input,
    scalar: Option<ScalarCase>,
    high_level: bool,
    same_source: bool,
    tensor_rows: usize,
    scalar_specs: usize,
    scalar_results: usize,
}
impl Case {
    fn expected(&self) -> Values {
        let a = self.lhs.logical();
        let b = self.rhs.logical();
        let shape = if self.scalar.is_some() {
            a.shape.clone()
        } else {
            broadcast_shape(&a.shape, &b.shape)
        };
        let table = pairs(self.lhs.dtype);
        let values = (0..shape.iter().product())
            .map(|i| {
                let lhs = a.broadcast_at(&shape, i);
                if let Some(row) = &self.scalar {
                    return row.expected[row.inputs.iter().position(|&x| x == lhs).expect("scalar geometry fixture")];
                }
                let rhs = b.broadcast_at(&shape, i);
                table.iter().find(|p| p.lhs == lhs && p.rhs == rhs).expect("tensor geometry fixture").expected
            })
            .collect();
        Values::new(&shape, values)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    pub tensor_cases: usize,
    pub scalar_cases: usize,
    /// Public wrappers, counted within tensor_cases/scalar_cases too.
    pub high_level_cases: usize,
    pub chain_cases: usize,
    pub chain_steps: usize,
    /// Original numerical corpus coverage, excluding public duplicates/geometry.
    pub tensor_rows: usize,
    pub scalar_specs: usize,
    pub scalar_results: usize,
    pub source_checks: usize,
    pub alias_checks: usize,
    pub poison_checks: usize,
    pub values: usize,
    pub exact_values: usize,
    pub nan_values: usize,
}
impl Summary {
    pub fn cases(&self) -> usize {
        self.tensor_cases + self.scalar_cases + self.chain_cases
    }
    fn results(&mut self, dtype: DType, values: &[u64]) {
        let nan = values.iter().filter(|&&x| is_nan(dtype, x)).count();
        self.values += values.len();
        self.nan_values += nan;
        self.exact_values += values.len() - nan;
    }
    fn record(&mut self, case: &Case) {
        if case.scalar.is_some() {
            self.scalar_cases += 1;
        } else {
            self.tensor_cases += 1;
        }
        self.high_level_cases += usize::from(case.high_level);
        self.tensor_rows += case.tensor_rows;
        self.scalar_specs += case.scalar_specs;
        self.scalar_results += case.scalar_results;
        let sources = if case.scalar.is_some() { 1 } else { 2 };
        self.source_checks += sources;
        self.alias_checks += sources;
        self.poison_checks += 1;
        self.results(case.lhs.dtype, &case.expected().values);
    }
    fn chain(&mut self, dtype: DType, chain: &ChainFixture) {
        self.chain_cases += 1;
        self.chain_steps += chain.rhs.len();
        self.source_checks += 1 + chain.rhs.len();
        self.alias_checks += 1 + chain.rhs.len();
        self.poison_checks += 1;
        self.results(dtype, &chain.expected);
    }
}
fn selected(dtype: DType, options: &Options) -> bool {
    (!options.no_flex32 || dtype != DType::Flex32) && options.dtype.is_none_or(|only| only == dtype)
}
fn admissible(dtype: DType, a: u64, b: u64, expected: u64, finite: bool) -> bool {
    !finite || (is_finite(dtype, a) && is_finite(dtype, b) && is_finite(dtype, expected))
}
fn scalar_selection(dtype: DType, finite: bool) -> Vec<ScalarCase> {
    scalar_fixtures(dtype)
        .into_iter()
        .filter_map(|mut row| {
            let retained: Vec<_> = row
                .inputs
                .iter()
                .copied()
                .zip(row.expected.iter().copied())
                .filter(|&(a, expected)| admissible(dtype, a, row.scalar_bits, expected, finite))
                .collect();
            (row.inputs, row.expected) = retained.into_iter().unzip();
            (!row.inputs.is_empty()).then_some(row)
        })
        .collect()
}
fn flat_pair_case(dtype: DType, rows: &[Pair], high_level: bool, numerical: bool) -> Case {
    Case {
        label: if numerical {
            "numerical tensor group"
        } else {
            "tensor boundary/public vector"
        },
        lhs: Input::flat(dtype, rows.iter().map(|r| r.lhs).collect()),
        rhs: Input::flat(dtype, rows.iter().map(|r| r.rhs).collect()),
        scalar: None,
        high_level,
        same_source: false,
        tensor_rows: if numerical { rows.len() } else { 0 },
        scalar_specs: 0,
        scalar_results: 0,
    }
}
fn scalar_case(dtype: DType, row: ScalarCase, high_level: bool, numerical: bool) -> Case {
    Case {
        label: if numerical {
            "numerical original Scalar variant"
        } else {
            "public scalar Float conversion"
        },
        lhs: Input::flat(dtype, row.inputs.clone()),
        rhs: Input::flat(dtype, vec![]),
        scalar_specs: usize::from(numerical),
        scalar_results: if numerical { row.inputs.len() } else { 0 },
        scalar: Some(row),
        high_level,
        same_source: false,
        tensor_rows: 0,
    }
}
fn view_input(dtype: DType, values: &[u64], geometry: usize, rhs: bool) -> Input {
    let value = |i: usize| values[i % values.len()];
    match geometry {
        0 => Input::shaped(dtype, &[], vec![value(0)]),
        1 => Input::shaped(dtype, &[2, 0, 7], vec![]),
        2 => {
            let mut input = Input::shaped(dtype, &[3, 7], (0..21).map(value).collect());
            input.views.push(View::Permute(vec![1, 0]));
            input.views.push(View::Flip(vec![0, 1]));
            input
        }
        3 => {
            let mut input = Input::shaped(dtype, &[5, 9], (0..45).map(value).collect());
            input.views.push(View::Slice {
                starts: vec![1, 1],
                steps: vec![2, 2],
                shape: vec![2, 3],
            });
            input
        }
        4 | 5 => {
            let mut input = Input::flat(dtype, (0..11).map(value).collect());
            input.views.push(View::Slice {
                starts: vec![1],
                steps: vec![if geometry == 4 { 1 } else { 2 }],
                shape: vec![5],
            });
            if geometry == 5 {
                input.views.push(View::Flip(vec![0]));
            }
            input
        }
        6 | 7 => {
            let short = rhs == (geometry == 7);
            Input::shaped(
                dtype,
                &[if short { 1 } else { 2 }, 3],
                (0..if short { 3 } else { 6 }).map(|i| value(i % 3)).collect(),
            )
        }
        8 => {
            let mut input = Input::shaped(dtype, &[1, 2, 1, 1, 1, 2, 1, 3], (0..12).map(value).collect());
            input.views.push(View::Flip(vec![7]));
            input.views.push(View::Expand(vec![1, 2, 1, 2, 1, 2, 1, 3]));
            input
        }
        9 => Input::shaped(dtype, &[3, 7], vec![value(0); 21]),
        _ => unreachable!(),
    }
}
fn cases(dtype: DType, options: &Options) -> Vec<Case> {
    let selected: Vec<_> = pairs(dtype)
        .into_iter()
        .filter(|p| options.group.is_none_or(|g| p.group == g) && admissible(dtype, p.lhs, p.rhs, p.expected, options.finite))
        .collect();
    let mut cases = Vec::new();
    for group in GROUPS {
        let rows: Vec<_> = selected.iter().copied().filter(|p| p.group == group).collect();
        if !rows.is_empty() {
            cases.push(flat_pair_case(dtype, &rows, false, true));
        }
    }
    if options.group.is_some() {
        return cases;
    }
    cases.push(flat_pair_case(dtype, &selected, true, false));
    let scalar_rows = scalar_selection(dtype, options.finite);
    for row in scalar_rows {
        cases.push(scalar_case(dtype, row.clone(), false, true));
        cases.push(scalar_case(dtype, row, true, false));
    }
    for row in storage_scalar_fixtures(dtype) {
        cases.push(scalar_case(dtype, row.clone(), false, false));
        cases.push(scalar_case(dtype, row, true, false));
    }
    let geometry_pairs: Vec<_> = selected.iter().copied().filter(|p| admissible(dtype, p.lhs, p.rhs, p.expected, true)).collect();
    // Int(-3) is finite in every format and has ten independently stored inputs.
    let scalar = scalar_selection(dtype, true)
        .into_iter()
        .find(|r| matches!(r.scalar, Scalar::Int(-3)))
        .expect("geometry scalar");
    for length in LENGTHS {
        cases.push(flat_pair_case(
            dtype,
            &(0..length).map(|i| selected[i % selected.len()]).collect::<Vec<_>>(),
            false,
            false,
        ));
        let mut case = scalar_case(dtype, scalar.clone(), false, false);
        case.label = "scalar writer boundary";
        case.lhs = Input::flat(dtype, (0..length).map(|i| scalar.inputs[i % scalar.inputs.len()]).collect());
        cases.push(case);
    }
    for geometry in 0..10 {
        if options.skip_rank_zero && geometry == 0 {
            continue;
        }
        let rows: Vec<_> = if geometry == 9 {
            vec![*geometry_pairs.iter().find(|p| p.lhs == p.rhs && p.lhs != 0).expect("same source row")]
        } else {
            geometry_pairs.clone()
        };
        let mut case = flat_pair_case(dtype, &rows, false, false);
        case.label = "tensor geometry";
        case.same_source = geometry == 9;
        case.lhs = view_input(dtype, &rows.iter().map(|p| p.lhs).collect::<Vec<_>>(), geometry, false);
        case.rhs = view_input(dtype, &rows.iter().map(|p| p.rhs).collect::<Vec<_>>(), geometry, true);
        cases.push(case);
        let mut case = scalar_case(dtype, scalar.clone(), false, false);
        case.label = "scalar geometry";
        case.lhs = view_input(dtype, &scalar.inputs, geometry, false);
        cases.push(case);
    }
    cases
}
/// No device or runtime initialization. Uses exactly the same selected work as run.
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if !selected(dtype, options) {
            continue;
        }
        let cases = cases(dtype, options);
        if cases.is_empty() && options.group.is_some() {
            continue;
        }
        summary.dtypes += 1;
        for case in cases {
            summary.record(&case);
        }
        if options.group.is_none() {
            for chain in chain_fixtures(dtype) {
                summary.chain(dtype, chain);
            }
        }
    }
    summary
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
            "storage metadata {:?} {:?} bytes={}",
            data.dtype(),
            data.shape(),
            data.as_bytes().len()
        ));
    }
    let mut errors = Vec::new();
    for (i, (actual, &wanted)) in unpack(dtype, data.as_bytes()).into_iter().zip(&expected.values).enumerate() {
        if actual != wanted && (raw || !is_nan(dtype, actual) || !is_nan(dtype, wanted)) {
            errors.push(format!("element={i} actual={actual:#018x} expected={wanted:#018x}"));
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
}
fn poison(dtype: DType) -> Input {
    let nan = pairs(dtype).into_iter().find(|r| is_nan(dtype, r.lhs)).expect("payload fixture").lhs;
    Input::flat(dtype, vec![nan; 37])
}
fn public_input(device: &Device, input: &Input) -> FloatTensor<Dispatch> {
    Tensor::<1>::from_data(
        TensorData::from_bytes_vec(pack(input.dtype, &input.root.values), input.root.shape.clone(), input.dtype),
        TensorCreationOptions::new(device.clone()).with_dtype(input.dtype),
    )
    .into_dispatch()
}
fn public_scalar(input: FloatTensor<Dispatch>, scalar: Scalar) -> FloatTensor<Dispatch> {
    // Public Burn converts each primitive argument with Scalar::new(dtype).
    // Direct Dispatch below separately exercises the original Scalar variant.
    let tensor = Tensor::<1>::from_dispatch(input);
    match scalar {
        Scalar::Float(v) => tensor.remainder_scalar(v),
        Scalar::Int(v) => tensor.remainder_scalar(v),
        Scalar::UInt(v) => tensor.remainder_scalar(v),
        Scalar::Bool(v) => tensor.remainder_scalar(v),
    }
    .into_dispatch()
}
async fn run_case(device: &Device, case: &Case, summary: &mut Summary) -> Result<(), String> {
    let dtype = case.lhs.dtype;
    let expected = case.expected();
    let (source_a, a) = if case.high_level {
        let a = public_input(device, &case.lhs);
        (a.clone(), a)
    } else {
        case.lhs.build::<Dispatch>(device.as_dispatch())
    };
    let alias_a = a.clone();
    let rhs = if case.scalar.is_some() {
        None
    } else if case.same_source {
        Some((source_a.clone(), a.clone()))
    } else if case.high_level {
        let b = public_input(device, &case.rhs);
        Some((b.clone(), b))
    } else {
        Some(case.rhs.build::<Dispatch>(device.as_dispatch()))
    };
    let alias_b = rhs.as_ref().map(|(_, b)| b.clone());
    let retained_b = rhs.as_ref().map(|(source, _)| source.clone());
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build::<Dispatch>(device.as_dispatch());
    let output = if let Some(row) = &case.scalar {
        if case.high_level {
            public_scalar(a, row.scalar)
        } else {
            Dispatch::float_remainder_scalar(a, row.scalar)
        }
    } else {
        let (_, b) = rhs.expect("tensor rhs");
        if case.high_level {
            Tensor::<1>::from_dispatch(a).remainder(Tensor::<1>::from_dispatch(b)).into_dispatch()
        } else {
            Dispatch::float_remainder(a, b)
        }
    };
    // Check every selected vector and retained source even when arithmetic fails.
    let mut errors = Vec::new();
    for (label, result) in [
        ("output", check::<Dispatch>(output, dtype, &expected, false).await),
        ("lhs source", check::<Dispatch>(source_a, dtype, &case.lhs.root, true).await),
        ("lhs alias", check::<Dispatch>(alias_a, dtype, &case.lhs.logical(), true).await),
        ("retained poison", check::<Dispatch>(retained_poison, dtype, &poison.root, true).await),
    ] {
        if let Err(e) = result {
            errors.push(format!("{label}: {e}"));
        }
    }
    if let (Some(source), Some(alias)) = (retained_b, alias_b) {
        for (label, result) in [
            ("rhs source", check::<Dispatch>(source, dtype, &case.rhs.root, true).await),
            ("rhs alias", check::<Dispatch>(alias, dtype, &case.rhs.logical(), true).await),
        ] {
            if let Err(e) = result {
                errors.push(format!("{label}: {e}"));
            }
        }
    }
    summary.record(case);
    if errors.is_empty() { Ok(()) } else { Err(errors.join("\n")) }
}
async fn run_chain(device: &Device, dtype: DType, chain: &ChainFixture, summary: &mut Summary) -> Result<(), String> {
    let input = Input::flat(dtype, vec![chain.initial]);
    let (source, mut current) = input.build::<Dispatch>(device.as_dispatch());
    let alias = current.clone();
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build::<Dispatch>(device.as_dispatch());
    let mut errors = Vec::new();
    let mut intermediates = Vec::new();
    let mut sources = Vec::new();
    for (step, (&rhs, &expected)) in chain.rhs.iter().zip(&chain.expected).enumerate() {
        let divisor = Input::flat(dtype, vec![rhs]);
        let (source_b, b) = divisor.build::<Dispatch>(device.as_dispatch());
        let alias_b = b.clone();
        current = Dispatch::float_remainder(current, b);
        // Reading each stored destination prevents Fusion from contracting stages.
        if let Err(e) = check::<Dispatch>(current.clone(), dtype, &Values::new(&[1], vec![expected]), false).await {
            errors.push(format!("chain step={step}: {e}"));
        }
        intermediates.push(current.clone());
        sources.push((source_b, alias_b, divisor.root));
    }
    drop(current);
    for (label, tensor, wanted) in [
        ("chain source", source, input.root.clone()),
        ("chain alias", alias, input.root),
        ("chain poison", retained_poison, poison.root),
    ] {
        if let Err(e) = check::<Dispatch>(tensor, dtype, &wanted, true).await {
            errors.push(format!("{label}: {e}"));
        }
    }
    for (source, alias, wanted) in sources {
        for tensor in [source, alias] {
            if let Err(e) = check::<Dispatch>(tensor, dtype, &wanted, true).await {
                errors.push(format!("chain divisor preservation: {e}"));
            }
        }
    }
    drop(intermediates);
    summary.chain(dtype, chain);
    if errors.is_empty() { Ok(()) } else { Err(errors.join("\n")) }
}
pub async fn run(device: &Device, options: &Options) -> Result<Summary, String> {
    let mut summary = Summary::default();
    let mut errors = Vec::new();
    for dtype in DTYPES {
        if !selected(dtype, options) {
            continue;
        }
        let cases = cases(dtype, options);
        if cases.is_empty() && options.group.is_some() {
            continue;
        }
        summary.dtypes += 1;
        for (i, case) in cases.into_iter().enumerate() {
            if let Err(e) = run_case(device, &case, &mut summary).await {
                errors.push(format!("remainder {dtype:?} case={i} {}: {e}", case.label));
            }
        }
        if options.group.is_none() {
            for (i, chain) in chain_fixtures(dtype).iter().enumerate() {
                if let Err(e) = run_chain(device, dtype, chain, &mut summary).await {
                    errors.push(format!("remainder {dtype:?} chain={i}: {e}"));
                }
            }
        }
    }
    let expected = expected_summary(options);
    if summary != expected {
        errors.push(format!("remainder corpus totals {summary:?}, expected {expected:?}"));
    }
    if errors.is_empty() { Ok(summary) } else { Err(errors.join("\n")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_selection_checks_effective_operands_and_all_geometry_expectations() {
        for dtype in DTYPES {
            for case in cases(
                dtype,
                &Options {
                    finite: true,
                    ..Options::default()
                },
            ) {
                assert!(case.lhs.logical().values.iter().all(|&v| is_finite(dtype, v)));
                if let Some(row) = &case.scalar {
                    assert!(is_finite(dtype, row.scalar_bits));
                } else {
                    assert!(case.rhs.logical().values.iter().all(|&v| is_finite(dtype, v)));
                }
                assert!(case.expected().values.iter().all(|&v| is_finite(dtype, v)));
            }
            let all = expected_summary(&Options {
                dtype: Some(dtype),
                ..Options::default()
            });
            let grouped: usize = GROUPS
                .iter()
                .map(|&group| {
                    let options = Options {
                        dtype: Some(dtype),
                        group: Some(group),
                        ..Options::default()
                    };
                    let s = expected_summary(&options);
                    assert_eq!((s.scalar_specs, s.chain_cases, s.high_level_cases), (0, 0, 0));
                    s.tensor_rows
                })
                .sum();
            assert_eq!(grouped, all.tensor_rows);
        }
    }
    #[test]
    fn independent_coordinate_traversal_handles_both_broadcast_directions_and_signed_views() {
        let root = Values::new(&[3, 7], (0..21).collect());
        let transformed = root.permute(&[1, 0]).flip(&[0, 1]);
        for row in 0..7 {
            for col in 0..3 {
                assert_eq!(transformed.values[row * 3 + col], ((2 - col) * 7 + 6 - row) as u64);
            }
        }
        let short = Values::new(&[1, 3], vec![10, 11, 12]);
        for (a, b) in [(&[1, 3][..], &[2, 3][..]), (&[2, 3][..], &[1, 3][..])] {
            assert_eq!(broadcast_shape(a, b), vec![2, 3]);
        }
        assert_eq!(short.expand(&[2, 3]).values, vec![10, 11, 12, 10, 11, 12]);
        assert_eq!(broadcast_shape(&[2, 0, 7], &[1, 1, 7]), vec![2, 0, 7]);
        assert_eq!(root.slice(&[1, 1], &[1, 2], &[2, 3]).values, vec![8, 10, 12, 15, 17, 19]);
    }
}
