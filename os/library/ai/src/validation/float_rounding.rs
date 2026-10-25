//! Bounded sqrt and integral rounding against independent exact fixtures.
//!
//! Finite results, infinities and signed zeros require exact stored bits. Public
//! NaNs require classification. Raw integral payload checks are separate.
//! Retained poison checks certify bytes, not physical padding or adjacency.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{Backend as BurnBackend, DType, Dispatch, TensorData, TensorMetadata, tensor::FloatTensor},
    tensor::{Device, Slice, Tensor, TensorCreationOptions},
};

pub const DTYPES: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
pub const POOL_LENGTH: usize = 64;
pub const CORPUS_VERSION: u32 = 1;
const LENGTHS: [usize; 10] = [0, 1, 2, 3, 127, 128, 129, 255, 256, 257];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Operation {
    Sqrt,
    Floor,
    Ceil,
    Round,
    Trunc,
}
impl Operation {
    pub const ALL: [Self; 5] = [Self::Sqrt, Self::Floor, Self::Ceil, Self::Round, Self::Trunc];
}

/// Default selects the full five-format, five-operation corpus.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub no_flex32: bool,
    /// Omit actual [] inputs, without replacing them with [1].
    pub skip_rank_zero: bool,
    /// Require finite logical operands AND finite expected results.
    /// Unused source slots and independent retained poison can still be NaNs.
    pub finite: bool,
    pub dtype: Option<DType>,
    /// Selecting an operation omits mixed-operation chains.
    pub operation: Option<Operation>,
}
#[derive(Clone, Copy, Debug)]
pub struct ChainStep {
    pub operation: Operation,
    pub expected: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct ChainFixture {
    pub initial: u64,
    pub steps: &'static [ChainStep],
}
include!("float_rounding_fixtures.rs");

fn format_index(dtype: DType) -> usize {
    match dtype {
        DType::F64 => 0,
        DType::F32 | DType::Flex32 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        _ => panic!("float rounding fixture dtype required"),
    }
}
fn read_word(start: usize, size: usize) -> u64 {
    let mut bytes = [0; 8];
    bytes[..size].copy_from_slice(&FIXTURE_BYTES[start..start + size]);
    u64::from_le_bytes(bytes)
}
fn word(dtype: DType, index: usize) -> u64 {
    read_word(POOL_OFFSETS[format_index(dtype)] + index * dtype.size(), dtype.size())
}
/// Immutable bounded inputs. Flex32 shares F32 encodings and retains its dtype.
pub fn pool(dtype: DType) -> Vec<u64> {
    (0..POOL_LENGTH).map(|i| word(dtype, i)).collect()
}
/// Independent fixtures for every half code; None for other dtypes.
pub fn half_expected_bits(dtype: DType, operation: Operation, input: u16) -> Option<u16> {
    let index = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        _ => return None,
    };
    Some(read_word(HALF_OFFSETS[index].0 + (operation as usize * 65536 + usize::from(input)) * 2, 2) as u16)
}
/// Exact F32 sqrt followed by exact destination-half RNE, not Linux approximation.
pub fn staged_sqrt_bits(dtype: DType, input: u16) -> Option<u16> {
    let index = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        _ => return None,
    };
    Some(read_word(HALF_OFFSETS[index].1 + usize::from(input) * 2, 2) as u16)
}
/// Look up any half code or a bounded F32/F64 pool input. Runtime never computes
/// expected arithmetic. NaN bits here do not strengthen the public class contract.
pub fn expected_bits(dtype: DType, operation: Operation, input: u64) -> Option<u64> {
    if !DTYPES.contains(&dtype) {
        return None;
    }
    if matches!(dtype, DType::F16 | DType::BF16) {
        return half_expected_bits(dtype, operation, u16::try_from(input).ok()?).map(u64::from);
    }
    let index = (0..POOL_LENGTH).find(|&i| word(dtype, i) == input)?;
    Some(word(dtype, (operation as usize + 1) * POOL_LENGTH + index))
}
/// Separate raw bitwise integral extension. Do not apply this to Linux public
/// comparisons or to preserved intrinsic F32 floor/ceil/trunc implementations.
pub fn raw_integral_nan_bits(dtype: DType, operation: Operation, input: u64) -> Option<u64> {
    if DTYPES.contains(&dtype) && operation != Operation::Sqrt && is_nan(dtype, input) && (dtype.size() == 8 || input < (1u64 << (dtype.size() * 8))) {
        Some(input)
    } else {
        None
    }
}
pub fn chain_fixtures(dtype: DType) -> &'static [ChainFixture] {
    CHAINS[format_index(dtype)]
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
    operation: Operation,
    input: Input,
    high_level: bool,
}
impl Case {
    fn expected(&self) -> Values {
        let logical = self.input.logical();
        Values::new(
            &logical.shape,
            logical
                .values
                .iter()
                .map(|&x| expected_bits(self.input.dtype, self.operation, x).expect("fixture covers public input"))
                .collect(),
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    /// Primitive calls, including every materialized chain step.
    pub operations: [usize; 5],
    pub unary_cases: usize,
    pub high_level_cases: usize,
    pub chain_cases: usize,
    pub chain_steps: usize,
    pub source_checks: usize,
    pub alias_checks: usize,
    pub poison_checks: usize,
    /// Destination values, including each checked chain intermediate.
    pub values: usize,
    pub exact_values: usize,
    pub nan_values: usize,
}
impl Summary {
    pub fn cases(&self) -> usize {
        self.unary_cases + self.high_level_cases + self.chain_cases
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
            self.unary_cases += 1;
        }
        self.source_checks += 1;
        self.alias_checks += 1;
        self.poison_checks += 1;
        self.results(case.input.dtype, &case.expected().values);
    }
    fn chain(&mut self, dtype: DType, chain: &ChainFixture) {
        self.chain_cases += 1;
        self.chain_steps += chain.steps.len();
        self.source_checks += 1;
        self.alias_checks += 1;
        self.poison_checks += 1;
        for step in chain.steps {
            self.operations[step.operation as usize] += 1;
            self.results(dtype, &[step.expected]);
        }
    }
}
fn selected(dtype: DType, options: &Options) -> bool {
    (!options.no_flex32 || dtype != DType::Flex32) && options.dtype.is_none_or(|only| only == dtype)
}
fn finite_input(dtype: DType, operation: Operation, input: u64) -> bool {
    is_finite(dtype, input) && is_finite(dtype, expected_bits(dtype, operation, input).expect("fixture lookup"))
}
fn flat_case(dtype: DType, operation: Operation, length: usize, finite: bool, high_level: bool) -> Case {
    let values: Vec<_> = pool(dtype).into_iter().filter(|&x| !finite || finite_input(dtype, operation, x)).collect();
    Case {
        label: if high_level {
            "high-level IEEE diagnostic pool"
        } else {
            "writer boundary length"
        },
        operation,
        input: Input::flat(dtype, (0..length).map(|i| values[i % values.len()]).collect()),
        high_level,
    }
}
fn geometry(dtype: DType, operation: Operation, geometry: usize) -> Case {
    // Positive finite, distinguishable neighbors isolate addressing. Domain and
    // class boundaries occur in every diagnostic vector and all long flat cases.
    let p: Vec<_> = pool(dtype)
        .into_iter()
        .filter(|&x| is_finite(dtype, x) && !is_nan(dtype, expected_bits(dtype, Operation::Sqrt, x).unwrap()))
        .collect();
    let value = |i: usize| p[(i * 7 + 3) % p.len()];
    let (input, label) = match geometry {
        0 => (Input::shaped(dtype, &[], vec![value(9)]), "rank zero"),
        1 | 3 => {
            let mut root = vec![pool(dtype)[4]; 41];
            for i in 0..17 {
                root[1 + 2 * i] = value(i);
            }
            let mut input = Input::flat(dtype, root);
            input.views.push(View::Slice {
                starts: vec![1],
                steps: vec![2],
                shape: vec![17],
            });
            if geometry == 3 {
                input.views.push(View::Flip(vec![0]));
            }
            (
                input,
                if geometry == 1 {
                    "odd offset and positive step"
                } else {
                    "reversed offset and step"
                },
            )
        }
        2 => {
            let mut input = Input::shaped(dtype, &[3, 5], (0..15).map(value).collect());
            input.views.push(View::Permute(vec![1, 0]));
            (input, "permuted distinguishable neighbors")
        }
        4 => {
            let mut input = Input::shaped(dtype, &[1, 3], (0..3).map(value).collect());
            input.views.push(View::Expand(vec![4, 3]));
            (input, "zero-stride expanded reader")
        }
        5 => {
            let mut input = Input::shaped(dtype, &[1, 2, 1, 1, 1, 2, 1, 3], (0..12).map(value).collect());
            input.views.push(View::Flip(vec![7]));
            input.views.push(View::Expand(vec![1, 2, 1, 2, 1, 2, 1, 3]));
            (input, "rank eight reversal and singleton expansion")
        }
        _ => unreachable!(),
    };
    Case {
        label,
        operation,
        input,
        high_level: false,
    }
}
fn cases(dtype: DType, options: &Options) -> Vec<Case> {
    let mut cases = Vec::new();
    for operation in Operation::ALL {
        if options.operation.is_some_and(|only| only != operation) {
            continue;
        }
        for length in LENGTHS {
            cases.push(flat_case(dtype, operation, length, options.finite, false));
        }
        cases.push(Case {
            label: "empty internal zero axis",
            operation,
            input: Input::shaped(dtype, &[2, 0, 7], vec![]),
            high_level: false,
        });
        for view in 0..6 {
            if options.skip_rank_zero && view == 0 {
                continue;
            }
            cases.push(geometry(dtype, operation, view));
        }
        let length = pool(dtype)
            .into_iter()
            .filter(|&x| !options.finite || finite_input(dtype, operation, x))
            .count();
        cases.push(flat_case(dtype, operation, length, options.finite, true));
    }
    cases
}
fn selected_chains<'a>(dtype: DType, options: &'a Options) -> impl Iterator<Item = &'static ChainFixture> + 'a {
    chain_fixtures(dtype).iter().filter(move |chain| {
        options.operation.is_none() && (!options.finite || (is_finite(dtype, chain.initial) && chain.steps.iter().all(|s| is_finite(dtype, s.expected))))
    })
}
/// Device-free enumeration. Filters change real work, never the comparison rule.
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
fn apply_unary<B: BurnBackend>(op: Operation, input: FloatTensor<B>) -> FloatTensor<B> {
    match op {
        Operation::Sqrt => B::float_sqrt(input),
        Operation::Floor => B::float_floor(input),
        Operation::Ceil => B::float_ceil(input),
        Operation::Round => B::float_round(input),
        Operation::Trunc => B::float_trunc(input),
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
    Input::flat(dtype, (0..37).map(|i| p[4 + i % 4]).collect())
}
async fn run_case(device: &Device, case: &Case, summary: &mut Summary) -> Result<(), String> {
    let dtype = case.input.dtype;
    let expected = case.expected();
    let (source, input) = if case.high_level {
        let tensor = Tensor::<1>::from_data(
            TensorData::from_bytes_vec(pack(dtype, &case.input.root.values), case.input.root.shape.clone(), dtype),
            TensorCreationOptions::new(device.clone()).with_dtype(dtype),
        );
        (tensor.clone().into_dispatch(), tensor.into_dispatch())
    } else {
        case.input.build::<Dispatch>(device.as_dispatch())
    };
    let alias = input.clone();
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build::<Dispatch>(device.as_dispatch());
    let output = if case.high_level {
        let tensor = Tensor::<1>::from_dispatch(input);
        match case.operation {
            Operation::Sqrt => tensor.sqrt(),
            Operation::Floor => tensor.floor(),
            Operation::Ceil => tensor.ceil(),
            Operation::Round => tensor.round(),
            Operation::Trunc => tensor.trunc(),
        }
        .into_dispatch()
    } else {
        apply_unary::<Dispatch>(case.operation, input)
    };
    check::<Dispatch>(output, dtype, &expected, false).await?;
    check::<Dispatch>(source, dtype, &case.input.root, true)
        .await
        .map_err(|e| format!("source preservation: {e}"))?;
    check::<Dispatch>(alias, dtype, &case.input.logical(), true)
        .await
        .map_err(|e| format!("alias preservation: {e}"))?;
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
    let mut intermediates = Vec::new();
    for (i, step) in chain.steps.iter().enumerate() {
        current = apply_unary::<Dispatch>(step.operation, current);
        // Read the stored destination before constructing the next primitive.
        // This also forces Linux Fusion to respect destination precision.
        check::<Dispatch>(current.clone(), dtype, &Values::new(&[1], vec![step.expected]), false)
            .await
            .map_err(|e| format!("chain step {i} {:?}: {e}", step.operation))?;
        intermediates.push(current.clone());
    }
    drop(current);
    check::<Dispatch>(source, dtype, &input.root, true).await?;
    check::<Dispatch>(alias, dtype, &input.root, true).await?;
    check::<Dispatch>(retained_poison, dtype, &poison.root, true).await?;
    // Keep every destination alive through all consumers, including the sqrt
    // witness. Source and intermediate references prevent mutable reuse.
    drop(intermediates);
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
                .map_err(|e| format!("rounding case={i} {dtype:?} {:?} {}: {e}", case.operation, case.label))?;
        }
        for (i, chain) in selected_chains(dtype, options).enumerate() {
            run_chain(device, dtype, chain, &mut summary)
                .await
                .map_err(|e| format!("rounding chain={i} {dtype:?}: {e}"))?;
        }
    }
    let expected = expected_summary(options);
    if summary != expected {
        return Err(format!("rounding corpus totals {summary:?}, expected {expected:?}"));
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_corpus_and_filters_cover_only_selected_finite_work() {
        let full = expected_summary(&Options::default());
        assert_eq!(full.cases(), 465);
        assert_eq!(full.chain_cases, 15);
        assert_eq!(full.chain_steps, 40);
        assert_eq!(full.high_level_cases, 25);
        assert_eq!(full.exact_values + full.nan_values, full.values);
        assert_eq!(full.operations.iter().sum::<usize>(), full.cases() - full.chain_cases + full.chain_steps);
        for dtype in DTYPES {
            for finite in [false, true] {
                let options = Options { finite, ..Options::default() };
                for case in cases(dtype, &options) {
                    assert!(case.expected().values.len() <= 257);
                    if finite {
                        assert!(case.input.logical().values.iter().all(|&x| is_finite(dtype, x)));
                        assert!(case.expected().values.iter().all(|&x| is_finite(dtype, x)));
                    }
                }
            }
            let selected = expected_summary(&Options {
                dtype: Some(dtype),
                ..Options::default()
            });
            assert_eq!(selected.dtypes, 1);
            assert_eq!(selected.cases(), 93);
            assert!(
                cases(
                    dtype,
                    &Options {
                        skip_rank_zero: true,
                        ..Options::default()
                    }
                )
                .iter()
                .all(|c| !c.input.logical().shape.is_empty())
            );
        }
        assert_eq!(
            expected_summary(&Options {
                finite: true,
                ..Options::default()
            })
            .nan_values,
            0
        );
        assert_eq!(
            full.cases()
                - expected_summary(&Options {
                    skip_rank_zero: true,
                    ..Options::default()
                })
                .cases(),
            25
        );
        assert_eq!(
            full.cases()
                - expected_summary(&Options {
                    no_flex32: true,
                    ..Options::default()
                })
                .cases(),
            93
        );
        for op in Operation::ALL {
            let s = expected_summary(&Options {
                operation: Some(op),
                ..Options::default()
            });
            assert_eq!(s.chain_cases, 0);
            assert_eq!(s.cases(), 90);
            for other in Operation::ALL {
                assert_eq!(s.operations[other as usize], if op == other { 90 } else { 0 });
            }
        }
    }
}
