//! Bounded typed ReLU checks with an independent integer IEEE reference.
//! Every result byte is exact, including NaN payloads and signed zeros.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::{
        Backend as BurnBackend, BoolDType, DType, Dispatch, TensorData, TensorMetadata,
        ops::{ActivationOps, FloatTensorOps},
        tensor::FloatTensor,
    },
    tensor::{Device, Slice, Tensor, TensorCreationOptions},
};

pub const CORPUS_VERSION: u32 = 1;
pub const DTYPES: [DType; 5] = [DType::F64, DType::F32, DType::Flex32, DType::F16, DType::BF16];
pub const POOL_LENGTH: usize = 32;
const LENGTHS: [usize; 6] = [0, 1, 127, 128, 129, 257];

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub no_flex32: bool,
    pub skip_rank_zero: bool,
    pub dtype: Option<DType>,
}

fn layout(dtype: DType) -> (u32, u32) {
    match dtype {
        DType::F64 => (11, 52),
        DType::F32 | DType::Flex32 => (8, 23),
        DType::F16 => (5, 10),
        DType::BF16 => (8, 7),
        _ => panic!("ReLU reference requires a stored float dtype"),
    }
}
fn masks(dtype: DType) -> (u64, u64, u64) {
    let (exponent, fraction) = layout(dtype);
    (1 << (exponent + fraction), ((1 << exponent) - 1) << fraction, (1 << fraction) - 1)
}
/// Integer IEEE classification. No host float or device result enters the oracle.
pub fn is_nan(dtype: DType, bits: u64) -> bool {
    let (_, exponent, fraction) = masks(dtype);
    bits & exponent == exponent && bits & fraction != 0
}
/// Ordered <= +0: NaNs are unordered, and both signed zeros compare equal.
pub fn lower_equal_zero(dtype: DType, bits: u64) -> bool {
    let (sign, _, _) = masks(dtype);
    !is_nan(dtype, bits) && (bits & sign != 0 || bits & (sign - 1) == 0)
}
/// ReLU keeps positive and unordered stored bits and replaces <=0 with +0.
pub fn expected_bits(dtype: DType, bits: u64) -> u64 {
    if lower_equal_zero(dtype, bits) { 0 } else { bits }
}
/// Pinned ActivationOps backward masks the stored forward output, not the input.
pub fn backward_bits(dtype: DType, output: u64, gradient: u64) -> u64 {
    if lower_equal_zero(dtype, output) { 0 } else { gradient }
}
/// Signed pairs cover subnormals, normal boundaries, infinities and quiet/signaling NaNs.
pub fn pool(dtype: DType) -> Vec<u64> {
    let (e, f) = layout(dtype);
    let (sign, exponent, fraction) = masks(dtype);
    let one = ((1 << (e - 1)) - 1) << f;
    let quiet = 1 << (f - 1);
    [
        0,
        1,
        2,
        fraction,
        1 << f,
        one - 1,
        one,
        one + 1,
        one + (1 << f),
        one + (10 << f),
        exponent - 1,
        exponent,
        exponent | quiet | 1,
        exponent | quiet | 0x35,
        exponent | 1,
        exponent | 0x25,
    ]
    .into_iter()
    .flat_map(|x| [x, x | sign])
    .collect()
}
/// Nonuniform destination-encoded gradients include signs, tiny values and payloads.
pub fn gradient_pool(dtype: DType) -> Vec<u64> {
    let p = pool(dtype);
    (0..p.len()).map(|i| p[(i * 11 + 7) % p.len()]).collect()
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
    input: Input,
    high_level: bool,
    unique: bool,
}
fn flat(dtype: DType, length: usize) -> Input {
    let p = pool(dtype);
    Input::flat(dtype, (0..length).map(|i| p[i % p.len()]).collect())
}
fn cases(dtype: DType, options: &Options) -> Vec<Case> {
    let mut cases: Vec<_> = LENGTHS
        .into_iter()
        .map(|length| Case {
            label: "writer boundary",
            input: flat(dtype, length),
            high_level: false,
            unique: false,
        })
        .collect();
    cases.push(Case {
        label: "empty interior axis",
        input: Input::shaped(dtype, &[2, 0, 7], vec![]),
        high_level: false,
        unique: false,
    });
    let p = pool(dtype);
    let value = |i: usize| p[(i * 7 + 3) % p.len()];
    for geometry in 0..6 {
        if geometry == 0 && options.skip_rank_zero {
            continue;
        }
        let (input, label) = match geometry {
            0 => (Input::shaped(dtype, &[], vec![p[15]]), "true rank zero"),
            1 | 3 => {
                let mut root = vec![p[25]; 41];
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
                        "odd offset and stride"
                    } else {
                        "signed reversed offset view"
                    },
                )
            }
            2 => {
                let mut input = Input::shaped(dtype, &[3, 5], (0..15).map(value).collect());
                input.views.push(View::Permute(vec![1, 0]));
                (input, "transposed view")
            }
            4 => {
                let mut input = Input::shaped(dtype, &[1, 3], (0..3).map(value).collect());
                input.views.push(View::Expand(vec![4, 3]));
                (input, "broadcast zero stride")
            }
            5 => {
                let mut input = Input::shaped(dtype, &[1, 2, 1, 1, 1, 2, 1, 3], (0..12).map(value).collect());
                input.views.push(View::Flip(vec![7]));
                input.views.push(View::Expand(vec![1, 2, 1, 2, 1, 2, 1, 3]));
                (input, "rank eight signed broadcast view")
            }
            _ => unreachable!(),
        };
        cases.push(Case {
            label,
            input,
            high_level: false,
            unique: false,
        });
    }
    cases.push(Case {
        label: "public Tensor explicit dtype",
        input: flat(dtype, POOL_LENGTH),
        high_level: true,
        unique: false,
    });
    cases.push(Case {
        label: "unique Dispatch input",
        input: flat(dtype, POOL_LENGTH),
        high_level: false,
        unique: true,
    });
    cases
}
fn forward(dtype: DType, values: &Values) -> Values {
    Values::new(&values.shape, values.values.iter().map(|&x| expected_bits(dtype, x)).collect())
}
fn gradient(dtype: DType, shape: &[usize]) -> Values {
    let p = gradient_pool(dtype);
    Values::new(shape, (0..shape.iter().product()).map(|i| p[i % p.len()]).collect())
}
fn backward(dtype: DType, output: &Values, grad: &Values) -> Values {
    Values::new(
        &output.shape,
        output.values.iter().zip(&grad.values).map(|(&x, &g)| backward_bits(dtype, x, g)).collect(),
    )
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub dtypes: usize,
    /// Includes high-level and unique cases.
    pub forward_cases: usize,
    pub high_level_cases: usize,
    pub unique_cases: usize,
    pub backward_cases: usize,
    pub idempotence_cases: usize,
    /// Explicit <=0 and mask-fill compositions, one per bool storage encoding.
    pub mask_cases: usize,
    /// Materialized forward/backward/idempotence and bool/mask destinations.
    pub chain_steps: usize,
    pub source_checks: usize,
    pub alias_checks: usize,
    pub gradient_checks: usize,
    pub output_checks: usize,
    pub poison_checks: usize,
    /// Float result values only. Every value requires exact stored bits.
    pub values: usize,
    pub nan_values: usize,
    pub bool_values: usize,
}
impl Summary {
    pub fn cases(&self) -> usize {
        self.forward_cases + self.mask_cases
    }
    fn results(&mut self, dtype: DType, values: &Values) {
        self.values += values.values.len();
        self.nan_values += values.values.iter().filter(|&&x| is_nan(dtype, x)).count();
    }
    fn record(&mut self, case: &Case) {
        let dtype = case.input.dtype;
        let output = forward(dtype, &case.input.logical());
        let grad = gradient(dtype, &output.shape);
        self.forward_cases += 1;
        self.high_level_cases += usize::from(case.high_level);
        self.unique_cases += usize::from(case.unique);
        self.backward_cases += 1;
        self.idempotence_cases += 1;
        self.chain_steps += 3;
        self.source_checks += usize::from(!case.unique);
        self.alias_checks += usize::from(!case.unique);
        self.gradient_checks += 1;
        self.output_checks += 1;
        self.poison_checks += 1;
        self.results(dtype, &output);
        self.results(dtype, &backward(dtype, &output, &grad));
        self.results(dtype, &output);
    }
    fn mask(&mut self, dtype: DType) {
        self.mask_cases += 1;
        self.chain_steps += 2;
        self.source_checks += 1;
        self.alias_checks += 1;
        self.poison_checks += 1;
        self.bool_values += POOL_LENGTH;
        self.results(dtype, &forward(dtype, &flat(dtype, POOL_LENGTH).root));
    }
}
fn selected(dtype: DType, options: &Options) -> bool {
    (!options.no_flex32 || dtype != DType::Flex32) && options.dtype.is_none_or(|only| only == dtype)
}
/// Enumerate exactly the same bounded work without constructing a device.
pub fn expected_summary(options: &Options) -> Summary {
    let mut summary = Summary::default();
    for dtype in DTYPES {
        if !selected(dtype, options) {
            continue;
        }
        summary.dtypes += 1;
        for case in cases(dtype, options) {
            summary.record(&case);
        }
        for _ in [BoolDType::U8, BoolDType::U32] {
            summary.mask(dtype);
        }
    }
    summary
}
async fn check(tensor: FloatTensor<Dispatch>, dtype: DType, expected: &Values) -> Result<(), String> {
    if tensor.dtype() != dtype || tensor.shape().as_slice() != expected.shape {
        return Err(format!(
            "metadata {:?} {:?}, expected {dtype:?} {:?}",
            tensor.dtype(),
            tensor.shape(),
            expected.shape
        ));
    }
    let data = Dispatch::float_into_data(tensor).await.map_err(|e| format!("read: {e:?}"))?;
    let wanted = pack(dtype, &expected.values);
    if data.dtype() != dtype || data.shape().as_slice() != expected.shape || data.as_bytes().len() != wanted.len() {
        return Err(format!(
            "read metadata {:?} {:?} bytes={}, expected bytes={}",
            data.dtype(),
            data.shape(),
            data.as_bytes().len(),
            wanted.len()
        ));
    }
    if let Some(i) = data.as_bytes().iter().zip(&wanted).position(|(a, b)| a != b) {
        let actual = unpack(dtype, data.as_bytes());
        let element = i / dtype.size();
        return Err(format!(
            "exact byte mismatch element={element} byte={i} actual={:#018x} expected={:#018x}",
            actual[element], expected.values[element]
        ));
    }
    Ok(())
}
fn poison(dtype: DType) -> Input {
    let p = pool(dtype);
    Input::flat(dtype, (0..37).map(|i| p[24 + i % 8]).collect())
}
async fn run_case(device: &Device, case: &Case, summary: &mut Summary) -> Result<(), String> {
    let dtype = case.input.dtype;
    let expected = forward(dtype, &case.input.logical());
    let (source, input) = if case.high_level {
        let tensor = Tensor::<1>::from_data(
            TensorData::from_bytes_vec(pack(dtype, &case.input.root.values), case.input.root.shape.clone(), dtype),
            TensorCreationOptions::new(device.clone()).with_dtype(dtype),
        );
        (tensor.clone().into_dispatch(), tensor.into_dispatch())
    } else {
        case.input.build::<Dispatch>(device.as_dispatch())
    };
    let (source, alias) = if case.unique {
        drop(source);
        (None, None)
    } else {
        (Some(source), Some(input.clone()))
    };
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build::<Dispatch>(device.as_dispatch());
    let output = if case.high_level {
        burn::tensor::activation::relu(Tensor::<1>::from_dispatch(input)).into_dispatch()
    } else {
        <Dispatch as ActivationOps<Dispatch>>::relu(input)
    };
    // Read each destination before constructing its consumer. Fusion cannot hide
    // intermediate storage precision, metadata or payload changes.
    check(output.clone(), dtype, &expected).await.map_err(|e| format!("forward: {e}"))?;
    let grad = gradient(dtype, &expected.shape);
    let (_, grad_tensor) = Input::shaped(dtype, &grad.shape, grad.values.clone()).build::<Dispatch>(device.as_dispatch());
    let grad_alias = grad_tensor.clone();
    let result = <Dispatch as ActivationOps<Dispatch>>::relu_backward(output.clone(), grad_tensor);
    check(result, dtype, &backward(dtype, &expected, &grad))
        .await
        .map_err(|e| format!("backward: {e}"))?;
    let repeated = <Dispatch as ActivationOps<Dispatch>>::relu(output.clone());
    check(repeated, dtype, &expected).await.map_err(|e| format!("idempotence: {e}"))?;
    check(output, dtype, &expected).await.map_err(|e| format!("retained output: {e}"))?;
    check(grad_alias, dtype, &grad).await.map_err(|e| format!("gradient preservation: {e}"))?;
    if let Some(source) = source {
        check(source, dtype, &case.input.root).await.map_err(|e| format!("source preservation: {e}"))?;
    }
    if let Some(alias) = alias {
        check(alias, dtype, &case.input.logical())
            .await
            .map_err(|e| format!("alias preservation: {e}"))?;
    }
    check(retained_poison, dtype, &poison.root)
        .await
        .map_err(|e| format!("poison preservation: {e}"))?;
    summary.record(case);
    Ok(())
}
async fn run_mask(device: &Device, dtype: DType, bool_dtype: BoolDType, summary: &mut Summary) -> Result<(), String> {
    use burn::backend::ops::BoolTensorOps;
    let input = flat(dtype, POOL_LENGTH);
    let (source, tensor) = input.build::<Dispatch>(device.as_dispatch());
    let alias = tensor.clone();
    let poison = poison(dtype);
    let (retained_poison, _) = poison.build::<Dispatch>(device.as_dispatch());
    let mask = Dispatch::float_lower_equal_elem(tensor.clone(), 0.into(), bool_dtype);
    let data = Dispatch::bool_into_data(mask.clone()).await.map_err(|e| format!("bool read: {e:?}"))?;
    let expected_mask: Vec<_> = input
        .root
        .values
        .iter()
        .flat_map(|&x| u32::from(lower_equal_zero(dtype, x)).to_le_bytes()[..DType::Bool(bool_dtype).size()].to_vec())
        .collect();
    if data.dtype() != DType::Bool(bool_dtype) || data.shape().as_slice() != input.root.shape || data.as_bytes() != expected_mask {
        return Err(format!("exact {bool_dtype:?} mask mismatch: {data:?}"));
    }
    let output = Dispatch::float_mask_fill(tensor, mask, 0.into());
    check(output, dtype, &forward(dtype, &input.root)).await?;
    check(source, dtype, &input.root).await?;
    check(alias, dtype, &input.root).await?;
    check(retained_poison, dtype, &poison.root).await?;
    summary.mask(dtype);
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
                .map_err(|e| format!("ReLU {dtype:?} case={i} {}: {e}", case.label))?;
        }
        for bool_dtype in [BoolDType::U8, BoolDType::U32] {
            run_mask(device, dtype, bool_dtype, &mut summary)
                .await
                .map_err(|e| format!("ReLU {dtype:?} {bool_dtype:?}: {e}"))?;
        }
    }
    let expected = expected_summary(options);
    if summary != expected {
        return Err(format!("ReLU totals {summary:?}, expected {expected:?}"));
    }
    Ok(summary)
}
