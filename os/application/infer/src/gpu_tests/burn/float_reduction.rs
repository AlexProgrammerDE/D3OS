//! Serial typed floating reductions: packed writers, public routing and residency.
//! Hardware execution remains gated on the root's final source/corpus approval.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Shape, Slice, TensorData, TensorMetadata, ops::FloatTensorOps, tensor::FloatTensor},
        tensor::try_read_sync,
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::float_reduction::{self as oracle, Input, Operation, Scope, Values, View},
};
use d3os_compute::{
    Backend as _, Error,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{self as abi, Kernel, cast::Format, float_reduction as raw, reduction::Reduction};

type Result<T> = core::result::Result<T, String>;
// Only ROOT may approve these identities and enable runtime acceptance.
const FROZEN: bool = true;
const REFERENCE_SHA256: &str = "84021711460c5ae060eb774326cef0c6f69a44300b5ba4d778f066c6abaea249";
const GUARD: u32 = 0x6bad_beef;
const DTYPES: [DType; 4] = [DType::F64, DType::Flex32, DType::F16, DType::BF16];
const FORMATS: [Format; 4] = [Format::F64, Format::Flex32, Format::F16, Format::BF16];
const OPERATIONS: [Operation; 3] = [Operation::Sum, Operation::Mean, Operation::Product];
fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("float reduction: {e}"))
}
fn format_of(dtype: DType) -> Format {
    FORMATS[DTYPES.iter().position(|&d| d == dtype).expect("new numeric dtype")]
}
fn raw_op(op: Operation) -> raw::Operation {
    match op {
        Operation::Sum => raw::Operation::Sum,
        Operation::Mean => raw::Operation::Mean,
        Operation::Product => raw::Operation::Product,
    }
}
fn reduction(op: Operation) -> Reduction {
    match op {
        Operation::Sum => Reduction::Sum,
        Operation::Mean => Reduction::Mean,
        Operation::Product => Reduction::Product,
    }
}
fn data(dtype: DType, values: &Values) -> TensorData {
    TensorData::from_bytes_vec(oracle::pack(dtype, &values.values), Shape::from(values.shape.clone()), dtype)
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect()
}
fn output_words(dtype: DType, values: &Values) -> Vec<u32> {
    words(&oracle::pack(dtype, &values.values))
}
fn check_data(actual: &TensorData, dtype: DType, expected: &Values) -> Result<()> {
    if actual.dtype() != dtype || actual.shape().as_slice() != expected.shape || actual.as_bytes() != oracle::pack(dtype, &expected.values) {
        return Err(format!(
            "dtype={dtype:?} shape={:?} actual_dtype={:?} actual_shape={:?} expected_bits={:?} actual_bits={:?}",
            expected.shape,
            actual.dtype(),
            actual.shape(),
            &expected.values[..expected.values.len().min(8)],
            &actual.as_bytes()[..actual.as_bytes().len().min(32)]
        ));
    }
    Ok(())
}
fn dense(shape: &[usize], values: Vec<u64>) -> Input {
    Input {
        root: Values::new(shape, values),
        views: vec![],
    }
}
fn slices(starts: &[usize], steps: &[usize], shape: &[usize]) -> Vec<Slice> {
    starts
        .iter()
        .zip(steps)
        .zip(shape)
        .map(|((&start, &step), &length)| {
            Slice::new(
                start as isize,
                Some((start + length.saturating_sub(1) * step + usize::from(length > 0)) as isize),
                step as isize,
            )
        })
        .collect()
}
// Build signed strides on the physical root, independently of Input::logical.
fn layout(input: &Input) -> Result<Layout> {
    let mut layout = gpu(Layout::contiguous(Shape::from(input.root.shape.clone())))?;
    for view in &input.views {
        layout = gpu(match view {
            View::Permute(axes) => layout.permute(axes),
            View::Flip(axes) => layout.flip(axes),
            View::Slice { starts, steps, shape } => layout.slice(&slices(starts, steps, shape)),
            View::Expand(shape) => layout.expand(Shape::from(shape.clone())),
        })?;
    }
    Ok(layout)
}

#[derive(Clone)]
struct Case {
    id: String,
    dtype: DType,
    input: Input,
    op: Operation,
    scope: Scope,
}
impl Case {
    fn expected(&self) -> Result<Values> {
        oracle::reference(self.dtype, &self.input.logical(), self.op, self.scope).map_err(|e| format!("reference: {e:?}"))
    }
    fn parameters(&self) -> Result<[u32; 24]> {
        let Scope::Axis(axis) = self.scope else {
            return Err(String::from("raw ALL is forbidden"));
        };
        let layout = layout(&self.input)?;
        let mut p = [0; 24];
        p[..8].copy_from_slice(&[
            self.expected()?.values.len() as u32,
            layout.shape().num_dims() as u32,
            raw::control(raw_op(self.op), format_of(self.dtype)).map_err(|e| format!("control: {e:?}"))?,
            axis as u32,
            layout.offset() as u32,
            self.input.root.values.len() as u32,
            layout.shape()[axis] as u32,
            raw::SERIAL_POLICY,
        ]);
        p[8..16].fill(1);
        for (a, (&dim, &stride)) in layout.shape().iter().zip(layout.strides()).enumerate() {
            p[8 + a] = dim as u32;
            p[16 + a] = stride as i32 as u32;
        }
        Ok(p)
    }
}
fn case(dtype: DType, input: Input, op: Operation, scope: Scope, id: impl Into<String>) -> Case {
    Case {
        id: id.into(),
        dtype,
        input,
        op,
        scope,
    }
}
fn signed(dtype: DType, n: i64) -> u64 {
    oracle::integer_bits(dtype, n.unsigned_abs()) | if n < 0 { 1u64 << (dtype.size() * 8 - 1) } else { 0 }
}
// Each row isolates a witness: a NaN cannot conceal a finite arithmetic fault.
fn fixture(dtype: DType, length: usize) -> Input {
    let one = signed(dtype, 1);
    let sign = 1u64 << (dtype.size() * 8 - 1);
    let (huge, maximum, normal, inf, payload) = match dtype {
        DType::F64 => (
            0x4340000000000000,
            0x7fefffffffffffff,
            0x0010000000000000,
            0x7ff0000000000000,
            0xfff0000000000123,
        ),
        DType::Flex32 => (0x4b800000, 0x7f7fffff, 0x00800000, 0x7f800000, 0xff800123),
        DType::F16 => (0x7400, 0x7bff, 0x0400, 0x7c00, 0xfc23),
        DType::BF16 => (0x4b80, 0x7f7f, 0x0080, 0x7f80, 0xff83),
        _ => unreachable!(),
    };
    let half = oracle::ieee_binary(dtype, one, signed(dtype, 2), oracle::Arithmetic::Divide);
    let patterns: [&[u64]; 8] = [
        &[huge, one, huge | sign],
        &[0, sign, 0],
        &[maximum, maximum, one],
        &[normal, half, half],
        &[1, 1, sign | 1],
        &[inf, inf | sign, one],
        &[payload, one, sign],
        &[one | 1, one | 2, one | sign],
    ];
    let values = patterns.iter().flat_map(|row| (0..length).map(move |i| row[i % row.len()])).collect();
    dense(&[8, length], values)
}
fn plan() -> Vec<Case> {
    let mut cases = Vec::new();
    for (d, dtype) in DTYPES.into_iter().enumerate() {
        for op in OPERATIONS {
            cases.push(case(dtype, fixture(dtype, 3), op, Scope::Axis(1), format!("projection/{dtype:?}/{op:?}")));
        }
        for (i, length) in [1, 127, 128, 129, 257].into_iter().enumerate() {
            cases.push(case(
                dtype,
                fixture(dtype, length),
                OPERATIONS[(d + i) % 3],
                Scope::Axis(1),
                format!("length/{dtype:?}/{length}"),
            ));
        }
        let mut input = fixture(dtype, 129);
        input.views.push(View::Flip(vec![1]));
        cases.push(case(dtype, input, Operation::Sum, Scope::Axis(1), format!("reverse/{dtype:?}")));
        let mut input = fixture(dtype, 127);
        input.views.push(View::Permute(vec![1, 0]));
        cases.push(case(dtype, input, Operation::Mean, Scope::Axis(0), format!("transpose/{dtype:?}")));
        let input = Input {
            root: Values::new(&[3, 1], vec![signed(dtype, 2), 1, signed(dtype, -1)]),
            views: vec![View::Expand(vec![3, 129])],
        };
        cases.push(case(dtype, input, Operation::Product, Scope::Axis(1), format!("broadcast/{dtype:?}")));
        let input = Input {
            root: Values::new(&[2, 1, 1, 1, 1, 1, 3, 7], (0..42).map(|i| signed(dtype, i % 5 - 2)).collect()),
            views: vec![
                View::Slice {
                    starts: vec![0, 0, 0, 0, 0, 0, 1, 1],
                    steps: vec![1, 1, 1, 1, 1, 1, 1, 2],
                    shape: vec![2, 1, 1, 1, 1, 1, 2, 3],
                },
                View::Flip(vec![0, 7]),
            ],
        };
        cases.push(case(dtype, input, Operation::Sum, Scope::Axis(7), format!("signed-offset-rank8/{dtype:?}")));
        for op in OPERATIONS {
            cases.push(case(dtype, dense(&[3, 0], vec![]), op, Scope::Axis(1), format!("no-read/{dtype:?}/{op:?}")));
        }
        // Finite-only rows expose final narrowing and gradual underflow.
        for op in [Operation::Mean, Operation::Product] {
            let input = dense(
                &[3, 3],
                vec![signed(dtype, 1), signed(dtype, 2), signed(dtype, 3), 1, 1, 1, signed(dtype, -1), 1, 1],
            );
            cases.push(case(dtype, input, op, Scope::Axis(1), format!("finite/{dtype:?}/{op:?}")));
        }
    }
    for (id, dtype, op, shape) in [
        ("R34", DType::F16, Operation::Sum, [3, 31]),
        ("R45", DType::F16, Operation::Mean, [129, 2]),
        ("R46", DType::BF16, Operation::Product, [257, 3]),
    ] {
        let n = shape.iter().product();
        let values = (0..n).map(|i| signed(dtype, if i % 3 == 0 { -1 } else { 1 })).collect();
        cases.push(case(dtype, dense(&shape, values), op, Scope::Axis(1), id));
    }
    let mut values = vec![0; 65536];
    values[0] = signed(DType::F16, 1);
    cases.push(case(
        DType::F16,
        dense(&[1, 65536], values),
        Operation::Mean,
        Scope::Axis(1),
        "count65536/raw-only",
    ));
    cases
}
#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    cases: usize,
    guarded: usize,
    fresh: usize,
    native: usize,
    globals: usize,
    empty: usize,
    rejected: usize,
    deferred: usize,
    queued: usize,
    values: usize,
}
// Independently enumerated host plan; never infer expectations from GPU counters.
const EXPECTED: Counts = Counts {
    cases: 72,
    guarded: 144,
    fresh: 72,
    native: 71,
    globals: 78,
    empty: 15,
    rejected: 308,
    deferred: 12,
    queued: 8,
    values: 3313,
};
#[derive(Default)]
struct Observed {
    counts: Counts,
    failed: usize,
    examples: Vec<String>,
}
impl Observed {
    fn record(&mut self, context: &str, result: Result<()>) {
        if let Err(error) = result {
            self.failed += 1;
            if self.examples.len() < 12 {
                self.examples.push(format!("{context}: {error}"));
            }
        }
    }
}

#[derive(Clone)]
struct Work {
    kernels: [u64; abi::KERNEL_COUNT],
    services: [u64; abi::SERVICE_OPERATIONS],
}
impl Default for Work {
    fn default() -> Self {
        Self {
            kernels: [0; abi::KERNEL_COUNT],
            services: [0; abi::SERVICE_OPERATIONS],
        }
    }
}
impl Work {
    fn scope(&mut self) {
        self.services[abi::BEGIN_COMMANDS as usize] += 1;
        self.services[abi::END_COMMANDS as usize] += 1;
    }
    fn launch(&mut self, kernel: Kernel, fresh: bool) {
        self.kernels[kernel as usize] += 1;
        self.services[if fresh { abi::LAUNCH_ALLOCATE } else { abi::LAUNCH } as usize] += 1;
    }
    fn empty(&mut self) {
        self.services[abi::ALLOCATE as usize] += 1;
        self.kernels[Kernel::Zero as usize] += 1;
    }
}
struct Snapshot {
    kernels: [abi::ProfileEntry; abi::KERNEL_COUNT],
    services: [abi::ServiceEntry; abi::SERVICE_OPERATIONS],
}
impl Snapshot {
    fn take(backend: &Nvidia) -> Result<Self> {
        Ok(Self {
            kernels: gpu(backend.profile())?,
            services: gpu(backend.service_profile())?,
        })
    }
    fn check(&self, backend: &Nvidia, work: &Work) -> Result<()> {
        let after = Self::take(backend)?;
        check_counters(self, &after, work)
    }
}
fn check_counters(before: &Snapshot, after: &Snapshot, work: &Work) -> Result<()> {
    for (i, ((old, new), want)) in before.kernels.iter().zip(&after.kernels).zip(work.kernels).enumerate() {
        if new.calls.checked_sub(old.calls) != Some(want) {
            return Err(format!("kernel={i} expected={want} before={} after={}", old.calls, new.calls));
        }
    }
    for (i, ((old, new), &work_calls)) in before.services.iter().zip(&after.services).zip(&work.services).enumerate() {
        // PROFILE is read once between service snapshots. SERVICE_PROFILE
        // copies before incrementing its own counter, so its delta is one.
        let want = work_calls + u64::from(i == abi::PROFILE as usize || i == abi::SERVICE_PROFILE as usize);
        if new.calls.checked_sub(old.calls) != Some(want) {
            return Err(format!("service={i} expected={want} before={} after={}", old.calls, new.calls));
        }
    }
    Ok(())
}
fn unchanged(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    let before = Snapshot::take(backend)?;
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    let mut work = Work::default();
    work.services[abi::DOWNLOAD as usize] = 1;
    before.check(backend, &work)?;
    if actual != expected {
        return Err(String::from("physical root/guard changed"));
    }
    Ok(())
}
fn poison(expected: &[u32], alternate: bool) -> Vec<u32> {
    // Both masks change every byte, including both U64 words and narrow tails.
    expected.iter().map(|&word| word ^ if alternate { 0x5ac3_6db7 } else { 0xa53c_9248 }).collect()
}
fn validate_words(actual: &[u32], expected: &[u32], guarded: bool) -> Result<()> {
    let offset = usize::from(guarded);
    if actual.len() != expected.len() + offset * 2 {
        return Err(String::from("physical extent changed"));
    }
    if guarded && (actual[0] != GUARD || actual[actual.len() - 1] != GUARD) {
        return Err(String::from("output guard changed"));
    }
    if actual[offset..offset + expected.len()] != *expected {
        let at = actual[offset..offset + expected.len()].iter().zip(expected).position(|(a, b)| a != b).unwrap();
        return Err(format!("physical word={at} actual={:#x} expected={:#x}", actual[offset + at], expected[at]));
    }
    Ok(())
}
fn check_words(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32], guarded: bool) -> Result<()> {
    let mut actual = vec![0; expected.len() + usize::from(guarded) * 2];
    let before = Snapshot::take(backend)?;
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    let mut work = Work::default();
    work.services[abi::DOWNLOAD as usize] = 1;
    before.check(backend, &work)?;
    validate_words(&actual, expected, guarded)
}
fn source_words(case: &Case) -> Vec<u32> {
    let mut packed = output_words(case.dtype, &case.input.root);
    if packed.is_empty() {
        packed.push(0xa53c_9248);
    } else if case.dtype.size() == 2 && case.input.root.values.len() % 2 == 1 {
        *packed.last_mut().unwrap() |= 0xdead_0000;
    }
    packed
}
fn raw_case(backend: &Nvidia, case: &Case, counts: &mut Counts) -> Result<()> {
    let p = case.parameters()?;
    let g = raw::geometry(&p).map_err(|e| format!("geometry: {e:?}"))?;
    let expected = case.expected()?;
    let packed = output_words(case.dtype, &expected);
    let mut source = vec![GUARD];
    source.extend(source_words(case));
    source.push(GUARD);
    let input = gpu(backend.upload_words(&source))?;
    if g.input_words as usize != source.len() - 2 || g.output_words as usize != packed.len() {
        return Err(String::from("physical extent differs from reference"));
    }
    let params = p.map(|x| x as usize);
    for alternate in [false, true] {
        let mut initial = vec![GUARD];
        initial.extend(poison(&packed, alternate));
        initial.push(GUARD);
        let out = gpu(backend.upload_words(&initial))?;
        let before = Snapshot::take(backend)?;
        gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatReduction, &[(&input, 1), (&out, 1)], &params)))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::FloatReduction, false);
        before.check(backend, &work)?;
        // Original reduction allocation, including its unused halfword. No cast.
        check_words(backend, &out, &packed, true)?;
        unchanged(backend, &input, &source)?;
        counts.guarded += 1;
        counts.values += expected.values.len();
    }
    let dirty = gpu(backend.upload_words(&poison(&packed, true)))?;
    drop(dirty);
    let before = Snapshot::take(backend)?;
    let fresh = gpu(backend.with_commands(|| backend.produce_words(packed.len(), Kernel::FloatReduction, &[(&input, 1)], &params)))?;
    let mut work = Work::default();
    work.scope();
    work.launch(Kernel::FloatReduction, true);
    before.check(backend, &work)?;
    check_words(backend, &fresh, &packed, false)?;
    unchanged(backend, &input, &source)?;
    counts.fresh += 1;
    counts.values += expected.values.len();
    counts.cases += 1;
    Ok(())
}
fn read_float(tensor: FloatTensor<Native>) -> Result<TensorData> {
    try_read_sync(Native::float_into_data(tensor))
        .ok_or("float read pending")?
        .map_err(|e| format!("float read: {e:?}"))
}
fn native_view(mut tensor: FloatTensor<Native>, input: &Input) -> FloatTensor<Native> {
    for view in &input.views {
        tensor = match view {
            View::Permute(axes) => Native::float_permute(tensor, axes),
            View::Flip(axes) => Native::float_flip(tensor, axes),
            View::Slice { starts, steps, shape } => Native::float_slice(tensor, &slices(starts, steps, shape)),
            View::Expand(shape) => Native::float_expand(tensor, Shape::from(shape.clone())),
        };
    }
    tensor
}
fn native_apply(input: FloatTensor<Native>, op: Operation, scope: Scope) -> FloatTensor<Native> {
    match (scope, op) {
        (Scope::Axis(a), Operation::Sum) => Native::float_sum_dim(input, a),
        (Scope::Axis(a), Operation::Mean) => Native::float_mean_dim(input, a),
        (Scope::Axis(a), Operation::Product) => Native::float_prod_dim(input, a),
        (Scope::Global, Operation::Sum) => Native::float_sum(input),
        (Scope::Global, Operation::Mean) => Native::float_mean(input),
        (Scope::Global, Operation::Product) => Native::float_prod(input),
    }
}
fn native_work(case: &Case) -> Work {
    let mut work = Work::default();
    work.scope();
    let input = case.input.logical();
    match case.scope {
        Scope::Axis(_) => {
            if case.expected().expect("valid primitive").values.is_empty() {
                if case.dtype != DType::F32 {
                    work.empty();
                }
            } else {
                work.launch(
                    if case.dtype == DType::F32 {
                        Kernel::TensorReduction
                    } else {
                        Kernel::FloatReduction
                    },
                    true,
                );
            }
        }
        Scope::Global => {
            if input.shape.is_empty() {
                return work;
            }
            if input.values.is_empty() && case.dtype != DType::F32 {
                work.launch(if case.dtype == DType::Flex32 { Kernel::Fill } else { Kernel::TensorStorage }, true);
            } else {
                let stages = input.shape.len();
                work.scope();
                for _ in 0..stages {
                    work.launch(
                        if case.dtype == DType::F32 {
                            Kernel::TensorReduction
                        } else {
                            Kernel::FloatReduction
                        },
                        true,
                    );
                }
                work.services[abi::FREE as usize] += stages.saturating_sub(1) as u64;
                if case.dtype == DType::F32 && case.op == Operation::Mean {
                    work.launch(Kernel::TensorElementwise, true);
                    work.services[abi::FREE as usize] += 1;
                }
            }
        }
    }
    work
}
fn native_case(backend: &Nvidia, case: &Case, counts: &mut Counts) -> Result<()> {
    let expected = case.expected()?;
    let source = Native::float_from_data(data(case.dtype, &case.input.root), &NativeDevice);
    let root = native_view(source.clone(), &case.input);
    let alias = root.clone();
    let before = Snapshot::take(backend)?;
    let output = gpu(backend.with_commands(|| Ok::<_, Error>(native_apply(root.clone(), case.op, case.scope))))?;
    before.check(backend, &native_work(case))?;
    check_data(&read_float(output)?, case.dtype, &expected)?;
    check_data(&read_float(alias)?, case.dtype, &case.input.logical())?;
    check_data(&read_float(source)?, case.dtype, &case.input.root)?;
    if case.scope == Scope::Global {
        counts.globals += 1;
    } else {
        counts.native += 1;
    }
    counts.values += expected.values.len();
    Ok(())
}
fn global_plan() -> Vec<Case> {
    let mut cases = Vec::new();
    for dtype in DTYPES {
        let sign = 1u64 << (dtype.size() * 8 - 1);
        let huge = match dtype {
            DType::F64 => 0x4350000000000000,
            DType::Flex32 => 0x4c000000,
            DType::F16 => 0x6800,
            DType::BF16 => 0x4380,
            _ => unreachable!(),
        };
        let one = signed(dtype, 1);
        let inputs = [
            dense(&[2, 3], vec![huge, one, huge | sign, one, one, one]),
            dense(&[3, 2, 2], (0..12).map(|i| signed(dtype, i % 5 - 2)).collect()),
            dense(&[], vec![sign]),
            dense(&[], vec![oracle::canonical_nan(dtype) | sign | 3]),
            dense(&[0, 3], vec![]),
            dense(&[0, 0], vec![]),
        ];
        for (i, input) in inputs.into_iter().enumerate() {
            for op in OPERATIONS {
                cases.push(case(dtype, input.clone(), op, Scope::Global, format!("global/{dtype:?}/{op:?}/{i}")));
            }
        }
    }
    for shape in [[2, 3], [3, 2]] {
        for op in OPERATIONS {
            cases.push(case(
                DType::F32,
                dense(&shape, vec![0x3f800000; 6]),
                op,
                Scope::Global,
                format!("legacy/{op:?}/{shape:?}"),
            ));
        }
    }
    cases
}
fn malformed(p: [u32; 24]) -> Vec<Vec<u32>> {
    let mut cases = Vec::new();
    for (field, value) in [
        (0, 2),
        (1, 0),
        (1, 9),
        (3, 2),
        (3, u32::MAX),
        (4, u32::MAX),
        (5, 0),
        (5, 0x80000000),
        (6, 2),
        (7, 0),
        (7, 2),
        (10, 2),
        (18, 1),
        (8, u32::MAX),
        (16, i32::MAX as u32),
        (17, (-1i32) as u32),
    ] {
        let mut bad = p;
        bad[field] = value;
        cases.push(bad.to_vec());
    }
    for control in [
        p[2] ^ 1 << 24,
        (p[2] & !0xff) | 3,
        (p[2] & !0xff) | 0xff,
        (p[2] & !0xff00) | (Format::F32 as u32) << 8,
        (p[2] & !0xff00) | (Format::I32 as u32) << 8,
        (p[2] & !0xff0000) | (Format::F32 as u32) << 16,
        (p[2] & !0xff000000) | (Format::F16 as u32) << 24,
        (p[2] & !0xff00) | (Format::Bool8 as u32) << 8,
        (p[2] & !0xff00) | (Format::Bool32 as u32) << 8,
        (p[2] & !0xff00) | 0xff << 8,
        (p[2] & !0xff0000) | (Format::I32 as u32) << 16,
        (p[2] & !0xff000000) | 0xff << 24,
    ] {
        let mut bad = p;
        bad[2] = control;
        cases.push(bad.to_vec());
    }
    cases.push(p[..23].to_vec());
    let mut long = p.to_vec();
    long.push(0);
    cases.push(long);
    let mut rank = p;
    rank[1] = 8;
    rank[8..16].fill(65536);
    rank[16..24].fill(1);
    cases.push(rank.to_vec());
    let mut empty = p;
    empty[0] = 0;
    empty[8] = 0;
    empty[5] = 0;
    cases.push(empty.to_vec());
    empty[9] = 0;
    empty[6] = 0;
    cases.push(empty.to_vec());
    cases
}
fn reject<T>(result: d3os_compute::Result<T>) -> Result<()> {
    match result {
        Err(Error::InvalidParameter | Error::ShapeMismatch | Error::SizeOverflow) => Ok(()),
        Err(e) => Err(format!("noncanonical rejection: {e}")),
        Ok(_) => Err(String::from("invalid launch accepted")),
    }
}
fn rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in DTYPES {
        let c = case(dtype, dense(&[3, 3], vec![signed(dtype, 1); 9]), Operation::Sum, Scope::Axis(1), "reject");
        let p = c.parameters()?;
        let params = p.map(|v| v as usize);
        let source = output_words(dtype, &c.input.root);
        let expected = output_words(dtype, &c.expected()?);
        let a = gpu(backend.upload_words(&source))?;
        let initial = poison(&expected, false);
        let out = gpu(backend.upload_words(&initial))?;
        let short = gpu(backend.upload_words(&[GUARD]))?;
        let before = Snapshot::take(backend)?;
        for bad in malformed(p) {
            let bad: Vec<_> = bad.into_iter().map(|v| v as usize).collect();
            reject(backend.dispatch_words(Kernel::FloatReduction, &[(&a, 0), (&out, 0)], &bad))?;
            reject(backend.produce_words(expected.len(), Kernel::FloatReduction, &[(&a, 0)], &bad))?;
            counts.rejected += 2;
        }
        for buffers in [
            vec![(&a, 0)],
            vec![(&a, 0), (&out, 0), (&short, 0)],
            vec![(&a, 0), (&a, 0)],
            vec![(&a, 0), (&a, 1)],
            vec![(&short, 0), (&out, 0)],
            vec![(&a, 0), (&out, 1)],
            vec![(&a, 1), (&out, 0)],
        ] {
            reject(backend.dispatch_words(Kernel::FloatReduction, &buffers, &params))?;
            counts.rejected += 1;
        }
        for size in [0, expected.len() - 1, expected.len() + 1] {
            let result = backend.produce_words(size, Kernel::FloatReduction, &[(&a, 0)], &params);
            if size == 0 {
                if !matches!(result, Err(Error::InvalidShape)) {
                    return Err(String::from("zero output allocation did not reject InvalidShape"));
                }
            } else {
                reject(result)?;
            }
            counts.rejected += 1;
        }
        reject(backend.produce_words(expected.len(), Kernel::FloatReduction, &[(&short, 0)], &params))?;
        counts.rejected += 1;
        // Two alias requests reach driver validation. No kernel may run.
        let mut work = Work::default();
        work.services[abi::LAUNCH as usize] = 2;
        before.check(backend, &work)?;
        unchanged(backend, &a, &source)?;
        unchanged(backend, &out, &initial)?;
        unchanged(backend, &short, &[GUARD])?;
        let before = Snapshot::take(backend)?;
        gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatReduction, &[(&a, 0), (&out, 0)], &params)))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::FloatReduction, false);
        before.check(backend, &work)?;
        check_words(backend, &out, &expected, false)?;
        unchanged(backend, &a, &source)?;
    }
    Ok(())
}
fn empty(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        let values = Values::new(&[0, 3], vec![]);
        let root = gpu(NativeTensor::from_data(data(dtype, &values)))?;
        for op in OPERATIONS {
            if !root.can_mut() {
                return Err(String::from("empty source lacks unique ownership"));
            }
            let before = Snapshot::take(backend)?;
            let output = gpu(root.reduce(1, reduction(op)))?;
            let mut work = Work::default();
            if dtype != DType::F32 {
                work.empty();
            }
            before.check(backend, &work)?;
            let fresh = dtype != DType::F32;
            if root.can_mut() != fresh || output.can_mut() != fresh {
                return Err(String::from("empty alias/fresh allocation contract changed"));
            }
            check_data(&gpu(output.read())?, dtype, &Values::new(&[0, 1], vec![]))?;
            drop(output);
            if !root.can_mut() {
                return Err(String::from("empty output retained source ownership"));
            }
            counts.empty += 1;
        }
    }
    Ok(())
}
fn deferred(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in DTYPES {
        let values = Values::new(&[3, 3], vec![signed(dtype, 1); 9]);
        let source = Native::float_from_data(data(dtype, &values), &NativeDevice);
        for op in OPERATIONS {
            let alias = source.clone();
            let before = Snapshot::take(backend)?;
            let (invalid, propagated) = gpu(backend.with_commands(|| {
                backend.record_commands(|| {
                    let middle = native_apply(source.clone(), op, Scope::Axis(1));
                    let invalid = native_apply(middle, op, Scope::Axis(2));
                    let propagated = native_apply(invalid.clone(), op, Scope::Global);
                    Ok::<_, Error>((invalid, propagated))
                })
            }))?;
            let mut work = Work::default();
            work.scope();
            work.scope();
            work.launch(Kernel::FloatReduction, true);
            work.services[abi::FREE as usize] = 1;
            before.check(backend, &work)?;
            if invalid.dtype() != dtype || invalid.shape().as_slice() != [3, 1] {
                return Err(String::from("second-stage invalid axis lost metadata"));
            }
            if propagated.dtype() != dtype || propagated.shape().as_slice() != [1] {
                return Err(String::from("deferred global error lost metadata"));
            }
            let before = Snapshot::take(backend)?;
            if read_float(invalid).is_ok() || read_float(propagated).is_ok() {
                return Err(String::from("invalid reduction failed to defer its error"));
            }
            before.check(backend, &Work::default())?;
            check_data(&read_float(alias)?, dtype, &values)?;
            let valid = case(dtype, dense(&[3, 3], values.values.clone()), op, Scope::Axis(1), "recovery");
            let before = Snapshot::take(backend)?;
            let output = gpu(backend.with_commands(|| Ok::<_, Error>(native_apply(source.clone(), op, Scope::Axis(1)))))?;
            before.check(backend, &native_work(&valid))?;
            check_data(&read_float(output)?, dtype, &valid.expected()?)?;
            counts.deferred += 1;
            counts.values += 3;
        }
        check_data(&read_float(source)?, dtype, &values)?;
    }
    Ok(())
}
fn queued(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in DTYPES {
        let first = case(
            dtype,
            dense(&[3, 3], (0..9).map(|i| signed(dtype, i % 3 + 1)).collect()),
            Operation::Mean,
            Scope::Axis(1),
            "queued-first",
        );
        let expected = first.expected()?;
        let second = case(
            dtype,
            dense(&expected.shape, expected.values.clone()),
            Operation::Sum,
            Scope::Axis(0),
            "queued-second",
        );
        let expected2 = second.expected()?;
        let third = case(
            dtype,
            dense(&expected2.shape, expected2.values.clone()),
            Operation::Product,
            Scope::Axis(1),
            "resident-final-consumer",
        );
        let final_expected = third.expected()?;
        let p = first.parameters()?.map(|v| v as usize);
        let p2 = second.parameters()?.map(|v| v as usize);
        let p3 = third.parameters()?.map(|v| v as usize);
        let first_words = output_words(dtype, &expected).len();
        let final_words = output_words(dtype, &final_expected).len();
        let input = gpu(backend.upload_words(&output_words(dtype, &first.input.root)))?;
        let before = Snapshot::take(backend)?;
        let output = gpu(backend.with_commands(|| {
            let temporary = backend.record_commands(|| {
                let temporary = backend.produce_words(first_words, Kernel::FloatReduction, &[(&input, 0)], &p)?;
                drop(input);
                Ok::<_, Error>(temporary)
            })?;
            let middle = backend.record_commands(|| {
                let middle = backend.produce_words(final_words, Kernel::FloatReduction, &[(&temporary, 0)], &p2)?;
                drop(temporary);
                let dirty = backend.produce_words(first_words, Kernel::Fill, &[], &[first_words, 0xdead_beef])?;
                drop(dirty);
                Ok::<_, Error>(middle)
            })?;
            let output = backend.produce_words(final_words, Kernel::FloatReduction, &[(&middle, 0)], &p3)?;
            drop(middle);
            Ok::<_, Error>(output)
        }))?;
        let mut work = Work::default();
        for _ in 0..3 {
            work.scope();
            work.launch(Kernel::FloatReduction, true);
        }
        work.launch(Kernel::Fill, true);
        work.services[abi::FREE as usize] = 4;
        before.check(backend, &work)?;
        check_words(backend, &output, &output_words(dtype, &final_expected), false)?;
        counts.queued += 1;
        counts.values += 1;
        // Public globals nest their own recorder. Last owners disappear before
        // completion, and only the final consumer is read after the outer scope.
        let source = Native::float_from_data(data(dtype, &first.input.root), &NativeDevice);
        let global = oracle::reference(dtype, &first.input.root, Operation::Mean, Scope::Global).map_err(|e| format!("queued global: {e:?}"))?;
        let final_expected = oracle::reference(dtype, &global, Operation::Sum, Scope::Axis(0)).map_err(|e| format!("queued consumer: {e:?}"))?;
        let before = Snapshot::take(backend)?;
        let output = gpu(backend.with_commands(|| {
            backend.record_commands(|| {
                let middle = Native::float_mean(source);
                Ok::<_, Error>(Native::float_sum_dim(middle, 0))
            })
        }))?;
        let mut work = Work::default();
        for _ in 0..3 {
            work.scope();
            work.launch(Kernel::FloatReduction, true);
        }
        work.services[abi::FREE as usize] = 3;
        before.check(backend, &work)?;
        check_data(&read_float(output)?, dtype, &final_expected)?;
        counts.queued += 1;
        counts.values += 1;
    }
    Ok(())
}
fn native_options() -> oracle::Options {
    oracle::Options {
        native_error_extensions: true,
        ..Default::default()
    }
}
fn expected_shared() -> oracle::Summary {
    oracle::Summary {
        options: native_options(),
        execution_scope: oracle::ExecutionScope::NativeErrorExtensions,
        cases: 80,
        standalone_cases: 72,
        chain_cases: 8,
        device_cases: 71,
        model_cases: 7,
        diagnostic_cases: 2,
        steps: 84,
        values: 356,
        checks: oracle::Checks {
            input: 73,
            source: 73,
            alias: 73,
            output: 71,
            error: 2,
            recovery: 2,
            model: 7,
        },
    }
}
fn expected_counts(cases: &[Case]) -> Result<Counts> {
    let mut counts = Counts::default();
    for case in cases {
        let n = case.expected()?.values.len();
        counts.cases += 1;
        counts.guarded += 2;
        counts.fresh += 1;
        counts.values += 3 * n;
        if case.id != "count65536/raw-only" {
            counts.native += 1;
            counts.values += n;
        }
    }
    for case in global_plan() {
        counts.globals += 1;
        counts.values += case.expected()?.values.len();
    }
    counts.empty = 15;
    counts.deferred = 12;
    counts.queued = 8;
    counts.values += 36 + 8;
    for dtype in DTYPES {
        let c = case(dtype, dense(&[3, 3], vec![signed(dtype, 1); 9]), Operation::Sum, Scope::Axis(1), "reject");
        counts.rejected += 2 * malformed(c.parameters()?).len() + 7 + 3 + 1;
    }
    Ok(counts)
}
fn audit_geometry(case: &Case) -> Result<()> {
    let p = case.parameters()?;
    let expected = case.expected()?;
    let n = expected.values.len();
    let layout = layout(&case.input)?;
    let logical = case.input.logical();
    let Scope::Axis(axis) = case.scope else {
        return Err(String::from("unexpected ALL"));
    };
    let input_words = if logical.values.is_empty() {
        1
    } else {
        (case.input.root.values.len() * case.dtype.size()).div_ceil(4)
    };
    let output_words = (n * case.dtype.size()).div_ceil(4).max(1);
    let owners = if case.dtype == DType::F64 { n } else { output_words };
    let g = raw::geometry(&p).map_err(|e| format!("geometry: {e:?}"))?;
    let w = Kernel::FloatReduction.work(&p).map_err(|e| format!("registered work: {e:?}"))?;
    if raw::PARAMETERS != 24
        || raw::THREADS != 128
        || raw::SERIAL_POLICY != 1
        || g.input_words != input_words as u64
        || g.output_words != output_words as u64
        || g.owners as usize != owners
        || g.output_count as usize != n
        || g.reduction_length as usize != logical.shape[axis]
        || raw::validate(&p).ok() != Some([input_words as u64, output_words as u64])
        || raw::work_items(&p).ok() != Some(owners as u32)
        || w.threads != 128
        || w.blocks as usize != owners.div_ceil(128)
        || w.extents[..2] != [input_words as u64, output_words as u64]
        || (w.pointers, w.output, w.in_place, w.mutated) != (2, 1, false, 0)
        || Kernel::FloatReduction.entry(&p) != "burn_float_reduce"
        || Kernel::FloatReduction.symbol() != "burn_float_reduce"
    {
        return Err(format!("registered serial work mismatch: {}", case.id));
    }
    for (i, &bits) in logical.values.iter().enumerate() {
        if case.input.root.values[gpu(layout.element_offset(i))?] != bits {
            return Err(format!("signed address mismatch: {} at {i}", case.id));
        }
    }
    let start = 0x100004;
    let end = start + w.extents[0] * 4;
    if w.validate_aliases(&[start, end]).is_err() || w.validate_aliases(&[start, start]).is_ok() || w.validate_aliases(&[start, end - 4]).is_ok() {
        return Err(String::from("exact/displaced alias contract changed"));
    }
    Ok(())
}
fn contract(cases: &[Case]) -> Result<()> {
    if !matches!((abi::VERSION, abi::KERNEL_COUNT), (53, 85) | (54, 88))
        || abi::SERVICE_OPERATIONS != 14
        || Kernel::FloatReduction as usize != 84
        || raw::FORMATS != FORMATS
        || oracle::CORPUS_VERSION != 1
    {
        return Err(String::from("float reduction protocol/registration mismatch"));
    }
    for case in cases {
        audit_geometry(case)?;
    }
    if expected_counts(cases)? != EXPECTED {
        return Err(format!("raw count freeze pending: {:?}", expected_counts(cases)?));
    }
    if expected_shared() != oracle::expected_summary(&native_options()) {
        return Err(String::from("shared Native scope/counts changed"));
    }
    Ok(())
}
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    let cases = plan();
    contract(&cases)?;
    let shared_expected = expected_shared();
    crate::report(format_args!(
        "NVIDIA_BURN_FLOAT_REDUCTION_PLAN frozen={FROZEN} reference_sha256={REFERENCE_SHA256} raw={EXPECTED:?} shared={}",
        oracle::summary_json(&shared_expected)
    ));
    if !FROZEN {
        return Err(String::from("float reduction UNFROZEN: ROOT source/API/corpus/count approval required"));
    }
    let mut observed = Observed::default();
    for case in &cases {
        let result = raw_case(backend, case, &mut observed.counts);
        observed.record(&format!("raw/{}", case.id), result);
        if case.id != "count65536/raw-only" {
            let result = native_case(backend, case, &mut observed.counts);
            observed.record(&format!("native/{}", case.id), result);
        }
    }
    for case in global_plan() {
        let result = native_case(backend, &case, &mut observed.counts);
        observed.record(&case.id, result);
    }
    for (name, suite) in [
        ("rejections", rejections as fn(&Nvidia, &mut Counts) -> Result<()>),
        ("empty", empty),
        ("deferred", deferred),
        ("queued", queued),
    ] {
        let result = suite(backend, &mut observed.counts);
        observed.record(name, result);
    }
    let device = d3os_ai::native_device().map_err(|e| format!("float reduction device: {e}"))?;
    let options = native_options();
    let report = try_read_sync(oracle::run(&device, &options)).ok_or("shared float reduction pending")?;
    for failure in report.failures.iter().take(12) {
        observed.record(&format!("shared/{}", failure.case_id), Err(failure.message.clone()));
    }
    let footer = oracle::success_footer(&options, &report);
    let accepted = observed.failed == 0 && observed.counts == EXPECTED && report.expected == shared_expected && footer.is_ok();
    crate::report(format_args!(
        "NVIDIA_BURN_FLOAT_REDUCTION_OBSERVED accepted={accepted} counts={:?} failed={} shared_attempted={} shared_passed={} shared_errors={} shared_failed={} examples={:?}",
        observed.counts, observed.failed, report.attempted, report.passed, report.observed_errors, report.failed, observed.examples
    ));
    if !accepted {
        return Err(format!(
            "float reduction failed expected={EXPECTED:?} observed={:?} failures={} shared_failed={}",
            observed.counts, observed.failed, report.failed
        ));
    }
    crate::report(format_args!("{} os=d3os backend=native fusion=disabled", footer?));
    crate::report(format_args!(
        "NVIDIA_BURN_FLOAT_REDUCTION_STRICT_OK reference_sha256={REFERENCE_SHA256} protocol={} kernels={} expected={EXPECTED:?} observed={:?}",
        abi::VERSION,
        abi::KERNEL_COUNT,
        observed.counts
    ));
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_registered_plan_owns_every_logical_output_and_physical_word() {
        let cases = plan();
        let counts = expected_counts(&cases).unwrap();
        println!("raw plan: {counts:?}");
        println!("shared Native plan: {}", oracle::summary_json(&expected_shared()));
        for case in &cases {
            audit_geometry(case).unwrap();
            let g = raw::geometry(&case.parameters().unwrap()).unwrap();
            let slots = if case.dtype.size() == 8 { 1 } else { 4 / case.dtype.size() };
            let mut seen = vec![0; g.output_count as usize];
            for owner in 0..g.owners as usize {
                for slot in 0..slots {
                    let index = owner * slots + slot;
                    // The padded half uses a safe logical coordinate before
                    // evaluating any input address, even when the lane is masked.
                    let safe = if index < seen.len() { index } else { 0 };
                    assert!(safe < seen.len());
                    if index < seen.len() {
                        seen[index] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&n| n == 1));
        }
        for (id, shape, dtype, op) in [
            ("R34", [3, 31], DType::F16, Operation::Sum),
            ("R45", [129, 2], DType::F16, Operation::Mean),
            ("R46", [257, 3], DType::BF16, Operation::Product),
        ] {
            let case = cases.iter().find(|c| c.id == id).unwrap();
            assert_eq!(case.input.root.shape, shape);
            assert_eq!((case.dtype, case.op), (dtype, op));
            let packed = output_words(dtype, &case.expected().unwrap());
            assert_eq!(packed.last().unwrap() >> 16, 0);
        }
        contract(&cases).unwrap();
    }
    #[test]
    fn poison_gate_rejects_partial_writes_guards_and_wide_high_word_faults() {
        for case in plan() {
            let expected = output_words(case.dtype, &case.expected().unwrap());
            let a = poison(&expected, false);
            let b = poison(&expected, true);
            for ((&a, &b), &e) in a.iter().zip(&b).zip(&expected) {
                for ((a, b), e) in a.to_le_bytes().into_iter().zip(b.to_le_bytes()).zip(e.to_le_bytes()) {
                    assert!(a != b && a != e && b != e);
                }
            }
            if ["R34", "R45", "R46"].contains(&case.id.as_str()) || case.dtype == DType::F64 {
                let mut correct = vec![GUARD];
                correct.extend(&expected);
                correct.push(GUARD);
                validate_words(&correct, &expected, true).unwrap();
                for pattern in [a, b] {
                    for (i, &poison) in pattern.iter().enumerate() {
                        let mut omitted = correct.clone();
                        omitted[i + 1] = poison;
                        assert!(validate_words(&omitted, &expected, true).is_err());
                    }
                }
                for i in [0, correct.len() - 1] {
                    let mut damaged = correct.clone();
                    damaged[i] ^= 1;
                    assert!(validate_words(&damaged, &expected, true).is_err());
                }
                assert!(validate_words(&correct[..correct.len() - 1], &expected, true).is_err());
            }
        }
    }
    #[test]
    fn metadata_rejects_before_launch_and_empty_geometry_keeps_one_dummy_word() {
        for dtype in DTYPES {
            let c = case(dtype, dense(&[3, 3], vec![signed(dtype, 1); 9]), Operation::Sum, Scope::Axis(1), "metadata");
            for bad in malformed(c.parameters().unwrap()) {
                assert!(Kernel::FloatReduction.work(&bad).is_err(), "{dtype:?} {bad:?}");
            }
            for shape in [[0, 3], [0, 0]] {
                let empty = case(dtype, dense(&shape, vec![]), Operation::Mean, Scope::Axis(1), "N0");
                let p = empty.parameters().unwrap();
                let g = raw::geometry(&p).unwrap();
                assert_eq!((g.input_words, g.output_words, g.owners), (1, 1, 0));
                assert!(raw::validate(&p).is_err());
                assert!(raw::work_items(&p).is_err());
                assert!(Kernel::FloatReduction.work(&p).is_err());
            }
            for op in OPERATIONS {
                let empty = case(dtype, dense(&[3, 0], vec![]), op, Scope::Axis(1), "L0");
                let mut p = empty.parameters().unwrap();
                p[4] = 8;
                p[5] = 8;
                p[16] = i32::MIN as u32;
                p[17] = i32::MAX as u32;
                let g = raw::geometry(&p).unwrap();
                assert_eq!(g.input_words, 1);
                assert_eq!(raw::validate(&p).unwrap()[0], 1);
                p[4] = 9;
                assert!(raw::geometry(&p).is_err());
            }
            let mut wide = c.parameters().unwrap();
            wide[0] = 1;
            wide[1] = 1;
            wide[3] = 0;
            wide[4] = i32::MAX as u32 - 1;
            wide[5] = i32::MAX as u32;
            wide[6] = 1;
            wide[8..16].fill(1);
            wide[16..24].fill(0);
            wide[16] = 1;
            let g = raw::geometry(&wide).unwrap();
            assert_eq!(g.input_words, format_of(dtype).width().words(i32::MAX as u32));
            if dtype == DType::F64 {
                assert_eq!(g.input_words, 4294967294);
            }
            // Product must notice the zero before overflowing another dimension.
            wide[1] = 3;
            wide[0] = 0;
            wide[3] = 1;
            wide[6] = i32::MAX as u32;
            wide[8] = 0;
            wide[9] = i32::MAX as u32;
            wide[10] = i32::MAX as u32;
            assert_eq!(raw::geometry(&wide).unwrap().owners, 0);
            assert!(Kernel::FloatReduction.work(&wide).is_err());
        }
        for format in Format::ALL {
            if !FORMATS.contains(&format) {
                for op in raw::Operation::ALL {
                    assert!(raw::control(op, format).is_err());
                }
            }
        }
        assert!(matches!(Nvidia.produce_words(0, Kernel::FloatReduction, &[], &[]), Err(Error::InvalidShape)));
    }
    #[test]
    fn all_kernel_and_service_counters_detect_extra_work_and_resets() {
        let snapshot = |work: &Work| Snapshot {
            kernels: core::array::from_fn(|i| abi::ProfileEntry {
                calls: 7 + work.kernels[i],
                ..Default::default()
            }),
            services: core::array::from_fn(|i| abi::ServiceEntry {
                calls: 7 + work.services[i] + u64::from(i == abi::PROFILE as usize || i == abi::SERVICE_PROFILE as usize),
                ..Default::default()
            }),
        };
        let before = Snapshot {
            kernels: [abi::ProfileEntry {
                calls: 7,
                ..Default::default()
            }; abi::KERNEL_COUNT],
            services: [abi::ServiceEntry {
                calls: 7,
                ..Default::default()
            }; abi::SERVICE_OPERATIONS],
        };
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::FloatReduction, true);
        check_counters(&before, &snapshot(&work), &work).unwrap();
        for i in 0..abi::KERNEL_COUNT {
            let mut after = snapshot(&work);
            after.kernels[i].calls += 1;
            assert!(check_counters(&before, &after, &work).is_err());
            after.kernels[i].calls = 0;
            assert!(check_counters(&before, &after, &work).is_err());
        }
        for i in 0..abi::SERVICE_OPERATIONS {
            let mut after = snapshot(&work);
            after.services[i].calls += 1;
            assert!(check_counters(&before, &after, &work).is_err());
            after.services[i].calls = 0;
            assert!(check_counters(&before, &after, &work).is_err());
        }
    }
    #[test]
    fn count_and_stage_witnesses_use_exact_accumulators_and_preserve_scalar_bits() {
        for case in global_plan()
            .into_iter()
            .filter(|c| c.dtype != DType::F32 && c.input.root.shape == [2, 3] && c.op == Operation::Sum)
        {
            let reverse = oracle::reference(case.dtype, &case.input.root, Operation::Sum, Scope::Axis(1)).unwrap();
            let reverse = oracle::reference(case.dtype, &reverse, Operation::Sum, Scope::Axis(0)).unwrap();
            assert_ne!(case.expected().unwrap().values, reverse.values);
        }
        let case = plan().into_iter().find(|c| c.id == "count65536/raw-only").unwrap();
        assert_eq!(case.expected().unwrap().values, [0x0100]);
        assert_eq!(oracle::scalar_count(DType::F16, 65536), 0x7c00);
        for dtype in DTYPES {
            let sign = 1u64 << (dtype.size() * 8 - 1);
            for bits in [sign, oracle::canonical_nan(dtype) | sign | 3, 1] {
                let input = Values::new(&[], vec![bits]);
                for op in OPERATIONS {
                    assert_eq!(oracle::reference(dtype, &input, op, Scope::Global).unwrap().values, [bits]);
                }
            }
            let fixture = fixture(dtype, 3);
            for row in fixture.root.values.chunks_exact(3) {
                for op in OPERATIONS {
                    let expected = oracle::axis_reference(dtype, row, op);
                    let reduced = oracle::reference(dtype, &Values::new(&[3], row.to_vec()), op, Scope::Axis(0)).unwrap();
                    assert_eq!(reduced.values, [expected]);
                }
            }
        }
    }
}
