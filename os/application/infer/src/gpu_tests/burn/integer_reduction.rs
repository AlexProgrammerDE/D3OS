//! Bounded exact integer reduction conformance and resident command contracts.
//! Host checks validate plans and references; only the parent runs the GPU suite.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{
            DType, IntDType, Shape, Slice, TensorData, TensorMetadata,
            ops::{FloatTensorOps, IntTensorOps},
            tensor::IntTensor,
        },
        tensor::try_read_sync,
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::integer_reduction::{self as oracle, Expected, Input, Operation, Scope, Values, View},
};
use d3os_compute::{
    Backend as _, Error,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{self as abi, Kernel, cast::Format, integer_reduction as raw, reduction::Reduction};

type Result<T> = core::result::Result<T, String>;
// Parent approval of source identities, protocol and this plan is required.
const FROZEN: bool = true;
const REFERENCE_SHA256: &str = "e192ec54472ba33dfeee193ebf7bee1463a3df1f876142a93ff36c3f35da2a24";
const GUARD: u32 = 0x6bad_beef;
const FORMATS: [Format; 8] = [
    Format::I64,
    Format::I32,
    Format::I16,
    Format::I8,
    Format::U64,
    Format::U32,
    Format::U16,
    Format::U8,
];

fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("integer reduction: {e}"))
}
fn format_of(dtype: DType) -> Format {
    FORMATS[oracle::DTYPES.iter().position(|&d| d == dtype).expect("integer dtype")]
}
fn raw_op(op: Operation) -> raw::Operation {
    match op {
        Operation::Sum => raw::Operation::Sum,
        Operation::Product => raw::Operation::Product,
        Operation::Mean => raw::Operation::Mean,
        Operation::Min => raw::Operation::Min,
        Operation::Max => raw::Operation::Max,
        Operation::ArgMin => raw::Operation::ArgMin,
        Operation::ArgMax => raw::Operation::ArgMax,
        _ => unreachable!("raw pairs are two independent reductions"),
    }
}
fn reduction(op: Operation) -> Reduction {
    match op {
        Operation::Sum => Reduction::Sum,
        Operation::Product => Reduction::Product,
        Operation::Mean => Reduction::Mean,
        Operation::Min => Reduction::Min,
        Operation::Max => Reduction::Max,
        Operation::ArgMin => Reduction::ArgMin,
        Operation::ArgMax => Reduction::ArgMax,
        _ => unreachable!("single reduction"),
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
fn read_int(tensor: IntTensor<Native>) -> Result<TensorData> {
    try_read_sync(Native::int_into_data(tensor))
        .ok_or("integer reduction read pending")?
        .map_err(|e| format!("integer read: {e:?}"))
}
fn dense(shape: &[usize], values: Vec<u64>) -> Input {
    Input {
        root: Values::new(shape, values),
        views: vec![],
    }
}
fn generated(dtype: DType, n: usize) -> Vec<u64> {
    let pool = [0, 1, u64::MAX, 2, 1u64 << 63, 0x7fff_ffff_ffff_ffff, 0xffff_ffff, 127, 128, 257, 0x8000_0000];
    (0..n).map(|i| pool[(i * 7 + i / 11) % pool.len()] & oracle::mask(dtype)).collect()
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
    fn expected(&self) -> Result<Expected> {
        let logical = self.input.logical();
        if self.scope == Scope::Global && self.op == Operation::Mean {
            // Raw ALL Mean has no stored-input-dtype intermediate. Axis(0)
            // on the independent flattened logical reference models that fold.
            oracle::reference(self.dtype, &Values::new(&[logical.values.len()], logical.values), self.op, Scope::Axis(0))
        } else {
            oracle::reference(self.dtype, &logical, self.op, self.scope)
        }
        .map_err(|e| format!("reference: {e:?}"))
    }
    fn parameters(&self) -> Result<[u32; 24]> {
        let layout = layout(&self.input)?;
        let mut p = [0; 24];
        let expected = self.expected()?;
        p[..8].copy_from_slice(&[
            expected.output.values.len() as u32,
            layout.shape().num_dims().max(1) as u32,
            raw::control(raw_op(self.op), format_of(self.dtype), format_of(self.dtype)).map_err(|e| format!("control: {e:?}"))?,
            match self.scope {
                Scope::Axis(axis) => axis as u32,
                Scope::Global => raw::ALL_AXES,
            },
            layout.offset() as u32,
            self.input.root.values.len() as u32,
            match self.scope {
                Scope::Axis(axis) => layout.shape()[axis] as u32,
                Scope::Global => layout.len() as u32,
            },
            0,
        ]);
        p[8..16].fill(1);
        for (axis, (&dim, &stride)) in layout.shape().iter().zip(layout.strides()).enumerate() {
            p[8 + axis] = dim as u32;
            p[16 + axis] = stride as i32 as u32;
        }
        if layout.shape().num_dims() == 0 {
            p[16] = 1;
        }
        Ok(p)
    }
}
fn plan() -> Vec<Case> {
    let mut cases = Vec::new();
    for (d, dtype) in oracle::DTYPES.into_iter().enumerate() {
        for op in Operation::AXIS {
            cases.push(Case {
                id: format!("canonical/{dtype:?}/{op:?}"),
                dtype,
                input: dense(&[3, 3], generated(dtype, 9)),
                op,
                scope: Scope::Axis(1),
            });
        }
        // Every format sees both schedules, with operations rotated once.
        for (i, length) in oracle::LENGTHS.into_iter().enumerate() {
            cases.push(Case {
                id: format!("length/{dtype:?}/{length}"),
                dtype,
                input: dense(&[3, length], generated(dtype, 3 * length)),
                op: Operation::AXIS[(d + i) % 7],
                scope: Scope::Axis(1),
            });
        }
        for op in [Operation::Sum, Operation::Product] {
            cases.push(Case {
                id: format!("empty/{dtype:?}/{op:?}"),
                dtype,
                input: dense(&[2, 0], vec![]),
                op,
                scope: Scope::Axis(1),
            });
        }
        // Identity-equal endpoints remain valid winners, including U64::MAX
        // and I64::MIN. Ties cross tree lanes and select logical index zero.
        for op in [Operation::Min, Operation::ArgMin, Operation::Max, Operation::ArgMax] {
            let minimum = matches!(op, Operation::Min | Operation::ArgMin);
            let bits = if minimum {
                if dtype.is_int() { oracle::mask(dtype) >> 1 } else { oracle::mask(dtype) }
            } else if dtype.is_int() {
                1 << (oracle::width(dtype) - 1)
            } else {
                0
            };
            cases.push(Case {
                id: format!("endpoint-tie/{dtype:?}/{op:?}"),
                dtype,
                input: dense(&[3, 129], vec![bits; 387]),
                op,
                scope: Scope::Axis(1),
            });
        }
        for op in Operation::GLOBAL {
            cases.push(Case {
                id: format!("all/{dtype:?}/{op:?}"),
                dtype,
                input: dense(&[3, 3], generated(dtype, 9)),
                op,
                scope: Scope::Global,
            });
        }
    }
    for dtype in [DType::I64, DType::U8] {
        cases.push(Case {
            id: format!("rank-eight/{dtype:?}"),
            dtype,
            input: dense(&[2; 8], generated(dtype, 256)),
            op: Operation::ArgMax,
            scope: Scope::Axis(7),
        });
    }
    // Reuse shared view fixtures, but upload their actual physical roots.
    for case in oracle::corpus(&oracle::Options::default())
        .into_iter()
        .filter(|c| !c.input.views.is_empty() && c.expected().is_ok())
    {
        let step = case.steps[0];
        let ops = match step.operation {
            Operation::MinWithIndices => vec![Operation::Min, Operation::ArgMin],
            Operation::MaxWithIndices => vec![Operation::Max, Operation::ArgMax],
            op => vec![op],
        };
        for op in ops {
            cases.push(Case {
                id: format!("view/{}/{op:?}", case.id),
                dtype: case.dtype,
                input: case.input.clone(),
                op,
                scope: step.scope,
            });
        }
    }
    cases
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    cases: usize,
    guarded: usize,
    fresh: usize,
    native: usize,
    globals: usize,
    pairs: usize,
    queued: usize,
    rejected: usize,
    values: usize,
}
// CPU-enumerated plan totals. These never come from device observations.
const EXPECTED: Counts = Counts {
    cases: 243,
    guarded: 486,
    fresh: 486,
    native: 211,
    globals: 40,
    pairs: 17,
    queued: 24,
    rejected: 464,
    values: 4445,
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
fn native_work(case: &Case) -> Result<Work> {
    let mut work = Work::default();
    work.scope();
    if case.expected()?.output.values.is_empty() {
        if case.dtype != DType::I32 {
            work.empty();
        }
    } else {
        work.launch(
            if case.dtype == DType::I32 {
                Kernel::TensorReduction
            } else {
                Kernel::IntegerReduction
            },
            true,
        );
    }
    Ok(work)
}
fn unchanged(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if actual != expected {
        return Err(String::from("physical root/guard changed"));
    }
    Ok(())
}
fn poison(expected: &[u32], alternate: bool) -> Vec<u32> {
    // Both masks change every byte, including both U64 words and narrow tails.
    expected.iter().map(|&word| word ^ if alternate { 0x5ac3_6db7 } else { 0xa53c_9248 }).collect()
}
fn check_words(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32], guarded: bool) -> Result<()> {
    let offset = usize::from(guarded);
    let mut actual = vec![0; expected.len() + offset * 2];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if guarded && (actual[0] != GUARD || actual[actual.len() - 1] != GUARD) {
        return Err(String::from("output guard changed"));
    }
    // Compare entire words, including all required zero tail bytes.
    if actual[offset..offset + expected.len()] != *expected {
        let at = actual[offset..offset + expected.len()].iter().zip(expected).position(|(a, b)| a != b).unwrap();
        return Err(format!("physical word={at} actual={:#x} expected={:#x}", actual[offset + at], expected[at]));
    }
    Ok(())
}
fn raw_case(backend: &Nvidia, case: &Case, observed: &mut Counts) -> Result<()> {
    let p = case.parameters()?;
    let g = raw::geometry(&p).map_err(|e| format!("geometry: {e:?}"))?;
    let expected = case.expected()?;
    let packed = output_words(case.dtype, &expected.output);
    let mut source = vec![GUARD];
    let mut storage = output_words(case.dtype, &case.input.root);
    // Empty U64 input binds exactly one dummy word, not two.
    if storage.is_empty() {
        storage.push(0xa53c_9248);
    }
    source.extend(storage);
    source.push(GUARD);
    let input = gpu(backend.upload_words(&source))?;
    if g.input_words as usize != source.len() - 2 || g.output_words as usize != packed.len() {
        return Err(String::from("physical geometry/reference extent mismatch"));
    }
    let params = p.map(|x| x as usize);
    for alternate in [false, true] {
        let mut initial = vec![GUARD];
        initial.extend(poison(&packed, alternate));
        initial.push(GUARD);
        let out = gpu(backend.upload_words(&initial))?;
        let before = Snapshot::take(backend)?;
        gpu(backend.with_commands(|| backend.dispatch_words(Kernel::IntegerReduction, &[(&input, 1), (&out, 1)], &params)))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::IntegerReduction, false);
        before.check(backend, &work)?;
        check_words(backend, &out, &packed, true)?;
        unchanged(backend, &input, &source)?;
        observed.guarded += 1;
        observed.values += expected.output.values.len();
        // Recycle independently dirty storage before the full-writer allocation.
        let dirty = gpu(backend.upload_words(&poison(&packed, alternate)))?;
        drop(dirty);
        let before = Snapshot::take(backend)?;
        let fresh = gpu(backend.with_commands(|| backend.produce_words(packed.len(), Kernel::IntegerReduction, &[(&input, 1)], &params)))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::IntegerReduction, true);
        before.check(backend, &work)?;
        check_words(backend, &fresh, &packed, false)?;
        unchanged(backend, &input, &source)?;
        observed.fresh += 1;
        observed.values += expected.output.values.len();
    }
    observed.cases += 1;
    Ok(())
}
fn native_case(backend: &Nvidia, case: &Case, counts: &mut Counts) -> Result<()> {
    let Scope::Axis(axis) = case.scope else {
        return Ok(());
    };
    let expected = case.expected()?;
    let root = gpu(NativeTensor::from_data(data(case.dtype, &case.input.root)))?;
    let input = gpu(root.with_layout(layout(&case.input)?))?;
    let alias = input.clone();
    let before = Snapshot::take(backend)?;
    let output = gpu(backend.with_commands(|| input.reduce(axis, reduction(case.op))))?;
    before.check(backend, &native_work(case)?)?;
    check_data(&gpu(output.read())?, case.dtype, &expected.output)?;
    check_data(&gpu(root.read())?, case.dtype, &case.input.root)?;
    check_data(&gpu(alias.read())?, case.dtype, &case.input.logical())?;
    counts.native += 1;
    counts.values += expected.output.values.len();
    Ok(())
}

fn malformed(p: [u32; 24]) -> Vec<Vec<u32>> {
    let mut cases = Vec::new();
    for (field, value) in [
        (0, 2),
        (1, 0),
        (1, 9),
        (3, 2),
        (4, u32::MAX),
        (5, 0),
        (5, i32::MAX as u32 + 1),
        (6, 2),
        (7, 1),
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
        p[2] ^ (1 << 24),
        (p[2] & !0xff) | 7,
        (p[2] & !0xff00) | ((Format::F32 as u32) << 8),
        (p[2] & !0xff0000) | ((Format::F32 as u32) << 16),
    ] {
        let mut bad = p;
        bad[2] = control;
        cases.push(bad.to_vec());
    }
    cases.push(p[..23].to_vec());
    let mut long = p.to_vec();
    long.push(0);
    cases.push(long);
    // Rank-eight product overflow and signed spans must fail without allocation.
    let mut rank = p;
    rank[1] = 8;
    rank[8..16].fill(65536);
    rank[16..24].fill(1);
    cases.push(rank.to_vec());
    let mut empty = p;
    empty[0] = 0;
    empty[8] = 0;
    cases.push(empty.to_vec()); // geometry is host-valid; raw dispatch rejects N=0.
    let mut zero_axis = p;
    zero_axis[0] = 0;
    zero_axis[8] = 0;
    zero_axis[9] = 0;
    zero_axis[6] = 0;
    zero_axis[2] = raw::control(
        raw::Operation::Mean,
        Format::from_raw((p[2] >> 8) & 0xff).unwrap(),
        Format::from_raw((p[2] >> 8) & 0xff).unwrap(),
    )
    .unwrap();
    cases.push(zero_axis.to_vec());
    cases
}
fn reject<T>(result: d3os_compute::Result<T>) -> Result<()> {
    match result {
        Err(Error::InvalidParameter | Error::ShapeMismatch | Error::SizeOverflow) => Ok(()),
        Err(error) => Err(format!("noncanonical rejection: {error}")),
        Ok(_) => Err(String::from("invalid launch accepted")),
    }
}
fn reject_output_size<T>(result: d3os_compute::Result<T>, words: usize) -> Result<()> {
    if words == 0 {
        match result {
            Err(Error::InvalidShape) => Ok(()),
            Err(error) => Err(format!("zero output shape rejection: {error}")),
            Ok(_) => Err(String::from("zero output allocation accepted")),
        }
    } else {
        reject(result)
    }
}
fn rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        let case = Case {
            id: format!("reject/{dtype:?}"),
            dtype,
            input: dense(&[3, 3], generated(dtype, 9)),
            op: Operation::Sum,
            scope: Scope::Axis(1),
        };
        let p = case.parameters()?;
        let expected = output_words(dtype, &case.expected()?.output);
        let source = output_words(dtype, &case.input.root);
        let a = gpu(backend.upload_words(&source))?;
        let out_initial = poison(&expected, false);
        let out = gpu(backend.upload_words(&out_initial))?;
        let short = gpu(backend.upload_words(&[GUARD]))?;
        let before = Snapshot::take(backend)?;
        for bad in malformed(p) {
            let bad: Vec<_> = bad.into_iter().map(|x| x as usize).collect();
            reject(backend.dispatch_words(Kernel::IntegerReduction, &[(&a, 0), (&out, 0)], &bad))?;
            reject(backend.produce_words(expected.len(), Kernel::IntegerReduction, &[(&a, 0)], &bad))?;
            counts.rejected += 2;
        }
        let params = p.map(|x| x as usize);
        for buffers in [
            vec![(&a, 0)],
            vec![(&a, 0), (&out, 0), (&short, 0)],
            vec![(&a, 0), (&a, 0)],
            vec![(&a, 0), (&a, 1)],
            vec![(&short, 0), (&out, 0)],
            vec![(&a, 0), (&out, 1)],
            vec![(&a, 1), (&out, 0)],
        ] {
            reject(backend.dispatch_words(Kernel::IntegerReduction, &buffers, &params))?;
            counts.rejected += 1;
        }
        for size in [0, expected.len() - 1, expected.len() + 1] {
            reject_output_size(backend.produce_words(size, Kernel::IntegerReduction, &[(&a, 0)], &params), size)?;
            counts.rejected += 1;
        }
        reject(backend.produce_words(expected.len(), Kernel::IntegerReduction, &[(&short, 0)], &params))?;
        counts.rejected += 1;
        // Two overlapping bindings reach driver-side CPU alias validation.
        // Neither submits a kernel; all other malformed calls stop locally.
        let mut work = Work::default();
        work.services[abi::LAUNCH as usize] = 2;
        before.check(backend, &work)?;
        unchanged(backend, &a, &source)?;
        unchanged(backend, &out, &out_initial)?;
        // A valid writer on the same allocations must succeed after each batch.
        let before = Snapshot::take(backend)?;
        gpu(backend.with_commands(|| backend.dispatch_words(Kernel::IntegerReduction, &[(&a, 0), (&out, 0)], &params)))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::IntegerReduction, false);
        before.check(backend, &work)?;
        check_words(backend, &out, &expected, false)?;
    }
    Ok(())
}
fn globals(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        for op in Operation::GLOBAL {
            let values = if dtype == DType::I8 && op == Operation::Mean {
                Values::new(&[2], vec![127, 127])
            } else {
                Values::new(&[2, 3], generated(dtype, 6))
            };
            let expected = oracle::reference(dtype, &values, op, Scope::Global).map_err(|e| format!("global oracle: {e:?}"))?;
            let root = Native::int_from_data(data(dtype, &values), &NativeDevice);
            let alias = root.clone();
            let before = Snapshot::take(backend)?;
            let output = gpu(backend.with_commands(|| {
                Ok::<_, Error>(match op {
                    Operation::Sum => Native::int_sum(root.clone()),
                    Operation::Product => Native::int_prod(root.clone()),
                    Operation::Mean => Native::int_mean(root.clone()),
                    Operation::Min => Native::int_min(root.clone()),
                    Operation::Max => Native::int_max(root.clone()),
                    _ => unreachable!(),
                })
            }))?;
            let mut work = Work::default();
            work.scope();
            let stages = if dtype == DType::I32 { values.shape.len() } else { 1 };
            for _ in 0..stages {
                work.launch(
                    if dtype == DType::I32 {
                        Kernel::TensorReduction
                    } else {
                        Kernel::IntegerReduction
                    },
                    true,
                );
            }
            if dtype == DType::I32 {
                work.scope();
                work.services[abi::FREE as usize] += (stages - 1) as u64;
            }
            if op == Operation::Mean {
                work.scope();
                work.launch(
                    if dtype == DType::I32 {
                        Kernel::TensorWords
                    } else {
                        Kernel::IntegerArithmetic
                    },
                    true,
                );
                work.services[abi::FREE as usize] += 1;
            }
            before.check(backend, &work)?;
            check_data(&read_int(output)?, dtype, &expected.output)?;
            check_data(&read_int(alias)?, dtype, &values)?;
            counts.globals += 1;
            counts.values += expected.output.values.len();
        }
    }
    Ok(())
}
fn pairs(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        for op in Operation::PAIRS {
            let winning = match op {
                Operation::MinWithIndices if dtype.is_int() => 1 << (oracle::width(dtype) - 1),
                Operation::MinWithIndices => 0,
                _ if dtype.is_int() => oracle::mask(dtype) >> 1,
                _ => oracle::mask(dtype),
            };
            // I16/U16 also cross their exposed coordinate width without a
            // Cartesian expansion. Only these two long fixtures are large.
            let coordinate = if matches!(dtype, DType::I16 | DType::U16) { 65536 } else { 256 };
            let mut bits = vec![1; coordinate + 1];
            bits[coordinate] = winning;
            let values = Values::new(&[1, bits.len()], bits);
            let expected = oracle::reference(dtype, &values, op, Scope::Axis(1)).map_err(|e| format!("pair reference: {e:?}"))?;
            let root = Native::int_from_data(data(dtype, &values), &NativeDevice);
            let alias = root.clone();
            let before = Snapshot::take(backend)?;
            let (value, index, value_sum, index_sum) = gpu(backend.with_commands(|| {
                let (value, index) = match op {
                    Operation::MinWithIndices => Native::int_min_dim_with_indices(root, 1),
                    _ => Native::int_max_dim_with_indices(root, 1),
                };
                // Queue consumers before any read; pair coordinates are never
                // fed back into a gather after exposed-index narrowing.
                let value_sum = Native::int_sum_dim(value.clone(), 0);
                let index_sum = Native::int_sum_dim(index.clone(), 0);
                Ok::<_, Error>((value, index, value_sum, index_sum))
            }))?;
            let mut work = Work::default();
            work.scope();
            work.scope();
            for _ in 0..4 {
                work.launch(
                    if dtype == DType::I32 {
                        Kernel::TensorReduction
                    } else {
                        Kernel::IntegerReduction
                    },
                    true,
                );
            }
            // root's last command-local owner drops, alias keeps storage alive.
            before.check(backend, &work)?;
            check_data(&read_int(value)?, dtype, &expected.output)?;
            check_data(&read_int(value_sum)?, dtype, &expected.output)?;
            let indices = expected.indices.as_ref().ok_or("missing pair indices")?;
            check_data(&read_int(index)?, dtype, indices)?;
            check_data(&read_int(index_sum)?, dtype, indices)?;
            check_data(&read_int(alias)?, dtype, &values)?;
            if op == Operation::MaxWithIndices {
                let last = Native::int_from_data(data(dtype, &values), &NativeDevice);
                let clone = last.clone();
                drop(last);
                let before = Snapshot::take(backend)?;
                let (value, index) = gpu(backend.with_commands(|| {
                    let (value, index) = Native::int_max_dim_with_indices(clone, 1);
                    Ok::<_, Error>((Native::int_sum_dim(value, 0), Native::int_sum_dim(index, 0)))
                }))?;
                let mut work = Work::default();
                work.scope();
                work.scope();
                for _ in 0..4 {
                    work.launch(
                        if dtype == DType::I32 {
                            Kernel::TensorReduction
                        } else {
                            Kernel::IntegerReduction
                        },
                        true,
                    );
                }
                work.services[abi::FREE as usize] = 3;
                before.check(backend, &work)?;
                check_data(&read_int(value)?, dtype, &expected.output)?;
                check_data(&read_int(index)?, dtype, indices)?;
                counts.queued += 1;
                counts.values += 2;
            }
            counts.pairs += 1;
            counts.values += 4;
        }
    }
    // A float value must come from the winner, independently of an I8 index.
    let root = Native::float_from_data(TensorData::new((0..201).map(|i| i as f32).collect::<Vec<_>>(), [1, 201]), &NativeDevice);
    let alias = root.clone();
    let before = Snapshot::take(backend)?;
    let (value, index, consumer, index_consumer) = gpu(backend.with_commands(|| {
        let (value, index) = Native::float_max_dim_with_indices(root, 1, IntDType::I8);
        let consumer = Native::float_sum_dim(value.clone(), 0);
        let index_consumer = Native::int_sum_dim(index.clone(), 0);
        Ok::<_, Error>((value, index, consumer, index_consumer))
    }))?;
    let mut work = Work::default();
    work.scope();
    work.scope();
    for _ in 0..3 {
        work.launch(Kernel::TensorReduction, true);
    }
    work.launch(Kernel::TensorCast, true);
    work.launch(Kernel::IntegerReduction, true);
    work.services[abi::FREE as usize] = 1; // temporary I32 winner before cast
    before.check(backend, &work)?;
    for tensor in [value, consumer] {
        let actual = try_read_sync(Native::float_into_data(tensor))
            .ok_or("float read pending")?
            .map_err(|e| format!("float pair read: {e:?}"))?;
        if actual.dtype() != DType::F32 || actual.shape().as_slice() != [1, 1] || actual.as_bytes() != 200f32.to_le_bytes() {
            return Err(String::from("F32 pair used narrowed I8 coordinate as gather index"));
        }
    }
    check_data(&read_int(index)?, DType::I8, &Values::new(&[1, 1], vec![200]))?;
    check_data(&read_int(index_consumer)?, DType::I8, &Values::new(&[1, 1], vec![200]))?;
    let source = try_read_sync(Native::float_into_data(alias))
        .ok_or("float source pending")?
        .map_err(|e| format!("float source: {e:?}"))?;
    if source.as_bytes() != (0..201).flat_map(|i| (i as f32).to_le_bytes()).collect::<Vec<_>>() {
        return Err(String::from("float pair changed source"));
    }
    counts.pairs += 1;
    counts.values += 4;
    Ok(())
}
fn queued(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        let values = Values::new(&[3, 129], generated(dtype, 387));
        let first = oracle::reference(dtype, &values, Operation::Sum, Scope::Axis(1)).map_err(|e| format!("queued reference: {e:?}"))?;
        let second = oracle::reference(dtype, &first.output, Operation::Product, Scope::Axis(0)).map_err(|e| format!("queued reference: {e:?}"))?;
        let case = Case {
            id: format!("queued/{dtype:?}"),
            dtype,
            input: dense(&values.shape, values.values.clone()),
            op: Operation::Sum,
            scope: Scope::Axis(1),
        };
        let p = case.parameters()?.map(|x| x as usize);
        let next = Case {
            id: String::from("consumer"),
            dtype,
            input: dense(&first.output.shape, first.output.values.clone()),
            op: Operation::Product,
            scope: Scope::Axis(0),
        };
        let p2 = next.parameters()?.map(|x| x as usize);
        let a = gpu(backend.upload_words(&output_words(dtype, &values)))?;
        let before = Snapshot::take(backend)?;
        let output = gpu(backend.with_commands(|| {
            let temporary = backend.produce_words(output_words(dtype, &first.output).len(), Kernel::IntegerReduction, &[(&a, 0)], &p)?;
            drop(a); // last input owner before command completion
            let survivor = backend.produce_words(output_words(dtype, &second.output).len(), Kernel::IntegerReduction, &[(&temporary, 0)], &p2)?;
            drop(temporary); // last intermediate owner while its consumer is queued
            Ok::<_, Error>(survivor)
        }))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::IntegerReduction, true);
        work.launch(Kernel::IntegerReduction, true);
        work.services[abi::FREE as usize] = 2;
        before.check(backend, &work)?;
        check_words(backend, &output, &output_words(dtype, &second.output), false)?;
        counts.queued += 1;
        counts.values += second.output.values.len();
        let root = Native::int_from_data(data(dtype, &values), &NativeDevice);
        let clone = root.clone();
        drop(root);
        let before = Snapshot::take(backend)?;
        let output = gpu(backend.with_commands(|| {
            let temporary = Native::int_sum_dim(clone, 1);
            let alias = temporary.clone();
            drop(temporary);
            Ok::<_, Error>(Native::int_prod_dim(alias, 0))
        }))?;
        let mut work = Work::default();
        work.scope();
        let kernel = if dtype == DType::I32 {
            Kernel::TensorReduction
        } else {
            Kernel::IntegerReduction
        };
        work.launch(kernel, true);
        work.launch(kernel, true);
        work.services[abi::FREE as usize] = 2;
        before.check(backend, &work)?;
        check_data(&read_int(output)?, dtype, &second.output)?;
        counts.queued += 1;
        counts.values += second.output.values.len();
    }
    Ok(())
}
fn empty_and_deferred(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        let values = Values::new(&[0, 3], vec![]);
        let case = Case {
            id: format!("outer-empty/{dtype:?}"),
            dtype,
            input: dense(&[0, 3], vec![]),
            op: Operation::Mean,
            scope: Scope::Axis(1),
        };
        // The legacy I32 path reuses empty storage without GPU work. Other
        // integer formats allocate a fresh placeholder and run only Zero.
        native_case(backend, &case, counts)?;
        let source = Native::int_from_data(data(dtype, &values), &NativeDevice);
        let alias = source.clone();
        let before = Snapshot::take(backend)?;
        let invalid = Native::int_sum_dim(source, 8);
        if invalid.dtype() != dtype || invalid.shape().as_slice() != [0, 3] {
            return Err(String::from("deferred invalid-axis metadata changed"));
        }
        before.check(backend, &Work::default())?;
        let error = try_read_sync(Native::int_into_data(invalid))
            .ok_or("deferred read pending")?
            .err()
            .ok_or("invalid axis produced readable data")?;
        if error.is_device_poisoned() {
            return Err(String::from("shape error poisoned device"));
        }
        check_data(&read_int(alias)?, dtype, &values)?;
        let good = Native::int_from_data(data(dtype, &Values::new(&[1], vec![1])), &NativeDevice);
        check_data(&read_int(Native::int_sum(good))?, dtype, &Values::new(&[1], vec![1]))?;
        counts.rejected += 1;
    }
    Ok(())
}

fn expected_counts(cases: &[Case]) -> Result<Counts> {
    let mut counts = Counts::default();
    for case in cases {
        let n = case.expected()?.output.values.len();
        counts.cases += 1;
        counts.guarded += 2;
        counts.fresh += 2;
        counts.values += n * 4;
        if matches!(case.scope, Scope::Axis(_)) {
            counts.native += 1;
            counts.values += n;
        }
    }
    counts.native += 8; // positive-axis empty outers
    counts.globals = 40;
    counts.values += 40;
    counts.pairs = 17;
    counts.values += 68;
    counts.queued = 24;
    counts.values += 32;
    counts.rejected = oracle::DTYPES
        .iter()
        .map(|&dtype| {
            let case = Case {
                id: String::new(),
                dtype,
                input: dense(&[3, 3], generated(dtype, 9)),
                op: Operation::Sum,
                scope: Scope::Axis(1),
            };
            case.parameters().map(|p| malformed(p).len() * 2 + 11 + 1)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum();
    Ok(counts)
}
fn audit_geometry(case: &Case) -> Result<()> {
    let p = case.parameters()?;
    let expected = case.expected()?;
    let n = expected.output.values.len();
    let length = match case.scope {
        Scope::Axis(axis) => case.input.logical().shape[axis],
        Scope::Global => case.input.logical().values.len(),
    };
    let output_count = (n * case.dtype.size()).div_ceil(4);
    let owners = if case.dtype.size() == 8 { n } else { output_count };
    let work_items = owners * if length >= 128 { 128 } else { 1 };
    let g = raw::geometry(&p).map_err(|e| format!("geometry: {e:?}"))?;
    let w = Kernel::IntegerReduction.work(&p).map_err(|e| format!("work: {e:?}"))?;
    let layout = layout(&case.input)?;
    let logical = case.input.logical();
    for (i, bits) in logical.values.iter().enumerate() {
        let physical = gpu(layout.element_offset(i))?;
        if case.input.root.values[physical] != *bits {
            return Err(format!("physical view mismatch at {i}"));
        }
    }
    if raw::PARAMETERS != 24
        || raw::THREADS != 128
        || g.output_count as usize != n
        || g.reduction_length as usize != length
        || g.owners as usize != owners
        || raw::work_items(&p).ok() != Some(work_items as u32)
        || g.schedule != if length >= 128 { raw::Schedule::Tree } else { raw::Schedule::Short }
        || w.threads != 128
        || w.blocks as usize != work_items.div_ceil(128)
        || w.extents[..2]
            != [
                if logical.values.is_empty() {
                    1
                } else {
                    (case.input.root.values.len() * case.dtype.size()).div_ceil(4) as u64
                },
                output_count as u64,
            ]
        || (w.pointers, w.output, w.in_place, w.mutated) != (2, 1, false, 0)
        || Kernel::IntegerReduction.entry(&p)
            != if length >= 128 {
                "burn_integer_reduce_tree"
            } else {
                "burn_integer_reduce_short"
            }
    {
        return Err(format!("work/metadata mismatch {}", case.id));
    }
    let input_start = 0x100004;
    let output_start = input_start + w.extents[0] * 4;
    if w.validate_aliases(&[input_start, output_start]).is_err()
        || w.validate_aliases(&[input_start, input_start]).is_ok()
        || w.validate_aliases(&[input_start, output_start - 4]).is_ok()
    {
        return Err(String::from("nonempty alias or four-byte alignment contract changed"));
    }
    Ok(())
}
fn contract(cases: &[Case]) -> Result<()> {
    if abi::VERSION < 51 || abi::KERNEL_COUNT < 83 || Kernel::IntegerReduction as usize != 82 || oracle::CORPUS_VERSION != 1 {
        return Err(String::from("integer reduction protocol/registration mismatch"));
    }
    let shared = oracle::expected_summary(&oracle::Options::default());
    if (
        shared.dtypes,
        shared.cases,
        shared.valid_cases,
        shared.diagnostic_cases,
        shared.steps,
        shared.public_cases,
        shared.pair_steps,
        shared.chain_cases,
        shared.chain_steps,
        shared.values,
        shared.source_checks,
        shared.alias_checks,
    ) != (8, 365, 323, 42, 347, 155, 52, 16, 40, 2360, 323, 323)
    {
        return Err(String::from("shared integer semantic identity/counts changed"));
    }
    if raw::FORMATS != FORMATS || !(100..=250).contains(&cases.len()) {
        return Err(String::from("raw bounded plan changed"));
    }
    for case in cases {
        audit_geometry(case)?;
    }
    if expected_counts(cases)? != EXPECTED {
        return Err(String::from("raw expected counts changed"));
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<()> {
    // No device call is allowed before the parent accepts the freeze/API plan.
    let cases = plan();
    contract(&cases)?;
    let expected = expected_counts(&cases)?;
    crate::report(format_args!(
        "NVIDIA_BURN_INTEGER_REDUCTION_PLAN frozen={FROZEN} reference_sha256={REFERENCE_SHA256} expected={expected:?}"
    ));
    if !FROZEN {
        return Err(String::from("integer reduction UNFROZEN: parent source/API/protocol approval required"));
    }
    let mut observed = Observed::default();
    for case in &cases {
        let result = raw_case(backend, case, &mut observed.counts);
        observed.record(
            &format!(
                "raw/{} {:?} {:?} {:?} shape={:?} root={:?} views={:?}",
                case.id,
                case.dtype,
                case.op,
                case.scope,
                case.input.logical().shape,
                case.input.root.shape,
                case.input.views
            ),
            result,
        );
        let result = native_case(backend, case, &mut observed.counts);
        observed.record(&format!("native/{}", case.id), result);
    }
    for (name, suite) in [
        ("rejections", rejections as fn(&Nvidia, &mut Counts) -> Result<()>),
        ("globals", globals),
        ("pairs", pairs),
        ("queued", queued),
        ("empty-deferred", empty_and_deferred),
    ] {
        let result = suite(backend, &mut observed.counts);
        observed.record(name, result);
    }
    // The shared runner exercises actual public Tensor APIs and deferred Native
    // diagnostics. Its CPU expected summary is never an observed GPU count.
    let device = d3os_ai::native_device().map_err(|e| format!("integer device: {e}"))?;
    let options = oracle::Options {
        native_error_extensions: true,
        ..Default::default()
    };
    let report = try_read_sync(oracle::run(&device, &options)).ok_or("shared integer run pending")?;
    for failure in report.failures.iter().take(12) {
        observed.record(&format!("shared/{}", failure.case_id), Err(failure.message.clone()));
    }
    let footer = oracle::success_footer(&options, &report);
    let accepted = observed.failed == 0 && observed.counts == expected && footer.is_ok();
    crate::report(format_args!(
        "NVIDIA_BURN_INTEGER_REDUCTION_OBSERVED accepted={accepted} counts={:?} failed={} shared_attempted={} shared_passed={} shared_errors={} shared_failed={} examples={:?}",
        observed.counts, observed.failed, report.attempted, report.passed, report.observed_errors, report.failed, observed.examples
    ));
    if !accepted {
        return Err(format!(
            "integer reduction conformance failed; expected={expected:?}; observed={:?}; failed={}; shared_failed={}",
            observed.counts, observed.failed, report.failed
        ));
    }
    crate::report(format_args!("{} os=d3os backend=native fusion=disabled", footer?));
    crate::report(format_args!(
        "NVIDIA_BURN_INTEGER_REDUCTION_STRICT_OK reference_sha256={REFERENCE_SHA256} protocol={} kernels={} expected={expected:?} observed={:?}",
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
    fn bounded_plan_checks_physical_views_schedules_and_full_word_poison() {
        let cases = plan();
        contract(&cases).unwrap();
        let counts = expected_counts(&cases).unwrap();
        assert_eq!(counts, EXPECTED);
        println!("integer reduction pure expected: {counts:?}");
        for case in &cases {
            let expected = output_words(case.dtype, &case.expected().unwrap().output);
            let a: Vec<_> = poison(&expected, false).iter().flat_map(|w| w.to_le_bytes()).collect();
            let b: Vec<_> = poison(&expected, true).iter().flat_map(|w| w.to_le_bytes()).collect();
            let target: Vec<_> = expected.iter().flat_map(|w| w.to_le_bytes()).collect();
            assert!(a.iter().zip(&b).zip(target).all(|((&a, &b), target)| a != b && a != target && b != target));
        }
        for dtype in oracle::DTYPES {
            for op in Operation::AXIS {
                assert!(cases.iter().any(|c| c.dtype == dtype && c.op == op));
            }
            for length in oracle::LENGTHS {
                assert!(cases.iter().any(|c| c.dtype == dtype && c.input.logical().shape == [3, length]));
            }
        }
    }
    #[test]
    fn malformed_metadata_empty_outputs_and_narrowing_boundaries_are_independent() {
        for dtype in oracle::DTYPES {
            let case = Case {
                id: String::new(),
                dtype,
                input: dense(&[3, 3], generated(dtype, 9)),
                op: Operation::Sum,
                scope: Scope::Axis(1),
            };
            let p = case.parameters().unwrap();
            for bad in malformed(p) {
                assert!(Kernel::IntegerReduction.work(&bad).is_err(), "{dtype:?} {bad:?}");
            }
            let empty = Case {
                input: dense(&[0, 3], vec![]),
                ..case.clone()
            };
            let p = empty.parameters().unwrap();
            assert!(raw::geometry(&p).is_ok());
            assert!(raw::validate(&p).is_err());
            let wide = Case {
                input: dense(&[2, 0], vec![]),
                op: Operation::Product,
                ..case
            };
            let g = raw::geometry(&wide.parameters().unwrap()).unwrap();
            assert_eq!(g.input_words, 1);
            assert_eq!(g.output_words as usize, (dtype.size() * 2).div_ceil(4));
        }
        let values = Values::new(&[2], vec![127, 127]);
        assert_eq!(
            oracle::reference(DType::I8, &values, Operation::Mean, Scope::Axis(0)).unwrap().output.values,
            [127]
        );
        assert_eq!(
            oracle::reference(DType::I8, &values, Operation::Mean, Scope::Global).unwrap().output.values,
            [255]
        );
    }
    #[test]
    fn zero_output_size_uses_the_real_allocator_shape_error() {
        let result = Nvidia.produce_words(0, Kernel::IntegerReduction, &[], &[]);
        assert!(matches!(result, Err(Error::InvalidShape)));
        reject_output_size(result, 0).unwrap();
        assert!(reject_output_size::<()>(Err(Error::InvalidParameter), 0).is_err());
        assert!(reject_output_size::<()>(Err(Error::InvalidShape), 1).is_err());
        assert!(reject_output_size(Ok(()), 0).is_err());
    }
    #[test]
    fn all_kernel_and_service_counters_detect_hidden_work_and_resets() {
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
        work.launch(Kernel::IntegerReduction, true);
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
}
