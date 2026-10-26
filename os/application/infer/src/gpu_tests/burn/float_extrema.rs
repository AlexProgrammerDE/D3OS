//! Exact floating extrema storage, routing, and resident command contracts.
//! CPU checks enumerate plans. GPU execution requires the parent's source freeze.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{
            DType, IntDType, Shape, Slice, TensorData, TensorMetadata,
            ops::{FloatTensorOps, IntTensorOps},
            tensor::{FloatTensor, IntTensor},
        },
        tensor::try_read_sync,
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::float_extrema::{self as oracle, Expected, Input, Operation, Scope, Values, View},
};
use d3os_compute::{
    Backend as _, Error,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{self as abi, Kernel, cast::Format, float_extrema as raw, reduction::Reduction};

type Result<T> = core::result::Result<T, String>;
// Only the parent may approve the exact shared/source/API/corpus identities.
const FROZEN: bool = true;
const REFERENCE_SHA256: &str = "2635fdb60ee613403139d1acf1e26d2822c1170d4cb6a3cce14f17f081ea71df";
const GUARD: u32 = 0x6bad_beef;
const FORMATS: [Format; 5] = [Format::F64, Format::F32, Format::Flex32, Format::F16, Format::BF16];
const INDEX_FORMATS: [Format; 8] = [
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
    value.map_err(|e| format!("float extrema: {e}"))
}
fn format_of(dtype: DType) -> Format {
    if let Some(index) = oracle::DTYPES.iter().position(|&d| d == dtype) {
        FORMATS[index]
    } else {
        INDEX_FORMATS[oracle::INDEX_DTYPES.iter().position(|&d| d == dtype).expect("index dtype")]
    }
}
fn raw_op(op: Operation) -> raw::Operation {
    match op {
        Operation::Min => raw::Operation::Min,
        Operation::Max => raw::Operation::Max,
        Operation::ArgMin => raw::Operation::ArgMin,
        Operation::ArgMax => raw::Operation::ArgMax,
        _ => unreachable!("pairs use two independent resident folds"),
    }
}
fn reduction(op: Operation) -> Reduction {
    match op {
        Operation::Min => Reduction::Min,
        Operation::Max => Reduction::Max,
        Operation::ArgMin => Reduction::ArgMin,
        Operation::ArgMax => Reduction::ArgMax,
        _ => unreachable!("single extrema"),
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
        .ok_or("float extrema read pending")?
        .map_err(|e| format!("integer read: {e:?}"))
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
    index_dtype: DType,
}
impl Case {
    fn output_dtype(&self) -> DType {
        if self.op.is_arg() { self.index_dtype } else { self.dtype }
    }
    fn expected(&self) -> Result<Expected> {
        oracle::reference(self.dtype, self.index_dtype, &self.input.logical(), self.op, self.scope).map_err(|e| format!("reference: {e:?}"))
    }
    fn parameters(&self) -> Result<[u32; 24]> {
        let layout = layout(&self.input)?;
        let mut p = [0; 24];
        let expected = self.expected()?;
        p[..8].copy_from_slice(&[
            expected.output.values.len() as u32,
            layout.shape().num_dims().max(1) as u32,
            raw::control(raw_op(self.op), format_of(self.dtype), format_of(self.output_dtype())).map_err(|e| format!("control: {e:?}"))?,
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
// Each row isolates a numerical witness so NaNs cannot hide numeric failures.
fn fixture(dtype: DType, length: usize) -> Input {
    let p = oracle::pool(dtype);
    let sign = 1u64 << (oracle::width(dtype) - 1);
    let mut values = vec![p[6]; 7 * length];
    for row in 0..7 {
        let witness: &[u64] = match row {
            0 => &p[6..9],
            1 => &p[..2],
            2 => &p[2..6],
            3 => &p[10..14],
            4 | 5 => &p[6..7],
            _ => &p[12..13],
        };
        for i in 0..length {
            values[row * length + i] = witness[(i + row) % witness.len()];
        }
    }
    let first = usize::from(length > 1);
    values[4 * length + first] = p[14]; // positive signaling NaN wins over numbers
    values[4 * length + 129.min(length - 1)] = if length > 129 { p[15] } else { p[14] };
    values[5 * length + 129.min(length - 1)] = sign | p[14] | 3; // negative signaling payload
    if length > 130 {
        values[5 * length + 130] = p[15];
    }
    dense(&[7, length], values)
}
fn case(dtype: DType, input: Input, op: Operation, scope: Scope, index_dtype: DType, id: String) -> Case {
    Case {
        id,
        dtype,
        input,
        op,
        scope,
        index_dtype: if op.is_arg() { index_dtype } else { DType::I32 },
    }
}
fn plan() -> Vec<Case> {
    let mut cases = Vec::new();
    for (d, dtype) in oracle::DTYPES.into_iter().enumerate() {
        for op in Operation::AXIS {
            let outputs = if op.is_arg() {
                &oracle::INDEX_DTYPES[..]
            } else {
                core::slice::from_ref(&dtype)
            };
            for (i, &index_dtype) in outputs.iter().enumerate() {
                let length = if (i + usize::from(op.is_min())).is_multiple_of(2) { 3 } else { 129 };
                cases.push(case(
                    dtype,
                    fixture(dtype, length),
                    op,
                    Scope::Axis(1),
                    index_dtype,
                    format!("projection/{dtype:?}/{op:?}/{index_dtype:?}"),
                ));
            }
        }
        for (i, length) in oracle::LENGTHS.into_iter().enumerate() {
            cases.push(case(
                dtype,
                fixture(dtype, length),
                Operation::AXIS[(d + i) % 4],
                Scope::Axis(1),
                DType::I32,
                format!("length/{dtype:?}/{length}"),
            ));
        }
        for op in Operation::AXIS {
            let mut input = fixture(dtype, 131);
            input.views.push(View::Flip(vec![1]));
            cases.push(case(dtype, input, op, Scope::Axis(1), DType::I32, format!("reverse/{dtype:?}/{op:?}")));
        }
        let mut transposed = fixture(dtype, 129);
        transposed.views.push(View::Permute(vec![1, 0]));
        cases.push(case(
            dtype,
            transposed,
            Operation::Min,
            Scope::Axis(0),
            DType::I32,
            format!("transpose/{dtype:?}"),
        ));
        let mut offset = fixture(dtype, 7);
        offset.views.push(View::Slice {
            starts: vec![0, 1],
            steps: vec![1, 2],
            shape: vec![7, 3],
        });
        cases.push(case(
            dtype,
            offset,
            Operation::Max,
            Scope::Axis(1),
            DType::I32,
            format!("offset-subword/{dtype:?}"),
        ));
        let mut broadcast = dense(&[1, 3], oracle::pool(dtype)[6..9].to_vec());
        broadcast.views.push(View::Expand(vec![7, 3]));
        cases.push(case(
            dtype,
            broadcast,
            Operation::ArgMax,
            Scope::Axis(1),
            DType::U8,
            format!("broadcast/{dtype:?}"),
        ));
        for op in Operation::GLOBAL {
            cases.push(case(
                dtype,
                fixture(dtype, 3),
                op,
                Scope::Global,
                DType::I32,
                format!("global/{dtype:?}/{op:?}"),
            ));
        }
        cases.push(case(
            dtype,
            dense(&[], vec![oracle::pool(dtype)[14]]),
            Operation::Min,
            Scope::Global,
            DType::I32,
            format!("scalar/{dtype:?}"),
        ));
    }
    for rank in 1..=8 {
        let dtype = oracle::DTYPES[rank % 5];
        let mut shape = vec![1; rank];
        shape[rank - 1] = 3;
        let mut input = dense(&shape, oracle::pool(dtype)[6..9].to_vec());
        if rank > 1 {
            let mut expanded = shape;
            expanded[0] = 2;
            input.views.push(View::Expand(expanded));
        }
        input.views.push(View::Flip(vec![rank - 1]));
        input.views.push(View::Permute((0..rank).rev().collect()));
        cases.push(case(dtype, input, Operation::ArgMin, Scope::Axis(0), DType::I64, format!("rank/{rank}")));
    }
    // Cross output-owner CTA boundaries without multiplying the input matrix.
    for (dtype, count, index_dtype) in [
        (DType::F64, 129, DType::I32),
        (DType::F32, 127, DType::I32),
        (DType::Flex32, 129, DType::I32),
        (DType::F16, 257, DType::I32),
        (DType::BF16, 513, DType::U8),
    ] {
        let op = if index_dtype == DType::U8 { Operation::ArgMax } else { Operation::Min };
        let bits = oracle::pool(dtype)[6];
        cases.push(case(
            dtype,
            dense(&[count, 1], vec![bits; count]),
            op,
            Scope::Axis(1),
            index_dtype,
            format!("owner-tail/{dtype:?}/{count}"),
        ));
        cases.push(case(
            dtype,
            dense(&[1, 129], vec![oracle::pool(dtype)[14]; 129]),
            Operation::Max,
            Scope::Axis(1),
            DType::I32,
            format!("all-signaling-nan/{dtype:?}"),
        ));
    }
    // Low-bit projection must never change the address used to reload the value.
    for dtype in oracle::DTYPES {
        let p = oracle::pool(dtype);
        let mut values = vec![p[6]; 201];
        values[200] = p[7];
        cases.push(case(
            dtype,
            dense(&[1, 201], values),
            Operation::ArgMax,
            Scope::Axis(1),
            DType::I8,
            format!("coordinate-200/{dtype:?}"),
        ));
    }
    cases
}
#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    cases: usize,
    guarded: usize,
    fresh: usize,
    native: usize,
    pairs: usize,
    queued: usize,
    empty: usize,
    rejected: usize,
    values: usize,
}
// Filled from the independently enumerated CPU plan, never GPU observations.
const EXPECTED: Counts = Counts {
    cases: 188,
    guarded: 376,
    fresh: 376,
    native: 188,
    pairs: 40,
    queued: 10,
    empty: 60,
    rejected: 360,
    values: 11400,
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
    let packed = output_words(case.output_dtype(), &expected.output);
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
        gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatExtrema, &[(&input, 1), (&out, 1)], &params)))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::FloatExtrema, false);
        before.check(backend, &work)?;
        check_words(backend, &out, &packed, true)?;
        unchanged(backend, &input, &source)?;
        observed.guarded += 1;
        observed.values += expected.output.values.len();
        // Recycle independently dirty storage before the full-writer allocation.
        let dirty = gpu(backend.upload_words(&poison(&packed, alternate)))?;
        drop(dirty);
        let before = Snapshot::take(backend)?;
        let fresh = gpu(backend.with_commands(|| backend.produce_words(packed.len(), Kernel::FloatExtrema, &[(&input, 1)], &params)))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::FloatExtrema, true);
        before.check(backend, &work)?;
        check_words(backend, &fresh, &packed, false)?;
        unchanged(backend, &input, &source)?;
        observed.fresh += 1;
        observed.values += expected.output.values.len();
    }
    observed.cases += 1;
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
fn int_dtype(dtype: DType) -> IntDType {
    match dtype {
        DType::I64 => IntDType::I64,
        DType::I32 => IntDType::I32,
        DType::I16 => IntDType::I16,
        DType::I8 => IntDType::I8,
        DType::U64 => IntDType::U64,
        DType::U32 => IntDType::U32,
        DType::U16 => IntDType::U16,
        DType::U8 => IntDType::U8,
        _ => unreachable!(),
    }
}
enum Output {
    Float(FloatTensor<Native>),
    Int(IntTensor<Native>),
}
impl Output {
    fn read(self) -> Result<TensorData> {
        match self {
            Self::Float(t) => read_float(t),
            Self::Int(t) => read_int(t),
        }
    }
}
fn native_apply(input: FloatTensor<Native>, op: Operation, scope: Scope, index_dtype: DType) -> Output {
    match (scope, op) {
        (Scope::Global, Operation::Min) => Output::Float(Native::float_min(input)),
        (Scope::Global, Operation::Max) => Output::Float(Native::float_max(input)),
        (Scope::Axis(axis), Operation::Min) => Output::Float(Native::float_min_dim(input, axis)),
        (Scope::Axis(axis), Operation::Max) => Output::Float(Native::float_max_dim(input, axis)),
        (Scope::Axis(axis), Operation::ArgMin) => Output::Int(Native::float_argmin(input, axis, int_dtype(index_dtype))),
        (Scope::Axis(axis), Operation::ArgMax) => Output::Int(Native::float_argmax(input, axis, int_dtype(index_dtype))),
        _ => unreachable!(),
    }
}
fn native_work(case: &Case) -> Work {
    let mut work = Work::default();
    work.scope();
    if case.dtype != DType::F32 {
        work.launch(Kernel::FloatExtrema, true);
    } else {
        let stages = match case.scope {
            Scope::Axis(_) => 1,
            Scope::Global => case.input.logical().shape.len(),
        };
        for _ in 0..stages {
            work.launch(Kernel::TensorReduction, true);
        }
        if case.scope == Scope::Global {
            work.scope();
            work.services[abi::FREE as usize] += stages.saturating_sub(1) as u64;
        }
        if case.op.is_arg() && case.index_dtype != DType::I32 {
            work.launch(Kernel::TensorCast, true);
            work.services[abi::FREE as usize] += 1;
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
    let output = gpu(backend.with_commands(|| Ok::<_, Error>(native_apply(root.clone(), case.op, case.scope, case.index_dtype))))?;
    before.check(backend, &native_work(case))?;
    check_data(&output.read()?, case.output_dtype(), &expected.output)?;
    check_data(&read_float(alias)?, case.dtype, &case.input.logical())?;
    check_data(&read_float(source)?, case.dtype, &case.input.root)?;
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
        (p[2] & !0xff00) | ((Format::I32 as u32) << 8),
        (p[2] & !0xff0000) | ((Format::I32 as u32) << 16),
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
        raw::Operation::Min,
        Format::from_raw((p[2] >> 8) & 0xff).unwrap(),
        Format::from_raw((p[2] >> 8) & 0xff).unwrap(),
    )
    .unwrap();
    cases.push(zero_axis.to_vec());
    zero_axis[0] = 3;
    zero_axis[8] = 3;
    cases.push(zero_axis.to_vec());
    let mut global_arg = p;
    global_arg[0] = 1;
    global_arg[3] = raw::ALL_AXES;
    global_arg[6] = 9;
    global_arg[2] = raw::control(raw::Operation::ArgMin, Format::from_raw((p[2] >> 8) & 0xff).unwrap(), Format::I32).unwrap();
    cases.push(global_arg.to_vec());
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
            input: dense(&[3, 3], fixture(dtype, 3).root.values[..9].to_vec()),
            op: Operation::Min,
            scope: Scope::Axis(1),
            index_dtype: DType::I32,
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
            reject(backend.dispatch_words(Kernel::FloatExtrema, &[(&a, 0), (&out, 0)], &bad))?;
            reject(backend.produce_words(expected.len(), Kernel::FloatExtrema, &[(&a, 0)], &bad))?;
            counts.rejected += 2;
        }
        // Empty F64 binds one actual word. N=0 must reject before a launch.
        let mut empty = p;
        empty[0] = 0;
        empty[8] = 0;
        empty[5] = 0;
        let empty_params = empty.map(|x| x as usize);
        reject(backend.dispatch_words(Kernel::FloatExtrema, &[(&short, 0), (&out, 0)], &empty_params))?;
        reject(backend.produce_words(1, Kernel::FloatExtrema, &[(&short, 0)], &empty_params))?;
        counts.rejected += 2;
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
            reject(backend.dispatch_words(Kernel::FloatExtrema, &buffers, &params))?;
            counts.rejected += 1;
        }
        for size in [0, expected.len() - 1, expected.len() + 1] {
            reject_output_size(backend.produce_words(size, Kernel::FloatExtrema, &[(&a, 0)], &params), size)?;
            counts.rejected += 1;
        }
        reject(backend.produce_words(expected.len(), Kernel::FloatExtrema, &[(&short, 0)], &params))?;
        counts.rejected += 1;
        // Two overlapping bindings reach driver-side CPU alias validation.
        // Neither submits a kernel; all other malformed calls stop locally.
        let mut work = Work::default();
        work.services[abi::LAUNCH as usize] = 2;
        before.check(backend, &work)?;
        unchanged(backend, &a, &source)?;
        unchanged(backend, &out, &out_initial)?;
        unchanged(backend, &short, &[GUARD])?;
        // A valid writer on the same allocations must succeed after each batch.
        let before = Snapshot::take(backend)?;
        gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatExtrema, &[(&a, 0), (&out, 0)], &params)))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::FloatExtrema, false);
        before.check(backend, &work)?;
        check_words(backend, &out, &expected, false)?;
    }
    Ok(())
}
fn pair_input(dtype: DType) -> Input {
    let p = oracle::pool(dtype);
    let mut bits = vec![p[6]; 201];
    bits[200] = p[7];
    dense(&[1, 201], bits)
}
fn pair_work(dtype: DType, index_dtype: DType, consumers: bool) -> Work {
    let mut work = Work::default();
    work.scope();
    work.scope();
    let kernel = if dtype == DType::F32 { Kernel::TensorReduction } else { Kernel::FloatExtrema };
    work.launch(kernel, true);
    work.launch(kernel, true);
    if dtype == DType::F32 && index_dtype != DType::I32 {
        work.launch(Kernel::TensorCast, true);
        work.services[abi::FREE as usize] += 1;
    }
    if consumers {
        work.launch(kernel, true);
        work.launch(
            if index_dtype == DType::I32 {
                Kernel::TensorReduction
            } else {
                Kernel::IntegerReduction
            },
            true,
        );
    }
    work
}
fn pairs(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        let input = pair_input(dtype);
        for index_dtype in oracle::INDEX_DTYPES {
            let expected =
                oracle::reference(dtype, index_dtype, &input.root, Operation::MaxWithIndices, Scope::Axis(1)).map_err(|e| format!("pair reference: {e:?}"))?;
            let source = Native::float_from_data(data(dtype, &input.root), &NativeDevice);
            let alias = source.clone();
            let before = Snapshot::take(backend)?;
            let (value, index, consumer, index_consumer) = gpu(backend.with_commands(|| {
                let (value, index) = Native::float_max_dim_with_indices(source, 1, int_dtype(index_dtype));
                let consumer = Native::float_max_dim(value.clone(), 0);
                let index_consumer = Native::int_max_dim(index.clone(), 0);
                Ok::<_, Error>((value, index, consumer, index_consumer))
            }))?;
            before.check(backend, &pair_work(dtype, index_dtype, true))?;
            for tensor in [value, consumer] {
                check_data(&read_float(tensor)?, dtype, &expected.output)?;
            }
            let indices = expected.indices.as_ref().ok_or("missing pair indices")?;
            for tensor in [index, index_consumer] {
                check_data(&read_int(tensor)?, index_dtype, indices)?;
            }
            check_data(&read_float(alias)?, dtype, &input.root)?;
            counts.pairs += 1;
            counts.values += 4;
        }
    }
    Ok(())
}
fn queued(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        let first = case(
            dtype,
            fixture(dtype, 129),
            Operation::Min,
            Scope::Axis(1),
            DType::I32,
            String::from("queued-first"),
        );
        let expected = first.expected()?;
        let next = case(
            dtype,
            dense(&expected.output.shape, expected.output.values.clone()),
            Operation::Max,
            Scope::Axis(0),
            DType::I32,
            String::from("queued-next"),
        );
        let final_expected = next.expected()?;
        let p = first.parameters()?.map(|x| x as usize);
        let p2 = next.parameters()?.map(|x| x as usize);
        let input = gpu(backend.upload_words(&output_words(dtype, &first.input.root)))?;
        let before = Snapshot::take(backend)?;
        let output = gpu(backend.with_commands(|| {
            let temporary = backend.produce_words(output_words(dtype, &expected.output).len(), Kernel::FloatExtrema, &[(&input, 0)], &p)?;
            drop(input);
            let output = backend.produce_words(output_words(dtype, &final_expected.output).len(), Kernel::FloatExtrema, &[(&temporary, 0)], &p2)?;
            drop(temporary);
            Ok::<_, Error>(output)
        }))?;
        let mut work = Work::default();
        work.scope();
        work.launch(Kernel::FloatExtrema, true);
        work.launch(Kernel::FloatExtrema, true);
        work.services[abi::FREE as usize] = 2;
        before.check(backend, &work)?;
        check_words(backend, &output, &output_words(dtype, &final_expected.output), false)?;
        counts.queued += 1;
        counts.values += 1;
        // Last public source and pair owners drop while their consumers remain queued.
        let input = pair_input(dtype);
        let source = Native::float_from_data(data(dtype, &input.root), &NativeDevice);
        let expected =
            oracle::reference(dtype, DType::I8, &input.root, Operation::MaxWithIndices, Scope::Axis(1)).map_err(|e| format!("queued pair: {e:?}"))?;
        let before = Snapshot::take(backend)?;
        let (value, index) = gpu(backend.with_commands(|| {
            let (value, index) = Native::float_max_dim_with_indices(source, 1, IntDType::I8);
            Ok::<_, Error>((Native::float_max_dim(value, 0), Native::int_max_dim(index, 0)))
        }))?;
        let mut work = pair_work(dtype, DType::I8, true);
        work.services[abi::FREE as usize] += 3;
        before.check(backend, &work)?;
        check_data(&read_float(value)?, dtype, &expected.output)?;
        check_data(&read_int(index)?, DType::I8, expected.indices.as_ref().ok_or("missing queued indices")?)?;
        counts.queued += 1;
        counts.values += 2;
    }
    Ok(())
}
fn empty(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for dtype in oracle::DTYPES {
        let input = dense(&[0, 3], vec![]);
        let root = gpu(NativeTensor::from_data(data(dtype, &input.root)))?;
        let source = Native::float_from_data(data(dtype, &input.root), &NativeDevice);
        for op in Operation::AXIS {
            let expected = oracle::reference(dtype, DType::I32, &input.root, op, Scope::Axis(1)).map_err(|e| format!("empty reference: {e:?}"))?;
            if !root.can_mut() {
                return Err(String::from("empty source was not uniquely owned"));
            }
            let before = Snapshot::take(backend)?;
            let output = gpu(root.reduce(1, reduction(op)))?;
            let mut work = Work::default();
            if dtype != DType::F32 {
                work.empty();
            }
            before.check(backend, &work)?;
            // Empty layouts are contiguous. Shared ownership identifies the
            // legacy source alias; fresh placeholders each keep one owner.
            let fresh = dtype != DType::F32;
            if output.can_mut() != fresh || root.can_mut() != fresh {
                return Err(String::from("empty single storage ownership changed"));
            }
            check_data(&gpu(output.read())?, if op.is_arg() { DType::I32 } else { dtype }, &expected.output)?;
            drop(output);
            if !root.can_mut() {
                return Err(String::from("empty single retained a source owner"));
            }
            counts.empty += 1;
        }
        for index_dtype in oracle::INDEX_DTYPES {
            if !source.can_mut() {
                return Err(String::from("empty pair source was not uniquely owned"));
            }
            let before = Snapshot::take(backend)?;
            let (value, index) = Native::float_max_dim_with_indices(source.clone(), 1, int_dtype(index_dtype));
            let mut work = Work::default();
            if dtype != DType::F32 {
                work.scope();
                work.empty();
                work.empty();
            }
            before.check(backend, &work)?;
            let fresh = dtype != DType::F32;
            if value.can_mut() != fresh || index.can_mut() != fresh || source.can_mut() != fresh {
                return Err(String::from("empty pair storage ownership changed"));
            }
            let expected = Values::new(&[0, 1], vec![]);
            check_data(&read_float(value)?, dtype, &expected)?;
            check_data(&read_int(index)?, index_dtype, &expected)?;
            if !source.can_mut() {
                return Err(String::from("empty pair retained a source owner"));
            }
            counts.empty += 1;
        }
        // Every error is recoverable and precedes placeholder allocation.
        for shape in [[2, 0], [0, 0]] {
            let bad = gpu(NativeTensor::from_data(data(dtype, &Values::new(&shape, vec![]))))?;
            let before = Snapshot::take(backend)?;
            for op in Operation::AXIS {
                reject(bad.reduce(1, reduction(op)))?;
                counts.rejected += 1;
            }
            before.check(backend, &Work::default())?;
            check_data(&gpu(bad.read())?, dtype, &Values::new(&shape, vec![]))?;
        }
        let before = Snapshot::take(backend)?;
        reject(root.reduce(2, Reduction::Min))?;
        counts.rejected += 1;
        before.check(backend, &Work::default())?;
        check_data(&gpu(root.read())?, dtype, &input.root)?;
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
        scope: oracle::ExecutionScope::NativeErrorExtensions,
        dtypes: 5,
        cases: 134,
        valid_cases: 116,
        diagnostic_cases: 18,
        standalone_cases: 126,
        steps: 132,
        public_cases: 49,
        pair_steps: 79,
        chain_cases: 8,
        chain_steps: 24,
        values: 735,
        checks: oracle::Checks {
            input: 134,
            source: 134,
            alias: 134,
            output: 211,
            chain_index: 16,
            recovery: 18,
        },
    }
}
fn expected_counts(cases: &[Case]) -> Result<Counts> {
    let mut counts = Counts::default();
    for case in cases {
        let n = case.expected()?.output.values.len();
        counts.cases += 1;
        counts.guarded += 2;
        counts.fresh += 2;
        counts.native += 1;
        counts.values += 5 * n;
    }
    counts.pairs = 40;
    counts.queued = 10;
    counts.empty = 60;
    counts.values += 160 + 15;
    for dtype in oracle::DTYPES {
        let c = case(
            dtype,
            dense(&[3, 3], fixture(dtype, 3).root.values[..9].to_vec()),
            Operation::Min,
            Scope::Axis(1),
            DType::I32,
            String::new(),
        );
        counts.rejected += malformed(c.parameters()?).len() * 2 + 7 + 3 + 1 + 2 + 9;
    }
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
    let output_count = (n * case.output_dtype().size()).div_ceil(4);
    let owners = if case.output_dtype().size() == 8 { n } else { output_count };
    let work_items = owners * if length >= 128 { 128 } else { 1 };
    let g = raw::geometry(&p).map_err(|e| format!("geometry: {e:?}"))?;
    let w = Kernel::FloatExtrema.work(&p).map_err(|e| format!("work: {e:?}"))?;
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
        || Kernel::FloatExtrema.entry(&p)
            != if length >= 128 {
                "burn_float_extrema_tree"
            } else {
                "burn_float_extrema_short"
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
    if !matches!((abi::VERSION, abi::KERNEL_COUNT), (52, 84) | (53, 85))
        || Kernel::FloatExtrema as usize != 83
        || oracle::CORPUS_VERSION != 1
        || raw::FORMATS != FORMATS
    {
        return Err(String::from("float extrema protocol/registration mismatch"));
    }
    let shared = expected_shared();
    if shared != oracle::expected_summary(&native_options())
        || shared.options != native_options()
        || shared.scope != oracle::ExecutionScope::NativeErrorExtensions
        || shared.checks.input != shared.cases
        || shared.checks.source != shared.cases
        || shared.checks.alias != shared.cases
        || shared.checks.recovery != shared.diagnostic_cases
    {
        return Err(String::from("shared Native execution scope/checks changed"));
    }
    for case in cases {
        audit_geometry(case)?;
    }
    if expected_counts(cases)? != EXPECTED {
        return Err(format!("raw expected counts changed: {:?}", expected_counts(cases)?));
    }
    Ok(())
}
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    let cases = plan();
    contract(&cases)?;
    let expected = expected_counts(&cases)?;
    let shared_expected = expected_shared();
    crate::report(format_args!(
        "NVIDIA_BURN_FLOAT_EXTREMA_PLAN frozen={FROZEN} reference_sha256={REFERENCE_SHA256} raw={expected:?} shared={}",
        oracle::summary_json(&shared_expected)
    ));
    if !FROZEN {
        return Err(String::from("float extrema UNFROZEN: parent source/API/corpus/protocol approval required"));
    }
    let mut observed = Observed::default();
    for case in &cases {
        let result = raw_case(backend, case, &mut observed.counts);
        observed.record(&format!("raw/{}", case.id), result);
        let result = native_case(backend, case, &mut observed.counts);
        observed.record(&format!("native/{}", case.id), result);
    }
    for (name, suite) in [
        ("rejections", rejections as fn(&Nvidia, &mut Counts) -> Result<()>),
        ("pairs", pairs),
        ("queued", queued),
        ("empty", empty),
    ] {
        let result = suite(backend, &mut observed.counts);
        observed.record(name, result);
    }
    let device = d3os_ai::native_device().map_err(|e| format!("float extrema device: {e}"))?;
    let options = native_options();
    let report = try_read_sync(oracle::run(&device, &options)).ok_or("shared float extrema run pending")?;
    for failure in report.failures.iter().take(12) {
        observed.record(&format!("shared/{}", failure.case_id), Err(failure.message.clone()));
    }
    let footer = oracle::success_footer(&options, &report);
    let accepted = observed.failed == 0 && observed.counts == expected && report.expected == shared_expected && footer.is_ok();
    crate::report(format_args!(
        "NVIDIA_BURN_FLOAT_EXTREMA_OBSERVED accepted={accepted} counts={:?} failed={} shared_attempted={} shared_passed={} shared_errors={} shared_failed={} examples={:?}",
        observed.counts, observed.failed, report.attempted, report.passed, report.observed_errors, report.failed, observed.examples
    ));
    if !accepted {
        return Err(format!(
            "float extrema failed: expected={expected:?} observed={:?} failed={} shared_failed={}",
            observed.counts, observed.failed, report.failed
        ));
    }
    crate::report(format_args!("{} os=d3os backend=native fusion=disabled", footer?));
    crate::report(format_args!(
        "NVIDIA_BURN_FLOAT_EXTREMA_STRICT_OK reference_sha256={REFERENCE_SHA256} protocol={} kernels={} expected={expected:?} observed={:?}",
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
    fn bounded_physical_plan_counts_masks_and_poison() {
        let cases = plan();
        let counts = expected_counts(&cases).unwrap();
        println!("raw plan: {counts:?}");
        println!("shared Native plan: {}", oracle::summary_json(&expected_shared()));
        for case in &cases {
            audit_geometry(case).unwrap();
            let expected = output_words(case.output_dtype(), &case.expected().unwrap().output);
            let a = poison(&expected, false);
            let b = poison(&expected, true);
            for ((a, b), expected) in a.into_iter().zip(b).zip(expected) {
                for ((a, b), expected) in a.to_le_bytes().into_iter().zip(b.to_le_bytes()).zip(expected.to_le_bytes()) {
                    assert!(a != b && a != expected && b != expected);
                }
            }
            let p = case.parameters().unwrap();
            let g = raw::geometry(&p).unwrap();
            let slots = if case.output_dtype().size() == 8 { 1 } else { 4 / case.output_dtype().size() };
            let count = g.output_count as usize;
            let mut seen = vec![0; count];
            for owner in 0..g.owners as usize {
                for slot in 0..slots {
                    let output = owner * slots + slot;
                    // Padded lanes sanitize the coordinate before any source address.
                    let safe = if output < count { output } else { 0 };
                    assert!(safe < count);
                    if output < count {
                        seen[output] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&writers| writers == 1));
        }
        for dtype in oracle::DTYPES {
            for index in oracle::INDEX_DTYPES {
                for op in [Operation::ArgMin, Operation::ArgMax] {
                    assert!(cases.iter().any(|c| c.dtype == dtype && c.index_dtype == index && c.op == op));
                }
            }
        }
        for rank in 0..=8 {
            assert!(cases.iter().any(|c| c.input.logical().shape.len() == rank));
        }
        // Bind the independently enumerated totals to the accepted pure plan.
        contract(&cases).unwrap();
    }
    #[test]
    fn exact_winners_include_signaling_payloads_ties_and_original_coordinates() {
        for dtype in oracle::DTYPES {
            let p = oracle::pool(dtype);
            for op in Operation::PAIRS {
                let input = fixture(dtype, 131);
                let result = oracle::reference(dtype, DType::I8, &input.root, op, Scope::Axis(1)).unwrap();
                assert_eq!(result.output.values[4], p[14]);
                assert_eq!(result.logical_indices[4], 1);
                let mut reversed = input;
                reversed.views.push(View::Flip(vec![1]));
                let result = oracle::reference(dtype, DType::U64, &reversed.logical(), op, Scope::Axis(1)).unwrap();
                assert_eq!(result.output.values[4], p[15]);
                assert_eq!(result.logical_indices[4], 1);
                assert_eq!(result.output.values[1], p[1]); // first logical zero keeps its sign
                assert_eq!(result.logical_indices[1], 0);
            }
            let input = pair_input(dtype);
            let expected = oracle::reference(dtype, DType::I8, &input.root, Operation::MaxWithIndices, Scope::Axis(1)).unwrap();
            assert_eq!(expected.output.values, [p[7]]);
            assert_eq!(expected.logical_indices, [200]);
            assert_eq!(oracle::decode_index(DType::I8, expected.indices.unwrap().values[0]), -56);
            for index_dtype in [DType::I64, DType::U64] {
                assert_eq!(output_words(index_dtype, &Values::new(&[1], vec![200])), [200, 0]);
            }
        }
    }
    #[test]
    fn malformed_metadata_empty_storage_aliases_and_allocator_errors_are_strict() {
        for dtype in oracle::DTYPES {
            let c = case(
                dtype,
                dense(&[3, 3], fixture(dtype, 3).root.values[..9].to_vec()),
                Operation::Min,
                Scope::Axis(1),
                DType::I32,
                String::new(),
            );
            for bad in malformed(c.parameters().unwrap()) {
                assert!(Kernel::FloatExtrema.work(&bad).is_err(), "{dtype:?} {bad:?}");
            }
            let empty = Case {
                input: dense(&[0, 3], vec![]),
                ..c
            };
            let p = empty.parameters().unwrap();
            let g = raw::geometry(&p).unwrap();
            assert_eq!((g.input_words, g.output_words, g.owners), (1, 1, 0));
            assert!(raw::validate(&p).is_err());
            assert!(raw::work_items(&p).is_err());
            assert!(raw::symbol(&p).is_err());
            for shape in [[0, 0], [2, 0]] {
                let mut zero = p;
                zero[8] = shape[0];
                zero[9] = shape[1];
                zero[0] = shape[0];
                zero[6] = 0;
                assert!(raw::geometry(&zero).is_err());
            }
        }
        // This returns locally before any host request or device access.
        let result = Nvidia.produce_words(0, Kernel::FloatExtrema, &[], &[]);
        assert!(matches!(result, Err(Error::InvalidShape)));
        reject_output_size(result, 0).unwrap();
        assert!(reject_output_size::<()>(Err(Error::InvalidParameter), 0).is_err());
        assert!(reject_output_size::<()>(Err(Error::InvalidShape), 1).is_err());
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
        work.launch(Kernel::FloatExtrema, true);
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
