//! Guarded typed Exp/Log qualification against the frozen independent corpus.
//! Accuracy limits are project targets. Runtime observations do not imply a
//! vendor guarantee, Linux parity, or instruction-level staging evidence.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{
            Backend as _, DType, Dispatch, Scalar, Shape, Slice, TensorData, TensorMetadata,
            ops::{FloatTensorOps, IntTensorOps},
            tensor::FloatTensor,
        },
        tensor::{Tensor, try_read_sync},
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::typed_exp_log::{self as oracle, Accuracy, Operation, Reference},
};
use d3os_compute::{
    Backend as _, Error,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{self as abi, Kernel, cast::Format, elementwise::Elementwise, float_arithmetic as float};

type Result<T> = core::result::Result<T, String>;
const GUARD: u32 = 0x6bad_beef;
const POISON: u32 = 0xa53c_d791;
const CHUNK: usize = 4096;
const COMPARISONS: usize = 861244;
const POLICY: &str = "f93ec90c97502770fc68be7c51192c4cf708d2bbe99d10cbcf3723010ed1bf83";
const FIXTURE: &str = "cda7620e1072aab7aa388f14d2e73142860fdaa21386cb47df8c9518b551437d";
const FORMATS: [(DType, Format); 5] = [
    (DType::F64, Format::F64),
    (DType::F32, Format::F32),
    (DType::Flex32, Format::Flex32),
    (DType::F16, Format::F16),
    (DType::BF16, Format::BF16),
];
const LENGTHS: [usize; 9] = [1, 2, 3, 127, 128, 129, 255, 256, 257];
#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    corpus_rows: usize,
    corpus_batches: usize,
    geometry: usize,
    raw: usize,
    fresh: usize,
    native: usize,
    public: usize,
    rejected: usize,
    empty: usize,
    queued: usize,
    f32_routes: usize,
    typed_routes: usize,
}
// Fixed plan, independent of observed accuracy maxima and classifications.
const EXPECTED: Counts = Counts {
    corpus_rows: 306572,
    corpus_batches: 78,
    geometry: 330,
    raw: 408,
    fresh: 418,
    native: 670,
    public: 1000,
    rejected: 595,
    empty: 200,
    queued: 30,
    f32_routes: 334,
    typed_routes: 1336,
};
#[derive(Default)]
struct Observed {
    accuracy: Accuracy,
    examples: Vec<String>,
}
impl Observed {
    fn record(&mut self, dtype: DType, op: Operation, rows: &[Reference], actual: &[u64], path: &str) -> Result<()> {
        if rows.len() != actual.len() {
            return Err(String::from("typed exp/log logical length mismatch"));
        }
        for (row, &bits) in rows.iter().zip(actual) {
            let comparison = self.accuracy.record(dtype, op, row, bits);
            if !comparison.passed && self.examples.len() < 12 {
                self.examples.push(format!(
                    "path={path} dtype={} op={} input={:#018x} actual={bits:#018x} staged={:#018x} ideal={:#018x} comparison={comparison:?}",
                    oracle::dtype_label(dtype),
                    op.label(),
                    row.input,
                    row.staged,
                    row.ideal,
                ));
            }
        }
        Ok(())
    }
}
fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("typed exp/log: {e}"))
}
fn op_code(op: Operation) -> Elementwise {
    match op {
        Operation::Exp => Elementwise::Exp,
        Operation::Log => Elementwise::Log,
    }
}
fn native_kernel(dtype: DType) -> Kernel {
    if dtype == DType::F32 {
        Kernel::TensorElementwise
    } else {
        Kernel::FloatArithmetic
    }
}
fn routed(counts: &mut Counts, dtype: DType) {
    if dtype == DType::F32 {
        counts.f32_routes += 1;
    } else {
        counts.typed_routes += 1;
    }
}
fn data(dtype: DType, shape: &[usize], bits: &[u64]) -> TensorData {
    TensorData::from_bytes_vec(oracle::pack(dtype, bits), Shape::from(shape), dtype)
}
fn packed(dtype: DType, bits: &[u64]) -> Vec<u32> {
    oracle::pack(dtype, bits)
        .chunks(4)
        .map(|chunk| {
            let mut bytes = [0; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect()
}
fn metadata(actual: &TensorData, dtype: DType, shape: &[usize], length: usize) -> Result<()> {
    if actual.dtype() != dtype || actual.shape().as_slice() != shape || actual.as_bytes().len() != length * dtype.size() {
        return Err(format!("typed exp/log metadata {dtype:?} {shape:?}: {:?} {:?}", actual.dtype(), actual.shape()));
    }
    Ok(())
}
fn source_data(actual: &TensorData, dtype: DType, shape: &[usize], expected: &[u64]) -> Result<()> {
    metadata(actual, dtype, shape, expected.len())?;
    if oracle::unpack(dtype, actual.as_bytes()) != expected {
        return Err(format!("typed exp/log changed source bytes or NaN payload {dtype:?} {shape:?}"));
    }
    Ok(())
}
fn output_data(actual: &TensorData, input: &Input, op: Operation, observed: &mut Observed, path: &str) -> Result<()> {
    metadata(actual, input.dtype, input.layout.shape().as_slice(), input.logical.len())?;
    observed.record(input.dtype, op, &input.references(op)?, &oracle::unpack(input.dtype, actual.as_bytes()), path)
}
fn primitive_data(value: FloatTensor<Native>) -> Result<TensorData> {
    try_read_sync(Native::float_into_data(value))
        .ok_or("typed exp/log Native read pending")?
        .map_err(|e| format!("typed exp/log Native read: {e:?}"))
}
fn unchanged(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if actual != expected {
        return Err(String::from("typed exp/log source/guard/unused RHS changed"));
    }
    Ok(())
}
fn output_words(backend: &Nvidia, buffer: &GpuBuffer<u32>, input: &Input, op: Operation, guarded: bool, observed: &mut Observed, path: &str) -> Result<()> {
    let count = (input.logical.len() * input.dtype.size()).div_ceil(4);
    let offset = usize::from(guarded);
    let mut actual = vec![0; count + 2 * offset];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if guarded && (actual[0] != GUARD || actual[count + 1] != GUARD) {
        return Err(String::from("typed exp/log output word guard changed"));
    }
    let words = &actual[offset..offset + count];
    if input.dtype.size() == 2 && input.logical.len() % 2 == 1 && words[count - 1] >> 16 != 0 {
        return Err(String::from("typed exp/log odd output upper half is not zero"));
    }
    let bytes: Vec<_> = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take(input.logical.len() * input.dtype.size())
        .collect();
    observed.record(input.dtype, op, &input.references(op)?, &oracle::unpack(input.dtype, &bytes), path)
}
/// Inspect every kernel counter per individual operation, including hidden copies.
fn resident(
    backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT], kernel: Kernel, fresh: bool,
) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE] {
        if after[service as usize].calls != before[service as usize].calls {
            return Err(format!("typed exp/log resident service={service}"));
        }
    }
    for (service, want) in [(abi::LAUNCH_ALLOCATE, u64::from(fresh)), (abi::LAUNCH, u64::from(!fresh))] {
        if after[service as usize].calls - before[service as usize].calls != want {
            return Err(format!("typed exp/log full writer service={service} expected={want}"));
        }
    }
    for (index, (old, new)) in profile.iter().zip(gpu(backend.profile())?).enumerate() {
        if new.calls - old.calls != u64::from(index == kernel as usize) {
            return Err(format!("typed exp/log route kernel={kernel:?} counter={index}"));
        }
    }
    Ok(())
}
fn no_work(backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT]) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
        if after[service as usize].calls != before[service as usize].calls {
            return Err(format!("typed exp/log invalid/empty service={service}"));
        }
    }
    if profile.iter().zip(gpu(backend.profile())?).any(|(a, b)| a.calls != b.calls) {
        return Err(String::from("typed exp/log invalid/empty kernel launched"));
    }
    Ok(())
}
#[derive(Clone, Copy)]
enum View {
    Dense,
    Strided { length: usize, reverse: bool },
    Offset,
    Transposed,
    Sliced,
    Broadcast,
}
struct Input {
    dtype: DType,
    format: Format,
    layout: Layout,
    storage: Vec<u64>,
    logical: Vec<u64>,
    view: View,
}
impl Input {
    fn flat(dtype: DType, format: Format, values: Vec<u64>) -> Result<Self> {
        Ok(Self {
            dtype,
            format,
            layout: gpu(Layout::contiguous(Shape::new([values.len()])))?,
            storage: values.clone(),
            logical: values,
            view: View::Dense,
        })
    }
    fn data(&self) -> TensorData {
        data(self.dtype, &[self.storage.len()], &self.storage)
    }
    fn tensor(&self) -> Result<NativeTensor> {
        gpu(NativeTensor::from_data(self.data()).and_then(|t| t.with_layout(self.layout.clone())))
    }
    fn references(&self, op: Operation) -> Result<Vec<Reference>> {
        self.logical
            .iter()
            .map(|&bits| {
                oracle::expected_bits(self.dtype, op, bits).ok_or_else(|| format!("typed exp/log missing reference {:?} {} {bits:#x}", self.dtype, op.label()))
            })
            .collect()
    }
    /// Apply actual public backend view operations before the measured launch.
    fn primitive(&self) -> (FloatTensor<Native>, FloatTensor<Native>) {
        let base = Native::float_from_data(self.data(), &NativeDevice);
        let source = match self.view {
            View::Dense => Native::float_reshape(base.clone(), self.layout.shape().clone()),
            View::Strided { length, reverse } => {
                let view = Native::float_slice(base.clone(), &[Slice::new(1, Some((2 * length) as isize), 2)]);
                if reverse { Native::float_flip(view, &[0]) } else { view }
            }
            View::Offset => Native::float_slice(base.clone(), &[Slice::new(1, Some(6), 1)]),
            View::Transposed => {
                let view = Native::float_reshape(base.clone(), Shape::new([3, 7]));
                Native::float_flip(Native::float_swap_dims(view, 0, 1), &[0, 1])
            }
            View::Sliced => {
                let view = Native::float_reshape(base.clone(), Shape::new([5, 9]));
                Native::float_slice(view, &[Slice::new(1, Some(5), 2), Slice::new(1, Some(7), 2)])
            }
            View::Broadcast => Native::float_expand(Native::float_slice(base.clone(), &[Slice::new(1, Some(2), 1)]), Shape::new([2, 3])),
        };
        (base, source)
    }
}
fn parameters(input: &Input, op: Operation) -> Result<[usize; 32]> {
    let mut p = gpu(input.layout.elementwise_parameters(&input.layout, Elementwise::Add, 0, false))?;
    p[2] = op.opcode() as usize | ((input.format as usize) << 8);
    p[3] = 0;
    p[5] = 0;
    p[7] = 0;
    p[24..32].fill(0);
    Ok(p)
}
fn source_words(input: &Input) -> Vec<u32> {
    let mut words = vec![GUARD];
    words.extend(packed(input.dtype, &input.storage));
    if input.dtype.size() == 2 && input.storage.len() % 2 == 1 {
        *words.last_mut().unwrap() |= 0xffff_0000;
    }
    words.push(GUARD);
    words
}
fn raw(backend: &Nvidia, input: &Input, op: Operation, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    let words = source_words(input);
    let source = gpu(backend.upload_words(&words))?;
    // Only one poison word exists, even for F64. Unary must never load RHS.
    let unused_words = [GUARD, POISON, GUARD];
    let rhs = gpu(backend.upload_words(&unused_words))?;
    let count = (input.logical.len() * input.dtype.size()).div_ceil(4);
    let mut poison = vec![POISON; count + 2];
    poison[0] = GUARD;
    poison[count + 1] = GUARD;
    let output = gpu(backend.upload_words(&poison))?;
    let p = parameters(input, op)?;
    let services = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatArithmetic, &[(&source, 1), (&rhs, 1), (&output, 1)], &p)))?;
    resident(backend, &services, &profile, Kernel::FloatArithmetic, false)?;
    output_words(backend, &output, input, op, true, observed, "raw")?;
    counts.raw += 1;
    // Return a dirty allocation before the fresh full writer obtains storage.
    gpu(backend.write_words(&output, 1, &vec![POISON; count]))?;
    drop(output);
    let services = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    let output = gpu(backend.with_commands(|| backend.produce_words(count, Kernel::FloatArithmetic, &[(&source, 1), (&rhs, 1)], &p)))?;
    resident(backend, &services, &profile, Kernel::FloatArithmetic, true)?;
    output_words(backend, &output, input, op, false, observed, "fresh")?;
    unchanged(backend, &source, &words)?;
    unchanged(backend, &rhs, &unused_words)?;
    counts.fresh += 1;
    Ok(())
}
fn native(backend: &Nvidia, input: &Input, op: Operation, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    let base = gpu(NativeTensor::from_data(input.data()))?;
    let source = gpu(base.with_layout(input.layout.clone()))?;
    for legacy in [false, true] {
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| if legacy { source.unary(op_code(op)) } else { source.float_unary(op_code(op)) }))?;
        resident(backend, &services, &profile, native_kernel(input.dtype), true)?;
        output_data(
            &gpu(output.read())?,
            input,
            op,
            observed,
            if legacy { "Native.unary" } else { "Native.float_unary" },
        )?;
        counts.native += 1;
        routed(counts, input.dtype);
    }
    source_data(&gpu(source.read())?, input.dtype, input.layout.shape().as_slice(), &input.logical)?;
    source_data(&gpu(base.read())?, input.dtype, &[input.storage.len()], &input.storage)
}
fn native_unary(source: FloatTensor<Native>, op: Operation) -> FloatTensor<Native> {
    match op {
        Operation::Exp => Native::float_exp(source),
        Operation::Log => Native::float_log(source),
    }
}
fn public(backend: &Nvidia, input: &Input, op: Operation, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for path in 0..3 {
        let (base, source) = input.primitive();
        let alias = source.clone();
        // Check the public view recipe independently before measuring the op.
        source_data(&primitive_data(alias.clone())?, input.dtype, input.layout.shape().as_slice(), &input.logical)?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        // Separate the dispatch from its readback for exact residency counters.
        let actual = if path == 0 {
            let output = native_unary(source, op);
            resident(backend, &services, &profile, native_kernel(input.dtype), true)?;
            primitive_data(output)?
        } else {
            let output = public_launch(source, op, path == 1, input.layout.shape().num_dims())?;
            resident(backend, &services, &profile, native_kernel(input.dtype), true)?;
            try_read_sync(Dispatch::float_into_data(output))
                .ok_or("typed exp/log public read pending")?
                .map_err(|e| format!("typed exp/log public read: {e:?}"))?
        };
        output_data(&actual, input, op, observed, ["Native.wrapper", "Dispatch", "Tensor"][path])?;
        source_data(&primitive_data(alias)?, input.dtype, input.layout.shape().as_slice(), &input.logical)?;
        source_data(&primitive_data(base)?, input.dtype, &[input.storage.len()], &input.storage)?;
        counts.public += 1;
        routed(counts, input.dtype);
    }
    Ok(())
}
fn public_launch(source: FloatTensor<Native>, op: Operation, dispatch: bool, rank: usize) -> Result<FloatTensor<Dispatch>> {
    fn launch<const D: usize>(source: FloatTensor<Native>, op: Operation, dispatch: bool) -> FloatTensor<Dispatch> {
        let source = Tensor::<D>::from_primitive::<Native>(source);
        if dispatch {
            match op {
                Operation::Exp => Dispatch::float_exp(source.into_dispatch()),
                Operation::Log => Dispatch::float_log(source.into_dispatch()),
            }
        } else {
            match op {
                Operation::Exp => source.exp(),
                Operation::Log => source.log(),
            }
            .into_dispatch()
        }
    }
    Ok(match rank {
        0 => launch::<0>(source, op, dispatch),
        1 => launch::<1>(source, op, dispatch),
        2 => launch::<2>(source, op, dispatch),
        3 => launch::<3>(source, op, dispatch),
        8 => launch::<8>(source, op, dispatch),
        _ => return Err(String::from("typed exp/log public rank missing")),
    })
}
/// Expected coordinates use explicit host indices, never Layout::element_offset.
fn geometry_inputs(dtype: DType, format: Format, pool: &[u64]) -> Result<Vec<Input>> {
    let values = |n: usize| (0..n).map(|i| pool[i % pool.len()]).collect::<Vec<_>>();
    let mut cases = Vec::new();
    for length in LENGTHS {
        cases.push(Input::flat(dtype, format, values(length))?);
        let storage = values(2 * length + 11);
        for reverse in [false, true] {
            let logical = (0..length).map(|i| storage[if reverse { 2 * length - 1 - 2 * i } else { 1 + 2 * i }]).collect();
            cases.push(Input {
                dtype,
                format,
                storage: storage.clone(),
                logical,
                layout: gpu(Layout::from_parts(
                    Shape::new([length]),
                    vec![if reverse { -2 } else { 2 }],
                    if reverse { (2 * length - 1) as i64 } else { 1 },
                    storage.len(),
                ))?,
                view: View::Strided { length, reverse },
            });
        }
    }
    let storage = values(11);
    cases.push(Input {
        dtype,
        format,
        logical: storage[1..6].to_vec(),
        storage,
        layout: gpu(Layout::from_parts(Shape::new([5]), vec![1], 1, 11))?,
        view: View::Offset,
    });
    let storage = values(21);
    let indices: Vec<_> = (0..7).rev().flat_map(|column| (0..3).rev().map(move |row| row * 7 + column)).collect();
    cases.push(Input {
        dtype,
        format,
        logical: indices.iter().map(|&i| storage[i]).collect(),
        storage,
        layout: gpu(Layout::contiguous(Shape::new([3, 7]))
            .and_then(|l| l.permute(&[1, 0]))
            .and_then(|l| l.flip(&[0, 1])))?,
        view: View::Transposed,
    });
    let storage = values(45);
    cases.push(Input {
        dtype,
        format,
        logical: [10, 12, 14, 28, 30, 32].iter().map(|&i| storage[i]).collect(),
        storage,
        layout: gpu(Layout::from_parts(Shape::new([2, 3]), vec![18, 2], 10, 45))?,
        view: View::Sliced,
    });
    for shape in [vec![], vec![1, 2, 1, 2, 1, 2, 1, 3]] {
        let mut input = Input::flat(dtype, format, values(shape.iter().product()))?;
        input.layout = gpu(Layout::contiguous(Shape::from(shape)))?;
        cases.push(input);
    }
    let storage = values(3);
    cases.push(Input {
        dtype,
        format,
        logical: vec![storage[1]; 6],
        storage,
        layout: gpu(Layout::from_parts(Shape::new([2, 3]), vec![0, 0], 1, 3))?,
        view: View::Broadcast,
    });
    Ok(cases)
}
fn corpus(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for (dtype, format) in FORMATS {
        for op in oracle::OPERATIONS {
            let rows = oracle::rows(dtype, op);
            for chunk in rows.chunks(CHUNK) {
                let input = Input::flat(dtype, format, chunk.iter().map(|row| row.input).collect())?;
                raw(backend, &input, op, counts, observed)?;
                counts.corpus_rows += chunk.len();
                counts.corpus_batches += 1;
            }
        }
    }
    Ok(())
}
fn layouts(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for (dtype, format) in FORMATS {
        for op in oracle::OPERATIONS {
            // Spread representative specials, payloads and finite values across views.
            let rows = oracle::rows(dtype, op);
            let mut pool: Vec<_> = rows.iter().step_by((rows.len() / 31).max(1)).map(|row| row.input).collect();
            pool.extend(rows.iter().take(3).map(|row| row.input));
            pool.extend(rows.iter().rev().take(3).map(|row| row.input));
            for input in geometry_inputs(dtype, format, &pool)? {
                raw(backend, &input, op, counts, observed)?;
                native(backend, &input, op, counts, observed)?;
                public(backend, &input, op, counts, observed)?;
                counts.geometry += 1;
            }
        }
    }
    Ok(())
}
fn malformed(p: [usize; 32]) -> Vec<[usize; 32]> {
    let mut cases = Vec::new();
    for (field, value) in [
        (0, 0),
        (0, i32::MAX as usize + 1),
        (1, 0),
        (1, 9),
        (3, 1),
        (4, 5),
        (5, 1),
        (6, 0),
        (7, 1),
        (8, 6),
        (9, 2),
        (16, i32::MAX as usize),
        (16, u32::MAX as usize),
        (17, 1),
        (24, 1),
        (31, 1),
    ] {
        let mut bad = p;
        bad[field] = value;
        cases.push(bad);
    }
    for bit in 16..32 {
        let mut bad = p;
        bad[2] |= 1 << bit; // Includes SCALAR_RHS, illegal for both unary ops.
        cases.push(bad);
    }
    for operation in [0, 5, 255] {
        let mut bad = p;
        bad[2] = (bad[2] & !255) | operation;
        cases.push(bad);
    }
    for format in [Format::I32 as usize, Format::Bool8 as usize, 255] {
        let mut bad = p;
        bad[2] = (bad[2] & !0xff00) | (format << 8);
        cases.push(bad);
    }
    cases
}
fn raw_rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        for op in oracle::OPERATIONS {
            let bits = oracle::rows(dtype, op)[0].input;
            let input = Input::flat(dtype, format, vec![bits; 5])?;
            let p = parameters(&input, op)?;
            let words = vec![POISON; packed(dtype, &input.storage).len() + 4];
            let a = gpu(backend.upload_words(&words))?;
            let b = gpu(backend.upload_words(&words))?;
            let output = gpu(backend.upload_words(&words))?;
            let short = gpu(backend.upload_words(&[POISON]))?;
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let queue = gpu(backend.queue_profile())?;
            for bad in malformed(p) {
                if backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &bad).is_ok() {
                    return Err(format!("typed exp/log malformed accepted {dtype:?} {} {bad:?}", op.label()));
                }
                counts.rejected += 1;
            }
            for bad in [p[..31].to_vec(), {
                let mut v = p.to_vec();
                v.push(0);
                v
            }] {
                if backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &bad).is_ok() {
                    return Err(String::from("typed exp/log parameter length accepted"));
                }
                counts.rejected += 1;
            }
            for buffers in [vec![(&a, 1), (&output, 1)], vec![(&a, 1), (&b, 1), (&output, 1), (&short, 0)]] {
                if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p).is_ok() {
                    return Err(String::from("typed exp/log pointer arity accepted"));
                }
                counts.rejected += 1;
            }
            // Reject aliases with the input and with the unused unary RHS.
            for (target, offset) in [(&a, 1), (&a, 2), (&b, 1), (&b, 0)] {
                if backend
                    .dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (target, offset)], &p)
                    .is_ok()
                {
                    return Err(String::from("typed exp/log output alias accepted"));
                }
                counts.rejected += 1;
            }
            for buffers in [
                [(&short, 0), (&b, 1), (&output, 1)],
                [(&a, 1), (&b, words.len()), (&output, 1)],
                [(&a, 1), (&b, 1), (&short, 0)],
                [(&a, words.len()), (&b, 1), (&output, 1)],
                [(&a, 1), (&b, 1), (&output, words.len())],
            ] {
                if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p).is_ok() {
                    return Err(String::from("typed exp/log physical word bounds accepted"));
                }
                counts.rejected += 1;
            }
            if dtype == DType::F64 {
                let one = parameters(&Input::flat(dtype, format, vec![bits])?, op)?;
                for buffers in [[(&short, 0), (&b, 1), (&output, 1)], [(&a, 1), (&b, 1), (&short, 0)]] {
                    if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &one).is_ok() {
                        return Err(String::from("typed exp/log missing F64 second word accepted"));
                    }
                    counts.rejected += 1;
                }
                let mut high = p;
                high[4] = i32::MAX as usize - 5;
                high[6] = i32::MAX as usize;
                if backend
                    .dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &high)
                    .is_ok()
                {
                    return Err(String::from("typed exp/log F64 physical high-word bounds accepted"));
                }
                counts.rejected += 1;
            }
            // Only four alias attempts reach the driver. None records GPU work.
            let after = gpu(backend.service_profile())?;
            for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
                let want = if service == abi::LAUNCH { 4 } else { 0 };
                if after[service as usize].calls - services[service as usize].calls != want {
                    return Err(format!("typed exp/log rejected service={service} expected={want}"));
                }
            }
            if profile.iter().zip(gpu(backend.profile())?).any(|(a, b)| a.calls != b.calls) || gpu(backend.queue_profile())?.launches != queue.launches {
                return Err(String::from("typed exp/log rejected request launched GPU work"));
            }
            unchanged(backend, &a, &words)?;
            unchanged(backend, &b, &words)?;
            unchanged(backend, &output, &words)?;
            unchanged(backend, &short, &[POISON])?;
        }
    }
    Ok(())
}
fn native_validation(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        for length in [0, 3] {
            let source = Input::flat(dtype, format, vec![0; length])?.tensor()?;
            for op in oracle::OPERATIONS {
                let services = gpu(backend.service_profile())?;
                let profile = gpu(backend.profile())?;
                for result in [
                    source.float_binary(&source, op_code(op)),
                    source.binary(&source, op_code(op)),
                    source.float_scalar(op_code(op), Scalar::Bool(true)),
                ] {
                    if !matches!(result, Err(Error::InvalidParameter)) {
                        return Err(String::from("typed exp/log unary accepted binary/scalar arity"));
                    }
                    counts.rejected += 1;
                }
                no_work(backend, &services, &profile)?;
            }
        }
    }
    for length in [0, 3] {
        let source = gpu(NativeTensor::from_data(data(DType::I32, &[length], &vec![0; length])))?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        for op in oracle::OPERATIONS {
            for result in [source.float_unary(op_code(op)), source.unary(op_code(op))] {
                if !matches!(result, Err(Error::InvalidParameter)) {
                    return Err(String::from("typed exp/log integer Native storage accepted"));
                }
                counts.rejected += 1;
            }
        }
        no_work(backend, &services, &profile)?;
    }
    for (shape, strides, offset, extent) in [(vec![1; 9], vec![1; 9], 0, 1), (vec![3], vec![-1], 1, 3), (vec![3], vec![1], 1, 3)] {
        if Layout::from_parts(Shape::from(shape), strides, offset, extent).is_ok() {
            return Err(String::from("typed exp/log invalid Native rank/bounds accepted"));
        }
        counts.rejected += 1;
    }
    Ok(())
}
fn empties(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, _) in FORMATS {
        for shape in [vec![0], vec![2, 0, 7], vec![0, 2, 3], vec![1, 1, 1, 1, 0, 1, 1, 3]] {
            for op in oracle::OPERATIONS {
                let empty = gpu(NativeTensor::empty(Shape::from(shape.clone()), dtype))?;
                let source = Native::float_from_data(data(dtype, &shape, &[]), &NativeDevice);
                let services = gpu(backend.service_profile())?;
                let profile = gpu(backend.profile())?;
                for output in [gpu(empty.float_unary(op_code(op)))?, gpu(empty.unary(op_code(op)))?] {
                    source_data(&gpu(output.read())?, dtype, &shape, &[])?;
                    counts.empty += 1;
                }
                source_data(&primitive_data(native_unary(source.clone(), op))?, dtype, &shape, &[])?;
                counts.empty += 1;
                for dispatch in [false, true] {
                    let output = public_launch(source.clone(), op, dispatch, shape.len())?;
                    let actual = try_read_sync(Dispatch::float_into_data(output))
                        .ok_or("typed exp/log empty read pending")?
                        .map_err(|e| format!("typed exp/log empty public: {e:?}"))?;
                    source_data(&actual, dtype, &shape, &[])?;
                    counts.empty += 1;
                }
                no_work(backend, &services, &profile)?;
            }
        }
    }
    Ok(())
}
/// Last owners disappear while the command scope still blocks completion.
/// No read/profile call occurs in the scope: those calls flush pending work.
fn lifetimes(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for (dtype, format) in FORMATS {
        for op in oracle::OPERATIONS {
            let values: Vec<_> = oracle::rows(dtype, op).iter().step_by(23).take(17).map(|row| row.input).collect();
            let input = Input::flat(dtype, format, values)?;
            let source = input.tensor()?;
            let alias = source.clone();
            if source.can_mut() || alias.can_mut() {
                return Err(String::from("typed exp/log retained Native alias reports unique ownership"));
            }
            drop(alias);
            if !source.can_mut() {
                return Err(String::from("typed exp/log last Native owner reports shared ownership"));
            }
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let output = gpu(backend.with_commands(|| {
                let output = source.float_unary(op_code(op))?;
                drop(source);
                Ok::<_, Error>(output)
            }))?;
            resident(backend, &services, &profile, native_kernel(dtype), true)?;
            output_data(&gpu(output.read())?, &input, op, observed, "queued.Native")?;
            counts.native += 1;
            counts.queued += 1;
            routed(counts, dtype);

            let source = gpu(backend.upload_words(&packed(dtype, &input.storage)))?;
            let rhs = gpu(backend.upload_words(&[POISON]))?;
            let p = parameters(&input, op)?;
            let count = (input.logical.len() * dtype.size()).div_ceil(4);
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let output = gpu(backend.with_commands(|| {
                let output = backend.produce_words(count, Kernel::FloatArithmetic, &[(&source, 0), (&rhs, 0)], &p)?;
                drop(source);
                drop(rhs);
                Ok::<_, Error>(output)
            }))?;
            resident(backend, &services, &profile, Kernel::FloatArithmetic, true)?;
            output_words(backend, &output, &input, op, false, observed, "queued.raw")?;
            counts.fresh += 1;
            counts.queued += 1;

            let (base, source) = input.primitive();
            drop(base);
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let output = gpu(backend.with_commands(|| Ok::<_, Error>(native_unary(source, op))))?;
            resident(backend, &services, &profile, native_kernel(dtype), true)?;
            output_data(&primitive_data(output)?, &input, op, observed, "queued.public")?;
            counts.public += 1;
            counts.queued += 1;
            routed(counts, dtype);
        }
    }
    Ok(())
}
fn acknowledge(failed: FloatTensor<Native>, length: usize) -> Result<()> {
    if failed.dtype() != DType::I32 || failed.shape().as_slice() != [length] {
        return Err(String::from("typed exp/log failed integer primitive metadata changed"));
    }
    if !Native::sync(&NativeDevice).is_err_and(|e| !e.is_device_poisoned()) {
        return Err(String::from("typed exp/log missing recoverable Native sync error"));
    }
    let error = try_read_sync(Native::float_into_data(failed)).ok_or("typed exp/log failed Native read pending")?;
    if !error.is_err_and(|e| !e.is_device_poisoned()) || Native::sync(&NativeDevice).is_err() {
        return Err(String::from("typed exp/log error acknowledgment poisoned Native device"));
    }
    Ok(())
}
fn public_errors(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("typed exp/log error device: {e}"))?;
    for length in [0, 3] {
        for op in oracle::OPERATIONS {
            let source = Native::int_from_data(data(DType::I32, &[length], &vec![0; length]), &NativeDevice);
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let failed = native_unary(source, op);
            no_work(backend, &services, &profile)?;
            acknowledge(failed, length)?;
            no_work(backend, &services, &profile)?;
            counts.rejected += 1;

            // A failed reshape retains floating dtype and its rank-nine
            // metadata. Dispatch propagates that recoverable root failure.
            let source = Native::float_from_data(data(DType::F32, &[length], &vec![0; length]), &NativeDevice);
            let mut shape = vec![1; 9];
            shape[8] = length;
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let source = Native::float_reshape(source, Shape::from(shape.clone()));
            let source = Tensor::<9>::from_primitive::<Native>(source).into_dispatch();
            let failed = match op {
                Operation::Exp => Dispatch::float_exp(source),
                Operation::Log => Dispatch::float_log(source),
            };
            no_work(backend, &services, &profile)?;
            if failed.dtype() != DType::F32 || failed.shape().as_slice() != shape {
                return Err(String::from("typed exp/log failed Dispatch metadata changed"));
            }
            if !device.sync().is_err_and(|e| !e.is_device_poisoned()) {
                return Err(String::from("typed exp/log missing recoverable Dispatch sync error"));
            }
            let error = try_read_sync(Dispatch::float_into_data(failed)).ok_or("typed exp/log failed Dispatch read pending")?;
            if !error.is_err_and(|e| !e.is_device_poisoned()) || device.sync().is_err() {
                return Err(String::from("typed exp/log error acknowledgment poisoned Dispatch device"));
            }
            no_work(backend, &services, &profile)?;
            counts.rejected += 1;
        }
    }
    // Recover with real nonempty operations after all error acknowledgments.
    lifetimes(backend, counts, observed)
}
fn exercise(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    corpus(backend, counts, observed)?;
    layouts(backend, counts, observed)?;
    raw_rejections(backend, counts)?;
    native_validation(backend, counts)?;
    empties(backend, counts)?;
    public_errors(backend, counts, observed)
}
pub fn run(backend: &Nvidia) -> Result<()> {
    // The current driver/SDK handshake remains strict as protocols add operations.
    if abi::VERSION < 46
        || abi::KERNEL_COUNT < 82
        || Kernel::FloatArithmetic as u32 != 81
        || oracle::CORPUS_VERSION != 1
        || oracle::POLICY_SHA256 != POLICY
        || oracle::FIXTURE_SHA256 != FIXTURE
        || float::Operation::Exp as u32 != 15
        || float::Operation::Log as u32 != 16
    {
        return Err(String::from("typed exp/log protocol/corpus contract changed"));
    }
    let mut counts = Counts::default();
    let mut observed = Observed::default();
    let result = exercise(backend, &mut counts, &mut observed);
    let compared: usize = observed.accuracy.cells.iter().map(|cell| cell.values).sum();
    // Emit all ten cells before either structural or numerical failure returns.
    crate::report(format_args!(
        "NVIDIA_BURN_TYPED_EXP_LOG_ACCURACY policy_sha256={POLICY} fixture_sha256={FIXTURE} qualification_passed={} comparisons={compared} counts={counts:?} accuracy={}",
        result.is_ok() && counts == EXPECTED && compared == COMPARISONS && observed.accuracy.failures == 0,
        oracle::accuracy_json(&observed.accuracy),
    ));
    for example in &observed.examples {
        crate::report(format_args!("NVIDIA_BURN_TYPED_EXP_LOG_FAILURE {example}"));
    }
    result?;
    if counts != EXPECTED {
        return Err(format!("typed exp/log structural counts {counts:?}, expected {EXPECTED:?}"));
    }
    if compared != COMPARISONS {
        return Err(format!("typed exp/log comparison count {compared}, expected {COMPARISONS}"));
    }
    if observed.accuracy.failures != 0 {
        return Err(format!("typed exp/log project qualification failed comparisons={}", observed.accuracy.failures));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_TYPED_EXP_LOG_OK operations=2 formats=5 corpus_rows=306572 corpus_batches=78 chunk=4096 comparisons={compared} f64_inputs_per_op=16972 f32_inputs_per_op=2621 flex32_inputs_per_op=2621 f16_inputs_per_op=65536 bf16_inputs_per_op=65536 raw={} fresh={} native={} public={} geometry={} rejected={} empty={} queued={} f32_routes={} typed_routes={} policy_sha256={POLICY} fixture_sha256={FIXTURE} corpus_version=1 protocol={} profiles={} kernel={} guards=ok tails=zero source_nan_payloads=preserved resident=true full_writer=verified raw_f32=FloatArithmetic native_f32=TensorElementwise queued_last_owner=verified errors=recoverable qualification=project_targets linux_parity=not_claimed backend=nvidia",
        counts.raw,
        counts.fresh,
        counts.native,
        counts.public,
        counts.geometry,
        counts.rejected,
        counts.empty,
        counts.queued,
        counts.f32_routes,
        counts.typed_routes,
        abi::VERSION,
        abi::KERNEL_COUNT,
        Kernel::FloatArithmetic as usize,
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_maps_explicit_coordinates_to_declared_storage() {
        for (dtype, format) in FORMATS {
            let pool: Vec<_> = oracle::rows(dtype, Operation::Exp).iter().step_by(17).take(41).map(|r| r.input).collect();
            let cases = geometry_inputs(dtype, format, &pool).unwrap();
            assert_eq!(cases.len(), 33);
            let mut ranks = [false; 9];
            for input in cases {
                ranks[input.layout.shape().num_dims()] = true;
                assert_eq!(input.logical.len(), input.layout.len());
                for (linear, &bits) in input.logical.iter().enumerate() {
                    assert_eq!(input.storage[input.layout.element_offset(linear).unwrap()], bits);
                }
                for op in oracle::OPERATIONS {
                    let p = parameters(&input, op).unwrap();
                    let extents = float::validate(&p.map(|v| v as u32)).unwrap();
                    assert_eq!(
                        extents,
                        [
                            format.width().words(input.storage.len() as u32),
                            1,
                            format.width().words(input.logical.len() as u32)
                        ]
                    );
                    assert_eq!(input.references(op).unwrap().len(), input.logical.len());
                }
            }
            assert!(ranks[0] && ranks[1] && ranks[2] && ranks[8]);
        }
    }

    #[test]
    fn unary_metadata_mutations_fail_before_dispatch() {
        for (dtype, format) in FORMATS {
            for op in oracle::OPERATIONS {
                let input = Input::flat(dtype, format, vec![0; 5]).unwrap();
                let p = parameters(&input, op).unwrap();
                assert_eq!(float::work_items(&p.map(|v| v as u32)).unwrap(), if dtype.size() == 2 { 3 } else { 5 });
                let work = Kernel::FloatArithmetic.work(&p.map(|v| v as u32)).unwrap();
                assert!(work.validate_aliases(&[0x1004, 0x2004, 0x3004]).is_ok());
                for output in [0x1004, 0x1008, 0x2004, 0x2000] {
                    assert!(work.validate_aliases(&[0x1004, 0x2004, output]).is_err());
                }
                let invalid = malformed(p);
                assert_eq!(invalid.len(), 38);
                for bad in invalid {
                    assert!(float::validate(&bad.map(|v| v as u32)).is_err());
                }
                assert!(float::validate(&p[..31].iter().map(|&v| v as u32).collect::<Vec<_>>()).is_err());
                let mut extended = p.map(|v| v as u32).to_vec();
                extended.push(0);
                assert!(float::validate(&extended).is_err());
            }
        }
    }

    #[test]
    fn frozen_plan_covers_every_row_with_bounded_launches() {
        let mut plan = Counts::default();
        let mut comparisons = 0;
        for ((dtype, format), expected_rows) in FORMATS.into_iter().zip([16972, 2621, 2621, 65536, 65536]) {
            for op in oracle::OPERATIONS {
                let rows = oracle::rows(dtype, op);
                assert_eq!(rows.len(), expected_rows);
                assert!(rows.windows(2).all(|pair| pair[0].input < pair[1].input));
                for chunk in rows.chunks(CHUNK) {
                    assert!(chunk.len() <= CHUNK);
                    plan.corpus_rows += chunk.len();
                    plan.corpus_batches += 1;
                    plan.raw += 1;
                    plan.fresh += 1;
                    comparisons += chunk.len() * 2;
                }
                let pool: Vec<_> = rows.iter().take(41).map(|r| r.input).collect();
                for input in geometry_inputs(dtype, format, &pool).unwrap() {
                    plan.geometry += 1;
                    plan.raw += 1;
                    plan.fresh += 1;
                    plan.native += 2;
                    plan.public += 3;
                    for _ in 0..5 {
                        routed(&mut plan, dtype);
                    }
                    comparisons += input.logical.len() * 7;
                }
                // One last-owner operation in each of the three API layers.
                plan.fresh += 1;
                plan.native += 1;
                plan.public += 1;
                plan.queued += 3;
                routed(&mut plan, dtype);
                routed(&mut plan, dtype);
                comparisons += 17 * 3;
                // Metadata, parameter/pointer arity, aliases, physical bounds.
                plan.rejected += 38 + 2 + 2 + 4 + 5 + if dtype == DType::F64 { 3 } else { 0 };
                // Two lengths, three invalid Native arities.
                plan.rejected += 2 * 3;
                plan.empty += 4 * 5;
            }
        }
        // Integer Native methods, invalid Native layouts, deferred API errors.
        plan.rejected += 2 * 2 * 2 + 3 + 2 * 2 * 2;
        assert_eq!(plan, EXPECTED);
        assert_eq!(comparisons, COMPARISONS);
        assert_eq!(plan.raw + plan.fresh + plan.native + plan.public, 2496);
    }
}
