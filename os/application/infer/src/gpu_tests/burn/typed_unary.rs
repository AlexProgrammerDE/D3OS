//! Guarded typed Log1p/Sinh/Cosh/Tan qualification against the frozen independent corpus.
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
    validation::typed_unary::{self as oracle, Accuracy, Operation, Reference},
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
// The parent confirmed these numerical identities before host qualification.
const FROZEN: bool = true;
const COMPARISONS: usize = 3443996;
// Counts only, derived by the independent CPU plan. No observed maxima belong here.
const ROW_COUNTS: [[usize; 4]; 5] = [
    [26816, 26824, 26818, 26859],
    [8607, 8616, 8614, 8770],
    [8607, 8616, 8614, 8770],
    [65536, 65536, 65536, 65536],
    [65536, 65536, 65536, 65536],
];
const CELL_VALUES: [[usize; 4]; 5] = [
    [139300, 139332, 139308, 139472],
    [66464, 66500, 66492, 67116],
    [66464, 66500, 66492, 67116],
    [294180, 294180, 294180, 294180],
    [294180, 294180, 294180, 294180],
];
const POLICY: &str = "deab0d270d6730f8b3b61c8b76c2d3aab0b1ba1013f90113c0970b8f73ef32f2";
const FIXTURE: &str = "bfe2bdf874be9973c058f4167d12e1b6be2b1fbb32f99e4ba872c7d2965c1c93";
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
    corpus_rows: 700819,
    corpus_batches: 180,
    geometry: 760,
    raw: 1880,
    fresh: 1920,
    native: 1540,
    public: 2300,
    rejected: 2155,
    empty: 400,
    queued: 80,
    f32_routes: 768,
    typed_routes: 3072,
};
#[derive(Default)]
struct Observed {
    accuracy: Accuracy,
    examples: Vec<String>,
}
impl Observed {
    fn record(&mut self, dtype: DType, op: Operation, rows: &[Reference], actual: &[u64], path: &str) -> Result<()> {
        if rows.len() != actual.len() {
            return Err(String::from("typed unary logical length mismatch"));
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
    value.map_err(|e| format!("typed unary: {e}"))
}
fn op_code(op: Operation) -> Elementwise {
    match op {
        Operation::Log1p => Elementwise::Log1p,
        Operation::Sinh => Elementwise::Sinh,
        Operation::Cosh => Elementwise::Cosh,
        Operation::Tan => Elementwise::Tan,
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
        return Err(format!("typed unary metadata {dtype:?} {shape:?}: {:?} {:?}", actual.dtype(), actual.shape()));
    }
    Ok(())
}
fn source_data(actual: &TensorData, dtype: DType, shape: &[usize], expected: &[u64]) -> Result<()> {
    metadata(actual, dtype, shape, expected.len())?;
    if oracle::unpack(dtype, actual.as_bytes()) != expected {
        return Err(format!("typed unary changed source bytes or NaN payload {dtype:?} {shape:?}"));
    }
    Ok(())
}
fn output_data(actual: &TensorData, input: &Input, op: Operation, observed: &mut Observed, path: &str) -> Result<()> {
    metadata(actual, input.dtype, input.layout.shape().as_slice(), input.logical.len())?;
    observed.record(input.dtype, op, &input.references(op)?, &oracle::unpack(input.dtype, actual.as_bytes()), path)
}
fn primitive_data(value: FloatTensor<Native>) -> Result<TensorData> {
    try_read_sync(Native::float_into_data(value))
        .ok_or("typed unary Native read pending")?
        .map_err(|e| format!("typed unary Native read: {e:?}"))
}
fn unchanged(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if actual != expected {
        return Err(String::from("typed unary source/guard/unused RHS changed"));
    }
    Ok(())
}
fn output_words(
    backend: &Nvidia, buffer: &GpuBuffer<u32>, input: &Input, op: Operation, guarded: bool, observed: &mut Observed, path: &str,
) -> Result<Vec<u8>> {
    let count = (input.logical.len() * input.dtype.size()).div_ceil(4);
    let offset = usize::from(guarded);
    let mut actual = vec![0; count + 2 * offset];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if guarded && (actual[0] != GUARD || actual[count + 1] != GUARD) {
        return Err(String::from("typed unary output word guard changed"));
    }
    let words = &actual[offset..offset + count];
    if input.dtype.size() == 2 && input.logical.len() % 2 == 1 && words[count - 1] >> 16 != 0 {
        return Err(String::from("typed unary odd output upper half is not zero"));
    }
    let bytes: Vec<_> = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take(input.logical.len() * input.dtype.size())
        .collect();
    observed.record(input.dtype, op, &input.references(op)?, &oracle::unpack(input.dtype, &bytes), path)?;
    Ok(bytes)
}
/// Inspect every kernel counter per individual operation, including hidden copies.
fn resident(
    backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT], kernel: Kernel, fresh: bool,
) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE] {
        if after[service as usize].calls != before[service as usize].calls {
            return Err(format!("typed unary resident service={service}"));
        }
    }
    for (service, want) in [(abi::LAUNCH_ALLOCATE, u64::from(fresh)), (abi::LAUNCH, u64::from(!fresh))] {
        if after[service as usize].calls - before[service as usize].calls != want {
            return Err(format!("typed unary full writer service={service} expected={want}"));
        }
    }
    for (index, (old, new)) in profile.iter().zip(gpu(backend.profile())?).enumerate() {
        if new.calls - old.calls != u64::from(index == kernel as usize) {
            return Err(format!("typed unary route kernel={kernel:?} counter={index}"));
        }
    }
    Ok(())
}
fn no_work(backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT]) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
        if after[service as usize].calls != before[service as usize].calls {
            return Err(format!("typed unary invalid/empty service={service}"));
        }
    }
    if profile.iter().zip(gpu(backend.profile())?).any(|(a, b)| a.calls != b.calls) {
        return Err(String::from("typed unary invalid/empty kernel launched"));
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
                oracle::expected_bits(self.dtype, op, bits).ok_or_else(|| format!("typed unary missing reference {:?} {} {bits:#x}", self.dtype, op.label()))
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
/// Each poison value must fail this row's comparator, even for NaN targets.
/// Vary the sign and fraction so every F64 word changes between poisons.
/// Complement the target exponent; CPU audits prove numerical rejection for
/// every immutable fixture row, including unbounded finite outputs and NaNs.
fn output_poison(input: &Input, op: Operation, alternate: bool) -> Result<Vec<u32>> {
    let sign = 1u64 << (input.dtype.size() * 8 - 1);
    let mask = sign | (sign - 1);
    let fraction = match input.dtype {
        DType::F64 => 0x000f_ffff_ffff_ffff,
        DType::F32 | DType::Flex32 => 0x007f_ffff,
        DType::F16 => 0x03ff,
        DType::BF16 => 0x007f,
        _ => return Err(String::from("typed unary poison unsupported dtype")),
    };
    let bits: Vec<_> = input
        .references(op)?
        .iter()
        .map(|row| (!row.staged & mask) ^ if alternate { sign | fraction } else { 0 })
        .collect();
    let mut words = packed(input.dtype, &bits);
    if input.dtype.size() == 2 && input.logical.len() % 2 == 1 {
        *words.last_mut().unwrap() |= 0xffff_0000;
    }
    Ok(words)
}
fn identical_raw_bytes(expected: &mut Option<Vec<u8>>, actual: Vec<u8>) -> Result<()> {
    if let Some(expected) = expected {
        if *expected != actual {
            return Err(String::from("typed unary raw output depends on initial output bytes, including NaN payloads"));
        }
    } else {
        *expected = Some(actual);
    }
    Ok(())
}
fn raw(backend: &Nvidia, input: &Input, op: Operation, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    let words = source_words(input);
    let source = gpu(backend.upload_words(&words))?;
    // Only one poison word exists, even for F64. Unary must never load RHS.
    let unused_words = [GUARD, POISON, GUARD];
    let rhs = gpu(backend.upload_words(&unused_words))?;
    let count = (input.logical.len() * input.dtype.size()).div_ceil(4);
    let p = parameters(input, op)?;
    let mut expected_bytes = None;
    for alternate in [false, true] {
        let poison = output_poison(input, op, alternate)?;
        let mut guarded = vec![GUARD];
        guarded.extend_from_slice(&poison);
        guarded.push(GUARD);
        let output = gpu(backend.upload_words(&guarded))?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatArithmetic, &[(&source, 1), (&rhs, 1), (&output, 1)], &p)))?;
        resident(backend, &services, &profile, Kernel::FloatArithmetic, false)?;
        identical_raw_bytes(&mut expected_bytes, output_words(backend, &output, input, op, true, observed, "raw")?)?;
        unchanged(backend, &source, &words)?;
        unchanged(backend, &rhs, &unused_words)?;
        counts.raw += 1;
        // Release a dirty allocation before the fresh full writer obtains storage.
        // Pool selection is not assumed. The guarded repetitions directly test
        // two known initial outputs; fresh results must match the same bytes.
        gpu(backend.write_words(&output, 1, &poison))?;
        drop(output);
        let dirty = gpu(backend.upload_words(&poison))?;
        drop(dirty);
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| backend.produce_words(count, Kernel::FloatArithmetic, &[(&source, 1), (&rhs, 1)], &p)))?;
        resident(backend, &services, &profile, Kernel::FloatArithmetic, true)?;
        identical_raw_bytes(&mut expected_bytes, output_words(backend, &output, input, op, false, observed, "fresh")?)?;
        unchanged(backend, &source, &words)?;
        unchanged(backend, &rhs, &unused_words)?;
        counts.fresh += 1;
    }
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
        Operation::Log1p => Native::float_log1p(source),
        Operation::Sinh => Native::float_sinh(source),
        Operation::Cosh => Native::float_cosh(source),
        Operation::Tan => Native::float_tan(source),
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
                .ok_or("typed unary public read pending")?
                .map_err(|e| format!("typed unary public read: {e:?}"))?
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
                Operation::Log1p => Dispatch::float_log1p(source.into_dispatch()),
                Operation::Sinh => Dispatch::float_sinh(source.into_dispatch()),
                Operation::Cosh => Dispatch::float_cosh(source.into_dispatch()),
                Operation::Tan => Dispatch::float_tan(source.into_dispatch()),
            }
        } else {
            match op {
                Operation::Log1p => source.log1p(),
                Operation::Sinh => source.sinh(),
                Operation::Cosh => source.cosh(),
                Operation::Tan => source.tan(),
            }
            .into_dispatch()
        }
    }
    Ok(match rank {
        0 => launch::<0>(source, op, dispatch),
        1 => launch::<1>(source, op, dispatch),
        2 => launch::<2>(source, op, dispatch),
        3 => launch::<3>(source, op, dispatch),
        4 => launch::<4>(source, op, dispatch),
        5 => launch::<5>(source, op, dispatch),
        6 => launch::<6>(source, op, dispatch),
        7 => launch::<7>(source, op, dispatch),
        8 => launch::<8>(source, op, dispatch),
        _ => return Err(String::from("typed unary public rank missing")),
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
    let mut shapes = vec![vec![], vec![1, 2, 1, 2, 1, 2, 1, 3]];
    for rank in 3..8 {
        let mut shape = vec![1; rank];
        shape[rank - 1] = 3;
        shapes.push(shape);
    }
    for shape in shapes {
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
    for (fi, (dtype, format)) in FORMATS.into_iter().enumerate() {
        for op in oracle::OPERATIONS {
            let rows = oracle::rows(dtype, op);
            if rows.len() != ROW_COUNTS[fi][op.index()] {
                return Err(format!("typed unary fixture row count {dtype:?} {}: {}", op.label(), rows.len()));
            }
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
fn geometry_pool(dtype: DType, rows: &[Reference]) -> Vec<u64> {
    let mut pool = Vec::new();
    for class in [
        oracle::Class::Zero,
        oracle::Class::Nan,
        oracle::Class::Infinity,
        oracle::Class::Subnormal,
        oracle::Class::Normal,
    ] {
        pool.extend(rows.iter().filter(|r| oracle::classify(dtype, r.input) == class).take(3).map(|r| r.input));
        pool.extend(rows.iter().rev().filter(|r| oracle::classify(dtype, r.input) == class).take(3).map(|r| r.input));
    }
    pool.extend(rows.iter().step_by((rows.len() / 31).max(1)).map(|row| row.input));
    pool
}
fn layouts(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for (dtype, format) in FORMATS {
        for op in oracle::OPERATIONS {
            // Spread representative specials, payloads and finite values across views.
            let rows = oracle::rows(dtype, op);
            let pool = geometry_pool(dtype, &rows);
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
        bad[2] |= 1 << bit; // Includes SCALAR_RHS, illegal for every unary op.
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
/// Invalid requests must fail recoverably, including those sent to the driver.
fn rejected<T>(result: d3os_compute::Result<T>, context: &str) -> Result<()> {
    match result {
        Err(Error::InvalidParameter | Error::InvalidShape | Error::ShapeMismatch | Error::SizeOverflow) => Ok(()),
        Err(error) => Err(format!("typed unary {context} returned non-validation error: {error}")),
        Ok(_) => Err(format!("typed unary {context} accepted")),
    }
}
fn raw_rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        for op in oracle::OPERATIONS {
            let bits = oracle::rows(dtype, op)[0].input;
            let input = Input::flat(dtype, format, vec![bits; 5])?;
            let p = parameters(&input, op)?;
            let count = packed(dtype, &input.storage).len();
            let words = vec![POISON; count + 4];
            let a = gpu(backend.upload_words(&words))?;
            let b = gpu(backend.upload_words(&words))?;
            let output = gpu(backend.upload_words(&words))?;
            let short = gpu(backend.upload_words(&[POISON]))?;
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let queue = gpu(backend.queue_profile())?;
            for bad in malformed(p) {
                rejected(
                    backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &bad),
                    "raw malformed",
                )?;
                rejected(
                    backend.produce_words(count, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &bad),
                    "fresh malformed",
                )?;
                counts.rejected += 2;
            }
            for bad in [p[..31].to_vec(), {
                let mut v = p.to_vec();
                v.push(0);
                v
            }] {
                rejected(
                    backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &bad),
                    "raw parameter length",
                )?;
                rejected(
                    backend.produce_words(count, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &bad),
                    "fresh parameter length",
                )?;
                counts.rejected += 2;
            }
            for buffers in [vec![(&a, 1), (&output, 1)], vec![(&a, 1), (&b, 1), (&output, 1), (&short, 0)]] {
                rejected(backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p), "raw pointer arity")?;
                counts.rejected += 1;
            }
            for inputs in [vec![(&a, 1)], vec![(&a, 1), (&b, 1), (&short, 0)]] {
                rejected(backend.produce_words(count, Kernel::FloatArithmetic, &inputs, &p), "fresh pointer arity")?;
                counts.rejected += 1;
            }
            // Include zero work and mismatched full-writer word ownership.
            for length in [0, count - 1, count + 1] {
                rejected(
                    backend.produce_words(length, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &p),
                    "fresh output word count",
                )?;
                counts.rejected += 1;
            }
            // Reject overlapping outputs, including the inactive one-word RHS.
            for (target, offset) in [(&a, 1), (&a, 2), (&b, 1), (&b, 0)] {
                rejected(
                    backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (target, offset)], &p),
                    "raw alias",
                )?;
                counts.rejected += 1;
            }
            for buffers in [
                [(&short, 0), (&b, 1), (&output, 1)],
                [(&a, 1), (&b, words.len()), (&output, 1)],
                [(&a, 1), (&b, 1), (&short, 0)],
                [(&a, words.len()), (&b, 1), (&output, 1)],
                [(&a, 1), (&b, 1), (&output, words.len())],
            ] {
                rejected(backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p), "raw word bounds")?;
                counts.rejected += 1;
            }
            for inputs in [[(&short, 0), (&b, 1)], [(&a, 1), (&b, words.len())], [(&a, words.len()), (&b, 1)]] {
                rejected(backend.produce_words(count, Kernel::FloatArithmetic, &inputs, &p), "fresh word bounds")?;
                counts.rejected += 1;
            }
            if dtype == DType::F64 {
                let one = parameters(&Input::flat(dtype, format, vec![bits])?, op)?;
                for buffers in [[(&short, 0), (&b, 1), (&output, 1)], [(&a, 1), (&b, 1), (&short, 0)]] {
                    rejected(backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &one), "raw F64 second word")?;
                    counts.rejected += 1;
                }
                rejected(
                    backend.produce_words(2, Kernel::FloatArithmetic, &[(&short, 0), (&b, 1)], &one),
                    "fresh F64 second word",
                )?;
                counts.rejected += 1;
                let mut high = p;
                high[4] = i32::MAX as usize - 5;
                high[6] = i32::MAX as usize;
                rejected(
                    backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &high),
                    "raw F64 high-word bounds",
                )?;
                rejected(
                    backend.produce_words(count, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &high),
                    "fresh F64 high-word bounds",
                )?;
                counts.rejected += 2;
            }
            // Only four alias attempts reach the driver. None records GPU work.
            let after = gpu(backend.service_profile())?;
            for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
                let want = if service == abi::LAUNCH { 4 } else { 0 };
                if after[service as usize].calls - services[service as usize].calls != want {
                    return Err(format!("typed unary rejected service={service} expected={want}"));
                }
            }
            if profile.iter().zip(gpu(backend.profile())?).any(|(a, b)| a.calls != b.calls) || gpu(backend.queue_profile())?.launches != queue.launches {
                return Err(String::from("typed unary rejected request launched GPU work"));
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
                        return Err(String::from("typed unary unary accepted binary/scalar arity"));
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
                    return Err(String::from("typed unary integer Native storage accepted"));
                }
                counts.rejected += 1;
            }
        }
        no_work(backend, &services, &profile)?;
    }
    for (shape, strides, offset, extent) in [(vec![1; 9], vec![1; 9], 0, 1), (vec![3], vec![-1], 1, 3), (vec![3], vec![1], 1, 3)] {
        if Layout::from_parts(Shape::from(shape), strides, offset, extent).is_ok() {
            return Err(String::from("typed unary invalid Native rank/bounds accepted"));
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
                        .ok_or("typed unary empty read pending")?
                        .map_err(|e| format!("typed unary empty public: {e:?}"))?;
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
                return Err(String::from("typed unary retained Native alias reports unique ownership"));
            }
            drop(alias);
            if !source.can_mut() {
                return Err(String::from("typed unary last Native owner reports shared ownership"));
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

            let mut expected_bytes = None;
            for alternate in [false, true] {
                let source = gpu(backend.upload_words(&packed(dtype, &input.storage)))?;
                let rhs = gpu(backend.upload_words(&[POISON]))?;
                let p = parameters(&input, op)?;
                let count = (input.logical.len() * dtype.size()).div_ceil(4);
                let dirty = gpu(backend.upload_words(&output_poison(&input, op, alternate)?))?;
                drop(dirty);
                let services = gpu(backend.service_profile())?;
                let profile = gpu(backend.profile())?;
                let output = gpu(backend.with_commands(|| {
                    let output = backend.produce_words(count, Kernel::FloatArithmetic, &[(&source, 0), (&rhs, 0)], &p)?;
                    drop(source);
                    drop(rhs);
                    Ok::<_, Error>(output)
                }))?;
                resident(backend, &services, &profile, Kernel::FloatArithmetic, true)?;
                identical_raw_bytes(&mut expected_bytes, output_words(backend, &output, &input, op, false, observed, "queued.raw")?)?;
                counts.fresh += 1;
                counts.queued += 1;
            }

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
        return Err(String::from("typed unary failed integer primitive metadata changed"));
    }
    if !Native::sync(&NativeDevice).is_err_and(|e| !e.is_device_poisoned()) {
        return Err(String::from("typed unary missing recoverable Native sync error"));
    }
    let error = try_read_sync(Native::float_into_data(failed)).ok_or("typed unary failed Native read pending")?;
    if !error.is_err_and(|e| !e.is_device_poisoned()) || Native::sync(&NativeDevice).is_err() {
        return Err(String::from("typed unary error acknowledgment poisoned Native device"));
    }
    Ok(())
}
fn public_errors(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("typed unary error device: {e}"))?;
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
                Operation::Log1p => Dispatch::float_log1p(source),
                Operation::Sinh => Dispatch::float_sinh(source),
                Operation::Cosh => Dispatch::float_cosh(source),
                Operation::Tan => Dispatch::float_tan(source),
            };
            no_work(backend, &services, &profile)?;
            if failed.dtype() != DType::F32 || failed.shape().as_slice() != shape {
                return Err(String::from("typed unary failed Dispatch metadata changed"));
            }
            if !device.sync().is_err_and(|e| !e.is_device_poisoned()) {
                return Err(String::from("typed unary missing recoverable Dispatch sync error"));
            }
            let error = try_read_sync(Dispatch::float_into_data(failed)).ok_or("typed unary failed Dispatch read pending")?;
            if !error.is_err_and(|e| !e.is_device_poisoned()) || device.sync().is_err() {
                return Err(String::from("typed unary error acknowledgment poisoned Dispatch device"));
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
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    let mut counts = Counts::default();
    let mut observed = Observed::default();
    // Reject incompatible contracts before any GPU work, but still report all cells.
    let contract = if !FROZEN
        || abi::VERSION != 48
        || abi::KERNEL_COUNT != 82
        || Kernel::FloatArithmetic as u32 != 81
        || float::Operation::ALL.len() != 23
        || oracle::DTYPES != FORMATS.map(|(dtype, _)| dtype)
        || oracle::OPERATIONS.map(Operation::opcode) != [17, 30, 31, 32]
        || oracle::CORPUS_VERSION != 1
        || oracle::POLICY_SHA256 != POLICY
        || oracle::FIXTURE_SHA256 != FIXTURE
        || float::Operation::Log1p as u32 != 17
        || float::Operation::Sinh as u32 != 30
        || float::Operation::Cosh as u32 != 31
        || float::Operation::Tan as u32 != 32
    {
        Err(String::from("typed unary protocol/corpus contract changed"))
    } else {
        Ok(())
    };
    let result = contract.and_then(|()| exercise(backend, &mut counts, &mut observed));
    let compared: usize = observed.accuracy.cells.iter().map(|cell| cell.values).sum();
    let coherent = observed.accuracy.is_coherent();
    let cells_complete = observed.accuracy.cells.len() == 20
        && observed
            .accuracy
            .cells
            .iter()
            .zip(CELL_VALUES.iter().flatten())
            .all(|(cell, &expected)| cell.values == expected);
    // Emit all twenty cells before either structural or numerical failure returns.
    crate::report(format_args!(
        "NVIDIA_BURN_TYPED_UNARY_ACCURACY policy_sha256={POLICY} fixture_sha256={FIXTURE} qualification_passed={} coherent={coherent} comparisons={compared} counts={counts:?} accuracy={}",
        result.is_ok() && counts == EXPECTED && compared == COMPARISONS && cells_complete && coherent && observed.accuracy.failures == 0,
        oracle::accuracy_json(&observed.accuracy),
    ));
    for example in &observed.examples {
        crate::report(format_args!("NVIDIA_BURN_TYPED_UNARY_FAILURE {example}"));
    }
    result?;
    if counts != EXPECTED {
        return Err(format!("typed unary structural counts {counts:?}, expected {EXPECTED:?}"));
    }
    if compared != COMPARISONS {
        return Err(format!("typed unary comparison count {compared}, expected {COMPARISONS}"));
    }
    if !coherent || !cells_complete {
        return Err(format!("typed unary accuracy accounting coherent={coherent} cells_complete={cells_complete}"));
    }
    if observed.accuracy.failures != 0 {
        return Err(format!("typed unary project qualification failed comparisons={}", observed.accuracy.failures));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_TYPED_UNARY_OK operations=4 formats=5 corpus_rows={} corpus_batches={} chunk=4096 raw_poison_variants=2 comparisons={compared} rows_log1p_sinh_cosh_tan={ROW_COUNTS:?} raw={} fresh={} native={} public={} geometry={} rejected={} empty={} queued={} f32_routes={} typed_routes={} policy_sha256={POLICY} fixture_sha256={FIXTURE} corpus_version=1 sdk_operations=23 protocol={} profiles={} kernel={} guards=ok tails=zero raw_poison_independence=exact_bytes source_nan_payloads=preserved resident=true full_writer=verified raw_f32=FloatArithmetic native_f32=TensorElementwise queued_last_owner=verified errors=recoverable qualification=project_targets linux_parity=not_claimed backend=nvidia",
        counts.corpus_rows,
        counts.corpus_batches,
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

    /// Check actual pack/source/poison helpers for corpus, geometry and queued inputs.
    fn audit_input_encoding(input: &Input, op: Operation) {
        let original = oracle::pack(input.dtype, &input.logical);
        assert_eq!(oracle::unpack(input.dtype, &original), input.logical);
        let source = source_words(input);
        assert_eq!(source.first(), Some(&GUARD));
        assert_eq!(source.last(), Some(&GUARD));
        let source_bytes: Vec<_> = source[1..source.len() - 1]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .take(input.storage.len() * input.dtype.size())
            .collect();
        assert_eq!(oracle::unpack(input.dtype, &source_bytes), input.storage);
        if input.dtype.size() == 2 && input.storage.len() % 2 == 1 {
            assert_ne!(source[source.len() - 2] >> 16, 0);
        }
        let references = input.references(op).unwrap();
        assert_eq!(references.len(), input.logical.len());
        let poisons = [output_poison(input, op, false).unwrap(), output_poison(input, op, true).unwrap()];
        // Every physical F64 word differs, including its low fraction word.
        assert_eq!(poisons[0].len(), (input.logical.len() * input.dtype.size()).div_ceil(4));
        assert_eq!(poisons[0].len(), poisons[1].len());
        assert!(poisons[0].iter().zip(&poisons[1]).all(|(a, b)| a != b));
        let bytes = poisons.map(|words| {
            if input.dtype.size() == 2 && input.logical.len() % 2 == 1 {
                assert_ne!(words.last().unwrap() >> 16, 0);
            }
            words.iter().flat_map(|word| word.to_le_bytes()).take(original.len()).collect::<Vec<_>>()
        });
        assert!(bytes[0].iter().zip(&bytes[1]).all(|(a, b)| a != b));
        for poison in bytes {
            for (row, bits) in references.iter().zip(oracle::unpack(input.dtype, &poison)) {
                assert!(
                    !oracle::compare(input.dtype, op, row, bits).passed,
                    "accepted poison dtype={:?} op={} input={:#x} staged={:#x} poison={bits:#x}",
                    input.dtype,
                    op.label(),
                    row.input,
                    row.staged
                );
            }
        }
    }

    #[test]
    fn geometry_maps_independent_coordinates_and_all_public_ranks() {
        for (dtype, format) in FORMATS {
            for op in oracle::OPERATIONS {
                let rows = oracle::rows(dtype, op);
                let pool = geometry_pool(dtype, &rows);
                let cases = geometry_inputs(dtype, format, &pool).unwrap();
                assert_eq!(cases.len(), 38);
                assert_eq!(cases.iter().map(|input| input.logical.len()).sum::<usize>(), 3552);
                let mut ranks = [false; 9];
                for input in cases {
                    ranks[input.layout.shape().num_dims()] = true;
                    assert_eq!(input.logical.len(), input.layout.len());
                    for (linear, &bits) in input.logical.iter().enumerate() {
                        assert_eq!(input.storage[input.layout.element_offset(linear).unwrap()], bits);
                    }
                    let p = parameters(&input, op).unwrap();
                    let extents = float::validate(&p.map(|v| v as u32)).unwrap();
                    assert_eq!(
                        extents,
                        [
                            format.width().words(input.storage.len() as u32),
                            1,
                            format.width().words(input.logical.len() as u32),
                        ]
                    );
                    audit_input_encoding(&input, op);
                }
                assert!(ranks.into_iter().all(|covered| covered));
            }
        }
    }

    #[test]
    fn malformed_metadata_aliases_and_full_writer_extents_fail_before_launch() {
        for (dtype, format) in FORMATS {
            for op in oracle::OPERATIONS {
                let input = Input::flat(dtype, format, vec![0; 5]).unwrap();
                let p = parameters(&input, op).unwrap();
                let words = format.width().words(5);
                assert_eq!(float::work_items(&p.map(|v| v as u32)).unwrap(), if dtype.size() == 2 { 3 } else { 5 });
                let work = Kernel::FloatArithmetic.work(&p.map(|v| v as u32)).unwrap();
                assert!(Kernel::FloatArithmetic.allocates_output());
                assert_eq!(work.output, 2);
                assert_eq!(work.pointers, 3);
                assert_eq!(work.extents[..3], [words, 1, words]);
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
                if dtype == DType::F64 {
                    let mut high = p;
                    high[4] = i32::MAX as usize - 5;
                    high[6] = i32::MAX as usize;
                    let extents = float::validate(&high.map(|v| v as u32)).unwrap();
                    assert_eq!(extents[0], 2 * i32::MAX as u64);
                    assert!(extents[0] > words + 4);
                }
            }
        }
    }

    #[test]
    fn immutable_fixture_lookup_and_both_poison_classes_cover_every_row() {
        let mut accuracy = Accuracy::default();
        let mut total = 0;
        for (dtype, format) in FORMATS {
            for op in oracle::OPERATIONS {
                let rows = oracle::rows(dtype, op);
                assert!(rows.windows(2).all(|pair| pair[0].input < pair[1].input));
                for chunk in rows.chunks(CHUNK) {
                    let input = Input::flat(dtype, format, chunk.iter().map(|r| r.input).collect()).unwrap();
                    assert_eq!(input.references(op).unwrap(), chunk);
                    audit_input_encoding(&input, op);
                    for row in chunk {
                        assert!(accuracy.record(dtype, op, row, row.staged).passed);
                    }
                    total += chunk.len();
                }
                // Odd lengths exercise poisoned padding even for exhaustive
                // half corpora whose 4096-element chunks have no odd tail.
                let input = Input::flat(dtype, format, vec![rows[0].input; 3]).unwrap();
                audit_input_encoding(&input, op);
                let queued = Input::flat(dtype, format, rows.iter().step_by(23).take(17).map(|row| row.input).collect()).unwrap();
                assert_eq!(queued.logical.len(), 17);
                audit_input_encoding(&queued, op);
            }
        }
        assert_eq!(accuracy.cells.iter().map(|cell| cell.values).sum::<usize>(), total);
        assert_eq!(accuracy.failures, 0);
        assert!(accuracy.is_coherent());
    }

    #[test]
    fn numerical_failures_fill_all_cells_and_bound_examples() {
        let mut observed = Observed::default();
        for (dtype, format) in FORMATS {
            for op in oracle::OPERATIONS {
                let rows = oracle::rows(dtype, op);
                let input = Input::flat(dtype, format, rows.iter().step_by(rows.len() / 7).take(7).map(|row| row.input).collect()).unwrap();
                let bytes: Vec<_> = output_poison(&input, op, false)
                    .unwrap()
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take(input.logical.len() * dtype.size())
                    .collect();
                observed
                    .record(dtype, op, &input.references(op).unwrap(), &oracle::unpack(dtype, &bytes), "cpu.failure")
                    .unwrap();
            }
        }
        assert!(observed.accuracy.cells.iter().all(|cell| cell.values == 7 && cell.failures == 7));
        assert_eq!(observed.accuracy.failures, 20 * 7);
        assert_eq!(observed.examples.len(), 12);
        assert!(observed.accuracy.is_coherent());
    }

    #[test]
    fn independent_cpu_plan_freezes_launches_and_per_cell_observation_counts() {
        assert_eq!(oracle::POLICY_SHA256, POLICY);
        assert_eq!(oracle::FIXTURE_SHA256, FIXTURE);
        assert_eq!(oracle::DTYPES, FORMATS.map(|(dtype, _)| dtype));
        assert_eq!(oracle::OPERATIONS.map(Operation::opcode), [17, 30, 31, 32]);
        let mut plan = Counts::default();
        let mut cells = [[0; 4]; 5];
        // Independently enumerate geometry lengths, without calling the runtime
        // geometry generator, counters, expected arithmetic, or GPU APIs.
        let mut geometry_lengths = Vec::new();
        for n in [1, 2, 3, 127, 128, 129, 255, 256, 257] {
            geometry_lengths.extend([n; 3]);
        }
        geometry_lengths.extend([5, 21, 6, 1, 24, 3, 3, 3, 3, 3, 6]);
        assert_eq!(geometry_lengths.len(), 38);
        for (fi, (dtype, _)) in FORMATS.into_iter().enumerate() {
            for op in oracle::OPERATIONS {
                let rows = oracle::rows(dtype, op);
                assert_eq!(rows.len(), ROW_COUNTS[fi][op.index()]);
                let mut remaining = rows.len();
                while remaining != 0 {
                    let length = remaining.min(4096);
                    remaining -= length;
                    plan.corpus_rows += length;
                    plan.corpus_batches += 1;
                    plan.raw += 2;
                    plan.fresh += 2;
                    cells[fi][op.index()] += length * 4;
                }
                for &length in &geometry_lengths {
                    plan.geometry += 1;
                    plan.raw += 2;
                    plan.fresh += 2;
                    plan.native += 2;
                    plan.public += 3;
                    cells[fi][op.index()] += length * 9;
                }
                plan.native += 1;
                plan.public += 1;
                plan.fresh += 2;
                plan.queued += 4;
                cells[fi][op.index()] += 17 * 4;
                // Native and public launches, including two queued routes.
                let routed = geometry_lengths.len() * 5 + 2;
                if dtype == DType::F32 {
                    plan.f32_routes += routed;
                } else {
                    plan.typed_routes += routed;
                }
                // Both raw APIs: 38 metadata variants, two parameter lengths,
                // two pointer arities. Guarded aliases=4, bounds=5. Fresh has
                // three bad word counts and three bad input bounds.
                plan.rejected += 38 * 2 + 2 * 2 + 2 * 2 + 4 + 5 + 3 + 3;
                if dtype == DType::F64 {
                    plan.rejected += 5;
                }
                // Two lengths times three invalid Native arities.
                plan.rejected += 2 * 3;
                plan.empty += 4 * 5;
            }
        }
        // Integer Native methods, invalid Native layouts, deferred API errors.
        plan.rejected += 2 * 4 * 2 + 3 + 2 * 4 * 2;
        assert_eq!(plan, EXPECTED);
        assert_eq!(cells, CELL_VALUES);
        assert_eq!(cells.iter().flatten().sum::<usize>(), COMPARISONS);
        assert_eq!(plan.raw + plan.fresh + plan.native + plan.public, 6920 + 4 * plan.corpus_batches);
    }
}
