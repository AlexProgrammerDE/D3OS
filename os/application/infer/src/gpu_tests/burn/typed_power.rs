//! Guarded forward Power checks against independent stored-operand references.
//! Host audits establish fixture and launch-plan consistency, never GPU accuracy.
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
    validation::typed_power::{self as oracle, Reference, Schedule},
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
// Only the parent may authorize freezing the numerical identities and counts.
const FROZEN: bool = true;
const POLICY: &str = "98fa20bdf8ac17b6f6c2a2779b633b2b963448b11a71ee2f598d04a357d7b77b";
const FIXTURE: &str = "8767f0d381c19fe2c8ac90aece3a9bc96d36b5c9b2c042812364a442821290bd";
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
}
// Frozen independent CPU enumeration, never GPU-observed totals.
const EXPECTED_COUNTS: Counts = Counts {
    corpus_rows: 366933,
    corpus_batches: 91,
    geometry: 1340,
    raw: 3372,
    fresh: 3392,
    native: 1982,
    public: 8199,
    rejected: 1425,
    empty: 1740,
    queued: 90,
};
const COMPARISONS: usize = 5141923;
const CELLS: usize = 145;
fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("typed power: {e}"))
}
fn native_kernel(dtype: DType) -> Kernel {
    if dtype == DType::F32 {
        Kernel::TensorElementwise
    } else {
        Kernel::FloatArithmetic
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
        return Err(format!("typed power metadata {dtype:?} {shape:?}: {:?} {:?}", actual.dtype(), actual.shape()));
    }
    Ok(())
}
fn source_data(actual: &TensorData, dtype: DType, shape: &[usize], expected: &[u64]) -> Result<()> {
    metadata(actual, dtype, shape, expected.len())?;
    if oracle::unpack(dtype, actual.as_bytes()) != expected {
        return Err(format!("typed power changed source bytes or NaN payload {dtype:?} {shape:?}"));
    }
    Ok(())
}
fn primitive_data(value: FloatTensor<Native>) -> Result<TensorData> {
    try_read_sync(Native::float_into_data(value))
        .ok_or("typed power Native read pending")?
        .map_err(|e| format!("typed power Native read: {e:?}"))
}
fn unchanged(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if actual != expected {
        return Err(String::from("typed power source/guard/unused RHS changed"));
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
fn source_words(input: &Input) -> Vec<u32> {
    let mut words = vec![GUARD];
    words.extend(packed(input.dtype, &input.storage));
    if input.dtype.size() == 2 && input.storage.len() % 2 == 1 {
        *words.last_mut().unwrap() |= 0xffff_0000;
    }
    words.push(GUARD);
    words
}
fn identical_raw_bytes(expected: &mut Option<Vec<u8>>, actual: Vec<u8>) -> Result<()> {
    if let Some(expected) = expected {
        if *expected != actual {
            return Err(String::from("typed power raw output depends on initial output bytes, including NaN payloads"));
        }
    } else {
        *expected = Some(actual);
    }
    Ok(())
}

#[derive(Default)]
struct Observed {
    accuracy: oracle::Accuracy,
    examples: Vec<String>,
}
impl Observed {
    fn record(&mut self, dtype: DType, route: oracle::Route, rows: &[Reference], actual: &[u64], path: &str) -> Result<()> {
        if rows.len() != actual.len() {
            return Err(String::from("typed power logical length mismatch"));
        }
        for (row, &bits) in rows.iter().zip(actual) {
            let comparison = self.accuracy.record(dtype, route, row, bits);
            if !comparison.passed && self.examples.len() < 12 {
                self.examples.push(format!(
                    "path={path} dtype={dtype:?} route={route:?} schedule={:?} lhs={:#018x} rhs={:#018x} actual={bits:#018x} staged={:#018x} ideal={:#018x} comparison={comparison:?}",
                    row.schedule, row.lhs, row.rhs, row.staged, row.ideal,
                ));
            }
        }
        Ok(())
    }
}
fn references(dtype: DType, schedule: Schedule, lhs: &[u64], rhs: &[u64]) -> Result<Vec<Reference>> {
    if lhs.len() != rhs.len() {
        return Err(String::from("typed power reference operand length mismatch"));
    }
    lhs.iter()
        .zip(rhs)
        .map(|(&a, &b)| {
            oracle::expected_bits(dtype, schedule, a, b).ok_or_else(|| format!("typed power missing reference {dtype:?} {schedule:?} lhs={a:#x} rhs={b:#x}"))
        })
        .collect()
}
fn output_data(
    actual: &TensorData, dtype: DType, shape: &[usize], rows: &[Reference], route: oracle::Route, observed: &mut Observed, path: &str,
) -> Result<()> {
    metadata(actual, dtype, shape, rows.len())?;
    observed.record(dtype, route, rows, &oracle::unpack(dtype, actual.as_bytes()), path)
}
fn parameters(lhs: &Input, rhs: &Input, shape: &[usize], scalar: Option<u64>) -> Result<[usize; 32]> {
    let left = gpu(lhs.layout.expand(Shape::from(shape)))?;
    let right = if scalar.is_some() {
        left.clone()
    } else {
        gpu(rhs.layout.expand(Shape::from(shape)))?
    };
    let mut p = gpu(left.elementwise_parameters(&right, Elementwise::Add, 0, false))?;
    p[2] = Elementwise::Power as usize | ((lhs.format as usize) << 8);
    if let Some(bits) = scalar {
        p[2] |= float::SCALAR_RHS as usize;
        p[3] = bits as u32 as usize;
        p[5] = (bits >> 32) as usize;
        p[7] = 0;
        p[24..32].fill(0);
    }
    Ok(p)
}
/// Exact expected kernel multiplicities, including fills and exponent casts.
fn launch_plan(dtype: DType, schedule: Schedule, integer_tensor: bool) -> [u64; abi::KERNEL_COUNT] {
    let mut plan = [0; abi::KERNEL_COUNT];
    match schedule {
        Schedule::Identity => {}
        Schedule::Ones => plan[if dtype.size() == 4 { Kernel::Fill } else { Kernel::TensorStorage } as usize] = 1,
        Schedule::SquareReciprocal => plan[native_kernel(dtype) as usize] = 2,
        _ => plan[native_kernel(dtype) as usize] = 1,
    }
    if integer_tensor {
        plan[Kernel::TensorCast as usize] += 1;
    }
    plan
}
fn resident(
    backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT], plan: &[u64; abi::KERNEL_COUNT],
    fresh: bool,
) -> Result<()> {
    let launches: u64 = plan.iter().sum();
    check_service_deltas(before, &gpu(backend.service_profile())?, &resident_services(launches, fresh))?;
    check_kernel_deltas(profile, &gpu(backend.profile())?, plan)
}
fn resident_services(launches: u64, fresh: bool) -> [(u32, u64); 5] {
    [
        (abi::UPLOAD, 0),
        (abi::DOWNLOAD, 0),
        (abi::ALLOCATE, 0),
        (abi::LAUNCH_ALLOCATE, if fresh { launches } else { 0 }),
        (abi::LAUNCH, if fresh { 0 } else { launches }),
    ]
}
fn check_service_deltas(
    before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], after: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], expected: &[(u32, u64)],
) -> Result<()> {
    for &(service, want) in expected {
        let old = before[service as usize].calls;
        let new = after[service as usize].calls;
        if new.checked_sub(old) != Some(want) {
            return Err(format!("typed power resident service={service} expected={want} before={old} after={new}"));
        }
    }
    Ok(())
}
fn empty_scalar_work(route: oracle::Route, scalar: Scalar) -> ([(u32, u64); 5], [u64; abi::KERNEL_COUNT]) {
    // Wrapper exponent zero calls float_ones, which constructs new empty storage.
    // NativeTensor::empty requires one physical word. ALLOCATE clears its page
    // through the driver's Zero kernel, without a userspace launch or upload.
    // The impl routes always reuse empty input storage, even for exponent zero.
    let ones = matches!(
        route,
        oracle::Route::PowfScalar | oracle::Route::PowiScalar | oracle::Route::PublicPowfScalar | oracle::Route::PublicPowiScalar
    ) && scalar_schedule(scalar) == Schedule::Ones;
    let mut kernels = [0; abi::KERNEL_COUNT];
    kernels[Kernel::Zero as usize] = u64::from(ones);
    (
        [
            (abi::UPLOAD, 0),
            (abi::DOWNLOAD, 0),
            (abi::ALLOCATE, u64::from(ones)),
            (abi::LAUNCH_ALLOCATE, 0),
            (abi::LAUNCH, 0),
        ],
        kernels,
    )
}
fn check_kernel_deltas(
    before: &[abi::ProfileEntry; abi::KERNEL_COUNT], after: &[abi::ProfileEntry; abi::KERNEL_COUNT], plan: &[u64; abi::KERNEL_COUNT],
) -> Result<()> {
    for (index, ((old, new), &want)) in before.iter().zip(after).zip(plan).enumerate() {
        if new.calls.checked_sub(old.calls) != Some(want) {
            return Err(format!(
                "typed power route counter={index} expected={want} before={} after={}",
                old.calls, new.calls
            ));
        }
    }
    Ok(())
}
fn no_work(backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT]) -> Result<()> {
    resident(backend, before, profile, &[0; abi::KERNEL_COUNT], true)
}
/// Complement target exponent bits, then vary every byte for the second poison.
/// Both remain outside the strict class/envelope policy, even for NaN targets.
fn output_poison(dtype: DType, rows: &[Reference], alternate: bool) -> Vec<u32> {
    let sign = 1u64 << (dtype.size() * 8 - 1);
    let mask = sign | (sign - 1);
    let fraction = match dtype {
        DType::F64 => 0x000f_ffff_ffff_ffff,
        DType::F32 | DType::Flex32 => 0x007f_ffff,
        DType::F16 => 0x03ff,
        DType::BF16 => 0x007f,
        _ => unreachable!("power poison floating dtype"),
    };
    let bits: Vec<_> = rows
        .iter()
        .map(|row| (!row.staged & mask) ^ if alternate { sign | fraction } else { 0 })
        .collect();
    let mut words = packed(dtype, &bits);
    if dtype.size() == 2 && rows.len() % 2 == 1 {
        *words.last_mut().unwrap() |= if alternate { 0x5ac3_0000 } else { 0xa53c_0000 };
    }
    words
}
fn output_words(
    backend: &Nvidia, buffer: &GpuBuffer<u32>, dtype: DType, rows: &[Reference], guarded: bool, location: (oracle::Route, &str), observed: &mut Observed,
) -> Result<Vec<u8>> {
    let (route, path) = location;
    let count = (rows.len() * dtype.size()).div_ceil(4);
    let offset = usize::from(guarded);
    let mut actual = vec![0; count + 2 * offset];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if guarded && (actual[0] != GUARD || actual[count + 1] != GUARD) {
        return Err(String::from("typed power physical output guard changed"));
    }
    let words = &actual[offset..offset + count];
    if dtype.size() == 2 && rows.len() % 2 == 1 && words[count - 1] >> 16 != 0 {
        return Err(String::from("typed power odd output upper half is not zero"));
    }
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).take(rows.len() * dtype.size()).collect();
    observed.record(dtype, route, rows, &oracle::unpack(dtype, &bytes), path)?;
    Ok(bytes)
}
fn raw(
    backend: &Nvidia, inputs: [&Input; 2], shape: &[usize], scalar: Option<u64>, rows: &[Reference], counts: &mut Counts, observed: &mut Observed,
) -> Result<()> {
    let [lhs, rhs] = inputs;
    let aw = source_words(lhs);
    // A scalar never needs even the second F64 word of this independently guarded RHS.
    let bw = if scalar.is_some() { vec![GUARD, POISON, GUARD] } else { source_words(rhs) };
    let a = gpu(backend.upload_words(&aw))?;
    let b = gpu(backend.upload_words(&bw))?;
    let count = (rows.len() * lhs.dtype.size()).div_ceil(4);
    let p = parameters(lhs, rhs, shape, scalar)?;
    let mut bytes = None;
    let mut plan = [0; abi::KERNEL_COUNT];
    plan[Kernel::FloatArithmetic as usize] = 1;
    let route = if scalar.is_some() { oracle::Route::ScalarImpl } else { oracle::Route::Tensor };
    for alternate in [false, true] {
        let poison = output_poison(lhs.dtype, rows, alternate);
        let mut guarded = vec![GUARD];
        guarded.extend_from_slice(&poison);
        guarded.push(GUARD);
        let output = gpu(backend.upload_words(&guarded))?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &p)))?;
        resident(backend, &services, &profile, &plan, false)?;
        identical_raw_bytes(&mut bytes, output_words(backend, &output, lhs.dtype, rows, true, (route, "raw"), observed)?)?;
        unchanged(backend, &a, &aw)?;
        unchanged(backend, &b, &bw)?;
        counts.raw += 1;
        gpu(backend.write_words(&output, 1, &poison))?;
        drop(output);
        let dirty = gpu(backend.upload_words(&poison))?;
        drop(dirty);
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| backend.produce_words(count, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &p)))?;
        resident(backend, &services, &profile, &plan, true)?;
        identical_raw_bytes(&mut bytes, output_words(backend, &output, lhs.dtype, rows, false, (route, "fresh"), observed)?)?;
        unchanged(backend, &a, &aw)?;
        unchanged(backend, &b, &bw)?;
        counts.fresh += 1;
    }
    Ok(())
}
fn native(
    backend: &Nvidia, inputs: [&Input; 2], shape: &[usize], scalar: Option<Scalar>, rows: &[Reference], counts: &mut Counts, observed: &mut Observed,
) -> Result<()> {
    let [lhs, rhs] = inputs;
    let abase = gpu(NativeTensor::from_data(lhs.data()))?;
    let bbase = gpu(NativeTensor::from_data(rhs.data()))?;
    let a = gpu(abase.with_layout(lhs.layout.clone()))?;
    let b = gpu(bbase.with_layout(rhs.layout.clone()))?;
    for legacy in [false, true] {
        if legacy && scalar.is_some() {
            continue;
        }
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| {
            if let Some(scalar) = scalar {
                a.float_scalar(Elementwise::Power, scalar)
            } else if legacy {
                a.binary(&b, Elementwise::Power)
            } else {
                a.float_binary(&b, Elementwise::Power)
            }
        }))?;
        resident(backend, &services, &profile, &launch_plan(lhs.dtype, Schedule::Mathematical, false), true)?;
        output_data(
            &gpu(output.read())?,
            lhs.dtype,
            shape,
            rows,
            if scalar.is_some() { oracle::Route::ScalarImpl } else { oracle::Route::Tensor },
            observed,
            "NativeTensor",
        )?;
        counts.native += 1;
    }
    source_data(&gpu(a.read())?, lhs.dtype, lhs.layout.shape().as_slice(), &lhs.logical)?;
    source_data(&gpu(b.read())?, rhs.dtype, rhs.layout.shape().as_slice(), &rhs.logical)?;
    source_data(&gpu(abase.read())?, lhs.dtype, &[lhs.storage.len()], &lhs.storage)?;
    source_data(&gpu(bbase.read())?, rhs.dtype, &[rhs.storage.len()], &rhs.storage)
}
fn dispatch_data(value: FloatTensor<Dispatch>) -> Result<TensorData> {
    try_read_sync(Dispatch::float_into_data(value))
        .ok_or("typed power Dispatch read pending")?
        .map_err(|e| format!("typed power Dispatch read: {e:?}"))
}
fn public_tensor_launch(a: FloatTensor<Native>, b: FloatTensor<Native>, path: usize, rank: usize) -> Result<FloatTensor<Dispatch>> {
    fn launch<const D: usize>(a: FloatTensor<Native>, b: FloatTensor<Native>, path: usize) -> FloatTensor<Dispatch> {
        let a = Tensor::<D>::from_primitive::<Native>(a);
        let b = Tensor::<D>::from_primitive::<Native>(b);
        if path == 1 {
            Dispatch::float_powf(a.into_dispatch(), b.into_dispatch())
        } else if path == 3 {
            a.powi(b).into_dispatch()
        } else {
            a.powf(b).into_dispatch()
        }
    }
    Ok(match rank {
        0 => launch::<0>(a, b, path),
        1 => launch::<1>(a, b, path),
        2 => launch::<2>(a, b, path),
        3 => launch::<3>(a, b, path),
        4 => launch::<4>(a, b, path),
        5 => launch::<5>(a, b, path),
        6 => launch::<6>(a, b, path),
        7 => launch::<7>(a, b, path),
        8 => launch::<8>(a, b, path),
        _ => return Err(String::from("typed power public rank missing")),
    })
}
fn public_tensor(backend: &Nvidia, lhs: &Input, rhs: &Input, shape: &[usize], rows: &[Reference], counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for path in 0..4 {
        let (abase, a) = lhs.primitive();
        let (bbase, b) = rhs.primitive();
        let aa = a.clone();
        let ba = b.clone();
        source_data(&primitive_data(aa.clone())?, lhs.dtype, lhs.layout.shape().as_slice(), &lhs.logical)?;
        source_data(&primitive_data(ba.clone())?, rhs.dtype, rhs.layout.shape().as_slice(), &rhs.logical)?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let actual = if path == 0 {
            let output = Native::float_powf(a, b);
            resident(
                backend,
                &services,
                &profile,
                &if rows.is_empty() {
                    [0; abi::KERNEL_COUNT]
                } else {
                    launch_plan(lhs.dtype, Schedule::Mathematical, false)
                },
                true,
            )?;
            primitive_data(output)?
        } else {
            let output = public_tensor_launch(a, b, path, shape.len())?;
            resident(
                backend,
                &services,
                &profile,
                &if rows.is_empty() {
                    [0; abi::KERNEL_COUNT]
                } else {
                    launch_plan(lhs.dtype, Schedule::Mathematical, false)
                },
                true,
            )?;
            dispatch_data(output)?
        };
        output_data(
            &actual,
            lhs.dtype,
            shape,
            rows,
            if path >= 2 { oracle::Route::PublicTensor } else { oracle::Route::Tensor },
            observed,
            ["Native.powf", "Dispatch.powf", "Tensor.powf", "Tensor.powi.float_rhs"][path],
        )?;
        for (value, input, base) in [(aa, lhs, abase), (ba, rhs, bbase)] {
            source_data(&primitive_data(value)?, input.dtype, input.layout.shape().as_slice(), &input.logical)?;
            source_data(&primitive_data(base)?, input.dtype, &[input.storage.len()], &input.storage)?;
        }
        counts.public += 1;
    }
    Ok(())
}

fn scalar_schedule(scalar: Scalar) -> Schedule {
    match scalar {
        Scalar::Float(0.0) | Scalar::Int(0) | Scalar::UInt(0) | Scalar::Bool(false) => Schedule::Ones,
        Scalar::Float(1.0) | Scalar::Int(1) | Scalar::UInt(1) | Scalar::Bool(true) => Schedule::Identity,
        Scalar::Float(2.0) | Scalar::Int(2) | Scalar::UInt(2) => Schedule::Square,
        Scalar::Float(-1.0) | Scalar::Int(-1) => Schedule::Reciprocal,
        Scalar::Float(-2.0) | Scalar::Int(-2) => Schedule::SquareReciprocal,
        _ => Schedule::Mathematical,
    }
}
fn scalar_launch(source: FloatTensor<Native>, scalar: Scalar, route: oracle::Route) -> FloatTensor<Native> {
    match route {
        oracle::Route::ScalarImpl => Native::float_powf_scalar_impl(source, scalar),
        oracle::Route::PowiScalarImpl => Native::float_powi_scalar_impl(source, scalar),
        oracle::Route::PowfScalar => Native::float_powf_scalar(source, scalar),
        oracle::Route::PowiScalar => Native::float_powi_scalar(source, scalar),
        _ => unreachable!("Native scalar route"),
    }
}
fn public_scalar_launch(source: FloatTensor<Native>, scalar: Scalar, route: oracle::Route, rank: usize) -> Result<FloatTensor<Dispatch>> {
    fn launch<const D: usize>(source: FloatTensor<Native>, scalar: Scalar, route: oracle::Route) -> FloatTensor<Dispatch> {
        let source = Tensor::<D>::from_primitive::<Native>(source);
        let source = match route {
            oracle::Route::ScalarImpl => return Dispatch::float_powf_scalar_impl(source.into_dispatch(), scalar),
            oracle::Route::PowiScalarImpl => return Dispatch::float_powi_scalar_impl(source.into_dispatch(), scalar),
            oracle::Route::PowfScalar => return Dispatch::float_powf_scalar(source.into_dispatch(), scalar),
            oracle::Route::PowiScalar => return Dispatch::float_powi_scalar(source.into_dispatch(), scalar),
            oracle::Route::PublicPowfScalar => match scalar {
                Scalar::Float(v) => source.powf_scalar(v),
                Scalar::Int(v) => source.powf_scalar(v),
                Scalar::UInt(v) => source.powf_scalar(v),
                Scalar::Bool(v) => source.powf_scalar(v),
            },
            oracle::Route::PublicPowiScalar => match scalar {
                Scalar::Float(v) => source.powi_scalar(v),
                Scalar::Int(v) => source.powi_scalar(v),
                Scalar::UInt(v) => source.powi_scalar(v),
                Scalar::Bool(v) => source.powi_scalar(v),
            },
            _ => unreachable!("public scalar route"),
        };
        source.into_dispatch()
    }
    Ok(match rank {
        0 => launch::<0>(source, scalar, route),
        1 => launch::<1>(source, scalar, route),
        2 => launch::<2>(source, scalar, route),
        3 => launch::<3>(source, scalar, route),
        4 => launch::<4>(source, scalar, route),
        5 => launch::<5>(source, scalar, route),
        6 => launch::<6>(source, scalar, route),
        7 => launch::<7>(source, scalar, route),
        8 => launch::<8>(source, scalar, route),
        _ => return Err(String::from("typed power scalar public rank missing")),
    })
}
fn public_scalar(
    backend: &Nvidia, input: &Input, scalar: Scalar, route: oracle::Route, rows: &[Reference], counts: &mut Counts, observed: &mut Observed,
) -> Result<()> {
    let schedule = rows.first().map_or(Schedule::Mathematical, |r| r.schedule);
    if rows.iter().any(|r| r.schedule != schedule) {
        return Err(String::from("typed power mixed scalar schedules"));
    }
    let public = matches!(route, oracle::Route::PublicPowfScalar | oracle::Route::PublicPowiScalar);
    for dispatch in [false, true] {
        if public && !dispatch {
            continue;
        }
        let (base, source) = input.primitive();
        let alias = source.clone();
        source_data(&primitive_data(alias.clone())?, input.dtype, input.layout.shape().as_slice(), &input.logical)?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let actual = if dispatch {
            let output = public_scalar_launch(source, scalar, route, input.layout.shape().num_dims())?;
            if schedule == Schedule::Identity && (output.can_mut() || alias.can_mut()) {
                return Err(String::from("typed power Dispatch identity failed to retain original alias"));
            }
            resident(backend, &services, &profile, &launch_plan(input.dtype, schedule, false), true)?;
            dispatch_data(output)?
        } else {
            let output = scalar_launch(source, scalar, route);
            // Returning the original primitive is observable without a launch.
            if schedule == Schedule::Identity && (output.can_mut() || alias.can_mut()) {
                return Err(String::from("typed power identity failed to retain original alias"));
            }
            resident(backend, &services, &profile, &launch_plan(input.dtype, schedule, false), true)?;
            primitive_data(output)?
        };
        output_data(
            &actual,
            input.dtype,
            input.layout.shape().as_slice(),
            rows,
            route,
            observed,
            if dispatch { "Dispatch/Tensor.scalar" } else { "Native.scalar" },
        )?;
        if schedule == Schedule::Identity {
            source_data(&actual, input.dtype, input.layout.shape().as_slice(), &input.logical)?;
        }
        source_data(&primitive_data(alias)?, input.dtype, input.layout.shape().as_slice(), &input.logical)?;
        source_data(&primitive_data(base)?, input.dtype, &[input.storage.len()], &input.storage)?;
        counts.public += 1;
    }
    Ok(())
}
fn tensor_case(backend: &Nvidia, lhs: &Input, rhs: &Input, shape: &[usize], rows: &[Reference], counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    raw(backend, [lhs, rhs], shape, None, rows, counts, observed)?;
    native(backend, [lhs, rhs], shape, None, rows, counts, observed)?;
    public_tensor(backend, lhs, rhs, shape, rows, counts, observed)
}
fn scalar_case(backend: &Nvidia, input: &Input, scalar: Scalar, bits: u64, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    let math = references(input.dtype, Schedule::Mathematical, &input.logical, &vec![bits; input.logical.len()])?;
    raw(backend, [input, input], input.layout.shape().as_slice(), Some(bits), &math, counts, observed)?;
    native(backend, [input, input], input.layout.shape().as_slice(), Some(scalar), &math, counts, observed)?;
    public_scalar(backend, input, scalar, oracle::Route::ScalarImpl, &math, counts, observed)?;
    let wrapper = references(input.dtype, scalar_schedule(scalar), &input.logical, &vec![bits; input.logical.len()])?;
    public_scalar(backend, input, scalar, oracle::Route::PowfScalar, &wrapper, counts, observed)
}
fn corpus(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for (dtype, format) in FORMATS {
        let rows = oracle::rows(dtype);
        for chunk in rows.chunks(CHUNK) {
            let lhs = Input::flat(dtype, format, chunk.iter().map(|r| r.lhs).collect())?;
            let rhs = Input::flat(dtype, format, chunk.iter().map(|r| r.rhs).collect())?;
            // Every corpus pair runs raw, both Native entrypoints and public APIs.
            tensor_case(backend, &lhs, &rhs, &[chunk.len()], chunk, counts, observed)?;
            counts.corpus_rows += chunk.len();
            counts.corpus_batches += 1;
        }
        for fixture in oracle::scalar_fixtures(dtype) {
            for (batch, inputs) in fixture.inputs.chunks(CHUNK).enumerate() {
                let start = batch * CHUNK;
                let end = start + inputs.len();
                let input = Input::flat(dtype, format, inputs.to_vec())?;
                let math = &fixture.primitive[start..end];
                raw(backend, [&input, &input], &[inputs.len()], Some(fixture.scalar_bits), math, counts, observed)?;
                native(backend, [&input, &input], &[inputs.len()], Some(fixture.scalar), math, counts, observed)?;
                public_scalar(backend, &input, fixture.scalar, oracle::Route::ScalarImpl, math, counts, observed)?;
                public_scalar(
                    backend,
                    &input,
                    fixture.scalar,
                    oracle::Route::PowfScalar,
                    &fixture.wrapper[start..end],
                    counts,
                    observed,
                )?;
                public_scalar(
                    backend,
                    &input,
                    fixture.scalar,
                    oracle::Route::PublicPowfScalar,
                    &fixture.public_wrapper[start..end],
                    counts,
                    observed,
                )?;
                if fixture.valid_powi {
                    public_scalar(backend, &input, fixture.scalar, oracle::Route::PowiScalarImpl, math, counts, observed)?;
                    public_scalar(
                        backend,
                        &input,
                        fixture.scalar,
                        oracle::Route::PowiScalar,
                        &fixture.wrapper[start..end],
                        counts,
                        observed,
                    )?;
                }
                if fixture.public_powi_valid {
                    public_scalar(
                        backend,
                        &input,
                        fixture.scalar,
                        oracle::Route::PublicPowiScalar,
                        &fixture.public_powi_wrapper[start..end],
                        counts,
                        observed,
                    )?;
                }
            }
        }
    }
    Ok(())
}
fn geometry_rows(dtype: DType) -> Vec<Reference> {
    let rows = oracle::rows(dtype);
    rows.iter().step_by((rows.len() / 1024).max(1)).take(1024).copied().collect()
}
fn broadcast_cases(dtype: DType, format: Format) -> Result<Vec<(Input, Input, Vec<Reference>)>> {
    let fixtures = oracle::scalar_fixtures(dtype);
    let bits = |n| {
        fixtures
            .iter()
            .find(|f| matches!(f.scalar,Scalar::Int(v) if v==n))
            .map(|f| f.scalar_bits)
            .ok_or("typed power broadcast scalar missing")
    };
    let cube = bits(3)?;
    let lhs = vec![bits(-2)?, bits(-1)?, bits(2)?];
    let rhs = vec![bits(-1)?, bits(2)?, bits(3)?];
    let mut a = Input::flat(dtype, format, lhs.clone())?;
    let mut b = Input::flat(dtype, format, vec![cube; 6])?;
    a.layout = gpu(Layout::contiguous(Shape::new([1, 3])))?;
    b.layout = gpu(Layout::contiguous(Shape::new([2, 3])))?;
    let expected_left = references(dtype, Schedule::Mathematical, &lhs.repeat(2), &[cube; 6])?;
    let mut c = Input::flat(dtype, format, vec![lhs[0]; 6])?;
    let mut d = Input::flat(dtype, format, rhs.clone())?;
    c.layout = gpu(Layout::contiguous(Shape::new([2, 3])))?;
    d.layout = gpu(Layout::contiguous(Shape::new([1, 3])))?;
    let expected_right = references(dtype, Schedule::Mathematical, &[lhs[0]; 6], &rhs.repeat(2))?;
    Ok(vec![(a, b, expected_left), (c, d, expected_right)])
}
fn layouts(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for (dtype, format) in FORMATS {
        let rows = geometry_rows(dtype);
        let left: Vec<_> = rows.iter().map(|r| r.lhs).collect();
        let right: Vec<_> = rows.iter().map(|r| r.rhs).collect();
        for (a, b) in geometry_inputs(dtype, format, &left)?.into_iter().zip(geometry_inputs(dtype, format, &right)?) {
            let rows = references(dtype, Schedule::Mathematical, &a.logical, &b.logical)?;
            tensor_case(backend, &a, &b, a.layout.shape().as_slice(), &rows, counts, observed)?;
            counts.geometry += 1;
        }
        for exponent in [0, 1, 2, -1, -2, 3] {
            let fixture = oracle::scalar_fixtures(dtype)
                .into_iter()
                .find(|s| matches!(s.scalar, Scalar::Int(n) if n == exponent))
                .ok_or("typed power missing scalar geometry fixture")?;
            for input in geometry_inputs(dtype, format, &fixture.inputs)? {
                scalar_case(backend, &input, fixture.scalar, fixture.scalar_bits, counts, observed)?;
                counts.geometry += 1;
            }
        }
        for (a, b, rows) in broadcast_cases(dtype, format)? {
            tensor_case(backend, &a, &b, &[2, 3], &rows, counts, observed)?;
            counts.geometry += 1;
        }
    }
    Ok(())
}
fn malformed(p: [usize; 32], scalar: [usize; 32], dtype: DType) -> Vec<[usize; 32]> {
    let mut cases = Vec::new();
    for (field, value) in [
        (0, 0),
        (0, i32::MAX as usize + 1),
        (1, 0),
        (1, 9),
        (3, 1),
        (4, 5),
        (5, 5),
        (6, 0),
        (7, 0),
        (8, 6),
        (9, 2),
        (16, i32::MAX as usize),
        (16, u32::MAX as usize),
        (17, 1),
        (25, 1),
    ] {
        let mut bad = p;
        bad[field] = value;
        cases.push(bad);
    }
    for bit in (16..24).chain(25..32) {
        let mut bad = p;
        bad[2] |= 1 << bit;
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
    for field in [7, 24, 31] {
        let mut bad = scalar;
        bad[field] = 1;
        cases.push(bad);
    }
    let mut bad = scalar;
    bad[2] = (bad[2] & !255) | Elementwise::Negate as usize;
    cases.push(bad);
    if dtype.size() < 8 {
        let mut bad = scalar;
        bad[5] = 1;
        cases.push(bad);
        if dtype.size() == 2 {
            let mut bad = scalar;
            bad[3] |= 1 << 16;
            cases.push(bad);
        }
    }
    cases
}
/// Invalid requests must fail recoverably, including those sent to the driver.
fn rejected<T>(result: d3os_compute::Result<T>, context: &str) -> Result<()> {
    match result {
        Err(Error::InvalidParameter | Error::InvalidShape | Error::ShapeMismatch | Error::SizeOverflow) => Ok(()),
        Err(error) => Err(format!("typed power {context} returned non-validation error: {error}")),
        Ok(_) => Err(format!("typed power {context} accepted")),
    }
}
fn raw_rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let input = Input::flat(dtype, format, vec![0; 5])?;
        let tensor = parameters(&input, &input, &[5], None)?;
        let scalar = parameters(&input, &input, &[5], Some(0))?;
        let count = packed(dtype, &input.storage).len();
        let words = vec![POISON; count + 4];
        let a = gpu(backend.upload_words(&words))?;
        let b = gpu(backend.upload_words(&words))?;
        let out = gpu(backend.upload_words(&words))?;
        let short = gpu(backend.upload_words(&[POISON]))?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let queue = gpu(backend.queue_profile())?;
        for bad in malformed(tensor, scalar, dtype) {
            rejected(
                backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&out, 1)], &bad),
                "raw malformed",
            )?;
            rejected(
                backend.produce_words(count, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &bad),
                "fresh malformed",
            )?;
            counts.rejected += 2;
        }
        for p in [tensor, scalar] {
            for bad in [p[..31].to_vec(), {
                let mut v = p.to_vec();
                v.push(0);
                v
            }] {
                rejected(
                    backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&out, 1)], &bad),
                    "raw parameter arity",
                )?;
                rejected(
                    backend.produce_words(count, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &bad),
                    "fresh parameter arity",
                )?;
                counts.rejected += 2;
            }
            for buffers in [vec![(&a, 1), (&out, 1)], vec![(&a, 1), (&b, 1), (&out, 1), (&short, 0)]] {
                rejected(backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p), "raw pointer arity")?;
                counts.rejected += 1;
            }
            for inputs in [vec![(&a, 1)], vec![(&a, 1), (&b, 1), (&short, 0)]] {
                rejected(backend.produce_words(count, Kernel::FloatArithmetic, &inputs, &p), "fresh pointer arity")?;
                counts.rejected += 1;
            }
            for length in [0, count - 1, count + 1] {
                rejected(
                    backend.produce_words(length, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &p),
                    "fresh output word count",
                )?;
                counts.rejected += 1;
            }
            for (target, offset) in [(&a, 1), (&a, 2), (&b, 1), (&b, 0)] {
                rejected(
                    backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (target, offset)], &p),
                    "raw output alias",
                )?;
                counts.rejected += 1;
            }
            for buffers in [
                [(&short, 0), (&b, 1), (&out, 1)],
                [(&a, words.len()), (&b, 1), (&out, 1)],
                [(&a, 1), (&b, words.len()), (&out, 1)],
                [(&a, 1), (&b, 1), (&short, 0)],
                [(&a, 1), (&b, 1), (&out, words.len())],
            ] {
                rejected(backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p), "raw word bounds")?;
                counts.rejected += 1;
            }
            for inputs in [[(&short, 0), (&b, 1)], [(&a, words.len()), (&b, 1)], [(&a, 1), (&b, words.len())]] {
                rejected(backend.produce_words(count, Kernel::FloatArithmetic, &inputs, &p), "fresh word bounds")?;
                counts.rejected += 1;
            }
            if p[7] != 0 {
                rejected(
                    backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&short, 0), (&out, 1)], &p),
                    "raw RHS bounds",
                )?;
                rejected(
                    backend.produce_words(count, Kernel::FloatArithmetic, &[(&a, 1), (&short, 0)], &p),
                    "fresh RHS bounds",
                )?;
                counts.rejected += 2;
            }
            if dtype == DType::F64 {
                let one = Input::flat(dtype, format, vec![0])?;
                let one = parameters(&one, &one, &[1], if p[7] == 0 { Some(0) } else { None })?;
                for buffers in [[(&short, 0), (&b, 1), (&out, 1)], [(&a, 1), (&b, 1), (&short, 0)]] {
                    rejected(backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &one), "raw F64 second word")?;
                    counts.rejected += 1;
                }
                rejected(
                    backend.produce_words(2, Kernel::FloatArithmetic, &[(&short, 0), (&b, 1)], &one),
                    "fresh F64 second word",
                )?;
                counts.rejected += 1;
                if p[7] != 0 {
                    rejected(
                        backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&short, 0), (&out, 1)], &one),
                        "raw F64 RHS second word",
                    )?;
                    rejected(
                        backend.produce_words(2, Kernel::FloatArithmetic, &[(&a, 1), (&short, 0)], &one),
                        "fresh F64 RHS second word",
                    )?;
                    counts.rejected += 2;
                }
                for (offset, extent) in if p[7] == 0 { &[(4, 6)][..] } else { &[(4, 6), (5, 7)][..] } {
                    let mut high = p;
                    high[*offset] = i32::MAX as usize - 5;
                    high[*extent] = i32::MAX as usize;
                    rejected(
                        backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&out, 1)], &high),
                        "raw F64 high word bounds",
                    )?;
                    rejected(
                        backend.produce_words(count, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &high),
                        "fresh F64 high word bounds",
                    )?;
                    counts.rejected += 2;
                }
            }
        }
        // Only address-alias probes reach the service. They record no GPU work.
        let after = gpu(backend.service_profile())?;
        for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
            if after[service as usize].calls - services[service as usize].calls != if service == abi::LAUNCH { 8 } else { 0 } {
                return Err(format!("typed power rejected service={service}"));
            }
        }
        check_kernel_deltas(&profile, &gpu(backend.profile())?, &[0; abi::KERNEL_COUNT])?;
        if gpu(backend.queue_profile())?.launches != queue.launches {
            return Err(String::from("typed power rejected request recorded GPU work"));
        }
        unchanged(backend, &a, &words)?;
        unchanged(backend, &b, &words)?;
        unchanged(backend, &out, &words)?;
        unchanged(backend, &short, &[POISON])?;
    }
    Ok(())
}
fn acknowledge(failed: FloatTensor<Native>, dtype: DType, shape: &[usize]) -> Result<()> {
    if failed.dtype() != dtype || failed.shape().as_slice() != shape {
        return Err(String::from("typed power failed Native metadata changed"));
    }
    if !Native::sync(&NativeDevice).is_err_and(|e| !e.is_device_poisoned()) {
        return Err(String::from("typed power missing recoverable Native sync error"));
    }
    let error = try_read_sync(Native::float_into_data(failed)).ok_or("typed power failed Native read pending")?;
    if !error.is_err_and(|e| !e.is_device_poisoned()) || Native::sync(&NativeDevice).is_err() {
        return Err(String::from("typed power acknowledgment poisoned Native device"));
    }
    Ok(())
}
fn acknowledge_dispatch(failed: FloatTensor<Dispatch>, dtype: DType, shape: &[usize]) -> Result<()> {
    if failed.dtype() != dtype || failed.shape().as_slice() != shape {
        return Err(String::from("typed power failed Dispatch metadata changed"));
    }
    let device = d3os_ai::native_device().map_err(|e| format!("typed power error device: {e}"))?;
    if !device.sync().is_err_and(|e| !e.is_device_poisoned()) {
        return Err(String::from("typed power missing recoverable Dispatch sync error"));
    }
    let error = try_read_sync(Dispatch::float_into_data(failed)).ok_or("typed power failed Dispatch read pending")?;
    if !error.is_err_and(|e| !e.is_device_poisoned()) || device.sync().is_err() {
        return Err(String::from("typed power acknowledgment poisoned Dispatch device"));
    }
    Ok(())
}
fn invalid_powi(backend: &Nvidia, input: &Input, scalar: Scalar, public: bool, counts: &mut Counts) -> Result<()> {
    for route in if public {
        &[oracle::Route::PublicPowiScalar][..]
    } else {
        &[oracle::Route::PowiScalarImpl, oracle::Route::PowiScalar][..]
    } {
        for dispatch in [false, true] {
            if public && !dispatch {
                continue;
            }
            let (_, source) = input.primitive();
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            if dispatch {
                let failed = public_scalar_launch(source, scalar, *route, input.layout.shape().num_dims())?;
                no_work(backend, &services, &profile)?;
                acknowledge_dispatch(failed, input.dtype, input.layout.shape().as_slice())?;
            } else {
                let failed = scalar_launch(source, scalar, *route);
                no_work(backend, &services, &profile)?;
                acknowledge(failed, input.dtype, input.layout.shape().as_slice())?;
            }
            no_work(backend, &services, &profile)?;
            counts.rejected += 1;
        }
    }
    Ok(())
}
fn native_validation(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        for length in [0, 3] {
            let input = Input::flat(dtype, format, vec![0; length])?;
            let source = input.tensor()?;
            let wrong = gpu(NativeTensor::from_data(data(
                if dtype == DType::F32 { DType::Flex32 } else { DType::F32 },
                &[length],
                &vec![0; length],
            )))?;
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            for result in [
                source.float_unary(Elementwise::Power),
                source.unary(Elementwise::Power),
                source.float_binary(&wrong, Elementwise::Power),
                source.binary(&wrong, Elementwise::Power),
            ] {
                rejected(result, "Native invalid dtype/arity")?;
                counts.rejected += 1;
            }
            no_work(backend, &services, &profile)?;
            for fixture in oracle::scalar_fixtures(dtype) {
                if !fixture.valid_powi {
                    invalid_powi(backend, &input, fixture.scalar, false, counts)?;
                }
                if !fixture.public_powi_valid {
                    invalid_powi(backend, &input, fixture.scalar, true, counts)?;
                }
            }
        }
    }
    for length in [0, 3] {
        let source = gpu(NativeTensor::from_data(data(DType::I32, &[length], &vec![0; length])))?;
        let primitive = Native::int_from_data(data(DType::I32, &[length], &vec![0; length]), &NativeDevice);
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        for result in [
            source.float_binary(&source, Elementwise::Power),
            source.binary(&source, Elementwise::Power),
            source.float_scalar(Elementwise::Power, Scalar::Float(3.0)),
        ] {
            rejected(result, "Native integer Power storage")?;
            counts.rejected += 1;
        }
        // Inspect the failed I32 primitive directly. Wrapping it in Tensor<Float>
        // would panic on dtype validation before its deferred error is acknowledged.
        for route in [
            oracle::Route::ScalarImpl,
            oracle::Route::PowiScalarImpl,
            oracle::Route::PowfScalar,
            oracle::Route::PowiScalar,
        ] {
            let failed = scalar_launch(primitive.clone(), Scalar::Int(3), route);
            no_work(backend, &services, &profile)?;
            acknowledge(failed, DType::I32, &[length])?;
            counts.rejected += 1;
        }
        let failed = Native::float_powf(primitive.clone(), primitive);
        no_work(backend, &services, &profile)?;
        acknowledge(failed, DType::I32, &[length])?;
        counts.rejected += 1;
        no_work(backend, &services, &profile)?;
    }
    for (shape, strides, offset, extent) in [(vec![1; 9], vec![1; 9], 0, 1), (vec![3], vec![-1], 1, 3), (vec![3], vec![1], 1, 3)] {
        if Layout::from_parts(Shape::from(shape), strides, offset, extent).is_ok() {
            return Err(String::from("typed power invalid layout accepted"));
        }
        counts.rejected += 1;
    }
    Ok(())
}
fn empties(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        for shape in [vec![0], vec![2, 0, 7], vec![0, 2, 3], vec![1, 1, 1, 1, 0, 1, 1, 3]] {
            let mut input = Input::flat(dtype, format, vec![])?;
            input.layout = gpu(Layout::contiguous(Shape::from(shape.clone())))?;
            let empty = gpu(NativeTensor::empty(Shape::from(shape.clone()), dtype))?;
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            for output in [
                gpu(empty.float_binary(&empty, Elementwise::Power))?,
                gpu(empty.binary(&empty, Elementwise::Power))?,
                gpu(empty.float_scalar(Elementwise::Power, Scalar::Float(3.0)))?,
            ] {
                source_data(&gpu(output.read())?, dtype, &shape, &[])?;
                counts.empty += 1;
            }
            no_work(backend, &services, &profile)?;
            public_tensor(backend, &input, &input, &shape, &[], counts, &mut Observed::default())?;
            // Public empty launches have no output writer, regardless of shortcut.
            counts.empty += 4;
            for scalar in [
                Scalar::Int(0),
                Scalar::Int(1),
                Scalar::Int(2),
                Scalar::Int(-1),
                Scalar::Int(-2),
                Scalar::Float(3.0),
                Scalar::UInt(u64::MAX),
                Scalar::Float(f64::MAX),
            ] {
                for route in [
                    oracle::Route::ScalarImpl,
                    oracle::Route::PowiScalarImpl,
                    oracle::Route::PowfScalar,
                    oracle::Route::PowiScalar,
                    oracle::Route::PublicPowfScalar,
                    oracle::Route::PublicPowiScalar,
                ] {
                    for dispatch in [false, true] {
                        if matches!(route, oracle::Route::PublicPowfScalar | oracle::Route::PublicPowiScalar) && !dispatch {
                            continue;
                        }
                        let (_, source) = input.primitive();
                        let services = gpu(backend.service_profile())?;
                        let profile = gpu(backend.profile())?;
                        (|| {
                            let actual = if dispatch {
                                dispatch_data(public_scalar_launch(source, scalar, route, shape.len())?)?
                            } else {
                                primitive_data(scalar_launch(source, scalar, route))?
                            };
                            source_data(&actual, dtype, &shape, &[])?;
                            let (expected_services, expected_kernels) = empty_scalar_work(route, scalar);
                            check_service_deltas(&services, &gpu(backend.service_profile())?, &expected_services)?;
                            check_kernel_deltas(&profile, &gpu(backend.profile())?, &expected_kernels)
                        })()
                        .map_err(|e| format!("typed power empty dtype={dtype:?} shape={shape:?} dispatch={dispatch} route={route:?} scalar={scalar:?}: {e}"))?;
                        counts.empty += 1;
                    }
                }
            }
        }
    }
    Ok(())
}
fn lifetimes(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for (dtype, format) in FORMATS {
        let rows: Vec<_> = geometry_rows(dtype).into_iter().step_by(23).take(17).collect();
        if rows.len() != 17 {
            return Err(String::from("typed power missing queued rows"));
        }
        let lhs = Input::flat(dtype, format, rows.iter().map(|r| r.lhs).collect())?;
        let rhs = Input::flat(dtype, format, rows.iter().map(|r| r.rhs).collect())?;
        for scalar in [false, true] {
            let fixture = oracle::scalar_fixtures(dtype)
                .into_iter()
                .find(|f| matches!(f.scalar, Scalar::Int(3)))
                .ok_or("typed power queued raw scalar missing")?;
            let scalar_bits = fixture.scalar_bits;
            let scalar_input = Input::flat(dtype, format, fixture.inputs[..17].to_vec())?;
            let source = if scalar { &scalar_input } else { &lhs };
            let rows = if scalar { fixture.primitive[..17].to_vec() } else { rows.clone() };
            let p = parameters(source, &rhs, &[17], scalar.then_some(scalar_bits))?;
            let mut bytes = None;
            for alternate in [false, true] {
                let a = gpu(backend.upload_words(&packed(dtype, &source.storage)))?;
                let b = gpu(backend.upload_words(&if scalar { vec![POISON] } else { packed(dtype, &rhs.storage) }))?;
                let dirty = gpu(backend.upload_words(&output_poison(dtype, &rows, alternate)))?;
                drop(dirty);
                let services = gpu(backend.service_profile())?;
                let profile = gpu(backend.profile())?;
                let output = gpu(backend.with_commands(|| {
                    let out = backend.produce_words((17 * dtype.size()).div_ceil(4), Kernel::FloatArithmetic, &[(&a, 0), (&b, 0)], &p)?;
                    drop(a);
                    drop(b);
                    Ok::<_, Error>(out)
                }))?;
                let mut plan = [0; abi::KERNEL_COUNT];
                plan[Kernel::FloatArithmetic as usize] = 1;
                resident(backend, &services, &profile, &plan, true)?;
                identical_raw_bytes(
                    &mut bytes,
                    output_words(
                        backend,
                        &output,
                        dtype,
                        &rows,
                        false,
                        (if scalar { oracle::Route::ScalarImpl } else { oracle::Route::Tensor }, "queued.raw"),
                        observed,
                    )?,
                )?;
                counts.fresh += 1;
                counts.queued += 1;
            }
        }
        let a = lhs.tensor()?;
        let b = rhs.tensor()?;
        let alias = a.clone();
        if a.can_mut() || alias.can_mut() {
            return Err(String::from("typed power retained Native alias claims unique storage"));
        }
        drop(alias);
        if !a.can_mut() || !b.can_mut() {
            return Err(String::from("typed power last Native owners claim shared storage"));
        }
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let out = gpu(backend.with_commands(|| {
            let out = a.float_binary(&b, Elementwise::Power)?;
            drop(a);
            drop(b);
            Ok::<_, Error>(out)
        }))?;
        resident(backend, &services, &profile, &launch_plan(dtype, Schedule::Mathematical, false), true)?;
        output_data(&gpu(out.read())?, dtype, &[17], &rows, oracle::Route::Tensor, observed, "queued.Native")?;
        counts.native += 1;
        counts.queued += 1;
        let (abase, a) = lhs.primitive();
        let (bbase, b) = rhs.primitive();
        drop(abase);
        drop(bbase);
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let out = gpu(backend.with_commands(|| Ok::<_, Error>(Native::float_powf(a, b))))?;
        resident(backend, &services, &profile, &launch_plan(dtype, Schedule::Mathematical, false), true)?;
        output_data(&primitive_data(out)?, dtype, &[17], &rows, oracle::Route::Tensor, observed, "queued.wrapper")?;
        counts.public += 1;
        counts.queued += 1;
        for exponent in [0, 1, 2, -1, -2, 3] {
            let fixture = oracle::scalar_fixtures(dtype)
                .into_iter()
                .find(|f| matches!(f.scalar,Scalar::Int(n) if n==exponent))
                .ok_or("typed power queued scalar missing")?;
            let rows = &fixture.wrapper[..17];
            let input = Input::flat(dtype, format, rows.iter().map(|r| r.lhs).collect())?;
            for dispatch in [false, true] {
                let (base, source) = input.primitive();
                drop(base);
                if !source.can_mut() {
                    return Err(String::from("typed power queued scalar not last owner"));
                }
                let services = gpu(backend.service_profile())?;
                let profile = gpu(backend.profile())?;
                let actual = if dispatch {
                    let source = Tensor::<1>::from_primitive::<Native>(source).into_dispatch();
                    let out = gpu(backend.with_commands(|| Ok::<_, Error>(Dispatch::float_powf_scalar(source, fixture.scalar))))?;
                    resident(backend, &services, &profile, &launch_plan(dtype, rows[0].schedule, false), true)?;
                    dispatch_data(out)?
                } else {
                    let out = gpu(backend.with_commands(|| Ok::<_, Error>(Native::float_powf_scalar(source, fixture.scalar))))?;
                    resident(backend, &services, &profile, &launch_plan(dtype, rows[0].schedule, false), true)?;
                    primitive_data(out)?
                };
                output_data(&actual, dtype, &[17], rows, oracle::Route::PowfScalar, observed, "queued.scalar")?;
                if rows[0].schedule == Schedule::Identity {
                    source_data(&actual, dtype, &[17], &input.logical)?;
                }
                counts.public += 1;
                counts.queued += 1;
            }
        }
    }
    Ok(())
}
fn integer_tensors(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    for (dtype, _) in FORMATS {
        for fixture in oracle::integer_fixtures(dtype) {
            for (chunk, rows) in fixture.rows.chunks(CHUNK).enumerate() {
                let lhs: Vec<_> = rows.iter().map(|r| r.lhs).collect();
                let exponents = vec![fixture.exponent as u64; rows.len()];
                for dispatch in [false, true] {
                    let a = Native::float_from_data(data(dtype, &[rows.len()], &lhs), &NativeDevice);
                    let b = Native::int_from_data(data(DType::I64, &[rows.len()], &exponents), &NativeDevice);
                    let aa = a.clone();
                    let ba = b.clone();
                    let services = gpu(backend.service_profile())?;
                    let profile = gpu(backend.profile())?;
                    let result = (|| {
                        let actual = if dispatch {
                            let a = Tensor::<1>::from_primitive::<Native>(a).into_dispatch();
                            let b = Tensor::<1, d3os_ai::burn::tensor::Int>::from_primitive::<Native>(b).into_dispatch();
                            let out = Dispatch::float_powi(a, b);
                            resident(backend, &services, &profile, &launch_plan(dtype, Schedule::Mathematical, true), true)?;
                            dispatch_data(out)?
                        } else {
                            let out = Native::float_powi(a, b);
                            resident(backend, &services, &profile, &launch_plan(dtype, Schedule::Mathematical, true), true)?;
                            primitive_data(out)?
                        };
                        output_data(
                            &actual,
                            dtype,
                            &[rows.len()],
                            rows,
                            oracle::Route::TensorPowi,
                            observed,
                            "backend.powi.cast_then_powf",
                        )?;
                        source_data(&primitive_data(aa)?, dtype, &[rows.len()], &lhs)?;
                        let integers = try_read_sync(Native::int_into_data(ba))
                            .ok_or("typed power integer read pending")?
                            .map_err(|e| format!("typed power preserved integer source: {e:?}"))?;
                        source_data(&integers, DType::I64, &[rows.len()], &exponents)
                    })();
                    result.map_err(|e| format!(
                        "typed power integer_tensor dtype={dtype:?} dispatch={dispatch} route=TensorPowi cast=I64->{dtype:?} exponent={} stored={:#x} chunk={chunk} length={}: {e}",
                        fixture.exponent, fixture.stored, rows.len(),
                    ))?;
                    counts.public += 1;
                }
            }
        }
    }
    Ok(())
}
fn contract() -> Result<()> {
    if !FROZEN || POLICY == "UNFROZEN" || FIXTURE == "UNFROZEN" {
        return Err(String::from(
            "typed power UNFROZEN: parent numerical policy and fixture freeze required before GPU work",
        ));
    }
    if abi::VERSION != 49
        || abi::KERNEL_COUNT != 82
        || Kernel::FloatArithmetic as u32 != 81
        || float::Operation::ALL.len() != 24
        || float::Operation::Power as u32 != 29
        || float::PARAMETERS != 32
        || oracle::DTYPES != FORMATS.map(|(dtype, _)| dtype)
        || oracle::CORPUS_VERSION != 1
        || oracle::POLICY_SHA256 != POLICY
        || oracle::FIXTURE_SHA256 != FIXTURE
    {
        return Err(String::from("typed power protocol/corpus contract changed"));
    }
    Ok(())
}
fn exercise(backend: &Nvidia, counts: &mut Counts, observed: &mut Observed) -> Result<()> {
    fn stage(name: &str, result: Result<()>) -> Result<()> {
        result.map_err(|e| format!("typed power stage={name}: {e}"))
    }
    stage("corpus", corpus(backend, counts, observed))?;
    stage("layouts", layouts(backend, counts, observed))?;
    stage("integer_tensors", integer_tensors(backend, counts, observed))?;
    stage("raw_rejections", raw_rejections(backend, counts))?;
    stage("native_validation", native_validation(backend, counts))?;
    stage("empties", empties(backend, counts))?;
    // Valid queued work proves that all deferred error roots were acknowledged.
    stage("lifetimes", lifetimes(backend, counts, observed))
}
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    // The freeze gate runs before profiles, uploads, device opening or dispatch.
    contract()?;
    let plan = reference_plan();
    let expected = EXPECTED_COUNTS;
    if expected_counts() != expected || plan.len() != CELLS || plan.iter().map(|c| c.values).sum::<usize>() != COMPARISONS {
        return Err(String::from("typed power frozen CPU plan changed before GPU work"));
    }
    let mut counts = Counts::default();
    let mut observed = Observed::default();
    crate::report(format_args!("NVIDIA_BURN_TYPED_POWER_REFERENCE_PLAN {}", reference_plan_json(&plan)));
    let result = exercise(backend, &mut counts, &mut observed);
    let compared: usize = observed.accuracy.cells.iter().map(|c| c.accuracy.values).sum();
    crate::report(format_args!(
        "NVIDIA_BURN_TYPED_POWER_ACCURACY policy_sha256={POLICY} fixture_sha256={FIXTURE} qualification_passed={} totals={} accuracy={}",
        result.is_ok() && counts == expected && complete_cells(&observed.accuracy, &plan) && observed.accuracy.is_coherent() && observed.accuracy.failures == 0,
        reference_plan_json(&plan),
        oracle::accuracy_json(&observed.accuracy),
    ));
    for example in &observed.examples {
        crate::report(format_args!("NVIDIA_BURN_TYPED_POWER_FAILURE {example}"));
    }
    result?;
    if counts != expected {
        return Err(format!("typed power structural counts {counts:?}, expected {expected:?}"));
    }
    if !complete_cells(&observed.accuracy, &plan) || !observed.accuracy.is_coherent() || observed.accuracy.failures != 0 {
        return Err(format!("typed power numerical qualification failures={}", observed.accuracy.failures));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_TYPED_POWER_OK formats=5 power=29 sdk_operations=24 protocol=49 profiles=82 kernel=81 chunk=4096 raw_poison_variants=2 comparisons={compared} counts={counts:?} policy_sha256={POLICY} fixture_sha256={FIXTURE} qualification=project_targets linux_parity=not_claimed autodiff=outside_scope backend=nvidia",
    ));
    Ok(())
}

/// Expected observations only. This never opens a device or records Accuracy.
fn plan_record(cells: &mut Vec<oracle::ReferenceCell>, dtype: DType, route: oracle::Route, rows: &[Reference], repetitions: usize) {
    for row in rows {
        let index = cells
            .iter()
            .position(|c| c.dtype == dtype && c.route == route && c.schedule == row.schedule && !c.chain)
            .unwrap_or_else(|| {
                cells.push(oracle::ReferenceCell {
                    dtype,
                    route,
                    schedule: row.schedule,
                    chain: false,
                    values: 0,
                    reference_classes: [0; 5],
                    ideal_classes: [0; 5],
                });
                cells.len() - 1
            });
        let cell = &mut cells[index];
        cell.values += repetitions;
        cell.reference_classes[oracle::classify(dtype, row.staged) as usize] += repetitions;
        cell.ideal_classes[oracle::classify(dtype, row.ideal) as usize] += repetitions;
    }
}
/// Coordinate enumeration independent of Layout and the runtime view recipes.
fn reference_indices() -> Vec<Vec<usize>> {
    let mut indices = Vec::new();
    for n in [1, 2, 3, 127, 128, 129, 255, 256, 257] {
        indices.push((0..n).collect());
        indices.push((0..n).map(|i| 1 + 2 * i).collect());
        indices.push((0..n).map(|i| 2 * n - 1 - 2 * i).collect());
    }
    indices.push((1..6).collect());
    indices.push((0..7).rev().flat_map(|col| (0..3).rev().map(move |row| row * 7 + col)).collect());
    indices.push(vec![10, 12, 14, 28, 30, 32]);
    for n in [1, 24, 3, 3, 3, 3, 3] {
        indices.push((0..n).collect());
    }
    indices.push(vec![1; 6]);
    indices
}
fn reference_plan() -> Vec<oracle::ReferenceCell> {
    use oracle::Route;
    let mut cells = Vec::new();
    for (dtype, format) in FORMATS {
        let rows = oracle::rows(dtype);
        plan_record(&mut cells, dtype, Route::Tensor, &rows, 8);
        plan_record(&mut cells, dtype, Route::PublicTensor, &rows, 2);
        let fixtures = oracle::scalar_fixtures(dtype);
        for f in &fixtures {
            plan_record(&mut cells, dtype, Route::ScalarImpl, &f.primitive, 7);
            plan_record(&mut cells, dtype, Route::PowfScalar, &f.wrapper, 2);
            plan_record(&mut cells, dtype, Route::PublicPowfScalar, &f.public_wrapper, 1);
            if f.valid_powi {
                plan_record(&mut cells, dtype, Route::PowiScalarImpl, &f.primitive, 2);
                plan_record(&mut cells, dtype, Route::PowiScalar, &f.wrapper, 2);
            }
            if f.public_powi_valid {
                plan_record(&mut cells, dtype, Route::PublicPowiScalar, &f.public_powi_wrapper, 1);
            }
        }
        let pool = geometry_rows(dtype);
        for indices in reference_indices() {
            let rows: Vec<_> = indices.iter().map(|&i| pool[i % pool.len()]).collect();
            plan_record(&mut cells, dtype, Route::Tensor, &rows, 8);
            plan_record(&mut cells, dtype, Route::PublicTensor, &rows, 2);
            for exponent in [0, 1, 2, -1, -2, 3] {
                let f = fixtures
                    .iter()
                    .find(|f| matches!(f.scalar,Scalar::Int(n) if n==exponent))
                    .expect("geometry scalar reference");
                let math: Vec<_> = indices.iter().map(|&i| f.primitive[i % f.primitive.len()]).collect();
                let wrapper: Vec<_> = indices.iter().map(|&i| f.wrapper[i % f.wrapper.len()]).collect();
                plan_record(&mut cells, dtype, Route::ScalarImpl, &math, 7);
                plan_record(&mut cells, dtype, Route::PowfScalar, &wrapper, 2);
            }
        }
        for (_, _, rows) in broadcast_cases(dtype, format).expect("broadcast reference plan") {
            plan_record(&mut cells, dtype, Route::Tensor, &rows, 8);
            plan_record(&mut cells, dtype, Route::PublicTensor, &rows, 2);
        }
        for fixture in oracle::integer_fixtures(dtype) {
            plan_record(&mut cells, dtype, Route::TensorPowi, &fixture.rows, 2);
        }
        let queued: Vec<_> = pool.iter().step_by(23).take(17).copied().collect();
        plan_record(&mut cells, dtype, Route::Tensor, &queued, 4);
        let cube = fixtures.iter().find(|f| matches!(f.scalar, Scalar::Int(3))).expect("queued scalar reference");
        plan_record(&mut cells, dtype, Route::ScalarImpl, &cube.primitive[..17], 2);
        for exponent in [0, 1, 2, -1, -2, 3] {
            let f = fixtures
                .iter()
                .find(|f| matches!(f.scalar,Scalar::Int(n) if n==exponent))
                .expect("queued wrapper reference");
            plan_record(&mut cells, dtype, Route::PowfScalar, &f.wrapper[..17], 2);
        }
    }
    cells
}
fn expected_counts() -> Counts {
    let mut plan = Counts::default();
    let views = reference_indices().len();
    for (dtype, _) in FORMATS {
        let n = oracle::rows(dtype).len();
        let batches = n.div_ceil(CHUNK);
        plan.corpus_rows += n;
        plan.corpus_batches += batches;
        plan.raw += 2 * batches;
        plan.fresh += 2 * batches;
        plan.native += 2 * batches;
        plan.public += 4 * batches;
        for f in oracle::scalar_fixtures(dtype) {
            let batches = f.inputs.len().div_ceil(CHUNK);
            plan.raw += 2 * batches;
            plan.fresh += 2 * batches;
            plan.native += batches;
            plan.public += (5 + 4 * usize::from(f.valid_powi) + usize::from(f.public_powi_valid)) * batches;
            plan.rejected += 2 * (4 * usize::from(!f.valid_powi) + usize::from(!f.public_powi_valid));
        }
        let tensors = views + 2;
        let scalars = 6 * views;
        plan.geometry += tensors + scalars;
        plan.raw += 2 * (tensors + scalars);
        plan.fresh += 2 * (tensors + scalars);
        plan.native += 2 * tensors + scalars;
        plan.public += 4 * (tensors + scalars);
        for f in oracle::integer_fixtures(dtype) {
            plan.public += 2 * f.rows.len().div_ceil(CHUNK);
        }
        let malformed = 40 + usize::from(dtype.size() < 8) + usize::from(dtype.size() == 2);
        plan.rejected += 2 * malformed + 48 + if dtype == DType::F64 { 14 } else { 0 };
        plan.rejected += 8;
        plan.empty += 4 * (3 + 4 + 8 * 10);
        plan.public += 16;
        plan.fresh += 4;
        plan.native += 1;
        plan.public += 13;
        plan.queued += 18;
    }
    plan.rejected += 19;
    plan
}
fn complete_cells(accuracy: &oracle::Accuracy, plan: &[oracle::ReferenceCell]) -> bool {
    accuracy.cells.len() == plan.len()
        && plan.iter().all(|expected| {
            accuracy
                .cells
                .iter()
                .find(|actual| {
                    actual.dtype == expected.dtype && actual.route == expected.route && actual.schedule == expected.schedule && actual.chain == expected.chain
                })
                .is_some_and(|actual| {
                    let a = &actual.accuracy;
                    a.values == expected.values
                        && (0..5).all(|class| {
                            a.transitions[class].iter().sum::<usize>() == expected.reference_classes[class]
                                && a.ideal_transitions[class].iter().sum::<usize>() == expected.ideal_classes[class]
                        })
                })
        })
}
fn reference_plan_json(cells: &[oracle::ReferenceCell]) -> String {
    let rows: Vec<_> = cells
        .iter()
        .map(|c| {
            format!(
                "{{\"dtype\":\"{}\",\"route\":\"{}\",\"schedule\":\"{}\",\"chain\":{},\"values\":{},\"reference_classes\":{:?},\"ideal_classes\":{:?}}}",
                oracle::dtype_label(c.dtype),
                c.route.label(),
                c.schedule.label(),
                c.chain,
                c.values,
                c.reference_classes,
                c.ideal_classes
            )
        })
        .collect();
    let n: usize = cells.iter().map(|c| c.values).sum();
    let c = &EXPECTED_COUNTS;
    format!(
        "{{\"values\":{n},\"nonchain_values\":{n},\"chain_values\":0,\"corpus_rows\":{},\"corpus_batches\":{},\"geometry\":{},\"raw\":{},\"fresh\":{},\"native\":{},\"public\":{},\"rejected\":{},\"empty\":{},\"queued\":{},\"cells\":[{}]}}",
        c.corpus_rows,
        c.corpus_batches,
        c.geometry,
        c.raw,
        c.fresh,
        c.native,
        c.public,
        c.rejected,
        c.empty,
        c.queued,
        rows.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn audit_poison(dtype: DType, rows: &[Reference]) {
        let poisons = [output_poison(dtype, rows, false), output_poison(dtype, rows, true)];
        assert_eq!(poisons[0].len(), (rows.len() * dtype.size()).div_ceil(4));
        assert!(poisons[0].iter().zip(&poisons[1]).all(|(a, b)| a != b));
        let bytes = poisons.map(|words| words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>());
        // Include physical padding and both words of every F64 output.
        assert!(bytes[0].iter().zip(&bytes[1]).all(|(a, b)| a != b));
        for bytes in bytes {
            if dtype.size() == 2 && rows.len() % 2 == 1 {
                assert_ne!(&bytes[bytes.len() - 2..], &[0, 0]);
            }
            for (row, bits) in rows.iter().zip(oracle::unpack(dtype, &bytes[..rows.len() * dtype.size()])) {
                assert!(
                    !oracle::compare(dtype, row, bits).passed,
                    "accepted poison {dtype:?} lhs={:#x} rhs={:#x} schedule={:?} staged={:#x} poison={bits:#x}",
                    row.lhs,
                    row.rhs,
                    row.schedule,
                    row.staged
                );
            }
        }
    }
    fn audit_sources(input: &Input) {
        let words = source_words(input);
        assert_eq!(words.first(), Some(&GUARD));
        assert_eq!(words.last(), Some(&GUARD));
        let bytes: Vec<_> = words[1..words.len() - 1]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .take(input.storage.len() * input.dtype.size())
            .collect();
        assert_eq!(oracle::unpack(input.dtype, &bytes), input.storage);
        for (linear, &bits) in input.logical.iter().enumerate() {
            assert_eq!(input.storage[input.layout.element_offset(linear).unwrap()], bits);
        }
    }
    #[test]
    fn geometry_coordinates_pairing_and_physical_extents_are_independent() {
        for (dtype, format) in FORMATS {
            let rows = geometry_rows(dtype);
            let left: Vec<_> = rows.iter().map(|r| r.lhs).collect();
            let right: Vec<_> = rows.iter().map(|r| r.rhs).collect();
            let mut ranks = [false; 9];
            for (a, b) in geometry_inputs(dtype, format, &left)
                .unwrap()
                .into_iter()
                .zip(geometry_inputs(dtype, format, &right).unwrap())
            {
                ranks[a.layout.shape().num_dims()] = true;
                audit_sources(&a);
                audit_sources(&b);
                let rows = references(dtype, Schedule::Mathematical, &a.logical, &b.logical).unwrap();
                let p = parameters(&a, &b, a.layout.shape().as_slice(), None).unwrap();
                assert_eq!(
                    float::validate(&p.map(|v| v as u32)).unwrap(),
                    [
                        format.width().words(a.storage.len() as u32),
                        format.width().words(b.storage.len() as u32),
                        format.width().words(rows.len() as u32)
                    ]
                );
                audit_poison(dtype, &rows);
            }
            assert!(ranks.into_iter().all(|covered| covered));
            for fixture in oracle::scalar_fixtures(dtype) {
                assert!(fixture.inputs.len() <= CHUNK);
                for input in geometry_inputs(dtype, format, &fixture.inputs).unwrap() {
                    audit_sources(&input);
                    let rows = references(dtype, Schedule::Mathematical, &input.logical, &vec![fixture.scalar_bits; input.logical.len()]).unwrap();
                    audit_poison(dtype, &rows);
                }
            }
        }
    }
    #[test]
    fn broadcasts_and_queued_last_owner_inputs_have_known_references() {
        for (dtype, format) in FORMATS {
            let cases = broadcast_cases(dtype, format).unwrap();
            assert_eq!(cases.len(), 2);
            for (index, (a, b, rows)) in cases.into_iter().enumerate() {
                assert_eq!(a.layout.shape().as_slice(), if index == 0 { &[1, 3] } else { &[2, 3] });
                assert_eq!(b.layout.shape().as_slice(), if index == 0 { &[2, 3] } else { &[1, 3] });
                let left = a.layout.expand(Shape::new([2, 3])).unwrap();
                let right = b.layout.expand(Shape::new([2, 3])).unwrap();
                for (i, row) in rows.iter().enumerate() {
                    assert_eq!(row.lhs, a.storage[left.element_offset(i).unwrap()]);
                    assert_eq!(row.rhs, b.storage[right.element_offset(i).unwrap()]);
                }
                let variable = if index == 0 { &a.storage } else { &b.storage };
                assert!(variable.windows(2).any(|w| w[0] != w[1]));
                audit_sources(&a);
                audit_sources(&b);
                audit_poison(dtype, &rows);
            }
            let queued: Vec<_> = geometry_rows(dtype).into_iter().step_by(23).take(17).collect();
            assert_eq!(queued.len(), 17);
            audit_poison(dtype, &queued);
            for exponent in [0, 1, 2, -1, -2, 3] {
                let f = oracle::scalar_fixtures(dtype)
                    .into_iter()
                    .find(|f| matches!(f.scalar,Scalar::Int(n) if n==exponent))
                    .unwrap();
                audit_poison(dtype, &f.wrapper[..17]);
            }
        }
    }
    #[test]
    fn immutable_lookup_and_poison_cover_every_corpus_and_scalar_schedule() {
        let mut accuracy = oracle::Accuracy::default();
        for (dtype, _) in FORMATS {
            let rows = oracle::rows(dtype);
            assert!(rows.windows(2).all(|w| (w[0].lhs, w[0].rhs) < (w[1].lhs, w[1].rhs)));
            for chunk in rows.chunks(CHUNK) {
                for row in chunk {
                    assert_eq!(oracle::expected_bits(dtype, row.schedule, row.lhs, row.rhs), Some(*row));
                    assert!(accuracy.record(dtype, oracle::Route::Tensor, row, row.staged).passed);
                }
                audit_poison(dtype, chunk);
            }
            for fixture in oracle::scalar_fixtures(dtype) {
                assert_eq!(fixture.inputs.len(), fixture.primitive.len());
                for rows in [
                    &fixture.primitive,
                    &fixture.wrapper,
                    &fixture.public_primitive,
                    &fixture.public_wrapper,
                    &fixture.public_powi_wrapper,
                ] {
                    for row in rows {
                        assert_eq!(oracle::expected_bits(dtype, row.schedule, row.lhs, row.rhs), Some(*row));
                        assert!(oracle::compare(dtype, row, row.staged).passed);
                    }
                    audit_poison(dtype, rows);
                }
                for row in &fixture.wrapper {
                    assert_eq!(row.schedule, scalar_schedule(fixture.scalar));
                }
                for row in &fixture.public_powi_wrapper {
                    assert_eq!(row.schedule, scalar_schedule(fixture.public_powi_scalar));
                }
            }
        }
        assert!(accuracy.is_coherent());
        assert_eq!(accuracy.failures, 0);
    }
    #[test]
    fn malformed_fields_word_aliases_and_full_writers_are_rejected() {
        for (dtype, format) in FORMATS {
            let input = Input::flat(dtype, format, vec![0; 5]).unwrap();
            let p = parameters(&input, &input, &[5], None).unwrap();
            let s = parameters(&input, &input, &[5], Some(0)).unwrap();
            for bad in malformed(p, s, dtype) {
                assert!(float::validate(&bad.map(|v| v as u32)).is_err(), "accepted {bad:?}");
            }
            for params in [p, s] {
                assert!(float::validate(&params[..31].iter().map(|&v| v as u32).collect::<Vec<_>>()).is_err());
                let mut extended = params.map(|v| v as u32).to_vec();
                extended.push(0);
                assert!(float::validate(&extended).is_err());
                let work = Kernel::FloatArithmetic.work(&params.map(|v| v as u32)).unwrap();
                assert_eq!(work.output, 2);
                assert_eq!(work.pointers, 3);
                assert!(Kernel::FloatArithmetic.allocates_output());
                assert!(work.validate_aliases(&[0x1004, 0x2004, 0x3004]).is_ok());
                for out in [0x1004, 0x1008, 0x2004, 0x2000] {
                    assert!(work.validate_aliases(&[0x1004, 0x2004, out]).is_err());
                }
                if dtype == DType::F64 {
                    let mut high = params;
                    high[4] = i32::MAX as usize - 5;
                    high[6] = i32::MAX as usize;
                    assert_eq!(float::validate(&high.map(|v| v as u32)).unwrap()[0], 2 * i32::MAX as u64);
                }
            }
        }
        for scalar in [
            Scalar::Float(2.5),
            Scalar::Float(f64::NAN),
            Scalar::Float(f64::INFINITY),
            Scalar::Float(f64::NEG_INFINITY),
        ] {
            assert_eq!(scalar_schedule(scalar), Schedule::Mathematical);
        }
    }
    #[test]
    fn exact_route_counters_detect_hidden_copies_wrong_classes_and_shortcuts() {
        let zero = [abi::ProfileEntry::default(); abi::KERNEL_COUNT];
        for (dtype, _) in FORMATS {
            for schedule in oracle::SCHEDULES {
                let plan = launch_plan(dtype, schedule, false);
                let mut after = zero;
                for (entry, &calls) in after.iter_mut().zip(&plan) {
                    entry.calls = calls;
                }
                assert!(check_kernel_deltas(&zero, &after, &plan).is_ok());
                for kernel in [
                    Kernel::FloatArithmetic,
                    Kernel::TensorElementwise,
                    Kernel::TensorCast,
                    Kernel::Fill,
                    Kernel::TensorStorage,
                ] {
                    let mut wrong = after;
                    wrong[kernel as usize].calls += 1;
                    assert!(check_kernel_deltas(&zero, &wrong, &plan).is_err());
                }
                assert_eq!(
                    plan.iter().sum::<u64>(),
                    match schedule {
                        Schedule::Identity => 0,
                        Schedule::SquareReciprocal => 2,
                        _ => 1,
                    }
                );
                let cast = launch_plan(dtype, Schedule::Mathematical, true);
                assert_eq!(cast[Kernel::TensorCast as usize], 1);
                assert_eq!(cast[native_kernel(dtype) as usize], 1);
            }
        }
    }
    #[test]
    fn empty_ones_storage_and_resident_cast_services_have_exact_contracts() {
        use oracle::Route;
        let before = [abi::ServiceEntry {
            calls: 11,
            ..Default::default()
        }; abi::SERVICE_OPERATIONS];
        let verify = |expected: [(u32, u64); 5]| {
            let mut after = before;
            for &(service, calls) in &expected {
                after[service as usize].calls += calls;
            }
            assert!(check_service_deltas(&before, &after, &expected).is_ok());
            for &(service, calls) in &expected {
                let mut extra = after;
                extra[service as usize].calls += 1;
                assert!(check_service_deltas(&before, &extra, &expected).is_err());
                if calls != 0 {
                    let mut missing = after;
                    missing[service as usize].calls -= 1;
                    assert!(check_service_deltas(&before, &missing, &expected).is_err());
                }
                let mut reset = after;
                reset[service as usize].calls = 0;
                assert!(check_service_deltas(&before, &reset, &expected).is_err());
            }
        };
        for (dtype, _) in FORMATS {
            for route in [
                Route::ScalarImpl,
                Route::PowiScalarImpl,
                Route::PowfScalar,
                Route::PowiScalar,
                Route::PublicPowfScalar,
                Route::PublicPowiScalar,
            ] {
                // Only wrapper zero constructs an output. Identity, square,
                // reciprocal and generic math all reuse the empty input storage.
                for (scalar, zero) in [
                    (Scalar::Int(0), true),
                    (Scalar::Float(-0.0), true),
                    (Scalar::UInt(0), true),
                    (Scalar::Bool(false), true),
                    (Scalar::Int(1), false),
                    (Scalar::Int(2), false),
                    (Scalar::Int(-1), false),
                    (Scalar::Int(-2), false),
                    (Scalar::Float(3.0), false),
                    (Scalar::UInt(u64::MAX), false),
                    (Scalar::Float(f64::MAX), false),
                ] {
                    let (expected, kernels) = empty_scalar_work(route, scalar);
                    let allocations = u64::from(zero && !matches!(route, Route::ScalarImpl | Route::PowiScalarImpl));
                    assert_eq!(
                        expected,
                        [
                            (abi::UPLOAD, 0),
                            (abi::DOWNLOAD, 0),
                            (abi::ALLOCATE, allocations),
                            (abi::LAUNCH_ALLOCATE, 0),
                            (abi::LAUNCH, 0)
                        ]
                    );
                    verify(expected);
                    for (index, calls) in kernels.iter().enumerate() {
                        assert_eq!(*calls, if index == Kernel::Zero as usize { allocations } else { 0 });
                    }
                }
            }
            // Native and Dispatch tensor-powi both cast I64 to the lhs dtype,
            // then launch Power. Neither route permits a standalone allocation.
            let plan = launch_plan(dtype, Schedule::Mathematical, true);
            assert_eq!(plan.iter().sum::<u64>(), 2);
            assert_eq!(plan[Kernel::TensorCast as usize], 1);
            assert_eq!(plan[native_kernel(dtype) as usize], 1);
            for fresh in [false, true] {
                let expected = resident_services(2, fresh);
                assert_eq!(
                    expected,
                    [
                        (abi::UPLOAD, 0),
                        (abi::DOWNLOAD, 0),
                        (abi::ALLOCATE, 0),
                        (abi::LAUNCH_ALLOCATE, if fresh { 2 } else { 0 }),
                        (abi::LAUNCH, if fresh { 0 } else { 2 })
                    ]
                );
                verify(expected);
            }
        }
        verify(resident_services(0, true));
    }
    #[test]
    fn failures_retain_both_operands_schedules_and_dynamic_cells() {
        let mut observed = Observed::default();
        for (dtype, _) in FORMATS {
            for fixture in oracle::scalar_fixtures(dtype).into_iter().take(8) {
                for (route, rows) in [(oracle::Route::ScalarImpl, fixture.primitive), (oracle::Route::PowfScalar, fixture.wrapper)] {
                    let bytes: Vec<_> = output_poison(dtype, &rows, false)
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .take(rows.len() * dtype.size())
                        .collect();
                    observed.record(dtype, route, &rows, &oracle::unpack(dtype, &bytes), "cpu.poison").unwrap();
                }
            }
        }
        assert_eq!(observed.examples.len(), 12);
        assert!(observed.accuracy.is_coherent());
        assert!(observed.accuracy.cells.len() > 20);
        assert!(observed.accuracy.cells.iter().all(|c| c.accuracy.values == c.accuracy.failures));
        for cell in &observed.accuracy.cells {
            if let Some(worst) = cell.accuracy.worst {
                assert_eq!(worst.schedule, cell.schedule);
                assert!(oracle::expected_bits(cell.dtype, worst.schedule, worst.lhs, worst.rhs).is_some());
            }
        }
    }
    #[test]
    fn freeze_gate_and_error_recipes_are_host_only() {
        if FROZEN {
            assert!(contract().is_ok());
        } else {
            assert!(contract().is_err());
        }
        for (shape, strides, offset, extent) in [(vec![1; 9], vec![1; 9], 0, 1), (vec![3], vec![-1], 1, 3), (vec![3], vec![1], 1, 3)] {
            assert!(Layout::from_parts(Shape::from(shape), strides, offset, extent).is_err());
        }
        for (dtype, _) in FORMATS {
            let fixtures = oracle::scalar_fixtures(dtype);
            assert!(fixtures.iter().any(|f| !f.valid_powi));
            assert!(fixtures.iter().any(|f| !f.public_powi_valid));
            assert!(
                fixtures
                    .iter()
                    .any(|f| matches!(f.scalar, Scalar::UInt(u64::MAX)) && f.valid_powi && f.public_powi_valid)
            );
            assert!(
                fixtures
                    .iter()
                    .any(|f| matches!(f.scalar,Scalar::Float(v) if v==f64::MAX) && f.valid_powi && f.public_powi_valid)
            );
        }
    }
    #[test]
    fn independent_cpu_plan_enumerates_routes_and_comparison_counts() {
        use oracle::Route;
        let mut plan = Counts::default();
        let mut cells: Vec<(DType, Route, Schedule, usize)> = Vec::new();
        let mut add = |dtype, route, schedule, n| {
            if let Some(cell) = cells.iter_mut().find(|c| c.0 == dtype && c.1 == route && c.2 == schedule) {
                cell.3 += n;
            } else {
                cells.push((dtype, route, schedule, n));
            }
        };
        // Enumerate lengths from coordinate recipes, without using geometry_inputs.
        let mut lengths = Vec::new();
        for n in [1, 2, 3, 127, 128, 129, 255, 256, 257] {
            lengths.extend([n; 3]);
        }
        lengths.extend([5, 21, 6, 1, 24, 3, 3, 3, 3, 3, 6]);
        assert_eq!(lengths.len(), 38);
        let geometry_values: usize = lengths.iter().sum();
        for (dtype, _) in FORMATS {
            let n = oracle::rows(dtype).len();
            let batches = n.div_ceil(4096);
            plan.corpus_rows += n;
            plan.corpus_batches += batches;
            plan.raw += 2 * batches;
            plan.fresh += 2 * batches;
            plan.native += 2 * batches;
            plan.public += 4 * batches;
            add(dtype, Route::Tensor, Schedule::Mathematical, n * 8);
            add(dtype, Route::PublicTensor, Schedule::Mathematical, n * 2);
            let fixtures = oracle::scalar_fixtures(dtype);
            for f in &fixtures {
                let n = f.inputs.len();
                let batches = n.div_ceil(4096);
                plan.raw += 2 * batches;
                plan.fresh += 2 * batches;
                plan.native += batches;
                plan.public += (5 + 4 * usize::from(f.valid_powi) + usize::from(f.public_powi_valid)) * batches;
                add(dtype, Route::ScalarImpl, Schedule::Mathematical, n * 7);
                add(dtype, Route::PowfScalar, f.wrapper[0].schedule, n * 2);
                add(dtype, Route::PublicPowfScalar, f.public_wrapper[0].schedule, n);
                if f.valid_powi {
                    add(dtype, Route::PowiScalarImpl, Schedule::Mathematical, n * 2);
                    add(dtype, Route::PowiScalar, f.wrapper[0].schedule, n * 2);
                }
                if f.public_powi_valid {
                    add(dtype, Route::PublicPowiScalar, f.public_powi_wrapper[0].schedule, n);
                }
                plan.rejected += 2 * (4 * usize::from(!f.valid_powi) + usize::from(!f.public_powi_valid));
            }
            // Tensor geometry and two real broadcasts.
            plan.geometry += 40;
            plan.raw += 80;
            plan.fresh += 80;
            plan.native += 80;
            plan.public += 160;
            add(dtype, Route::Tensor, Schedule::Mathematical, (geometry_values + 12) * 8);
            add(dtype, Route::PublicTensor, Schedule::Mathematical, (geometry_values + 12) * 2);
            // Six original Int scalar specs across all 38 coordinate recipes.
            for schedule in [
                Schedule::Ones,
                Schedule::Identity,
                Schedule::Square,
                Schedule::Reciprocal,
                Schedule::SquareReciprocal,
                Schedule::Mathematical,
            ] {
                plan.geometry += 38;
                plan.raw += 76;
                plan.fresh += 76;
                plan.native += 38;
                plan.public += 152;
                add(dtype, Route::ScalarImpl, Schedule::Mathematical, geometry_values * 7);
                add(dtype, Route::PowfScalar, schedule, geometry_values * 2);
                add(dtype, Route::PowfScalar, schedule, 34); // queued Native and Dispatch
            }
            for fixture in oracle::integer_fixtures(dtype) {
                plan.public += 2 * fixture.rows.len().div_ceil(4096);
                add(dtype, Route::TensorPowi, Schedule::Mathematical, fixture.rows.len() * 2);
            }
            // Physical/metadata negatives: 40 common malformed variants plus
            // one invalid high scalar word and one half high scalar lane.
            let malformed = 40 + usize::from(dtype.size() < 8) + usize::from(dtype.size() == 2);
            plan.rejected += 2 * malformed + 48 + if dtype == DType::F64 { 14 } else { 0 };
            plan.rejected += 8; // two lengths, four Native dtype/arity failures
            plan.empty += 4 * (3 + 4 + 8 * 10);
            plan.public += 4 * 4;
            plan.fresh += 4;
            plan.native += 1;
            plan.public += 13;
            plan.queued += 18;
            add(dtype, Route::Tensor, Schedule::Mathematical, 68);
            add(dtype, Route::ScalarImpl, Schedule::Mathematical, 34);
        }
        plan.rejected += 2 * (3 + 4 + 1) + 3; // integer Native/direct errors and invalid layouts
        let comparisons: usize = cells.iter().map(|c| c.3).sum();
        let reference = reference_plan();
        assert_eq!(reference.iter().map(|c| c.values).sum::<usize>(), comparisons);
        assert_eq!(reference.len(), cells.len());
        for cell in &cells {
            assert_eq!(
                reference
                    .iter()
                    .find(|c| c.dtype == cell.0 && c.route == cell.1 && c.schedule == cell.2 && !c.chain)
                    .unwrap()
                    .values,
                cell.3
            );
        }
        std::println!("POWER_REFERENCE_PLAN {}", reference_plan_json(&reference));
        assert_eq!(plan, expected_counts());
        assert_eq!(plan, EXPECTED_COUNTS);
        assert_eq!(comparisons, COMPARISONS);
        assert_eq!(reference.len(), CELLS);
        std::println!("POWER_PLAN_COUNTS {plan:?}");
        std::println!("POWER_PLAN_COMPARISONS {comparisons}");
        for (dtype, route, schedule, n) in cells {
            std::println!("POWER_PLAN_CELL {dtype:?} {route:?} {schedule:?} {n}");
        }
    }
}
