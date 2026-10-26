//! Guarded typed remainder with independent, immutable pair/scalar fixtures.
//! Half numerical agreement does not prove the required F32 staging. The
//! aggregate production IR/SASS needs its own audit. No Autodiff claim is made.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{
            Backend as _, DType, Dispatch, Scalar, Shape, TensorData, TensorMetadata,
            ops::{FloatTensorOps, IntTensorOps},
            tensor::FloatTensor,
        },
        tensor::{Tensor, TensorCreationOptions, try_read_sync},
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::typed_remainder as oracle,
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, cast::Format, elementwise::Elementwise, float_arithmetic as float};

type Result<T> = core::result::Result<T, String>;
const GUARD: u32 = 0x6bad_beef;
const POISON: u32 = 0xa53c_d791;
const OP: Elementwise = Elementwise::Remainder;
const FORMATS: [(DType, Format); 5] = [
    (DType::F64, Format::F64),
    (DType::F32, Format::F32),
    (DType::Flex32, Format::Flex32),
    (DType::F16, Format::F16),
    (DType::BF16, Format::BF16),
];
const LENGTHS: [usize; 9] = [1, 2, 3, 127, 128, 129, 255, 256, 257];
#[derive(Default, Debug, PartialEq, Eq)]
struct Counts {
    raw: usize,
    fresh: usize,
    native: usize,
    public: usize,
    rejected: usize,
    empty: usize,
    queued: usize,
    tensor_rows: usize,
    scalar_specs: usize,
    scalar_values: usize,
    storage_specs: usize,
    f32_routes: usize,
    typed_routes: usize,
}
// Freeze before GPU execution. Host audits enumerate this same bounded plan.
const EXPECTED: Counts = Counts {
    raw: 589,
    fresh: 604,
    native: 805,
    public: 762,
    rejected: 383,
    empty: 240,
    queued: 20,
    tensor_rows: 1608,
    scalar_specs: 221,
    scalar_values: 2210,
    storage_specs: 2,
    f32_routes: 302,
    typed_routes: 1265,
};
fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("typed remainder: {e}"))
}
fn data(dtype: DType, shape: &[usize], bits: &[u64]) -> TensorData {
    TensorData::from_bytes_vec(oracle::pack(dtype, bits), Shape::from(shape), dtype)
}
fn packed(dtype: DType, values: &[u64]) -> Vec<u32> {
    oracle::pack(dtype, values)
        .chunks(4)
        .map(|chunk| {
            let mut bytes = [0; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect()
}
fn check_bits(dtype: DType, actual: &[u64], expected: &[u64], source: bool) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(String::from("typed remainder logical length mismatch"));
    }
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if got != want && (source || !oracle::is_nan(dtype, got) || !oracle::is_nan(dtype, want)) {
            return Err(format!(
                "typed remainder {dtype:?} element={i} actual={got:#018x} expected={want:#018x} source={source}"
            ));
        }
    }
    Ok(())
}
fn check_data(actual: &TensorData, dtype: DType, shape: &[usize], expected: &[u64], source: bool) -> Result<()> {
    if actual.dtype() != dtype || actual.shape().as_slice() != shape || actual.as_bytes().len() != expected.len() * dtype.size() {
        return Err(format!(
            "typed remainder metadata mismatch {dtype:?} {shape:?}: {:?} {:?}",
            actual.dtype(),
            actual.shape()
        ));
    }
    check_bits(dtype, &oracle::unpack(dtype, actual.as_bytes()), expected, source)
}
fn check(tensor: &NativeTensor, dtype: DType, shape: &[usize], expected: &[u64], source: bool) -> Result<()> {
    check_data(&gpu(tensor.read())?, dtype, shape, expected, source)
}
fn read(value: FloatTensor<Native>, dtype: DType, shape: &[usize], expected: &[u64]) -> Result<()> {
    let actual = try_read_sync(Native::float_into_data(value))
        .ok_or("typed remainder backend read pending")?
        .map_err(|e| format!("typed remainder backend read: {e:?}"))?;
    check_data(&actual, dtype, shape, expected, false)
}
fn unchanged(backend: &Nvidia, buffer: &d3os_compute::nvidia::GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if actual != expected {
        return Err(String::from("typed remainder source/guard/unused RHS changed"));
    }
    Ok(())
}
fn check_words(backend: &Nvidia, buffer: &d3os_compute::nvidia::GpuBuffer<u32>, dtype: DType, expected: &[u64], guarded: bool) -> Result<()> {
    let count = packed(dtype, expected).len();
    let mut actual = vec![0; count + 2 * usize::from(guarded)];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if guarded && (actual[0] != GUARD || actual[count + 1] != GUARD) {
        return Err(String::from("typed remainder output guard changed"));
    }
    let words = &actual[usize::from(guarded)..usize::from(guarded) + count];
    if dtype.size() == 2 && expected.len() % 2 == 1 && words[count - 1] >> 16 != 0 {
        return Err(String::from("typed remainder odd output upper half is not zero"));
    }
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).take(expected.len() * dtype.size()).collect();
    check_bits(dtype, &oracle::unpack(dtype, &bytes), expected, false)
}
fn native_kernel(dtype: DType) -> Kernel {
    if dtype == DType::F32 {
        Kernel::TensorElementwise
    } else {
        Kernel::FloatArithmetic
    }
}
/// Check every kernel counter, including absence of a hidden clear/copy kernel.
fn resident(
    backend: &Nvidia, services: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT], kernel: Kernel, fresh: bool,
) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE] {
        if after[service as usize].calls != services[service as usize].calls {
            return Err(format!("typed remainder resident service={service}"));
        }
    }
    for (service, want) in [(abi::LAUNCH_ALLOCATE, u64::from(fresh)), (abi::LAUNCH, u64::from(!fresh))] {
        if after[service as usize].calls - services[service as usize].calls != want {
            return Err(format!("typed remainder full writer service={service} expected={want}"));
        }
    }
    for (i, (old, new)) in profile.iter().zip(gpu(backend.profile())?).enumerate() {
        if new.calls - old.calls != u64::from(i == kernel as usize) {
            return Err(format!("typed remainder route kernel={kernel:?} counter={i}"));
        }
    }
    Ok(())
}
fn routed(counts: &mut Counts, dtype: DType) {
    if dtype == DType::F32 {
        counts.f32_routes += 1;
    } else {
        counts.typed_routes += 1;
    }
}
fn no_work(backend: &Nvidia, services: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT]) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
        if after[service as usize].calls != services[service as usize].calls {
            return Err(format!("typed remainder invalid/empty service={service}"));
        }
    }
    if profile.iter().zip(gpu(backend.profile())?).any(|(a, b)| a.calls != b.calls) {
        return Err(String::from("typed remainder invalid/empty kernel launched"));
    }
    Ok(())
}
struct Input {
    dtype: DType,
    format: Format,
    layout: Layout,
    storage: Vec<u64>,
    logical: Vec<u64>,
}
impl Input {
    fn flat(dtype: DType, format: Format, values: Vec<u64>) -> Result<Self> {
        Ok(Self {
            dtype,
            format,
            layout: gpu(Layout::contiguous(Shape::new([values.len()])))?,
            storage: values.clone(),
            logical: values,
        })
    }
    fn data(&self) -> TensorData {
        data(self.dtype, &[self.storage.len()], &self.storage)
    }
    fn tensor(&self) -> Result<NativeTensor> {
        gpu(NativeTensor::from_data(self.data()).and_then(|v| v.with_layout(self.layout.clone())))
    }
}
fn parameters(lhs: &Input, rhs: &Input, shape: &[usize], scalar: Option<u64>) -> Result<[usize; 32]> {
    let left = gpu(lhs.layout.expand(Shape::from(shape)))?;
    let right = if scalar.is_some() {
        left.clone()
    } else {
        gpu(rhs.layout.expand(Shape::from(shape)))?
    };
    let mut p = gpu(left.elementwise_parameters(&right, Elementwise::Add, 0, false))?;
    p[2] = 34 | ((lhs.format as usize) << 8);
    if let Some(bits) = scalar {
        p[2] |= float::SCALAR_RHS as usize;
        p[3] = bits as u32 as usize;
        p[5] = (bits >> 32) as usize;
        p[7] = 0;
        p[24..32].fill(0);
    }
    Ok(p)
}
fn source(input: &Input) -> Vec<u32> {
    let mut words = vec![GUARD];
    words.extend(packed(input.dtype, &input.storage));
    if input.dtype.size() == 2 && input.storage.len() % 2 == 1 {
        *words.last_mut().unwrap() |= 0xffff_0000;
    }
    words.push(GUARD);
    words
}
fn raw(backend: &Nvidia, lhs: &Input, rhs: &Input, shape: &[usize], scalar: Option<u64>, expected: &[u64], counts: &mut Counts) -> Result<()> {
    let aw = source(lhs);
    // The unused scalar RHS has only one poison word, even for F64. Reading
    // either a tensor divisor or a second F64 word must not be necessary.
    let bw = if scalar.is_some() { vec![GUARD, POISON, GUARD] } else { source(rhs) };
    let a = gpu(backend.upload_words(&aw))?;
    let b = gpu(backend.upload_words(&bw))?;
    let words = packed(lhs.dtype, expected).len();
    let mut poison = vec![POISON; words + 2];
    poison[0] = GUARD;
    poison[words + 1] = GUARD;
    let output = gpu(backend.upload_words(&poison))?;
    let p = parameters(lhs, rhs, shape, scalar)?;
    let services = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &p)))?;
    resident(backend, &services, &profile, Kernel::FloatArithmetic, false)?;
    check_words(backend, &output, lhs.dtype, expected, true)?;
    counts.raw += 1;
    gpu(backend.write_words(&output, 1, &vec![POISON; words]))?;
    drop(output);
    let services = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    let output = gpu(backend.with_commands(|| backend.produce_words(words, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &p)))?;
    resident(backend, &services, &profile, Kernel::FloatArithmetic, true)?;
    check_words(backend, &output, lhs.dtype, expected, false)?;
    unchanged(backend, &a, &aw)?;
    unchanged(backend, &b, &bw)?;
    counts.fresh += 1;
    Ok(())
}
fn native(backend: &Nvidia, lhs: &Input, rhs: &Input, shape: &[usize], scalar: Option<Scalar>, expected: &[u64], counts: &mut Counts) -> Result<()> {
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
                a.float_scalar(OP, scalar)
            } else if legacy {
                a.binary(&b, OP)
            } else {
                a.float_binary(&b, OP)
            }
        }))?;
        resident(backend, &services, &profile, native_kernel(lhs.dtype), true)?;
        check(&output, lhs.dtype, shape, expected, false)?;
        counts.native += 1;
        routed(counts, lhs.dtype);
    }
    check(&a, lhs.dtype, lhs.layout.shape().as_slice(), &lhs.logical, true)?;
    check(&b, rhs.dtype, rhs.layout.shape().as_slice(), &rhs.logical, true)?;
    // Retained base aliases include every unselected input byte and NaN payload.
    check(&abase, lhs.dtype, &[lhs.storage.len()], &lhs.storage, true)?;
    check(&bbase, rhs.dtype, &[rhs.storage.len()], &rhs.storage, true)
}
fn both(backend: &Nvidia, lhs: &Input, rhs: &Input, shape: &[usize], scalar: Option<(Scalar, u64)>, expected: &[u64], counts: &mut Counts) -> Result<()> {
    raw(backend, lhs, rhs, shape, scalar.map(|v| v.1), expected, counts)?;
    native(backend, lhs, rhs, shape, scalar.map(|v| v.0), expected, counts)
}
fn lookup(dtype: DType, lhs: &[u64], rhs: &[u64]) -> Result<Vec<u64>> {
    if lhs.len() != rhs.len() {
        return Err(String::from("typed remainder reference length mismatch"));
    }
    lhs.iter()
        .zip(rhs)
        .map(|(&a, &b)| oracle::expected_bits(dtype, a, b).ok_or_else(|| format!("typed remainder missing independent pair {dtype:?} {a:#x} {b:#x}")))
        .collect()
}
fn public_scalar(source: Tensor<1>, scalar: Scalar) -> Tensor<1> {
    // Generic Tensor scalar arguments become Float through Scalar::new. Direct
    // Dispatch below separately tests preservation of Int, UInt, and Bool.
    match scalar {
        Scalar::Float(v) => source.remainder_scalar(v),
        Scalar::Int(v) => source.remainder_scalar(v),
        Scalar::UInt(v) => source.remainder_scalar(v),
        Scalar::Bool(v) => source.remainder_scalar(v),
    }
}
fn public(backend: &Nvidia, dtype: DType, lhs: &[u64], rhs: &[u64], scalar: Option<Scalar>, expected: &[u64], counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("typed remainder device: {e}"))?;
    for path in 0..3 {
        let a = Tensor::<1>::from_data(data(dtype, &[lhs.len()], lhs), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
        let b = Tensor::<1>::from_data(data(dtype, &[rhs.len()], rhs), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
        let pa = Native::float_from_data(data(dtype, &[lhs.len()], lhs), &NativeDevice);
        let pb = Native::float_from_data(data(dtype, &[rhs.len()], rhs), &NativeDevice);
        let alias = a.clone();
        let rhs_alias = b.clone();
        let pa_alias = pa.clone();
        let pb_alias = pb.clone();
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let actual = match path {
            0 => {
                let output = if let Some(s) = scalar {
                    Native::float_remainder_scalar(pa, s)
                } else {
                    Native::float_remainder(pa, pb)
                };
                resident(backend, &services, &profile, native_kernel(dtype), true)?;
                try_read_sync(Native::float_into_data(output))
                    .ok_or("typed remainder Native wrapper read pending")?
                    .map_err(|e| format!("typed remainder Native wrapper: {e:?}"))?
            }
            1 => {
                let output = if let Some(s) = scalar {
                    Dispatch::float_remainder_scalar(a.into_dispatch(), s)
                } else {
                    Dispatch::float_remainder(a.into_dispatch(), b.into_dispatch())
                };
                resident(backend, &services, &profile, native_kernel(dtype), true)?;
                try_read_sync(Dispatch::float_into_data(output))
                    .ok_or("typed remainder Dispatch read pending")?
                    .map_err(|e| format!("typed remainder Dispatch: {e:?}"))?
            }
            _ => {
                let output = if let Some(s) = scalar { public_scalar(a, s) } else { a.remainder(b) };
                resident(backend, &services, &profile, native_kernel(dtype), true)?;
                output.try_into_data().map_err(|e| format!("typed remainder public Tensor: {e:?}"))?
            }
        };
        check_data(&actual, dtype, &[expected.len()], expected, false)?;
        check_data(
            &alias.try_into_data().map_err(|e| format!("typed remainder public alias: {e:?}"))?,
            dtype,
            &[lhs.len()],
            lhs,
            true,
        )?;
        check_data(
            &rhs_alias.try_into_data().map_err(|e| format!("typed remainder RHS alias: {e:?}"))?,
            dtype,
            &[rhs.len()],
            rhs,
            true,
        )?;
        for (primitive, expected) in [(pa_alias, lhs), (pb_alias, rhs)] {
            let actual = try_read_sync(Native::float_into_data(primitive))
                .ok_or("typed remainder retained primitive read pending")?
                .map_err(|e| format!("typed remainder retained primitive: {e:?}"))?;
            check_data(&actual, dtype, &[expected.len()], expected, true)?;
        }
        counts.public += 1;
        routed(counts, dtype);
    }
    Ok(())
}
fn matrices(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let rows = oracle::pairs(dtype);
        let mut groups = Vec::new();
        for row in &rows {
            if !groups.contains(&row.group) {
                groups.push(row.group);
            }
        }
        for group in groups {
            let rows: Vec<_> = rows.iter().filter(|p| p.group == group).collect();
            let a = Input::flat(dtype, format, rows.iter().map(|p| p.lhs).collect())?;
            let b = Input::flat(dtype, format, rows.iter().map(|p| p.rhs).collect())?;
            let want: Vec<_> = rows.iter().map(|p| p.expected).collect();
            both(backend, &a, &b, &[want.len()], None, &want, counts)?;
            public(backend, dtype, &a.logical, &b.logical, None, &want, counts)?;
            counts.tensor_rows += want.len();
        }
        for fixture in oracle::scalar_fixtures(dtype) {
            if fixture.inputs.len() != 10 || fixture.expected.len() != 10 {
                return Err(String::from("typed remainder scalar fixture must retain ten dividends"));
            }
            let a = Input::flat(dtype, format, fixture.inputs)?;
            let b = Input::flat(dtype, format, vec![fixture.scalar_bits; a.logical.len()])?;
            both(
                backend,
                &a,
                &b,
                &[a.logical.len()],
                Some((fixture.scalar, fixture.scalar_bits)),
                &fixture.expected,
                counts,
            )?;
            public(backend, dtype, &a.logical, &b.logical, Some(fixture.scalar), &fixture.expected, counts)?;
            counts.scalar_specs += 1;
            counts.scalar_values += fixture.expected.len();
        }
        for fixture in oracle::storage_scalar_fixtures(dtype) {
            let a = Input::flat(dtype, format, fixture.inputs)?;
            let b = Input::flat(dtype, format, vec![fixture.scalar_bits; a.logical.len()])?;
            both(
                backend,
                &a,
                &b,
                &[a.logical.len()],
                Some((fixture.scalar, fixture.scalar_bits)),
                &fixture.expected,
                counts,
            )?;
            public(backend, dtype, &a.logical, &b.logical, Some(fixture.scalar), &fixture.expected, counts)?;
            counts.storage_specs += 1;
        }
    }
    Ok(())
}
/// Logical coordinates come from the independent shared Values traversal,
/// never Layout::element_offset or the production address calculation.
fn geometry_inputs(dtype: DType, format: Format, pool: &[u64]) -> Result<Vec<Input>> {
    let values = |n: usize| (0..n).map(|i| pool[i % pool.len()]).collect::<Vec<_>>();
    let mut cases = Vec::new();
    for length in LENGTHS {
        cases.push(Input::flat(dtype, format, values(length))?);
        let storage = values(2 * length + 11);
        for reverse in [false, true] {
            let view = oracle::Values::new(&[storage.len()], storage.clone()).slice(&[1], &[2], &[length]);
            let logical = if reverse { view.flip(&[0]).values } else { view.values };
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
            });
        }
    }
    let storage = values(11);
    let logical = oracle::Values::new(&[11], storage.clone()).slice(&[1], &[1], &[5]).values;
    cases.push(Input {
        dtype,
        format,
        storage,
        logical,
        layout: gpu(Layout::from_parts(Shape::new([5]), vec![1], 1, 11))?,
    });
    let storage = values(21);
    let logical = oracle::Values::new(&[3, 7], storage.clone()).permute(&[1, 0]).flip(&[0, 1]).values;
    cases.push(Input {
        dtype,
        format,
        storage,
        logical,
        layout: gpu(Layout::contiguous(Shape::new([3, 7]))
            .and_then(|l| l.permute(&[1, 0]))
            .and_then(|l| l.flip(&[0, 1])))?,
    });
    let storage = values(45);
    let logical = oracle::Values::new(&[5, 9], storage.clone()).slice(&[1, 1], &[2, 2], &[2, 3]).values;
    cases.push(Input {
        dtype,
        format,
        logical,
        storage,
        layout: gpu(Layout::from_parts(Shape::new([2, 3]), vec![18, 2], 10, 45))?,
    });
    for shape in [vec![], vec![1, 2, 1, 2, 1, 2, 1, 3]] {
        let mut input = Input::flat(dtype, format, values(shape.iter().product()))?;
        input.layout = gpu(Layout::contiguous(Shape::from(shape)))?;
        cases.push(input);
    }
    let storage = values(3);
    let logical = oracle::Values::new(&[3], storage.clone()).slice(&[1], &[1], &[1]).expand(&[2, 3]).values;
    cases.push(Input {
        dtype,
        format,
        storage,
        logical,
        layout: gpu(Layout::from_parts(Shape::new([2, 3]), vec![0, 0], 1, 3))?,
    });
    Ok(cases)
}
fn layouts(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let rows = oracle::pairs(dtype);
        let left: Vec<_> = rows.iter().map(|p| p.lhs).collect();
        let right: Vec<_> = rows.iter().map(|p| p.rhs).collect();
        for (a, b) in geometry_inputs(dtype, format, &left)?.into_iter().zip(geometry_inputs(dtype, format, &right)?) {
            let want = lookup(dtype, &a.logical, &b.logical)?;
            both(backend, &a, &b, a.layout.shape().as_slice(), None, &want, counts)?;
        }
        let fixture = oracle::scalar_fixtures(dtype)
            .into_iter()
            .find(|s| oracle::is_finite(dtype, s.scalar_bits) && s.scalar_bits != 0 && s.inputs.iter().any(|&a| a != 0))
            .ok_or("typed remainder missing geometry scalar")?;
        for a in geometry_inputs(dtype, format, &fixture.inputs)? {
            let want = lookup(dtype, &a.logical, &vec![fixture.scalar_bits; a.logical.len()])?;
            both(
                backend,
                &a,
                &a,
                a.layout.shape().as_slice(),
                Some((fixture.scalar, fixture.scalar_bits)),
                &want,
                counts,
            )?;
        }
        // Real operand broadcasting, rather than two pre-expanded Native views.
        let divisor = rows[0].rhs;
        let left: Vec<_> = rows.iter().filter(|p| p.rhs == divisor).take(3).map(|p| p.lhs).collect();
        if left.len() != 3 {
            return Err(String::from("typed remainder broadcast fixture missing"));
        }
        for lhs_broadcast in [false, true] {
            let mut a = Input::flat(dtype, format, if lhs_broadcast { left.clone() } else { left.repeat(2) })?;
            let mut b = Input::flat(dtype, format, vec![divisor; if lhs_broadcast { 6 } else { 3 }])?;
            a.layout = gpu(Layout::contiguous(Shape::new([if lhs_broadcast { 1 } else { 2 }, 3])))?;
            b.layout = gpu(Layout::contiguous(Shape::new([if lhs_broadcast { 2 } else { 1 }, 3])))?;
            let want = lookup(dtype, &left.repeat(2), &[divisor; 6])?;
            both(backend, &a, &b, &[2, 3], None, &want, counts)?;
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
fn raw_rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let bits = oracle::pairs(dtype)[0].lhs;
        let a = Input::flat(dtype, format, vec![bits; 5])?;
        let p = parameters(&a, &a, &[5], None)?;
        let s = parameters(&a, &a, &[5], Some(bits))?;
        let words = vec![POISON; packed(dtype, &a.storage).len() + 4];
        let input = gpu(backend.upload_words(&words))?;
        let rhs = gpu(backend.upload_words(&words))?;
        let output = gpu(backend.upload_words(&words))?;
        let short = gpu(backend.upload_words(&[POISON]))?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let queue = gpu(backend.queue_profile())?;
        for bad in malformed(p, s, dtype) {
            if backend
                .dispatch_words(Kernel::FloatArithmetic, &[(&input, 1), (&rhs, 1), (&output, 1)], &bad)
                .is_ok()
            {
                return Err(format!("typed remainder malformed accepted {dtype:?}: {bad:?}"));
            }
            counts.rejected += 1;
        }
        for bad in [p[..31].to_vec(), {
            let mut v = p.to_vec();
            v.push(0);
            v
        }] {
            if backend
                .dispatch_words(Kernel::FloatArithmetic, &[(&input, 1), (&rhs, 1), (&output, 1)], &bad)
                .is_ok()
            {
                return Err(String::from("typed remainder parameter length accepted"));
            }
            counts.rejected += 1;
        }
        for buffers in [vec![(&input, 1), (&output, 1)], vec![(&input, 1), (&rhs, 1), (&output, 1), (&short, 0)]] {
            if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p).is_ok() {
                return Err(String::from("typed remainder pointer count accepted"));
            }
            counts.rejected += 1;
        }
        // Output cannot overlap either source, even the unused scalar RHS.
        for (out, offset, params) in [(&input, 1, p), (&input, 2, p), (&rhs, 1, p), (&rhs, 2, p), (&rhs, 1, s), (&input, 1, s)] {
            if backend
                .dispatch_words(Kernel::FloatArithmetic, &[(&input, 1), (&rhs, 1), (out, offset)], &params)
                .is_ok()
            {
                return Err(String::from("typed remainder output alias accepted"));
            }
            counts.rejected += 1;
        }
        for buffers in [
            [(&short, 0), (&rhs, 1), (&output, 1)],
            [(&input, 1), (&short, 0), (&output, 1)],
            [(&input, 1), (&rhs, 1), (&short, 0)],
            [(&input, words.len()), (&rhs, 1), (&output, 1)],
            [(&input, 1), (&rhs, 1), (&output, words.len())],
        ] {
            if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p).is_ok() {
                return Err(String::from("typed remainder physical buffer bounds accepted"));
            }
            counts.rejected += 1;
        }
        if dtype == DType::F64 {
            let one = Input::flat(dtype, format, vec![bits])?;
            let one = parameters(&one, &one, &[1], None)?;
            for buffers in [
                [(&short, 0), (&rhs, 1), (&output, 1)],
                [(&input, 1), (&short, 0), (&output, 1)],
                [(&input, 1), (&rhs, 1), (&short, 0)],
            ] {
                if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &one).is_ok() {
                    return Err(String::from("typed remainder missing F64 second-word check"));
                }
                counts.rejected += 1;
            }
            let mut high = p;
            high[4] = i32::MAX as usize - 5;
            high[6] = i32::MAX as usize;
            if backend
                .dispatch_words(Kernel::FloatArithmetic, &[(&input, 1), (&rhs, 1), (&output, 1)], &high)
                .is_ok()
            {
                return Err(String::from("typed remainder F64 physical high-word bounds accepted"));
            }
            counts.rejected += 1;
        }
        // Six alias requests reach the service so the driver can compare GPU
        // addresses. Rejection must precede command recording and GPU launch.
        let after = gpu(backend.service_profile())?;
        for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
            let expected = if service == abi::LAUNCH { 6 } else { 0 };
            if after[service as usize].calls - services[service as usize].calls != expected {
                return Err(format!("typed remainder rejected request service={service} expected={expected}"));
            }
        }
        if profile.iter().zip(gpu(backend.profile())?).any(|(a, b)| a.calls != b.calls) || gpu(backend.queue_profile())?.launches != queue.launches {
            return Err(String::from("typed remainder rejected request launched GPU work"));
        }
        unchanged(backend, &input, &words)?;
        unchanged(backend, &rhs, &words)?;
        unchanged(backend, &output, &words)?;
        unchanged(backend, &short, &[POISON])?;
    }
    Ok(())
}
fn native_validation(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let wrong = if dtype == DType::F32 { FORMATS[2] } else { FORMATS[1] };
        for length in [0, 3] {
            let a = Input::flat(dtype, format, vec![0; length])?.tensor()?;
            let b = Input::flat(wrong.0, wrong.1, vec![0; length])?.tensor()?;
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            for result in [
                a.float_binary(&b, OP),
                a.binary(&b, OP),
                a.float_unary(OP),
                a.float_scalar(Elementwise::Exp, Scalar::Bool(true)),
            ] {
                if !matches!(result, Err(Error::InvalidParameter)) {
                    return Err(format!("typed remainder invalid Native request accepted {dtype:?} length={length}"));
                }
                counts.rejected += 1;
            }
            no_work(backend, &services, &profile)?;
        }
        let a = Input::flat(dtype, format, vec![0; 3])?.tensor()?;
        let b = Input::flat(dtype, format, vec![0; 5])?.tensor()?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        if a.float_binary(&b, OP).is_ok() {
            return Err(String::from("typed remainder incompatible shapes accepted"));
        }
        counts.rejected += 1;
        for (shape, strides, offset, extent) in [(vec![1; 9], vec![1; 9], 0, 1), (vec![3], vec![-1], 1, 3), (vec![3], vec![1], 1, 3)] {
            if Layout::from_parts(Shape::from(shape), strides, offset, extent).is_ok() {
                return Err(String::from("typed remainder invalid Native rank/bounds accepted"));
            }
            counts.rejected += 1;
        }
        no_work(backend, &services, &profile)?;
    }
    for length in [0, 3] {
        let source = gpu(NativeTensor::from_data(data(DType::I32, &[length], &vec![0; length])))?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        for result in [source.float_binary(&source, OP), source.float_scalar(OP, Scalar::Float(f64::NAN))] {
            if !matches!(result, Err(Error::InvalidParameter)) {
                return Err(String::from("typed remainder direct integer primitive accepted"));
            }
            counts.rejected += 1;
        }
        no_work(backend, &services, &profile)?;
    }
    Ok(())
}
fn empties(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("typed remainder empty device: {e}"))?;
    for (dtype, _) in FORMATS {
        for shape in [vec![0], vec![2, 0, 7], vec![0, 2, 3], vec![1, 1, 1, 1, 0, 1, 1, 3]] {
            let empty = gpu(NativeTensor::empty(Shape::from(shape.clone()), dtype))?;
            let primitive = Native::float_from_data(data(dtype, &shape, &[]), &NativeDevice);
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            for output in [gpu(empty.float_binary(&empty, OP))?, gpu(empty.binary(&empty, OP))?] {
                check(&output, dtype, &shape, &[], false)?;
                counts.empty += 1;
            }
            for scalar in [Scalar::Float(f64::NAN), Scalar::Int(i64::MIN), Scalar::UInt(u64::MAX), Scalar::Bool(false)] {
                check(&gpu(empty.float_scalar(OP, scalar))?, dtype, &shape, &[], false)?;
                counts.empty += 1;
            }
            read(Native::float_remainder(primitive.clone(), primitive.clone()), dtype, &shape, &[])?;
            read(Native::float_remainder_scalar(primitive, Scalar::Float(f64::INFINITY)), dtype, &shape, &[])?;
            counts.empty += 2;
            no_work(backend, &services, &profile)?;
            // Public rank-one empties complement direct Native rank-eight and
            // empty-middle shapes, without reshaping away the Native checks.
            let a = Tensor::<1>::from_data(data(dtype, &[0], &[]), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
            let b = a.clone();
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            for scalar in [false, true] {
                let output = if scalar {
                    Dispatch::float_remainder_scalar(a.clone().into_dispatch(), Scalar::Bool(false))
                } else {
                    Dispatch::float_remainder(a.clone().into_dispatch(), b.clone().into_dispatch())
                };
                let actual = try_read_sync(Dispatch::float_into_data(output))
                    .ok_or("typed remainder empty Dispatch read pending")?
                    .map_err(|e| format!("typed remainder empty Dispatch: {e:?}"))?;
                check_data(&actual, dtype, &[0], &[], false)?;
                let output = if scalar {
                    a.clone().remainder_scalar(f64::NEG_INFINITY)
                } else {
                    a.clone().remainder(b.clone())
                };
                check_data(
                    &output.try_into_data().map_err(|e| format!("typed remainder empty public: {e:?}"))?,
                    dtype,
                    &[0],
                    &[],
                    false,
                )?;
                counts.empty += 2;
            }
            no_work(backend, &services, &profile)?;
        }
    }
    Ok(())
}
fn aliases_and_lifetimes(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let rows = oracle::pairs(dtype);
        let equal: Vec<_> = rows.iter().filter(|p| p.lhs == p.rhs).take(9).map(|p| p.lhs).collect();
        if equal.is_empty() {
            return Err(String::from("typed remainder missing identical-source pairs"));
        }
        let want = lookup(dtype, &equal, &equal)?;
        let input = Input::flat(dtype, format, equal)?;
        let a = input.tensor()?;
        let alias = a.clone();
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(a.float_binary(&a, OP))?;
        resident(backend, &services, &profile, native_kernel(dtype), true)?;
        check(&output, dtype, &[want.len()], &want, false)?;
        check(&alias, dtype, &[input.logical.len()], &input.logical, true)?;
        counts.native += 1;
        routed(counts, dtype);
        let words = source(&input);
        let a = gpu(backend.upload_words(&words))?;
        let p = parameters(&input, &input, &[want.len()], None)?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.produce_words(packed(dtype, &want).len(), Kernel::FloatArithmetic, &[(&a, 1), (&a, 1)], &p))?;
        resident(backend, &services, &profile, Kernel::FloatArithmetic, true)?;
        check_words(backend, &output, dtype, &want, false)?;
        unchanged(backend, &a, &words)?;
        counts.fresh += 1;

        let fixture = oracle::scalar_fixtures(dtype)
            .into_iter()
            .find(|f| oracle::is_finite(dtype, f.scalar_bits) && f.scalar_bits != 0)
            .ok_or("typed remainder missing lifetime scalar")?;
        for scalar in [false, true] {
            let (a, b, want) = if scalar {
                (
                    Input::flat(dtype, format, fixture.inputs.clone())?,
                    Input::flat(dtype, format, vec![fixture.scalar_bits; 10])?,
                    fixture.expected.clone(),
                )
            } else {
                (
                    Input::flat(dtype, format, rows.iter().take(17).map(|p| p.lhs).collect())?,
                    Input::flat(dtype, format, rows.iter().take(17).map(|p| p.rhs).collect())?,
                    rows.iter().take(17).map(|p| p.expected).collect(),
                )
            };
            let left = a.tensor()?;
            let right = b.tensor()?;
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let output = gpu(backend.with_commands(|| {
                let output = if scalar {
                    left.float_scalar(OP, fixture.scalar)?
                } else {
                    left.float_binary(&right, OP)?
                };
                drop(left);
                drop(right);
                Ok::<_, Error>(output)
            }))?;
            resident(backend, &services, &profile, native_kernel(dtype), true)?;
            check(&output, dtype, &[want.len()], &want, false)?;
            counts.native += 1;
            counts.queued += 1;
            routed(counts, dtype);
            let left = gpu(backend.upload_words(&packed(dtype, &a.storage)))?;
            let right = gpu(backend.upload_words(&if scalar { vec![POISON] } else { packed(dtype, &b.storage) }))?;
            let p = parameters(&a, &b, &[want.len()], scalar.then_some(fixture.scalar_bits))?;
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let output = gpu(backend.with_commands(|| {
                let output = backend.produce_words(packed(dtype, &want).len(), Kernel::FloatArithmetic, &[(&left, 0), (&right, 0)], &p)?;
                drop(left);
                drop(right);
                Ok::<_, Error>(output)
            }))?;
            resident(backend, &services, &profile, Kernel::FloatArithmetic, true)?;
            check_words(backend, &output, dtype, &want, false)?;
            counts.fresh += 1;
            counts.queued += 1;
        }
    }
    Ok(())
}
/// A failed integer primitive keeps its actual metadata. Synchronize first,
/// then read the failure directly. Never construct a public Float around it.
fn acknowledge(failed: FloatTensor<Native>, dtype: DType, length: usize) -> Result<()> {
    if failed.dtype() != dtype || failed.shape().as_slice() != [length] {
        return Err(String::from("typed remainder failed primitive metadata changed"));
    }
    if !Native::sync(&NativeDevice).is_err_and(|e| !e.is_device_poisoned()) {
        return Err(String::from("typed remainder missing recoverable sync error"));
    }
    let error = try_read_sync(Native::float_into_data(failed)).ok_or("typed remainder failed read pending")?;
    if !error.is_err_and(|e| !e.is_device_poisoned()) || Native::sync(&NativeDevice).is_err() {
        return Err(String::from("typed remainder failure acknowledgment poisoned device"));
    }
    Ok(())
}
fn public_errors(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("typed remainder error device: {e}"))?;
    for length in [0, 3] {
        for scalar in [false, true] {
            let invalid = data(DType::I32, &[length], &vec![0; length]);
            let source = Native::int_from_data(invalid, &NativeDevice);
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let failed = if scalar {
                Native::float_remainder_scalar(source, Scalar::Float(f64::NAN))
            } else {
                Native::float_remainder(source.clone(), source)
            };
            no_work(backend, &services, &profile)?;
            acknowledge(failed, DType::I32, length)?;
            no_work(backend, &services, &profile)?;
            counts.rejected += 1;
        }
        for (dtype, format) in FORMATS {
            let wrong = if dtype == DType::F32 { DType::Flex32 } else { DType::F32 };
            let a = Input::flat(dtype, format, vec![0; length])?;
            let pa = Native::float_from_data(a.data(), &NativeDevice);
            let pb = Native::float_from_data(data(wrong, &[length], &vec![0; length]), &NativeDevice);
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let failed = Native::float_remainder(pa, pb);
            no_work(backend, &services, &profile)?;
            acknowledge(failed, dtype, length)?;
            no_work(backend, &services, &profile)?;
            counts.rejected += 1;
            for direct in [false, true] {
                let left = Tensor::<1>::from_data(a.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
                let right = Tensor::<1>::from_data(
                    data(wrong, &[length], &vec![0; length]),
                    TensorCreationOptions::new(device.clone()).with_dtype(wrong),
                );
                let services = gpu(backend.service_profile())?;
                let profile = gpu(backend.profile())?;
                let failed = if direct {
                    Dispatch::float_remainder(left.into_dispatch(), right.into_dispatch())
                } else {
                    left.remainder(right).into_dispatch()
                };
                no_work(backend, &services, &profile)?;
                if !device.sync().is_err_and(|e| !e.is_device_poisoned()) {
                    return Err(String::from("typed remainder Dispatch missing recoverable sync error"));
                }
                let error = try_read_sync(Dispatch::float_into_data(failed)).ok_or("typed remainder Dispatch failed read pending")?;
                if !error.is_err_and(|e| !e.is_device_poisoned()) || device.sync().is_err() {
                    return Err(String::from("typed remainder Dispatch error acknowledgment poisoned device"));
                }
                no_work(backend, &services, &profile)?;
                counts.rejected += 1;
            }
            // Recover with a real valid operation after all acknowledgments.
            let fixture = oracle::pairs(dtype)[0];
            let left = Input::flat(dtype, format, vec![fixture.lhs; length])?;
            let right = Input::flat(dtype, format, vec![fixture.rhs; length])?;
            if length != 0 {
                public(backend, dtype, &left.logical, &right.logical, None, &vec![fixture.expected; length], counts)?;
            }
        }
    }
    Ok(())
}
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    // Later protocol extensions retain this operation's original contract.
    if abi::VERSION < 45 || abi::KERNEL_COUNT < 82 || OP as u32 != 34 || oracle::CORPUS_VERSION != 1 {
        return Err(String::from("typed remainder protocol/corpus contract changed"));
    }
    let mut counts = Counts::default();
    matrices(backend, &mut counts)?;
    layouts(backend, &mut counts)?;
    raw_rejections(backend, &mut counts)?;
    native_validation(backend, &mut counts)?;
    empties(backend, &mut counts)?;
    aliases_and_lifetimes(backend, &mut counts)?;
    public_errors(backend, &mut counts)?;
    if counts != EXPECTED {
        return Err(format!("typed remainder counts {counts:?}, expected {EXPECTED:?}"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_TYPED_REMAINDER_OK operation=34 formats=5 raw={} fresh={} native={} public={} rejected={} empty={} queued={} tensor_rows={} scalar_specs={} scalar_values={} storage_specs={} f32_routes={} typed_routes={} oracle=independent_exact_fixtures exact_bits=true nan=class sources=exact guards=ok tails=zero resident=true half_schedule=separate_ir_audit f32_entry=separate errors=recoverable autodiff=not_claimed reductions=not_claimed backend=nvidia",
        counts.raw,
        counts.fresh,
        counts.native,
        counts.public,
        counts.rejected,
        counts.empty,
        counts.queued,
        counts.tensor_rows,
        counts.scalar_specs,
        counts.scalar_values,
        counts.storage_specs,
        counts.f32_routes,
        counts.typed_routes,
    ));
    Ok(())
}

#[cfg(test)]
mod host_tests {
    use super::*;

    #[test]
    fn views_follow_independent_coordinates_and_stay_inside_the_fixture_domain() {
        for (dtype, format) in FORMATS {
            let rows = oracle::pairs(dtype);
            let left: Vec<_> = rows.iter().map(|p| p.lhs).collect();
            let right: Vec<_> = rows.iter().map(|p| p.rhs).collect();
            let lhs = geometry_inputs(dtype, format, &left).unwrap();
            let rhs = geometry_inputs(dtype, format, &right).unwrap();
            assert_eq!(lhs.len(), 33);
            for (a, b) in lhs.iter().zip(&rhs) {
                for input in [a, b] {
                    let addressed: Vec<_> = (0..input.logical.len())
                        .map(|i| input.storage[input.layout.element_offset(i).unwrap()])
                        .collect();
                    assert_eq!(addressed, input.logical);
                }
                assert_eq!(lookup(dtype, &a.logical, &b.logical).unwrap().len(), a.logical.len());
                let p = parameters(a, b, a.layout.shape().as_slice(), None).unwrap();
                assert!(float::validate(&p.map(|v| v as u32)).is_ok());
            }
            let fixture = oracle::scalar_fixtures(dtype)
                .into_iter()
                .find(|s| oracle::is_finite(dtype, s.scalar_bits) && s.scalar_bits != 0)
                .unwrap();
            for input in geometry_inputs(dtype, format, &fixture.inputs).unwrap() {
                assert_eq!(
                    lookup(dtype, &input.logical, &vec![fixture.scalar_bits; input.logical.len()]).unwrap().len(),
                    input.logical.len()
                );
                let p = parameters(&input, &input, input.layout.shape().as_slice(), Some(fixture.scalar_bits)).unwrap();
                assert_eq!(p[3] as u64 | ((p[5] as u64) << 32), fixture.scalar_bits);
                assert!(float::validate(&p.map(|v| v as u32)).is_ok());
            }
        }
    }

    #[test]
    fn scalar_schedule_and_strict_exception_comparison_keep_their_witnesses() {
        let mut totals = (0, 0, 0, 0);
        for (dtype, format) in FORMATS {
            let mut variants = [false; 4];
            let rows = oracle::pairs(dtype);
            totals.0 += rows.len();
            for fixture in oracle::scalar_fixtures(dtype) {
                variants[match fixture.scalar {
                    Scalar::Float(_) => 0,
                    Scalar::Int(_) => 1,
                    Scalar::UInt(_) => 2,
                    Scalar::Bool(_) => 3,
                }] = true;
                totals.1 += 1;
                totals.2 += fixture.expected.len();
                assert_eq!(fixture.inputs.len(), 10);
                assert_eq!(lookup(dtype, &fixture.inputs, &[fixture.scalar_bits; 10]).unwrap(), fixture.expected);
                let input = Input::flat(dtype, format, fixture.inputs).unwrap();
                let p = parameters(&input, &input, &[10], Some(fixture.scalar_bits)).unwrap();
                assert_eq!(p[3] as u64 | ((p[5] as u64) << 32), fixture.scalar_bits);
                assert!(float::validate(&p.map(|v| v as u32)).is_ok());
                if matches!(dtype, DType::F32 | DType::Flex32) && matches!(fixture.scalar, Scalar::Int(v) if v == (1i64 << 62) + (1i64 << 38) + 1) {
                    assert_eq!(fixture.scalar_bits, 0x5e80_0000);
                    let i = input.logical.iter().position(|&v| v == 0xc040_0000).unwrap();
                    assert_eq!(fixture.expected[i], 0x5e80_0000);
                }
            }
            assert!(variants.into_iter().all(|v| v));
            for fixture in oracle::storage_scalar_fixtures(dtype) {
                totals.3 += 1;
                assert_eq!(fixture.expected, [0]);
                assert_eq!(lookup(dtype, &fixture.inputs, &[fixture.scalar_bits]).unwrap(), fixture.expected);
            }
            let ieee: Vec<_> = rows.iter().filter(|p| oracle::is_nan(dtype, p.lhs)).map(|p| p.lhs).collect();
            assert!(!ieee.is_empty());
            let sign = 1u64 << (dtype.size() * 8 - 1);
            assert!(check_bits(dtype, &[ieee[0]], &[ieee[0] ^ sign], false).is_ok());
            assert!(check_bits(dtype, &[ieee[0]], &[ieee[0] ^ sign], true).is_err());
            assert!(check_bits(dtype, &[sign], &[0], false).is_err());
            for row in rows.iter().filter(|p| !oracle::is_nan(dtype, p.expected)) {
                assert!(check_bits(dtype, &[row.expected ^ 1], &[row.expected], false).is_err());
            }
        }
        assert_eq!(totals, (1608, 221, 2210, 2));
    }

    #[test]
    fn bounded_plan_retains_all_rows_and_precise_route_counts() {
        let mut counts = Counts::default();
        for (dtype, format) in FORMATS {
            let rows = oracle::pairs(dtype);
            let mut groups = Vec::new();
            for row in &rows {
                if !groups.contains(&row.group) {
                    groups.push(row.group);
                }
            }
            let scalars = oracle::scalar_fixtures(dtype);
            let storage = oracle::storage_scalar_fixtures(dtype).len();
            let geometry = geometry_inputs(dtype, format, &[rows[0].lhs]).unwrap().len();
            counts.tensor_rows += rows.len();
            counts.scalar_specs += scalars.len();
            counts.scalar_values += scalars.iter().map(|s| s.expected.len()).sum::<usize>();
            counts.storage_specs += storage;
            counts.raw += groups.len() + scalars.len() + storage + 2 * geometry + 2;
            let native = 2 * groups.len() + scalars.len() + storage + 3 * geometry + 4 + 3;
            let public = 3 * (groups.len() + scalars.len() + storage) + 3;
            counts.native += native;
            counts.public += public;
            if dtype == DType::F32 {
                counts.f32_routes += native + public;
            } else {
                counts.typed_routes += native + public;
            }
            let input = Input::flat(dtype, format, vec![rows[0].lhs; 5]).unwrap();
            let p = parameters(&input, &input, &[5], None).unwrap();
            let scalar = parameters(&input, &input, &[5], Some(rows[0].lhs)).unwrap();
            counts.rejected += malformed(p, scalar, dtype).len() + 15 + 12 + 6;
            if dtype == DType::F64 {
                counts.rejected += 4;
            }
        }
        counts.rejected += 8;
        counts.fresh = counts.raw + 15;
        counts.empty = 5 * 4 * 12;
        counts.queued = 5 * 4;
        assert_eq!(counts, EXPECTED);
    }

    #[test]
    fn malformed_metadata_rejects_without_a_device() {
        let mut count = 0;
        for (dtype, format) in FORMATS {
            let input = Input::flat(dtype, format, vec![oracle::pairs(dtype)[0].lhs; 5]).unwrap();
            let p = parameters(&input, &input, &[5], None).unwrap();
            let scalar = parameters(&input, &input, &[5], Some(input.logical[0])).unwrap();
            assert!(float::validate(&p.map(|v| v as u32)).is_ok());
            assert!(float::validate(&scalar.map(|v| v as u32)).is_ok());
            for bad in malformed(p, scalar, dtype) {
                assert!(float::validate(&bad.map(|v| v as u32)).is_err(), "{dtype:?} {bad:?}");
                count += 1;
            }
        }
        assert_eq!(count, 206);
    }
}
