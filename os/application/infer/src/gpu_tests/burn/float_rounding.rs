//! Guarded sqrt and integral-rounding launches with independent exact fixtures.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Scalar, Shape, TensorData, ops::FloatTensorOps, tensor::FloatTensor},
        tensor::{Tensor, TensorCreationOptions},
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::float_rounding::{self as oracle, Operation},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, cast::Format, elementwise::Elementwise, float_arithmetic as float};

type Result<T> = core::result::Result<T, String>;
const GUARD: u32 = 0x6bad_beef;
const POISON: u32 = 0xa53c_d791;
const FORMATS: [(DType, Format); 5] = [
    (DType::F64, Format::F64),
    (DType::F32, Format::F32),
    (DType::Flex32, Format::Flex32),
    (DType::F16, Format::F16),
    (DType::BF16, Format::BF16),
];
// Independent of SDK legality tables and the frozen arithmetic test counts.
const OPS: [Elementwise; 5] = [Elementwise::Sqrt, Elementwise::Floor, Elementwise::Ceil, Elementwise::Round, Elementwise::Trunc];
#[derive(Default)]
struct Counts {
    raw: usize,
    fresh: usize,
    native: usize,
    rejected: usize,
    public: usize,
    exhaustive: usize,
    exhaustive_values: usize,
}
fn bytes(dtype: DType, values: &[u64]) -> Vec<u8> {
    oracle::pack(dtype, values)
}
fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("float rounding: {e}"))
}

fn packed(dtype: DType, values: &[u64]) -> Vec<u32> {
    bytes(dtype, values)
        .chunks(4)
        .map(|bytes| {
            let mut word = [0; 4];
            word[..bytes.len()].copy_from_slice(bytes);
            u32::from_le_bytes(word)
        })
        .collect()
}

/// The expected logical order is supplied separately from the GPU layout.
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
            logical: values.clone(),
            storage: values,
        })
    }

    fn tensor(&self) -> Result<NativeTensor> {
        gpu(NativeTensor::from_data(self.data()).and_then(|t| t.with_layout(self.layout.clone())))
    }

    fn data(&self) -> TensorData {
        TensorData::from_bytes_vec(bytes(self.dtype, &self.storage), Shape::new([self.storage.len()]), self.dtype)
    }
}

fn unchanged(backend: &Nvidia, buffer: &d3os_compute::nvidia::GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if actual != expected {
        return Err(String::from("rounding source/guard/poison changed"));
    }
    Ok(())
}

/// No operand readback, upload, separate allocation/clear, or hidden kernel.
fn resident(
    backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT], fresh: bool,
) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE] {
        if after[service as usize].calls != before[service as usize].calls {
            return Err(format!("rounding resident service={service}"));
        }
    }
    for (service, expected) in [(abi::LAUNCH_ALLOCATE, u64::from(fresh)), (abi::LAUNCH, u64::from(!fresh))] {
        if after[service as usize].calls - before[service as usize].calls != expected {
            return Err(format!("rounding full writer service={service} expected_calls={expected}"));
        }
    }
    let after = gpu(backend.profile())?;
    let mut launches = 0;
    for (index, (old, new)) in profile.iter().zip(&after).enumerate() {
        let delta = new.calls - old.calls;
        if delta != 0 && index != Kernel::FloatArithmetic as usize && index != Kernel::TensorElementwise as usize {
            return Err(format!("rounding operation used another kernel: {index}"));
        }
        launches += delta;
    }
    if launches != 1 {
        return Err(format!("rounding full writer kernel launches={launches}"));
    }
    Ok(())
}

fn no_launch(backend: &Nvidia, before: &abi::QueueProfile) -> Result<()> {
    if gpu(backend.queue_profile())?.launches != before.launches {
        return Err(String::from("invalid/empty rounding operation launched GPU work"));
    }
    Ok(())
}

fn parameters(input: &Input, op: Elementwise) -> Result<[usize; 32]> {
    let mut p = gpu(input.layout.elementwise_parameters(&input.layout, Elementwise::Add, 0, false))?;
    p[2] = op as usize | ((input.format as usize) << 8);
    p[3] = 0;
    p[5] = 0;
    p[7] = 0;
    p[24..32].fill(0);
    Ok(p)
}
/// IEEE encodings, with no host floating arithmetic used to predict results.
#[derive(Clone, Copy)]
struct Ieee {
    sign: u64,
    infinity: u64,
    one: u64,
    fraction: u32,
    bias: i32,
}
impl Ieee {
    fn new(dtype: DType) -> Self {
        let (sign, infinity, one, fraction, bias) = match dtype {
            DType::F64 => (1 << 63, 0x7ff0_0000_0000_0000, 0x3ff0_0000_0000_0000, 52, 1023),
            DType::F32 | DType::Flex32 => (1 << 31, 0x7f80_0000, 0x3f80_0000, 23, 127),
            DType::F16 => (1 << 15, 0x7c00, 0x3c00, 10, 15),
            DType::BF16 => (1 << 15, 0x7f80, 0x3f80, 7, 127),
            _ => unreachable!(),
        };
        Self {
            sign,
            infinity,
            one,
            fraction,
            bias,
        }
    }
    fn pow2(self, exponent: i32) -> u64 {
        ((exponent + self.bias) as u64) << self.fraction
    }
    fn nan(self) -> u64 {
        self.infinity | (1 << (self.fraction - 1))
    }
    fn is_nan(self, bits: u64) -> bool {
        bits & !self.sign > self.infinity
    }
}
fn check_bits(dtype: DType, actual: &[u64], expected: &[u64], exact_nan: bool) -> Result<()> {
    let ieee = Ieee::new(dtype);
    if actual.len() != expected.len() {
        return Err(String::from("rounding logical length mismatch"));
    }
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if got != want && (exact_nan || !ieee.is_nan(want) || !ieee.is_nan(got)) {
            return Err(format!(
                "rounding {dtype:?} element={index} actual={got:#018x} expected={want:#018x} exact_nan={exact_nan}"
            ));
        }
    }
    Ok(())
}
fn decode(dtype: DType, bytes: &[u8]) -> Vec<u64> {
    oracle::unpack(dtype, bytes)
}
fn check_data(data: &TensorData, dtype: DType, shape: &[usize], expected: &[u64], exact_nan: bool) -> Result<()> {
    if data.dtype() != dtype || data.shape().as_slice() != shape || data.as_bytes().len() != expected.len() * dtype.size() {
        return Err(format!(
            "rounding result metadata/bytes dtype={dtype:?} shape={shape:?} actual_dtype={:?} actual_shape={:?}",
            data.dtype(),
            data.shape()
        ));
    }
    check_bits(dtype, &decode(dtype, data.as_bytes()), expected, exact_nan)
}
fn check(tensor: &NativeTensor, dtype: DType, shape: &[usize], expected: &[u64], exact_nan: bool) -> Result<()> {
    check_data(&gpu(tensor.read())?, dtype, shape, expected, exact_nan)
}
fn check_words(backend: &Nvidia, buffer: &d3os_compute::nvidia::GpuBuffer<u32>, dtype: DType, expected: &[u64], guarded: bool, exact_nan: bool) -> Result<()> {
    let count = packed(dtype, expected).len();
    let mut actual = vec![0; count + if guarded { 2 } else { 0 }];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if guarded && (actual[0] != GUARD || actual[count + 1] != GUARD) {
        return Err(String::from("rounding output guard changed"));
    }
    let words = &actual[usize::from(guarded)..usize::from(guarded) + count];
    if dtype.size() == 2 && expected.len() % 2 == 1 && words[count - 1] >> 16 != 0 {
        return Err(String::from("rounding odd packed output tail is not zero"));
    }
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).take(expected.len() * dtype.size()).collect();
    check_bits(dtype, &decode(dtype, &bytes), expected, exact_nan)
}
fn operation(op: Elementwise) -> Operation {
    match op {
        Elementwise::Sqrt => Operation::Sqrt,
        Elementwise::Floor => Operation::Floor,
        Elementwise::Ceil => Operation::Ceil,
        Elementwise::Round => Operation::Round,
        Elementwise::Trunc => Operation::Trunc,
        _ => unreachable!(),
    }
}
fn native_exact_nan(dtype: DType, op: Elementwise) -> bool {
    op != Elementwise::Sqrt && (dtype != DType::F32 || op == Elementwise::Round)
}
fn reference(dtype: DType, op: Elementwise, input: &[u64]) -> Result<Vec<u64>> {
    input
        .iter()
        .map(|&bits| {
            // The generated portable fixtures classify NaNs. Raw integral routes
            // additionally promise to retain the original encoding, even for sNaNs.
            if let Some(bits) = oracle::raw_integral_nan_bits(dtype, operation(op), bits) {
                Ok(bits)
            } else {
                oracle::expected_bits(dtype, operation(op), bits).ok_or_else(|| format!("rounding oracle domain {dtype:?} {op:?} {bits:#x}"))
            }
        })
        .collect()
}
fn route(backend: &Nvidia, profile: &[abi::ProfileEntry; abi::KERNEL_COUNT], kernel: Kernel) -> Result<()> {
    let after = gpu(backend.profile())?;
    for (index, (old, new)) in profile.iter().zip(&after).enumerate() {
        if new.calls - old.calls != u64::from(index == kernel as usize) {
            return Err(format!("rounding route expected={kernel:?} changed_kernel={index}"));
        }
    }
    Ok(())
}
fn native_kernel(dtype: DType) -> Kernel {
    if dtype == DType::F32 {
        Kernel::TensorElementwise
    } else {
        Kernel::FloatArithmetic
    }
}

/// Each raw case has guarded sources, an unused poisoned RHS, and a poisoned
/// destination. The exhaustive cases use one large guarded launch apiece.
fn raw(backend: &Nvidia, input: &Input, op: Elementwise, expected: &[u64], exhaustive: bool, counts: &mut Counts) -> Result<()> {
    let mut words = vec![GUARD];
    words.extend(packed(input.dtype, &input.storage));
    if input.dtype.size() == 2 && input.storage.len() % 2 == 1 {
        *words.last_mut().unwrap() |= 0xffff_0000;
    }
    words.push(GUARD);
    let source = gpu(backend.upload_words(&words))?;
    let rhs_words = [GUARD, POISON, GUARD];
    let unused_rhs = gpu(backend.upload_words(&rhs_words))?;
    let word_count = packed(input.dtype, expected).len();
    let mut poison = vec![POISON; word_count + 2];
    poison[0] = GUARD;
    poison[word_count + 1] = GUARD;
    let output = gpu(backend.upload_words(&poison))?;
    let p = parameters(input, op)?;
    let before = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatArithmetic, &[(&source, 1), (&unused_rhs, 1), (&output, 1)], &p)))?;
    resident(backend, &before, &profile, false)?;
    route(backend, &profile, Kernel::FloatArithmetic)?;
    check_words(backend, &output, input.dtype, expected, true, op != Elementwise::Sqrt).map_err(|e| format!("raw {op:?} exhaustive={exhaustive}: {e}"))?;
    counts.raw += 1;
    if exhaustive {
        counts.exhaustive += 1;
        counts.exhaustive_values += expected.len();
    } else {
        gpu(backend.write_words(&output, 1, &vec![POISON; word_count]))?;
        drop(output);
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        // Input aliases are legal. The unused RHS must not be dereferenced.
        let output = gpu(backend.with_commands(|| backend.produce_words(word_count, Kernel::FloatArithmetic, &[(&source, 1), (&source, 1)], &p)))?;
        resident(backend, &before, &profile, true)?;
        route(backend, &profile, Kernel::FloatArithmetic)?;
        check_words(backend, &output, input.dtype, expected, false, op != Elementwise::Sqrt)?;
        counts.fresh += 1;
    }
    unchanged(backend, &source, &words)?;
    unchanged(backend, &unused_rhs, &rhs_words)?;
    Ok(())
}
fn native(backend: &Nvidia, input: &Input, op: Elementwise, expected: &[u64], counts: &mut Counts) -> Result<()> {
    let source = input.tensor()?;
    let alias = source.clone();
    for old in [false, true] {
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| if old { source.unary(op) } else { source.float_unary(op) }))?;
        resident(backend, &before, &profile, true)?;
        route(backend, &profile, native_kernel(input.dtype))?;
        check(
            &output,
            input.dtype,
            input.layout.shape().as_slice(),
            expected,
            native_exact_nan(input.dtype, op),
        )
        .map_err(|e| format!("Native {op:?} old={old}: {e}"))?;
        counts.native += 1;
    }
    check(&source, input.dtype, input.layout.shape().as_slice(), &input.logical, true)?;
    check(&alias, input.dtype, input.layout.shape().as_slice(), &input.logical, true)
}
fn both(backend: &Nvidia, input: &Input, op: Elementwise, expected: &[u64], counts: &mut Counts) -> Result<()> {
    raw(backend, input, op, expected, false, counts)?;
    native(backend, input, op, expected, counts)
}
impl Ieee {
    /// Encode an exactly representable small dyadic using integer arithmetic.
    fn dyadic(self, numerator: u64, exponent: i32) -> u64 {
        if numerator == 0 {
            return 0;
        }
        let top = 63 - numerator.leading_zeros();
        assert!(top <= self.fraction);
        self.pow2(top as i32 + exponent) | ((numerator - (1 << top)) << (self.fraction - top))
    }
}

/// Fixed results independent of the generated table and production bit helper.
fn witnesses(dtype: DType, op: Elementwise) -> Vec<(u64, u64)> {
    let f = Ieee::new(dtype);
    let mut cases = vec![(0, 0), (f.sign, f.sign), (f.infinity, f.infinity)];
    for nan in [f.infinity | 1, f.sign | f.infinity | 0x15, f.nan() | 0x31, f.sign | f.nan() | 0x53] {
        cases.push((nan, nan));
    }
    if op == Elementwise::Sqrt {
        let sqrt_two = match dtype {
            DType::F64 => 0x3ff6_a09e_667f_3bcd,
            DType::F32 | DType::Flex32 => 0x3fb5_04f3,
            DType::F16 => 0x3da8,
            DType::BF16 => 0x3fb5,
            _ => unreachable!(),
        };
        let sqrt_tiny = match dtype {
            DType::F64 => f.pow2(-537),
            DType::F32 | DType::Flex32 => 0x1a35_04f3,
            DType::F16 => f.pow2(-12),
            DType::BF16 => 0x1e35,
            _ => unreachable!(),
        };
        let exponent = match dtype {
            DType::F64 => 1000,
            DType::F16 => 14,
            _ => 100,
        };
        cases.extend([
            (f.one, f.one),
            (f.pow2(1), sqrt_two),
            (f.pow2(2), f.pow2(1)),
            (1, sqrt_tiny),
            (f.sign | 1, f.nan()),
            (f.sign | f.one, f.nan()),
            (f.sign | f.infinity, f.nan()),
            (f.pow2(exponent), f.pow2(exponent / 2)),
            (f.dyadic(25, -2) + 1, f.dyadic(5, -1)),
        ]);
        if dtype == DType::F64 {
            cases.push((0x3ff0_0000_0000_1000, 0x3ff0_0000_0000_0800));
        }
    } else {
        cases.push((f.sign | f.infinity, f.sign | f.infinity));
        // All ties n+0.5 for n=0..7 and both adjacent encodings, both signs.
        for n in 0..8 {
            let tie = f.dyadic(2 * n + 1, -1);
            for (bits, side) in [(tie - 1, -1), (tie, 0), (tie + 1, 1)] {
                for sign in [0, f.sign] {
                    let magnitude = match op {
                        Elementwise::Floor => n + u64::from(sign != 0),
                        Elementwise::Ceil => n + u64::from(sign == 0),
                        Elementwise::Trunc => n,
                        Elementwise::Round => n + u64::from(side > 0 || (side == 0 && n % 2 != 0)),
                        _ => unreachable!(),
                    };
                    cases.push((sign | bits, sign | f.dyadic(magnitude, 0)));
                }
            }
        }
        for sign in [0, f.sign] {
            for tiny in [1, (1 << f.fraction) - 1, 1 << f.fraction] {
                let up = (op == Elementwise::Floor && sign != 0) || (op == Elementwise::Ceil && sign == 0);
                cases.push((sign | tiny, sign | if up { f.one } else { 0 }));
            }
            for big in [f.pow2(f.fraction as i32), f.pow2(f.fraction as i32) + 1, f.infinity - 1] {
                cases.push((sign | big, sign | big));
            }
        }
        // Fractional ties near the last representable half-integers, including
        // neighbors that are already integral at the largest such exponent.
        for exponent in [f.fraction / 2, f.fraction - 2, f.fraction - 1] {
            let n = 1 << exponent;
            let tie = f.pow2(exponent as i32) | (1 << (f.fraction - exponent - 1));
            for (bits, side) in [(tie - 1, -1), (tie, 0), (tie + 1, 1)] {
                for sign in [0, f.sign] {
                    let integer = if exponent + 1 == f.fraction && side != 0 {
                        n + u64::from(side > 0)
                    } else {
                        n + u64::from(match op {
                            Elementwise::Floor => sign != 0,
                            Elementwise::Ceil => sign == 0,
                            Elementwise::Round => side > 0,
                            Elementwise::Trunc => false,
                            _ => unreachable!(),
                        })
                    };
                    cases.push((sign | bits, sign | f.dyadic(integer, 0)));
                }
            }
        }
        if dtype == DType::F64 {
            // +/- (2^32 + 0.5), including low-word fractional precision.
            let integer = 0x41f0_0000_0000_0000;
            for sign in [0, f.sign] {
                let increment = (op == Elementwise::Floor && sign != 0) || (op == Elementwise::Ceil && sign == 0);
                cases.push((sign | 0x41f0_0000_0008_0000, sign | (integer + if increment { 1 << 20 } else { 0 })));
            }
        }
    }
    // Every fixed case has an odd output count, exercising half-tail ownership.
    if cases.len() % 2 == 0 {
        cases.push((0, 0));
    }
    cases
}
fn matrices(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        for op in OPS {
            let fixed = witnesses(dtype, op);
            let input = Input::flat(dtype, format, fixed.iter().map(|v| v.0).collect())?;
            both(backend, &input, op, &fixed.iter().map(|v| v.1).collect::<Vec<_>>(), counts)?;
            let values = oracle::pool(dtype);
            let want = reference(dtype, op, &values)?;
            both(backend, &Input::flat(dtype, format, values)?, op, &want, counts)?;
        }
    }
    Ok(())
}
fn exhaustive_half(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in [FORMATS[3], FORMATS[4]] {
        for op in OPS {
            let values: Vec<_> = (0..=u16::MAX).map(u64::from).collect();
            let want = values
                .iter()
                .map(|&bits| {
                    if let Some(bits) = oracle::raw_integral_nan_bits(dtype, operation(op), bits) {
                        Ok(bits)
                    } else {
                        oracle::half_expected_bits(dtype, operation(op), bits as u16)
                            .map(u64::from)
                            .ok_or_else(|| format!("rounding missing full half table {dtype:?} {op:?} {bits:#x}"))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            raw(backend, &Input::flat(dtype, format, values)?, op, &want, true, counts)?;
        }
    }
    Ok(())
}
fn layouts(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let pool = oracle::pool(dtype);
        for (case, length) in [1, 2, 3, 4, 5, 127, 128, 129, 257].into_iter().enumerate() {
            let storage: Vec<_> = (0..length * 2 + 4).map(|i| pool[i % pool.len()]).collect();
            for step in [-1i64, 2] {
                let offset = if step < 0 { length as i64 } else { 1 };
                let logical = (0..length).map(|i| storage[(offset + i as i64 * step) as usize]).collect();
                let input = Input {
                    dtype,
                    format,
                    layout: gpu(Layout::from_parts(Shape::new([length]), vec![step], offset, storage.len()))?,
                    storage: storage.clone(),
                    logical,
                };
                let op = OPS[(case + usize::from(step > 0)) % OPS.len()];
                let want = reference(dtype, op, &input.logical)?;
                both(backend, &input, op, &want, counts)?;
            }
        }
        let storage: Vec<_> = (0..24).map(|i| pool[i % pool.len()]).collect();
        let permuted = Input {
            dtype,
            format,
            layout: gpu(Layout::contiguous(Shape::new([2, 3, 4]))
                .and_then(|l| l.permute(&[2, 0, 1]))
                .and_then(|l| l.flip(&[0, 2])))?,
            logical: (0..4)
                .flat_map(|i| (0..2).flat_map(move |j| (0..3).map(move |k| j * 12 + (2 - k) * 4 + 3 - i)))
                .map(|i| storage[i])
                .collect(),
            storage,
        };
        let broadcast = Input {
            dtype,
            format,
            layout: gpu(Layout::from_parts(Shape::new([3, 5]), vec![0, 0], 1, 3))?,
            storage: vec![POISON as u64 & (Ieee::new(dtype).sign - 1), Ieee::new(dtype).dyadic(5, -1), 0],
            logical: vec![Ieee::new(dtype).dyadic(5, -1); 15],
        };
        for input in [permuted, broadcast] {
            for op in OPS {
                let want = reference(dtype, op, &input.logical)?;
                both(backend, &input, op, &want, counts)?;
            }
        }
        for shape in [vec![], vec![1, 1, 1, 1, 1, 1, 1, 5]] {
            let length = shape.iter().product();
            let mut input = Input::flat(dtype, format, vec![Ieee::new(dtype).dyadic(5, -1); length])?;
            input.layout = gpu(Layout::contiguous(Shape::from(shape)))?;
            for op in OPS {
                let want = reference(dtype, op, &input.logical)?;
                both(backend, &input, op, &want, counts)?;
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
    let mut scalar = p;
    scalar[2] |= float::SCALAR_RHS as usize;
    cases.push(scalar);
    cases
}
fn raw_rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let input = Input::flat(dtype, format, vec![Ieee::new(dtype).one; 5])?;
        let words = vec![POISON; packed(dtype, &input.storage).len() + 4];
        let a = gpu(backend.upload_words(&words))?;
        let rhs = gpu(backend.upload_words(&words))?;
        let output = gpu(backend.upload_words(&words))?;
        let short = gpu(backend.upload_words(&[POISON]))?;
        let before = gpu(backend.queue_profile())?;
        for op in OPS {
            let p = parameters(&input, op)?;
            for bad in malformed(p) {
                if backend
                    .dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&rhs, 1), (&output, 1)], &bad)
                    .is_ok()
                {
                    return Err(format!("rounding malformed metadata accepted {dtype:?} {op:?}: {bad:?}"));
                }
                counts.rejected += 1;
            }
            for bad in [p[..31].to_vec(), {
                let mut v = p.to_vec();
                v.push(0);
                v
            }] {
                if backend
                    .dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&rhs, 1), (&output, 1)], &bad)
                    .is_ok()
                {
                    return Err(String::from("rounding parameter length accepted"));
                }
                counts.rejected += 1;
            }
            for buffers in [vec![(&a, 1), (&output, 1)], vec![(&a, 1), (&rhs, 1), (&output, 1), (&short, 0)]] {
                if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p).is_ok() {
                    return Err(String::from("rounding binding arity accepted"));
                }
                counts.rejected += 1;
            }
            // Even the unused RHS cannot alias the fresh output binding.
            for (out, offset) in [(&a, 1), (&a, 2), (&rhs, 1), (&rhs, 0)] {
                if backend
                    .dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&rhs, 1), (out, offset)], &p)
                    .is_ok()
                {
                    return Err(String::from("rounding exact/partial output alias accepted"));
                }
                counts.rejected += 1;
            }
            for buffers in [
                [(&short, 0), (&rhs, 1), (&output, 1)],
                [(&a, 1), (&short, 1), (&output, 1)],
                [(&a, 1), (&rhs, 1), (&short, 0)],
            ] {
                if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p).is_ok() {
                    return Err(String::from("rounding physical binding bounds accepted"));
                }
                counts.rejected += 1;
            }
            if dtype == DType::F64 {
                let one = parameters(&Input::flat(dtype, format, vec![0])?, op)?;
                for buffers in [[(&short, 0), (&rhs, 1), (&output, 1)], [(&a, 1), (&rhs, 1), (&short, 0)]] {
                    if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &one).is_ok() {
                        return Err(String::from("rounding F64 second physical word not checked"));
                    }
                    counts.rejected += 1;
                }
                let mut high = p;
                high[4] = i32::MAX as usize - 5;
                high[6] = i32::MAX as usize;
                if backend
                    .dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&rhs, 1), (&output, 1)], &high)
                    .is_ok()
                {
                    return Err(String::from("rounding F64 high physical bounds accepted"));
                }
                counts.rejected += 1;
            }
        }
        no_launch(backend, &before)?;
        unchanged(backend, &a, &words)?;
        unchanged(backend, &rhs, &words)?;
        unchanged(backend, &output, &words)?;
        unchanged(backend, &short, &[POISON])?;
        // A successful operation after rejected metadata must still work.
        let want = vec![Ieee::new(dtype).one; 5];
        both(backend, &input, Elementwise::Sqrt, &want, counts)?;
    }
    Ok(())
}
fn native_validation(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        for length in [0, 3] {
            let a = Input::flat(dtype, format, vec![Ieee::new(dtype).one; length])?.tensor()?;
            let before = gpu(backend.queue_profile())?;
            for op in OPS {
                for result in [a.float_binary(&a, op), a.float_scalar(op, Scalar::Float(1.0))] {
                    if !matches!(result, Err(Error::InvalidParameter)) {
                        return Err(format!("rounding unary arity accepted {dtype:?} {op:?}"));
                    }
                    counts.rejected += 1;
                }
            }
            for op in [Elementwise::Add, Elementwise::Equal, Elementwise::Remainder] {
                if !matches!(a.float_unary(op), Err(Error::InvalidParameter)) {
                    return Err(String::from("rounding invalid unary selector accepted"));
                }
                counts.rejected += 1;
            }
            no_launch(backend, &before)?;
        }
        let input = Input::flat(dtype, format, vec![Ieee::new(dtype).one; 3])?.tensor()?;
        let before = gpu(backend.queue_profile())?;
        for layout in [
            Layout::from_parts(Shape::new([4]), vec![1], 0, 4),
            Layout::from_parts(Shape::new([1]), vec![1], 3, 4),
            Layout::contiguous(Shape::new([1, 1, 1, 1, 1, 1, 1, 1, 1])),
        ] {
            if layout.and_then(|l| input.with_layout(l)).is_ok() {
                return Err(String::from("rounding invalid rank/physical view accepted"));
            }
            counts.rejected += 1;
        }
        if input.reshape(Shape::new([2, 2])).is_ok() {
            return Err(String::from("rounding invalid reshape accepted"));
        }
        counts.rejected += 1;
        no_launch(backend, &before)?;
    }
    for length in [0, 3] {
        let source = gpu(NativeTensor::from_data(TensorData::from_bytes_vec(
            vec![0; length * 4],
            Shape::new([length]),
            DType::I32,
        )))?;
        let before = gpu(backend.queue_profile())?;
        for op in OPS {
            for result in [source.float_unary(op), source.unary(op)] {
                if !matches!(result, Err(Error::InvalidParameter)) {
                    return Err(String::from("rounding nonfloat Native storage accepted"));
                }
                counts.rejected += 1;
            }
        }
        no_launch(backend, &before)?;
    }
    Ok(())
}
fn empty_and_lifetimes(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let empty = gpu(NativeTensor::empty(Shape::new([2, 0, 3]), dtype))?;
        let before = gpu(backend.queue_profile())?;
        for op in OPS {
            for old in [false, true] {
                let output = gpu(if old { empty.unary(op) } else { empty.float_unary(op) })?;
                check(&output, dtype, &[2, 0, 3], &[], true)?;
                counts.native += 1;
            }
        }
        no_launch(backend, &before)?;
        for op in OPS {
            let values = oracle::pool(dtype);
            let want = reference(dtype, op, &values)?;
            let input = Input::flat(dtype, format, values)?;
            let source = input.tensor()?;
            let before = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let output = gpu(backend.with_commands(|| {
                let result = source.float_unary(op)?;
                drop(source);
                Ok::<_, Error>(result)
            }))?;
            resident(backend, &before, &profile, true)?;
            route(backend, &profile, native_kernel(dtype))?;
            check(&output, dtype, &[input.logical.len()], &want, native_exact_nan(dtype, op))?;
            counts.native += 1;
            let source = gpu(backend.upload_words(&packed(dtype, &input.storage)))?;
            let p = parameters(&input, op)?;
            let before = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let output = gpu(backend.with_commands(|| {
                let result = backend.produce_words(packed(dtype, &want).len(), Kernel::FloatArithmetic, &[(&source, 0), (&source, 0)], &p)?;
                drop(source);
                Ok::<_, Error>(result)
            }))?;
            resident(backend, &before, &profile, true)?;
            route(backend, &profile, Kernel::FloatArithmetic)?;
            check_words(backend, &output, dtype, &want, false, op != Elementwise::Sqrt)?;
            counts.fresh += 1;
        }
    }
    Ok(())
}
fn dispatch_unary(source: Tensor<1>, op: Elementwise) -> Tensor<1> {
    match op {
        Elementwise::Sqrt => source.sqrt(),
        Elementwise::Floor => source.floor(),
        Elementwise::Ceil => source.ceil(),
        Elementwise::Round => source.round(),
        Elementwise::Trunc => source.trunc(),
        _ => unreachable!(),
    }
}
fn backend_dispatch(source: FloatTensor<Native>, op: Elementwise) -> Tensor<1> {
    let output = match op {
        Elementwise::Sqrt => Native::float_sqrt(source),
        Elementwise::Floor => Native::float_floor(source),
        Elementwise::Ceil => Native::float_ceil(source),
        Elementwise::Round => Native::float_round(source),
        Elementwise::Trunc => Native::float_trunc(source),
        _ => unreachable!(),
    };
    Tensor::<1>::from_primitive::<Native>(output)
}
fn backend_unary(input: &Input, op: Elementwise) -> Tensor<1> {
    backend_dispatch(Native::float_from_data(input.data(), &NativeDevice), op)
}
fn public_cases(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("rounding public device: {e}"))?;
    for (dtype, format) in FORMATS {
        for op in OPS {
            let values = oracle::pool(dtype);
            let want = reference(dtype, op, &values)?;
            let input = Input::flat(dtype, format, values)?;
            for direct in [false, true] {
                // Exclude construction/upload and readback from residency.
                let source = if direct {
                    None
                } else {
                    Some(Tensor::<1>::from_data(
                        input.data(),
                        TensorCreationOptions::new(device.clone()).with_dtype(dtype),
                    ))
                };
                if let Some(source) = source {
                    let before = gpu(backend.service_profile())?;
                    let profile = gpu(backend.profile())?;
                    let output = dispatch_unary(source, op);
                    resident(backend, &before, &profile, true)?;
                    route(backend, &profile, native_kernel(dtype))?;
                    check_data(
                        &output.try_into_data().map_err(|e| format!("rounding Dispatch {op:?}: {e:?}"))?,
                        dtype,
                        &[input.logical.len()],
                        &want,
                        native_exact_nan(dtype, op),
                    )?;
                } else {
                    let source = Native::float_from_data(input.data(), &NativeDevice);
                    let before = gpu(backend.service_profile())?;
                    let profile = gpu(backend.profile())?;
                    let output = backend_dispatch(source, op);
                    resident(backend, &before, &profile, true)?;
                    route(backend, &profile, native_kernel(dtype))?;
                    check_data(
                        &output.try_into_data().map_err(|e| format!("rounding backend {op:?}: {e:?}"))?,
                        dtype,
                        &[input.logical.len()],
                        &want,
                        native_exact_nan(dtype, op),
                    )?;
                }
                counts.public += 1;
            }
        }
        // Reading each cloned intermediate forces destination materialization.
        let f = Ieee::new(dtype);
        let input = Input::flat(dtype, format, vec![f.dyadic(25, -2) + 1; 3])?;
        let mut value = Tensor::<1>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
        for (op, expected) in [(Elementwise::Sqrt, f.dyadic(5, -1)), (Elementwise::Round, f.pow2(1))] {
            let before = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            value = dispatch_unary(value, op);
            resident(backend, &before, &profile, true)?;
            route(backend, &profile, native_kernel(dtype))?;
            check_data(
                &value.clone().try_into_data().map_err(|e| format!("rounding materialized {op:?}: {e:?}"))?,
                dtype,
                &[3],
                &[expected; 3],
                true,
            )?;
            counts.public += 1;
        }
    }
    Ok(())
}
fn public_errors(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("rounding error device: {e}"))?;
    for length in [0, 3] {
        for op in OPS {
            let invalid = TensorData::from_bytes_vec(vec![0; length * 4], Shape::new([length]), DType::I32);
            let before = gpu(backend.queue_profile())?;
            let source = Native::float_from_data(invalid, &NativeDevice);
            let failed = match op {
                Elementwise::Sqrt => Native::float_sqrt(source),
                Elementwise::Floor => Native::float_floor(source),
                Elementwise::Ceil => Native::float_ceil(source),
                Elementwise::Round => Native::float_round(source),
                Elementwise::Trunc => Native::float_trunc(source),
                _ => unreachable!(),
            };
            no_launch(backend, &before)?;
            // Failed I32 metadata cannot be reconstructed as a Float tensor.
            // Synchronize first, then acknowledge the failure at backend level.
            let sync_error = device.sync().is_err_and(|e| !e.is_device_poisoned());
            let failed_read = d3os_ai::burn::tensor::try_read_sync(Native::float_into_data(failed))
                .ok_or_else(|| String::from("rounding invalid dtype read did not complete"))?;
            if !sync_error || !failed_read.is_err_and(|e| !e.is_device_poisoned()) || device.sync().is_err() {
                return Err(format!("rounding invalid dtype acknowledgment failed {op:?} length={length}"));
            }
            no_launch(backend, &before)?;
            counts.rejected += 1;
            let input = Input::flat(DType::F64, Format::F64, vec![0x4019_0000_0000_0001; length])?;
            check_data(
                &backend_unary(&input, Elementwise::Sqrt)
                    .try_into_data()
                    .map_err(|e| format!("rounding backend recovery: {e:?}"))?,
                DType::F64,
                &[length],
                &vec![0x4004_0000_0000_0000; length],
                true,
            )?;
            counts.public += 1;
        }
        // Valid Float kinds retain high-level error propagation coverage.
        let a = Input::flat(DType::F32, Format::F32, vec![0x3f80_0000; length])?;
        let b = Input::flat(DType::Flex32, Format::Flex32, vec![0x3f80_0000; length])?;
        let left = Tensor::<1>::from_data(a.data(), TensorCreationOptions::new(device.clone()).with_dtype(a.dtype));
        let right = Tensor::<1>::from_data(b.data(), TensorCreationOptions::new(device.clone()).with_dtype(b.dtype));
        let before = gpu(backend.queue_profile())?;
        let failed = left + right;
        let dependent = failed.clone().sqrt().round();
        no_launch(backend, &before)?;
        if !device.sync().is_err_and(|e| !e.is_device_poisoned())
            || !dependent.try_into_data().is_err_and(|e| !e.is_device_poisoned())
            || !failed.try_into_data().is_err_and(|e| !e.is_device_poisoned())
            || device.sync().is_err()
        {
            return Err(format!("rounding mixed Float error recovery length={length}"));
        }
        no_launch(backend, &before)?;
        counts.rejected += 1;
        let valid = Tensor::<1>::from_data(a.data(), TensorCreationOptions::new(device.clone()).with_dtype(a.dtype))
            .sqrt()
            .round();
        check_data(
            &valid.try_into_data().map_err(|e| format!("rounding public recovery: {e:?}"))?,
            a.dtype,
            &[length],
            &a.logical,
            true,
        )?;
        counts.public += 1;
    }
    Ok(())
}
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    let mut counts = Counts::default();
    matrices(backend, &mut counts)?;
    layouts(backend, &mut counts)?;
    exhaustive_half(backend, &mut counts)?;
    raw_rejections(backend, &mut counts)?;
    native_validation(backend, &mut counts)?;
    empty_and_lifetimes(backend, &mut counts)?;
    public_cases(backend, &mut counts)?;
    public_errors(backend, &mut counts)?;
    let actual = (
        counts.raw,
        counts.fresh,
        counts.native,
        counts.rejected,
        counts.public,
        counts.exhaustive,
        counts.exhaustive_values,
    );
    if actual != (255, 270, 565, 1422, 72, 10, 655360) {
        return Err(format!("rounding case counts changed: {actual:?}"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_FLOAT_ROUNDING_OK operations=5 formats=5 raw={} fresh={} native={} rejected={} public={} exhaustive={} exhaustive_values={} oracle=exact_ieee_fixtures guards=ok tails=zero resident=true queued_lifetime=verified f32_fastpath=verified errors=recoverable backend=nvidia",
        counts.raw, counts.fresh, counts.native, counts.rejected, counts.public, counts.exhaustive, counts.exhaustive_values,
    ));
    Ok(())
}
