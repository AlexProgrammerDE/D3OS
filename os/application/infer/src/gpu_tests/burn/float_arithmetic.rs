//! Guarded typed float launches, IEEE edge vectors, and public Scalar preservation.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Scalar, Shape, TensorData, ops::FloatTensorOps},
        tensor::{Tensor, TensorCreationOptions},
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::float_arithmetic::{self as oracle, Operation},
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
// Independent of SDK legality tables: omitting an operation there must fail here.
const OPS: [Elementwise; 8] = [
    Elementwise::Add,
    Elementwise::Subtract,
    Elementwise::Multiply,
    Elementwise::Divide,
    Elementwise::Negate,
    Elementwise::Absolute,
    Elementwise::Reciprocal,
    Elementwise::Sign,
];
fn unary(op: Elementwise) -> bool {
    matches!(op, Elementwise::Negate | Elementwise::Absolute | Elementwise::Reciprocal | Elementwise::Sign)
}
#[derive(Default)]
struct Counts {
    raw: usize,
    fresh: usize,
    native: usize,
    rejected: usize,
    public: usize,
}
fn bytes(dtype: DType, values: &[u64]) -> Vec<u8> {
    oracle::pack(dtype, values)
}
fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("float arithmetic: {e}"))
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
        return Err(String::from("float source/guard/poison changed"));
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
            return Err(format!("float resident service={service}"));
        }
    }
    for (service, expected) in [(abi::LAUNCH_ALLOCATE, u64::from(fresh)), (abi::LAUNCH, u64::from(!fresh))] {
        if after[service as usize].calls - before[service as usize].calls != expected {
            return Err(format!("float full writer service={service} expected_calls={expected}"));
        }
    }
    let after = gpu(backend.profile())?;
    let mut launches = 0;
    for (index, (old, new)) in profile.iter().zip(&after).enumerate() {
        let delta = new.calls - old.calls;
        if delta != 0 && index != Kernel::FloatArithmetic as usize && index != Kernel::TensorElementwise as usize {
            return Err(format!("float operation used another kernel: {index}"));
        }
        launches += delta;
    }
    if launches != 1 {
        return Err(format!("float full writer kernel launches={launches}"));
    }
    Ok(())
}

fn no_launch(backend: &Nvidia, before: &abi::QueueProfile) -> Result<()> {
    if gpu(backend.queue_profile())?.launches != before.launches {
        return Err(String::from("invalid/empty float operation launched GPU work"));
    }
    Ok(())
}

fn parameters(lhs: &Input, rhs: &Input, op: Elementwise, scalar: Option<u64>) -> Result<[usize; 32]> {
    let right = if unary(op) || scalar.is_some() { &lhs.layout } else { &rhs.layout };
    let mut p = gpu(lhs.layout.elementwise_parameters(right, Elementwise::Add, 0, false))?;
    p[2] = op as usize | ((lhs.format as usize) << 8);
    if unary(op) || scalar.is_some() {
        p[5] = 0;
        p[7] = 0;
        p[24..32].fill(0);
    }
    if let Some(bits) = scalar {
        p[2] |= float::SCALAR_RHS as usize;
        p[3] = bits as u32 as usize;
        p[5] = (bits >> 32) as usize;
    }
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
    fn samples(self) -> Vec<u64> {
        vec![
            0,
            self.sign,
            1,
            self.sign | 1,
            (1 << self.fraction) - 1,
            1 << self.fraction,
            self.one,
            self.one + 1,
            self.sign | self.one,
            self.infinity - 1,
            self.infinity,
            self.sign | self.infinity,
            self.infinity | 1,
            self.sign | self.infinity | 0x15,
            self.nan() | 0x31,
            self.sign | self.nan() | 0x53,
            3,
        ]
    }
}
fn strict_nan(op: Elementwise) -> bool {
    matches!(op, Elementwise::Negate | Elementwise::Absolute)
}
fn check_bits(dtype: DType, actual: &[u64], expected: &[u64], exact_nan: bool) -> Result<()> {
    let ieee = Ieee::new(dtype);
    if actual.len() != expected.len() {
        return Err(String::from("float logical length mismatch"));
    }
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if got != want && (exact_nan || !ieee.is_nan(want) || !ieee.is_nan(got)) {
            return Err(format!(
                "float {dtype:?} element={index} actual={got:#018x} expected={want:#018x} exact_nan={exact_nan}"
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
            "float result metadata/bytes dtype={dtype:?} shape={shape:?} actual_dtype={:?} actual_shape={:?}",
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
        return Err(String::from("float output guard changed"));
    }
    let words = &actual[usize::from(guarded)..usize::from(guarded) + count];
    if dtype.size() == 2 && expected.len() % 2 == 1 && words[count - 1] >> 16 != 0 {
        return Err(String::from("float odd packed output tail is not zero"));
    }
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).take(expected.len() * dtype.size()).collect();
    check_bits(dtype, &decode(dtype, &bytes), expected, exact_nan)
}
fn scalar_value(dtype: DType, bits: u64) -> Scalar {
    let value = match dtype {
        DType::F64 => f64::from_bits(bits),
        DType::F32 | DType::Flex32 => f64::from(f32::from_bits(bits as u32)),
        DType::F16 => f64::from(d3os_ai::burn::backend::f16::from_bits(bits as u16).to_f32()),
        DType::BF16 => f64::from(d3os_ai::burn::backend::bf16::from_bits(bits as u16).to_f32()),
        _ => unreachable!(),
    };
    Scalar::Float(value)
}
fn raw(backend: &Nvidia, lhs: &Input, rhs: &Input, op: Elementwise, scalar: Option<u64>, expected: &[u64], counts: &mut Counts) -> Result<()> {
    let source = |input: &Input| {
        let mut words = vec![GUARD];
        words.extend(packed(input.dtype, &input.storage));
        if input.dtype.size() == 2 && input.storage.len() % 2 == 1 {
            *words.last_mut().unwrap() |= 0xffff_0000;
        }
        words.push(GUARD);
        words
    };
    let aw = source(lhs);
    let bw = source(rhs);
    let a = gpu(backend.upload_words(&aw))?;
    let b = gpu(backend.upload_words(&bw))?;
    let word_count = packed(lhs.dtype, expected).len();
    let mut poison = vec![POISON; word_count + 2];
    poison[0] = GUARD;
    poison[word_count + 1] = GUARD;
    let output = gpu(backend.upload_words(&poison))?;
    let p = parameters(lhs, rhs, op, scalar)?;
    let before = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    gpu(backend.with_commands(|| backend.dispatch_words(Kernel::FloatArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &p)))?;
    resident(backend, &before, &profile, false)?;
    check_words(backend, &output, lhs.dtype, expected, true, strict_nan(op)).map_err(|e| format!("raw {op:?} scalar={scalar:?}: {e}"))?;
    counts.raw += 1;
    gpu(backend.write_words(&output, 1, &vec![POISON; word_count]))?;
    drop(output);
    let before = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    let output = gpu(backend.with_commands(|| backend.produce_words(word_count, Kernel::FloatArithmetic, &[(&a, 1), (&b, 1)], &p)))?;
    resident(backend, &before, &profile, true)?;
    check_words(backend, &output, lhs.dtype, expected, false, strict_nan(op))?;
    unchanged(backend, &a, &aw)?;
    unchanged(backend, &b, &bw)?;
    counts.fresh += 1;
    Ok(())
}
fn route(backend: &Nvidia, profile: &[abi::ProfileEntry; abi::KERNEL_COUNT], dtype: DType) -> Result<()> {
    let after = gpu(backend.profile())?;
    for (kernel, launches) in [
        (Kernel::TensorElementwise, u64::from(dtype == DType::F32)),
        (Kernel::FloatArithmetic, u64::from(dtype != DType::F32)),
    ] {
        if after[kernel as usize].calls - profile[kernel as usize].calls != launches {
            return Err(format!("float Native route mismatch dtype={dtype:?} kernel={kernel:?}"));
        }
    }
    Ok(())
}
fn native(backend: &Nvidia, lhs: &Input, rhs: &Input, op: Elementwise, scalar: Option<u64>, expected: &[u64], counts: &mut Counts) -> Result<()> {
    let a = lhs.tensor()?;
    let b = rhs.tensor()?;
    // Check new entry points and the pre-existing binary/unary routing separately.
    for old in [false, true] {
        if old && scalar.is_some() {
            continue;
        }
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| {
            if unary(op) {
                if old { a.unary(op) } else { a.float_unary(op) }
            } else if let Some(bits) = scalar {
                a.float_scalar(op, scalar_value(lhs.dtype, bits))
            } else if old {
                a.binary(&b, op)
            } else {
                a.float_binary(&b, op)
            }
        }))?;
        resident(backend, &before, &profile, true)?;
        route(backend, &profile, lhs.dtype)?;
        check(&output, lhs.dtype, lhs.layout.shape().as_slice(), expected, strict_nan(op))
            .map_err(|e| format!("Native {op:?} old={old} scalar={scalar:?}: {e}"))?;
        counts.native += 1;
    }
    check(&a, lhs.dtype, lhs.layout.shape().as_slice(), &lhs.logical, true)?;
    check(&b, rhs.dtype, rhs.layout.shape().as_slice(), &rhs.logical, true)?;
    Ok(())
}
fn both(backend: &Nvidia, lhs: &Input, rhs: &Input, op: Elementwise, scalar: Option<u64>, expected: &[u64], counts: &mut Counts) -> Result<()> {
    raw(backend, lhs, rhs, op, scalar, expected, counts)?;
    native(backend, lhs, rhs, op, scalar, expected, counts)
}

/// Bounded exact triples. Ties, exceptional classes, and tiny results are encoded
/// directly rather than predicted with an F64 host calculation.
fn triples(dtype: DType, op: Elementwise) -> Vec<(u64, u64, u64)> {
    let f = Ieee::new(dtype);
    let (s, inf, one, two, half, nan, normal) = (f.sign, f.infinity, f.one, f.pow2(1), f.pow2(-1), f.nan(), 1 << f.fraction);
    match op {
        Elementwise::Add => vec![
            (0, 0, 0),
            (s, s, s),
            (one, s | one, 0),
            (1, 1, 2),
            (normal - 1, 1, normal),
            (inf - 1, inf - 1, inf),
            (inf, s | inf, nan),
            (nan | 0x15, one, nan),
            (one, f.pow2(-(f.fraction as i32) - 1), one),
            (one + 1, f.pow2(-(f.fraction as i32) - 1), one + 2),
            (s, 0, 0),
        ],
        Elementwise::Subtract => vec![
            (0, 0, 0),
            (s, 0, s),
            (s, s, 0),
            (one, one, 0),
            (normal, 1, normal - 1),
            (2, 1, 1),
            (inf, inf, nan),
            (one, inf, s | inf),
            (nan, one, nan),
            (one, s | one, two),
            (one, half, half),
        ],
        Elementwise::Multiply => vec![
            (one, two, two),
            (s, two, s),
            (s, s | two, 0),
            (normal, half, normal / 2),
            (1, half, 0),
            (3, half, 2),
            (s | 1, half, s),
            (inf - 1, two, inf),
            (0, inf, nan),
            (s | one, two, s | two),
            (nan, one, nan),
        ],
        Elementwise::Divide => {
            let third = match dtype {
                DType::F64 => 0x3fd5_5555_5555_5555,
                DType::F32 | DType::Flex32 => 0x3eaa_aaab,
                DType::F16 => 0x3555,
                DType::BF16 => 0x3eab,
                _ => unreachable!(),
            };
            vec![
                (one, two, half),
                (one, 0, inf),
                (one, s, s | inf),
                (s | one, inf, s),
                (normal, two, normal / 2),
                (1, two, 0),
                (3, two, 2),
                (s | 1, two, s),
                (0, 0, nan),
                (inf, inf, nan),
                (one, two + (1 << (f.fraction - 1)), third),
            ]
        }
        _ => unreachable!(),
    }
}
fn unary_expected(dtype: DType, op: Elementwise, values: &[u64]) -> Vec<u64> {
    let f = Ieee::new(dtype);
    values
        .iter()
        .map(|&bits| match op {
            Elementwise::Negate => bits ^ f.sign,
            Elementwise::Absolute => bits & !f.sign,
            Elementwise::Sign => {
                if bits & !f.sign == 0 || f.is_nan(bits) {
                    0
                } else {
                    f.one | (bits & f.sign)
                }
            }
            _ => unreachable!(),
        })
        .collect()
}
fn operation_matrix(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let f = Ieee::new(dtype);
        for op in OPS {
            if unary(op) {
                let (values, want) = if op == Elementwise::Reciprocal {
                    (
                        vec![0, f.sign, f.one, f.sign | f.one, f.pow2(1), f.infinity, f.sign | f.infinity, 1, f.nan()],
                        vec![
                            f.infinity,
                            f.sign | f.infinity,
                            f.one,
                            f.sign | f.one,
                            f.pow2(-1),
                            0,
                            f.sign,
                            f.infinity,
                            f.nan(),
                        ],
                    )
                } else {
                    let values = f.samples();
                    let want = unary_expected(dtype, op, &values);
                    (values, want)
                };
                let a = Input::flat(dtype, format, values)?;
                both(backend, &a, &a, op, None, &want, counts)?;
            } else {
                let cases = triples(dtype, op);
                let a = Input::flat(dtype, format, cases.iter().map(|c| c.0).collect())?;
                let b = Input::flat(dtype, format, cases.iter().map(|c| c.1).collect())?;
                both(backend, &a, &b, op, None, &cases.iter().map(|c| c.2).collect::<Vec<_>>(), counts)?;
                // Scalar has a real destination-format encoding and rounds before
                // arithmetic. Each binary op gets a distinct scalar launch.
                let (bits, want) = match op {
                    Elementwise::Add => (f.one, vec![f.one, f.pow2(1), 0]),
                    Elementwise::Subtract => (f.one, vec![f.sign | f.one, 0, f.sign | f.pow2(1)]),
                    Elementwise::Multiply => (f.pow2(1), vec![0, f.pow2(1), f.sign | f.pow2(1)]),
                    Elementwise::Divide => (f.pow2(1), vec![0, f.pow2(-1), f.sign | f.pow2(-1)]),
                    _ => unreachable!(),
                };
                let a = Input::flat(dtype, format, vec![0, f.one, f.sign | f.one])?;
                both(backend, &a, &a, op, Some(bits), &want, counts)?;
            }
        }
    }
    // F64 distinguishable input precision as well as high scalar words.
    let (dtype, format) = FORMATS[0];
    let a = Input::flat(dtype, format, vec![0x3ff0_0000_0000_1000, 0x54b2_49ad_2594_c37d, 1])?;
    let b = Input::flat(dtype, format, vec![0; 3])?;
    both(backend, &a, &b, Elementwise::Add, None, &a.logical, counts)?;
    let zero = Input::flat(dtype, format, vec![0; 3])?;
    for bits in [0x3ff0_0000_0000_1000, 0x54b2_49ad_2594_c37d, 1] {
        both(backend, &zero, &zero, Elementwise::Add, Some(bits), &[bits; 3], counts)?;
    }
    Ok(())
}
fn layouts(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let f = Ieee::new(dtype);
        for length in [1, 2, 3, 4, 5, 127, 128, 129, 257] {
            let storage: Vec<_> = (0..length * 2 + 4).map(|i| f.samples()[i % 17]).collect();
            for step in [-1i64, 2] {
                let offset = if step < 0 { length as i64 } else { 1 };
                let logical: Vec<_> = (0..length).map(|i| storage[(offset + i as i64 * step) as usize]).collect();
                let a = Input {
                    dtype,
                    format,
                    layout: gpu(Layout::from_parts(Shape::new([length]), vec![step], offset, storage.len()))?,
                    storage: storage.clone(),
                    logical,
                };
                let b = Input {
                    dtype,
                    format,
                    layout: gpu(Layout::from_parts(Shape::new([length]), vec![0], 1, 3))?,
                    storage: vec![0, f.one, 0],
                    logical: vec![f.one; length],
                };
                // Division by one preserves subnormals and signed zero; any NaN
                // remains a NaN. Source offset one crosses packed half words.
                both(backend, &a, &b, Elementwise::Divide, None, &a.logical, counts)?;
            }
        }
        let storage: Vec<_> = (0..24).map(|i| f.samples()[i % 17]).collect();
        let a = Input {
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
        let b = Input {
            dtype,
            format,
            layout: gpu(Layout::from_parts(Shape::new([4, 2, 3]), vec![0, 0, 0], 1, 3))?,
            storage: vec![0, f.one, 0],
            logical: vec![f.one; 24],
        };
        both(backend, &a, &b, Elementwise::Multiply, None, &a.logical, counts)?;
        both(backend, &b, &a, Elementwise::Multiply, None, &a.logical, counts)?;
        for shape in [vec![], vec![1, 1, 1, 1, 1, 1, 1, 5]] {
            let length = shape.iter().product();
            let mut a = Input::flat(dtype, format, vec![f.one; length])?;
            a.layout = gpu(Layout::contiguous(Shape::from(shape)))?;
            both(backend, &a, &a, Elementwise::Add, None, &vec![f.pow2(1); length], counts)?;
        }
    }
    Ok(())
}

fn malformed(p: [usize; 32], unary: [usize; 32], scalar: [usize; 32], dtype: DType) -> Vec<[usize; 32]> {
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
    for operation in [0, 5, 14, 255] {
        let mut bad = p;
        bad[2] = (bad[2] & !255) | operation;
        cases.push(bad);
    }
    for format in [Format::I32 as usize, Format::Bool8 as usize, 255] {
        let mut bad = p;
        bad[2] = (bad[2] & !0xff00) | (format << 8);
        cases.push(bad);
    }
    for field in [3, 5, 7, 24] {
        let mut bad = unary;
        bad[field] = 1;
        cases.push(bad);
    }
    let mut bad = unary;
    bad[2] |= float::SCALAR_RHS as usize;
    cases.push(bad);
    for field in [7, 24, 31] {
        let mut bad = scalar;
        bad[field] = 1;
        cases.push(bad);
    }
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
        let a = Input::flat(dtype, format, vec![Ieee::new(dtype).one; 5])?;
        let p = parameters(&a, &a, Elementwise::Add, None)?;
        let u = parameters(&a, &a, Elementwise::Negate, None)?;
        let s = parameters(&a, &a, Elementwise::Add, Some(Ieee::new(dtype).one))?;
        let words = vec![POISON; packed(dtype, &a.storage).len() + 4];
        let input = gpu(backend.upload_words(&words))?;
        let rhs = gpu(backend.upload_words(&words))?;
        let output = gpu(backend.upload_words(&words))?;
        let short = gpu(backend.upload_words(&[POISON]))?;
        let before = gpu(backend.queue_profile())?;
        for bad in malformed(p, u, s, dtype) {
            if backend
                .dispatch_words(Kernel::FloatArithmetic, &[(&input, 1), (&rhs, 1), (&output, 1)], &bad)
                .is_ok()
            {
                return Err(format!("float malformed metadata accepted {dtype:?}: {bad:?}"));
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
                return Err(String::from("float parameter length accepted"));
            }
            counts.rejected += 1;
        }
        // Exact/partial overlap and unused RHS aliasing are all forbidden.
        for (out, offset, params) in [(&input, 1, p), (&input, 2, p), (&rhs, 1, p), (&rhs, 2, p), (&rhs, 1, s), (&input, 1, u)] {
            if backend
                .dispatch_words(Kernel::FloatArithmetic, &[(&input, 1), (&rhs, 1), (out, offset)], &params)
                .is_ok()
            {
                return Err(String::from("float output alias accepted"));
            }
            counts.rejected += 1;
        }
        for buffers in [
            [(&short, 0), (&rhs, 1), (&output, 1)],
            [(&input, 1), (&short, 0), (&output, 1)],
            [(&input, 1), (&rhs, 1), (&short, 0)],
        ] {
            if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &p).is_ok() {
                return Err(String::from("float short physical buffer accepted"));
            }
            counts.rejected += 1;
        }
        if dtype == DType::F64 {
            // The low word alone must never satisfy a one-element F64 binding.
            let one = Input::flat(dtype, format, vec![0])?;
            let one = parameters(&one, &one, Elementwise::Add, None)?;
            for buffers in [
                [(&short, 0), (&rhs, 1), (&output, 1)],
                [(&input, 1), (&short, 0), (&output, 1)],
                [(&input, 1), (&rhs, 1), (&short, 0)],
            ] {
                if backend.dispatch_words(Kernel::FloatArithmetic, &buffers, &one).is_ok() {
                    return Err(String::from("F64 second physical word not checked"));
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
                return Err(String::from("F64 high physical word bounds accepted"));
            }
            counts.rejected += 1;
        }
        no_launch(backend, &before)?;
        unchanged(backend, &input, &words)?;
        unchanged(backend, &rhs, &words)?;
        unchanged(backend, &output, &words)?;
        unchanged(backend, &short, &[POISON])?;
    }
    Ok(())
}
fn native_validation(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let f = Ieee::new(dtype);
        let wrong = if dtype == DType::F32 { FORMATS[2] } else { FORMATS[1] };
        for length in [0, 3] {
            let a = Input::flat(dtype, format, vec![f.one; length])?.tensor()?;
            let b = Input::flat(wrong.0, wrong.1, vec![Ieee::new(wrong.0).one; length])?.tensor()?;
            let before = gpu(backend.queue_profile())?;
            for op in OPS {
                let bad = if unary(op) { a.float_binary(&a, op) } else { a.float_unary(op) };
                if !matches!(bad, Err(Error::InvalidParameter)) {
                    return Err(format!("float wrong arity accepted {dtype:?} {op:?}"));
                }
                counts.rejected += 1;
                if unary(op) {
                    if !matches!(a.float_scalar(op, Scalar::Float(1.0)), Err(Error::InvalidParameter)) {
                        return Err(String::from("float unary scalar accepted"));
                    }
                } else if !matches!(a.float_binary(&b, op), Err(Error::InvalidParameter)) {
                    return Err(String::from("float mismatched tensor dtype accepted"));
                }
                counts.rejected += 1;
            }
            for op in [Elementwise::Equal, Elementwise::MaskFill, Elementwise::Greater] {
                if !matches!(a.float_binary(&a, op), Err(Error::InvalidParameter))
                    || !matches!(a.float_unary(op), Err(Error::InvalidParameter))
                    || !matches!(a.float_scalar(op, Scalar::Bool(true)), Err(Error::InvalidParameter))
                {
                    return Err(String::from("float unsupported operation accepted"));
                }
                counts.rejected += 3;
            }
            no_launch(backend, &before)?;
        }
        let a = Input::flat(dtype, format, vec![f.one; 3])?.tensor()?;
        let b = Input::flat(dtype, format, vec![f.one; 5])?.tensor()?;
        let before = gpu(backend.queue_profile())?;
        if a.float_binary(&b, Elementwise::Add).is_ok() {
            return Err(String::from("float incompatible shapes accepted"));
        }
        counts.rejected += 1;
        no_launch(backend, &before)?;

        let a = gpu(Input::flat(dtype, format, vec![0, f.one])?.tensor()?.reshape(Shape::new([2, 1])))?;
        let b = gpu(Input::flat(dtype, format, vec![0, f.pow2(1), f.sign | f.pow2(1)])?
            .tensor()?
            .reshape(Shape::new([1, 3])))?;
        let three = f.pow2(1) | (1 << (f.fraction - 1));
        for (left, right, want) in [
            (&a, &b, vec![0, f.sign | f.pow2(1), f.pow2(1), f.one, f.sign | f.one, three]),
            (&b, &a, vec![0, f.pow2(1), f.sign | f.pow2(1), f.sign | f.one, f.one, f.sign | three]),
        ] {
            let before = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let result = gpu(left.float_binary(right, Elementwise::Subtract))?;
            resident(backend, &before, &profile, true)?;
            route(backend, &profile, dtype)?;
            check(&result, dtype, &[2, 3], &want, false)?;
            counts.native += 1;
        }
    }
    for length in [0, 3] {
        let source = gpu(NativeTensor::from_data(TensorData::from_bytes_vec(
            vec![0; length * 4],
            Shape::new([length]),
            DType::I32,
        )))?;
        let before = gpu(backend.queue_profile())?;
        for result in [
            source.float_binary(&source, Elementwise::Add),
            source.float_unary(Elementwise::Negate),
            source.float_scalar(Elementwise::Add, Scalar::Float(f64::NAN)),
        ] {
            if !matches!(result, Err(Error::InvalidParameter)) {
                return Err(String::from("nonfloat Native dtype accepted"));
            }
            counts.rejected += 1;
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
            let output = gpu(if unary(op) { empty.float_unary(op) } else { empty.float_binary(&empty, op) })?;
            check(&output, dtype, &[2, 0, 3], &[], true)?;
            counts.native += 1;
            if !unary(op) {
                for scalar in [Scalar::Float(f64::NAN), Scalar::Float(f64::INFINITY), Scalar::Float(f64::NEG_INFINITY)] {
                    check(&gpu(empty.float_scalar(op, scalar))?, dtype, &[2, 0, 3], &[], true)?;
                    counts.native += 1;
                }
            }
        }
        no_launch(backend, &before)?;
        let f = Ieee::new(dtype);
        let input = Input::flat(dtype, format, f.samples())?;
        let a = input.tensor()?;
        let b = Input::flat(dtype, format, vec![f.one; input.logical.len()])?.tensor()?;
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| {
            let result = a.float_binary(&b, Elementwise::Multiply)?;
            drop(a);
            drop(b);
            Ok::<_, Error>(result)
        }))?;
        resident(backend, &before, &profile, true)?;
        route(backend, &profile, dtype)?;
        check(&output, dtype, &[input.logical.len()], &input.logical, false)?;
        counts.native += 1;
        let a = gpu(backend.upload_words(&packed(dtype, &input.storage)))?;
        let b_input = Input::flat(dtype, format, vec![f.one; input.logical.len()])?;
        let b = gpu(backend.upload_words(&packed(dtype, &b_input.storage)))?;
        let p = parameters(&input, &b_input, Elementwise::Multiply, None)?;
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| {
            let result = backend.produce_words(packed(dtype, &input.logical).len(), Kernel::FloatArithmetic, &[(&a, 0), (&b, 0)], &p)?;
            drop(a);
            drop(b);
            Ok::<_, Error>(result)
        }))?;
        resident(backend, &before, &profile, true)?;
        check_words(backend, &output, dtype, &input.logical, false, false)?;
        counts.fresh += 1;
    }
    Ok(())
}

fn elementwise(op: Operation) -> Elementwise {
    match op {
        Operation::Add => Elementwise::Add,
        Operation::Sub => Elementwise::Subtract,
        Operation::Mul => Elementwise::Multiply,
        Operation::Div => Elementwise::Divide,
        Operation::Neg => Elementwise::Negate,
        Operation::Abs => Elementwise::Absolute,
        Operation::Recip => Elementwise::Reciprocal,
        Operation::Sign => Elementwise::Sign,
    }
}
fn reference(dtype: DType, op: Operation, a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    a.iter()
        .zip(b)
        .map(|(&lhs, &rhs)| oracle::expected_bits(dtype, op, lhs, rhs).ok_or_else(|| format!("float oracle domain {dtype:?} {op:?} {lhs:#x} {rhs:#x}")))
        .collect()
}
fn shared_matrix(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let pool = oracle::pool(dtype);
        for op in Operation::ALL {
            let (a, b) = if op.unary() {
                (pool.clone(), pool.clone())
            } else {
                (pool.iter().flat_map(|&v| vec![v; pool.len()]).collect(), pool.repeat(pool.len()))
            };
            let want = reference(dtype, op, &a, &b)?;
            let a = Input::flat(dtype, format, a)?;
            let b = Input::flat(dtype, format, b)?;
            both(backend, &a, &b, elementwise(op), None, &want, counts)?;
        }
    }
    Ok(())
}
fn backend_scalar(input: &Input, op: Elementwise, scalar: Scalar) -> Tensor<1> {
    let primitive = Native::float_from_data(input.data(), &NativeDevice);
    let output = match op {
        Elementwise::Add => Native::float_add_scalar(primitive, scalar),
        Elementwise::Subtract => Native::float_sub_scalar(primitive, scalar),
        Elementwise::Multiply => Native::float_mul_scalar(primitive, scalar),
        Elementwise::Divide => Native::float_div_scalar(primitive, scalar),
        _ => unreachable!(),
    };
    Tensor::<1>::from_primitive::<Native>(output)
}
fn dispatch_scalar(source: Tensor<1>, op: Elementwise, scalar: Scalar) -> Tensor<1> {
    // Scalar::new intentionally passes integer public arguments through F64.
    // Keep each variant at the real generic public API boundary.
    macro_rules! apply {
        ($value:expr) => {
            match op {
                Elementwise::Add => source.add_scalar($value),
                Elementwise::Subtract => source.sub_scalar($value),
                Elementwise::Multiply => source.mul_scalar($value),
                Elementwise::Divide => source.div_scalar($value),
                _ => unreachable!(),
            }
        };
    }
    match scalar {
        Scalar::Float(v) => apply!(v),
        Scalar::Int(v) => apply!(v),
        Scalar::UInt(v) => apply!(v),
        Scalar::Bool(v) => apply!(v),
    }
}
fn public_scalars(counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("float public device: {e}"))?;
    for (dtype, format) in FORMATS {
        let f = Ieee::new(dtype);
        // One operation isolates scalar encoding without reproducing the shared
        // public corpus. Zero observes normalized bits; one observes arithmetic.
        let input = Input::flat(dtype, format, vec![0, f.one, f.pow2(1) | (1 << (f.fraction - 1))])?;
        for fixture in oracle::scalar_fixtures(dtype) {
            let want = reference(dtype, Operation::Add, &input.logical, &[fixture.bits; 3])?;
            let result = backend_scalar(&input, Elementwise::Add, fixture.scalar);
            check_data(
                &result.try_into_data().map_err(|e| format!("float backend Scalar: {e:?}"))?,
                dtype,
                &[3],
                &want,
                false,
            )?;
            counts.public += 1;
            let source = Tensor::<1>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
            let result = dispatch_scalar(source, Elementwise::Add, fixture.scalar);
            check_data(
                &result.try_into_data().map_err(|e| format!("float Dispatch Scalar: {e:?}"))?,
                dtype,
                &[3],
                &want,
                false,
            )?;
            counts.public += 1;
        }
        // Half scalar rounding must precede multiplication; a late-only round
        // would change the result of 3 * (1 + 3*2^-13 / 3*2^-10).
        if matches!(dtype, DType::F16 | DType::BF16) {
            let scalar = if dtype == DType::F16 {
                f64::from_bits(0x3ff0_0180_0000_0000)
            } else {
                f64::from_bits(0x3ff0_0c00_0000_0000)
            };
            let three = f.pow2(1) | (1 << (f.fraction - 1));
            let input = Input::flat(dtype, format, vec![three; 3])?;
            check_data(
                &backend_scalar(&input, Elementwise::Multiply, Scalar::Float(scalar))
                    .try_into_data()
                    .map_err(|e| format!("float early half scalar: {e:?}"))?,
                dtype,
                &[3],
                &[three; 3],
                false,
            )?;
            counts.public += 1;
            let source = Tensor::<1>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
            check_data(
                &source
                    .mul_scalar(scalar)
                    .try_into_data()
                    .map_err(|e| format!("float Dispatch early half scalar: {e:?}"))?,
                dtype,
                &[3],
                &[three; 3],
                false,
            )?;
            counts.public += 1;
        }
        // Materialize every public intermediate to prove separate rounding.
        let chain = &oracle::chain_fixtures(dtype)[0];
        let input = Input::flat(dtype, format, vec![chain.initial; 3])?;
        let mut value = Tensor::<1>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
        for step in chain.steps {
            value = dispatch_scalar(value, elementwise(step.operation), scalar_value(dtype, step.rhs));
            check_data(
                &value.clone().try_into_data().map_err(|e| format!("float public rounding chain: {e:?}"))?,
                dtype,
                &[3],
                &[step.expected; 3],
                true,
            )?;
            counts.public += 1;
        }
    }
    // All four backend wrappers and the public Dispatch scalar type must retain
    // the F64 distinction. No expected result here is computed by host F64 math.
    for (op, lhs, scalar, want) in [
        (Elementwise::Add, 0x3ff0_0000_0000_0000, 0x3d70_0000_0000_0000, 0x3ff0_0000_0000_1000),
        (Elementwise::Subtract, 0x3ff0_0000_0000_0000, 0x3d70_0000_0000_0000, 0x3fef_ffff_ffff_e000),
        (Elementwise::Multiply, 0x3ff0_0000_0000_0000, 0x3ff0_0000_0000_1000, 0x3ff0_0000_0000_1000),
        (Elementwise::Divide, 0x3ff0_0000_0000_0000, 0x3ff0_0000_0000_1000, 0x3fef_ffff_ffff_e000),
    ] {
        let input = Input::flat(DType::F64, Format::F64, vec![lhs; 3])?;
        let scalar = Scalar::Float(f64::from_bits(scalar));
        let output = backend_scalar(&input, op, scalar);
        check_data(
            &output.try_into_data().map_err(|e| format!("float F64 backend: {e:?}"))?,
            DType::F64,
            &[3],
            &[want; 3],
            true,
        )?;
        counts.public += 1;
        let source = Tensor::<1>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(DType::F64));
        let output = dispatch_scalar(source, op, scalar);
        check_data(
            &output.try_into_data().map_err(|e| format!("float F64 Dispatch: {e:?}"))?,
            DType::F64,
            &[3],
            &[want; 3],
            true,
        )?;
        counts.public += 1;
    }
    Ok(())
}
fn public_errors(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("float error device: {e}"))?;
    for length in [0, 3] {
        for direct in [false, true] {
            let a = Input::flat(DType::F32, Format::F32, vec![0x3f80_0000; length])?;
            let b = Input::flat(DType::Flex32, Format::Flex32, vec![0x3f80_0000; length])?;
            let left = Tensor::<1>::from_data(a.data(), TensorCreationOptions::new(device.clone()).with_dtype(a.dtype));
            let right = Tensor::<1>::from_data(b.data(), TensorCreationOptions::new(device.clone()).with_dtype(b.dtype));
            let pa = Native::float_from_data(a.data(), &NativeDevice);
            let pb = Native::float_from_data(b.data(), &NativeDevice);
            let before = gpu(backend.queue_profile())?;
            let failed = if direct {
                Tensor::<1>::from_primitive::<Native>(Native::float_add(pa, pb))
            } else {
                left + right
            };
            let dependent = failed.clone().add_scalar(1.0);
            no_launch(backend, &before)?;
            if !device.sync().is_err_and(|e| !e.is_device_poisoned())
                || !dependent.try_into_data().is_err_and(|e| !e.is_device_poisoned())
                || !failed.try_into_data().is_err_and(|e| !e.is_device_poisoned())
                || device.sync().is_err()
            {
                return Err(format!("float incompatible dtype error recovery direct={direct} length={length}"));
            }
            no_launch(backend, &before)?;
            counts.rejected += 1;
            let valid = Tensor::<1>::from_data(a.data(), TensorCreationOptions::new(device.clone()).with_dtype(a.dtype)).add_scalar(1.0);
            check_data(
                &valid.try_into_data().map_err(|e| format!("float public recovery: {e:?}"))?,
                a.dtype,
                &[length],
                &vec![0x4000_0000; length],
                true,
            )?;
            counts.public += 1;
        }
        let invalid = TensorData::from_bytes_vec(vec![0; length * 4], Shape::new([length]), DType::I32);
        let before = gpu(backend.queue_profile())?;
        let failed = Native::float_add_scalar(Native::float_from_data(invalid, &NativeDevice), Scalar::Float(f64::NAN));
        no_launch(backend, &before)?;
        // The intentionally invalid input retains I32 failure metadata. Read the
        // backend failure directly instead of constructing a Float tensor.
        let sync_error = device.sync().is_err_and(|e| !e.is_device_poisoned());
        let failed_read =
            d3os_ai::burn::tensor::try_read_sync(Native::float_into_data(failed)).ok_or_else(|| String::from("invalid float input read did not complete"))?;
        if !sync_error || !failed_read.is_err_and(|e| !e.is_device_poisoned()) || device.sync().is_err() {
            return Err(String::from("float invalid dtype error acknowledgment poisoned device"));
        }
        no_launch(backend, &before)?;
        counts.rejected += 1;
        let input = Input::flat(DType::F64, Format::F64, vec![0; length])?;
        check_data(
            &backend_scalar(&input, Elementwise::Add, Scalar::Float(f64::from_bits(1)))
                .try_into_data()
                .map_err(|e| format!("float invalid dtype recovery: {e:?}"))?,
            DType::F64,
            &[length],
            &vec![1; length],
            true,
        )?;
        counts.public += 1;
    }
    Ok(())
}
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    let mut counts = Counts::default();
    operation_matrix(backend, &mut counts)?;
    shared_matrix(backend, &mut counts)?;
    layouts(backend, &mut counts)?;
    raw_rejections(backend, &mut counts)?;
    native_validation(backend, &mut counts)?;
    empty_and_lifetimes(backend, &mut counts)?;
    public_scalars(&mut counts)?;
    public_errors(backend, &mut counts)?;
    let actual = (counts.raw, counts.fresh, counts.native, counts.rejected, counts.public);
    if actual != (214, 219, 520, 557, 172) {
        return Err(format!("float arithmetic case count changed: {actual:?}"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_FLOAT_ARITHMETIC_OK operations=8 formats=5 raw={} fresh={} native={} rejected={} public={} oracle=exact_ieee_fixtures guards=ok tails=zero resident=true queued_lifetime=verified f32_fastpath=verified errors=recoverable backend=nvidia",
        counts.raw, counts.fresh, counts.native, counts.rejected, counts.public,
    ));
    Ok(())
}
