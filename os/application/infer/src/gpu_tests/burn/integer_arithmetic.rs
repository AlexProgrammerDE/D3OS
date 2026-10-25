//! Exact Native extensions, packed full writers, and recoverable integer errors.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Scalar, Shape, TensorData, TensorMetadata, ops::IntTensorOps},
        tensor::{Int, Tensor, TensorCreationOptions},
    },
    native::{Native, NativeDevice, NativeTensor, layout::Layout},
    validation::integer_arithmetic::{self as oracle, Operation},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, cast::Format, elementwise::Elementwise, integer_arithmetic as integer, words::WordOp};

type Result<T> = core::result::Result<T, String>;
const GUARD: u32 = 0x6bad_beef;
const POISON: u32 = 0xa53c_d791;
const FORMATS: [(DType, Format); 8] = [
    (DType::I8, Format::I8),
    (DType::I16, Format::I16),
    (DType::I32, Format::I32),
    (DType::I64, Format::I64),
    (DType::U8, Format::U8),
    (DType::U16, Format::U16),
    (DType::U32, Format::U32),
    (DType::U64, Format::U64),
];

#[derive(Default)]
struct Counts {
    raw: usize,
    fresh: usize,
    native: usize,
    rejected: usize,
    public: usize,
}

fn word(op: Operation) -> WordOp {
    match op {
        Operation::Add => WordOp::Add,
        Operation::Sub => WordOp::Subtract,
        Operation::Mul => WordOp::Multiply,
        Operation::Div => WordOp::Divide,
        Operation::FloorRem => WordOp::Remainder,
        Operation::Powi => WordOp::Power,
        Operation::Neg => WordOp::Negate,
        Operation::Abs => WordOp::Absolute,
        Operation::Sign => WordOp::Sign,
        Operation::And => WordOp::BitAnd,
        Operation::Or => WordOp::BitOr,
        Operation::Xor => WordOp::BitXor,
        Operation::Not => WordOp::BitNot,
        Operation::Shl => WordOp::ShiftLeft,
        Operation::Shr => WordOp::ShiftRight,
    }
}

fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("integer arithmetic: {e}"))
}

fn packed(dtype: DType, values: &[u64]) -> Vec<u32> {
    oracle::pack(dtype, values)
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
        TensorData::from_bytes_vec(oracle::pack(self.dtype, &self.storage), Shape::new([self.storage.len()]), self.dtype)
    }
}

fn parameters(lhs: &Input, rhs: &Input, op: Operation, scalar: Option<u64>) -> Result<[usize; 32]> {
    let mut p = gpu(lhs.layout.elementwise_parameters(&rhs.layout, Elementwise::Add, 0, false))?;
    p[2] = word(op) as usize | ((lhs.format as usize) << 8) | ((rhs.format as usize) << 16);
    if scalar.is_some() || op.unary() {
        p[2] = word(op) as usize | ((lhs.format as usize) << 8) | ((lhs.format as usize) << 16);
        p[5] = 0;
        p[7] = 0;
        p[24..32].fill(0);
    }
    if let Some(bits) = scalar {
        p[2] |= integer::SCALAR_RHS as usize;
        p[3] = bits as u32 as usize;
        p[5] = (bits >> 32) as usize;
    }
    Ok(p)
}

fn expected(lhs: &Input, rhs: &Input, op: Operation, scalar: Option<u64>) -> Vec<u64> {
    lhs.logical
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            oracle::reference(
                lhs.dtype,
                op,
                a,
                if scalar.is_some() { lhs.dtype } else { rhs.dtype },
                scalar.unwrap_or(rhs.logical[i]),
            )
        })
        .collect()
}

fn check_data(data: &TensorData, dtype: DType, shape: &[usize], expected: &[u64]) -> Result<()> {
    if data.dtype() != dtype || data.shape().as_slice() != shape || data.as_bytes() != oracle::pack(dtype, expected) {
        return Err(format!(
            "integer exact bytes/shape mismatch expected_dtype={dtype:?} expected_shape={shape:?} actual_dtype={:?} actual_shape={:?} actual_bytes={} expected_bytes={}",
            data.dtype(),
            data.shape(),
            data.as_bytes().len(),
            expected.len() * dtype.size()
        ));
    }
    Ok(())
}

fn check(tensor: &NativeTensor, dtype: DType, shape: &[usize], values: &[u64]) -> Result<()> {
    if tensor.dtype() != dtype || tensor.shape().as_slice() != shape {
        return Err(format!("integer result metadata dtype={dtype:?} shape={shape:?}"));
    }
    check_data(&gpu(tensor.read())?, dtype, shape, values)
}

fn unchanged(backend: &Nvidia, buffer: &d3os_compute::nvidia::GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if actual != expected {
        return Err(String::from("integer source/guard/poison changed"));
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
            return Err(format!("integer resident service={service}"));
        }
    }
    if fresh && after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1 {
        return Err(String::from("integer full writer did not allocate through one launch"));
    }
    let after = gpu(backend.profile())?;
    for kernel in [Kernel::Zero, Kernel::Fill, Kernel::TensorStorage] {
        if after[kernel as usize].calls != profile[kernel as usize].calls {
            return Err(format!("integer operation used a clear/fill/copy kernel: {kernel:?}"));
        }
    }
    Ok(())
}

fn raw(backend: &Nvidia, lhs: &Input, rhs: &Input, op: Operation, scalar: Option<u64>, counts: &mut Counts) -> Result<()> {
    let guard_source = |input: &Input| {
        let mut words = vec![GUARD];
        words.extend(packed(input.dtype, &input.storage));
        // Poison the unused source slots independently of the output tail.
        let bytes = input.storage.len() * input.dtype.size();
        if !bytes.is_multiple_of(4) {
            *words.last_mut().unwrap() |= u32::MAX << ((bytes % 4) * 8);
        }
        words.push(GUARD);
        words
    };
    let a_words = guard_source(lhs);
    let b_words = guard_source(rhs);
    let a = gpu(backend.upload_words(&a_words))?;
    let b = gpu(backend.upload_words(&b_words))?;
    let expected = packed(lhs.dtype, &expected(lhs, rhs, op, scalar));
    let mut poisoned = vec![POISON; expected.len() + 2];
    poisoned[0] = GUARD;
    *poisoned.last_mut().unwrap() = GUARD;
    let output = gpu(backend.upload_words(&poisoned))?;
    let p = parameters(lhs, rhs, op, scalar)?;
    let before = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    gpu(backend.with_commands(|| backend.dispatch_words(Kernel::IntegerArithmetic, &[(&a, 1), (&b, 1), (&output, 1)], &p)))?;
    resident(backend, &before, &profile, false)?;
    let mut guarded = vec![GUARD];
    guarded.extend(&expected);
    guarded.push(GUARD);
    unchanged(backend, &output, &guarded).map_err(|e| format!("raw {:?} {op:?} scalar={scalar:?}: {e}", lhs.dtype))?;
    counts.raw += 1;

    // Releasing dirty storage encourages allocator reuse; correctness does not
    // assume that a particular physical allocation will be reused.
    gpu(backend.write_words(&output, 1, &vec![POISON; expected.len()]))?;
    drop(output);
    let before = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    let output = gpu(backend.with_commands(|| backend.produce_words(expected.len(), Kernel::IntegerArithmetic, &[(&a, 1), (&b, 1)], &p)))?;
    resident(backend, &before, &profile, true)?;
    unchanged(backend, &output, &expected)?;
    unchanged(backend, &a, &a_words)?;
    unchanged(backend, &b, &b_words)?;
    counts.fresh += 1;
    Ok(())
}

fn as_scalar(dtype: DType, bits: u64) -> Scalar {
    if dtype.is_int() {
        Scalar::Int(oracle::decode(dtype, bits) as i64)
    } else {
        Scalar::UInt(bits)
    }
}

fn native(backend: &Nvidia, lhs: &Input, rhs: &Input, op: Operation, scalar: Option<u64>, counts: &mut Counts) -> Result<()> {
    let a = lhs.tensor()?;
    let b = rhs.tensor()?;
    let before = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    let output = gpu(backend.with_commands(|| {
        if op.unary() {
            a.integer_unary(word(op))
        } else if let Some(bits) = scalar {
            a.integer_scalar(word(op), as_scalar(lhs.dtype, bits))
        } else {
            a.integer_binary(&b, word(op))
        }
    }))?;
    resident(backend, &before, &profile, true)?;
    let after = gpu(backend.profile())?;
    let legacy = lhs.dtype == DType::I32 && (op.unary() || scalar.is_some() || rhs.dtype == DType::I32);
    for (kernel, launches) in [(Kernel::TensorWords, u64::from(legacy)), (Kernel::IntegerArithmetic, u64::from(!legacy))] {
        if after[kernel as usize].calls - profile[kernel as usize].calls != launches {
            return Err(format!("integer Native route mismatch dtype={:?} rhs={:?} op={op:?}", lhs.dtype, rhs.dtype));
        }
    }
    check(&output, lhs.dtype, lhs.layout.shape().as_slice(), &expected(lhs, rhs, op, scalar))
        .map_err(|e| format!("Native {:?} {op:?} scalar={scalar:?}: {e}", lhs.dtype))?;
    check(&a, lhs.dtype, lhs.layout.shape().as_slice(), &lhs.logical)?;
    check(&b, rhs.dtype, rhs.layout.shape().as_slice(), &rhs.logical)?;
    counts.native += 1;
    Ok(())
}

fn both(backend: &Nvidia, lhs: &Input, rhs: &Input, op: Operation, scalar: Option<u64>, counts: &mut Counts) -> Result<()> {
    raw(backend, lhs, rhs, op, scalar, counts)?;
    native(backend, lhs, rhs, op, scalar, counts)
}

fn edges(dtype: DType) -> Vec<u64> {
    let mask = oracle::mask(dtype);
    let sign = 1u64 << (oracle::width(dtype) - 1);
    [
        0,
        1,
        2,
        3,
        7,
        mask,
        mask - 1,
        mask - 2,
        mask - 6,
        sign,
        sign - 1,
        sign + 1,
        31,
        32,
        63,
        u64::from(oracle::width(dtype) - 1),
        u64::from(oracle::width(dtype)),
        u64::from(oracle::width(dtype) + 1),
        (1u64 << 32) + 1,
        (1u64 << 53) + 3,
        (1u64 << 63) + 0x0123_4567,
    ]
    .into_iter()
    .map(|v| v & mask)
    .collect()
}

fn operation_matrix(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let values = edges(dtype);
        let a = Input::flat(dtype, format, values.iter().flat_map(|&a| vec![a; values.len()]).collect())?;
        let b = Input::flat(dtype, format, values.repeat(values.len()))?;
        for op in Operation::ALL {
            both(backend, &a, &b, op, None, counts)?;
            if !op.unary() {
                // Wide masks retain the high word. Shifts exercise count N-1;
                // scalar division uses -3 for signed formats and 3 otherwise.
                let scalar = match op {
                    Operation::Div | Operation::FloorRem if dtype.is_int() => oracle::mask(dtype) - 2,
                    Operation::Div | Operation::FloorRem => 3,
                    Operation::Shl | Operation::Shr => u64::from(oracle::width(dtype) - 1),
                    Operation::Powi => 3,
                    _ => oracle::mask(dtype),
                };
                both(backend, &a, &b, op, Some(scalar), counts)?;
            }
        }
    }
    // All actual exponent formats, including narrow sign extension and wide
    // values that differ only above bit 31. This is a Native extension.
    for (dtype, format) in FORMATS {
        for (rhs_dtype, rhs_format) in FORMATS {
            let exponents = edges(rhs_dtype);
            let bases = edges(dtype);
            let a = Input::flat(dtype, format, (0..exponents.len() * bases.len()).map(|i| bases[i / exponents.len()]).collect())?;
            let b = Input::flat(rhs_dtype, rhs_format, exponents.repeat(bases.len()))?;
            both(backend, &a, &b, Operation::Powi, None, counts)?;
        }
    }
    Ok(())
}

fn layouts(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        // Packed tails and partial/inactive warp owners for every physical width.
        let mut lengths = vec![1, 2, 3, 4, 5, 127, 128, 129, 257];
        if dtype.size() == 1 {
            lengths.extend([511, 512, 513]);
        }
        for length in lengths {
            let storage: Vec<_> = (0..length + 4).map(|i| edges(dtype)[i % 21]).collect();
            let a = Input {
                dtype,
                format,
                layout: gpu(Layout::from_parts(Shape::new([length]), vec![-1], (length + 1) as i64, storage.len()))?,
                logical: (0..length).map(|i| storage[length + 1 - i]).collect(),
                storage,
            };
            let b = Input {
                dtype,
                format,
                layout: gpu(Layout::from_parts(Shape::new([length]), vec![0], 1, 3))?,
                storage: vec![0, oracle::mask(dtype), 7],
                logical: vec![oracle::mask(dtype); length],
            };
            // Always run division and shift so padded lanes must remain safe.
            for op in [Operation::Div, Operation::Shr] {
                both(backend, &a, &b, op, None, counts)?;
            }
        }
        let storage: Vec<_> = (0..24).map(|i| edges(dtype)[i % 21]).collect();
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
            layout: gpu(Layout::from_parts(Shape::new([4, 2, 3]), vec![0, 1, 0], 0, 2))?,
            storage: vec![3, oracle::mask(dtype) - 2],
            logical: (0..24).map(|i| if i / 3 % 2 == 0 { 3 } else { oracle::mask(dtype) - 2 }).collect(),
        };
        both(backend, &a, &b, Operation::FloorRem, None, counts)?;
        both(backend, &b, &a, Operation::Sub, None, counts)?;
        // Rank zero and rank eight are actual dispatches, not reshaped reads.
        for shape in [vec![], vec![1, 1, 1, 1, 1, 1, 1, 5]] {
            let length = shape.iter().product();
            let mut a = Input::flat(dtype, format, vec![oracle::mask(dtype) - 2; length])?;
            a.layout = gpu(Layout::contiguous(Shape::from(shape)))?;
            let mut b = Input::flat(dtype, format, vec![3; length])?;
            b.layout = a.layout.clone();
            both(backend, &a, &b, Operation::Powi, None, counts)?;
        }
    }
    Ok(())
}

fn no_launch(backend: &Nvidia, before: &abi::QueueProfile) -> Result<()> {
    if gpu(backend.queue_profile())?.launches != before.launches {
        return Err(String::from("invalid/empty integer operation launched GPU work"));
    }
    Ok(())
}

fn native_metadata_and_broadcast(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let a = Input::flat(dtype, format, vec![3, 7])?.tensor()?.reshape(Shape::new([2, 1]));
        let a = gpu(a)?;
        let b = gpu(Input::flat(dtype, format, vec![1, 2, 3])?.tensor()?.reshape(Shape::new([1, 3])))?;
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let first = gpu(a.integer_binary(&b, WordOp::Subtract))?;
        let second = gpu(b.integer_binary(&a, WordOp::Subtract))?;
        // Each broadcast direction gets a fresh output with no activation read.
        resident(backend, &before, &profile, false)?;
        if gpu(backend.service_profile())?[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 2 {
            return Err(String::from("integer bidirectional broadcast allocation"));
        }
        let want = |reverse| {
            [3, 7]
                .into_iter()
                .flat_map(|x| {
                    [1, 2, 3].into_iter().map(move |y| {
                        if reverse {
                            oracle::reference(dtype, Operation::Sub, y, dtype, x)
                        } else {
                            oracle::reference(dtype, Operation::Sub, x, dtype, y)
                        }
                    })
                })
                .collect::<Vec<_>>()
        };
        check(&first, dtype, &[2, 3], &want(false))?;
        check(&second, dtype, &[2, 3], &want(true))?;
        check(&a, dtype, &[2, 1], &[3, 7])?;
        check(&b, dtype, &[1, 3], &[1, 2, 3])?;
        counts.native += 2;

        let wrong_dtype = if dtype == DType::U8 { DType::I8 } else { DType::U8 };
        let mismatch = gpu(NativeTensor::from_data(TensorData::from_bytes_vec(vec![1; 2], Shape::new([2, 1]), wrong_dtype)))?;
        let bad_shape = Input::flat(dtype, format, vec![1; 5])?.tensor()?;
        let flat = Input::flat(dtype, format, vec![1; 3])?.tensor()?;
        let before = gpu(backend.queue_profile())?;
        for operation in Operation::ALL {
            let rejected = if operation.unary() {
                a.integer_binary(&a, word(operation))
            } else {
                a.integer_unary(word(operation))
            };
            if !matches!(rejected, Err(Error::InvalidParameter)) {
                return Err(format!("integer wrong arity accepted {operation:?}"));
            }
            counts.rejected += 1;
            if !operation.unary() && operation != Operation::Powi {
                if !matches!(a.integer_binary(&mismatch, word(operation)), Err(Error::InvalidParameter)) {
                    return Err(format!("integer mismatched ordinary dtype accepted {operation:?}"));
                }
                counts.rejected += 1;
            }
        }
        for operation in [WordOp::Equal, WordOp::IntToFloat, WordOp::BoolAnd, WordOp::IntMaskFill] {
            if !matches!(a.integer_scalar(operation, Scalar::Int(1)), Err(Error::InvalidParameter)) {
                return Err(String::from("nonarithmetic integer scalar operation accepted"));
            }
            counts.rejected += 1;
        }
        if flat.integer_binary(&bad_shape, WordOp::Power).is_ok() {
            return Err(String::from("integer nonbroadcastable power accepted"));
        }
        counts.rejected += 1;
        no_launch(backend, &before)?;
        // Shared read operands are permitted; neither input may be mutated.
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(a.integer_binary(&a, WordOp::Multiply))?;
        resident(backend, &before, &profile, true)?;
        check(&output, dtype, &[2, 1], &[9, 49])?;
        check(&a, dtype, &[2, 1], &[3, 7])?;
        counts.native += 1;
    }
    Ok(())
}

fn raw_rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let a = Input::flat(dtype, format, vec![3; 5])?;
        let p = parameters(&a, &a, Operation::Add, None)?;
        let words = vec![POISON; packed(dtype, &a.storage).len() + 4];
        let input = gpu(backend.upload_words(&words))?;
        let rhs = gpu(backend.upload_words(&words))?;
        let output = gpu(backend.upload_words(&words))?;
        let short = gpu(backend.upload_words(&[POISON]))?;
        let mut malformed = Vec::new();
        for (field, value) in [(0, 0), (1, 9), (3, 1), (4, 5), (6, 0), (8, 6), (16, i32::MAX as usize)] {
            let mut bad = p;
            bad[field] = value;
            malformed.push(bad);
        }
        for bits in [1usize << 25, 255, (Format::F32 as usize) << 8] {
            let mut bad = p;
            if bits == (Format::F32 as usize) << 8 {
                bad[2] = (bad[2] & !0xff00) | bits;
            } else {
                bad[2] |= bits;
            }
            malformed.push(bad);
        }
        let mut bad = p;
        bad[2] = (bad[2] & !0xff0000) | ((if format == Format::U8 { Format::I8 } else { Format::U8 } as usize) << 16);
        malformed.push(bad);
        let unary = parameters(&a, &a, Operation::Not, None)?;
        for field in [3, 5, 7, 24] {
            let mut bad = unary;
            bad[field] = 1;
            malformed.push(bad);
        }
        let mut bad = parameters(&a, &a, Operation::Add, Some(3))?;
        if dtype.size() < 8 {
            if dtype.size() == 4 {
                bad[5] = 1;
            } else {
                bad[3] |= 1 << (dtype.size() * 8);
            }
        } else {
            bad[24] = 1;
        }
        malformed.push(bad);
        let before = gpu(backend.queue_profile())?;
        for bad in malformed {
            if gpu(backend.with_commands(|| Ok::<_, Error>(backend.dispatch_words(Kernel::IntegerArithmetic, &[(&input, 1), (&rhs, 1), (&output, 1)], &bad))))?
                .is_ok()
            {
                return Err(format!("integer malformed metadata accepted: {bad:?}"));
            }
            counts.rejected += 1;
        }
        // Exact and partial aliases, including the unused scalar RHS pointer.
        for (out, offset, p) in [(&input, 1, p), (&rhs, 2, p), (&rhs, 1, parameters(&a, &a, Operation::Add, Some(3))?)] {
            if backend
                .dispatch_words(Kernel::IntegerArithmetic, &[(&input, 1), (&rhs, 1), (out, offset)], &p)
                .is_ok()
            {
                return Err(String::from("integer output alias accepted"));
            }
            counts.rejected += 1;
        }
        for buffers in [
            [(&short, 0), (&rhs, 1), (&output, 1)],
            [(&input, 1), (&short, 0), (&output, 1)],
            [(&input, 1), (&rhs, 1), (&short, 0)],
        ] {
            if backend.dispatch_words(Kernel::IntegerArithmetic, &buffers, &p).is_ok() {
                return Err(String::from("integer short physical buffer accepted"));
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

fn scalar_rejections(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let mut invalid = vec![
        (DType::U8, Format::U8, Scalar::UInt(256)),
        (DType::I8, Format::I8, Scalar::Int(128)),
        (DType::U64, Format::U64, Scalar::Int(-1)),
        (DType::I64, Format::I64, Scalar::UInt(u64::MAX)),
        (DType::U8, Format::U8, Scalar::Float(256.0)),
        (DType::I8, Format::I8, Scalar::Float(-129.0)),
        (DType::U64, Format::U64, Scalar::Float(-1.0)),
        (DType::U64, Format::U64, Scalar::Float(18446744073709551616.0)),
        (DType::I64, Format::I64, Scalar::Float(9223372036854775808.0)),
        (DType::I64, Format::I64, Scalar::Float(f64::from_bits(0xc3e0_0000_0000_0001))),
    ];
    for (dtype, format) in FORMATS {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            invalid.push((dtype, format, Scalar::Float(value)));
        }
    }
    let device = d3os_ai::native_device().map_err(|e| format!("integer device: {e}"))?;
    for (dtype, format, scalar) in invalid {
        for length in [0, 3] {
            let input = Input::flat(dtype, format, vec![3; length])?;
            let source = input.tensor()?;
            let before = gpu(backend.queue_profile())?;
            if !matches!(source.integer_scalar(WordOp::Add, scalar), Err(Error::InvalidParameter)) {
                return Err(format!("integer invalid received Scalar accepted {dtype:?} {scalar:?} empty={}", length == 0));
            }
            counts.rejected += 1;
            no_launch(backend, &before)?;
            check(&gpu(source.integer_scalar(WordOp::Add, Scalar::Int(1)))?, dtype, &[length], &vec![4; length])?;
            counts.native += 1;

            // Invoke the backend with Scalar directly. Generic Scalar::new can
            // panic before entry and is deliberately outside this contract.
            let primitive = Native::int_from_data(input.data(), &NativeDevice);
            let before = gpu(backend.queue_profile())?;
            let failed = Tensor::<1, Int>::from_primitive::<Native>(Native::int_add_scalar(primitive, scalar));
            let dependent = failed.clone() + 1;
            no_launch(backend, &before)?;
            if !device.sync().is_err_and(|e| !e.is_device_poisoned())
                || !dependent.try_into_data().is_err_and(|e| !e.is_device_poisoned())
                || !failed.try_into_data().is_err_and(|e| !e.is_device_poisoned())
                || device.sync().is_err()
            {
                return Err(format!("integer Scalar error acknowledgment {dtype:?} {scalar:?}"));
            }
            no_launch(backend, &before)?;
            counts.rejected += 1;
            let valid = Tensor::<1, Int>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype)) + 1;
            check_data(
                &valid.try_into_data().map_err(|e| format!("integer public recovery: {e:?}"))?,
                dtype,
                &[length],
                &vec![4; length],
            )?;
            counts.public += 1;
        }
    }
    Ok(())
}

fn scalar_boundaries(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format, scalar, bits) in [
        (DType::U8, Format::U8, Scalar::Float(-0.75), 0),
        (DType::U8, Format::U8, Scalar::Float(255.75), 255),
        (DType::I8, Format::I8, Scalar::Float(-128.75), 128),
        (DType::I8, Format::I8, Scalar::Float(127.75), 127),
        (DType::I64, Format::I64, Scalar::Float(-9223372036854775808.0), 1u64 << 63),
        (
            DType::I64,
            Format::I64,
            Scalar::Float(f64::from_bits(0x43df_ffff_ffff_ffff)),
            0x7fff_ffff_ffff_fc00,
        ),
        (
            DType::U64,
            Format::U64,
            Scalar::Float(f64::from_bits(0x43ef_ffff_ffff_ffff)),
            0xffff_ffff_ffff_f800,
        ),
        (DType::U64, Format::U64, Scalar::UInt(u64::MAX), u64::MAX),
        (DType::I64, Format::I64, Scalar::Int(i64::MIN), 1u64 << 63),
        (DType::U8, Format::U8, Scalar::Bool(true), 1),
        (DType::I8, Format::I8, Scalar::Float(-0.0), 0),
    ] {
        let input = Input::flat(dtype, format, edges(dtype))?;
        let source = input.tensor()?;
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(source.integer_scalar(WordOp::BitXor, scalar))?;
        resident(backend, &before, &profile, true)?;
        check(&output, dtype, &[input.logical.len()], &expected(&input, &input, Operation::Xor, Some(bits)))?;
        check(&source, dtype, &[input.logical.len()], &input.logical)?;
        counts.native += 1;
    }
    for (dtype, format) in FORMATS {
        let input = Input::flat(dtype, format, vec![3, 5, 7])?;
        let source = input.tensor()?;
        for (scalar, exponent) in [(Scalar::Float(-0.75), 0), (Scalar::Float(1.75), 1), (Scalar::Float(2.75), 2)] {
            let output = gpu(source.integer_powi_scalar(scalar))?;
            check(
                &output,
                dtype,
                &[3],
                &input.logical.iter().map(|&a| oracle::power_reference(dtype, a, exponent)).collect::<Vec<_>>(),
            )?;
            counts.native += 1;
        }
    }
    let source = Input::flat(DType::U8, Format::U8, vec![3, 5, 7])?.tensor()?;
    check(
        &gpu(source.integer_powi_scalar(Scalar::UInt(200)))?,
        DType::U8,
        &[3],
        &[3, 5, 7].map(|a| oracle::power_reference(DType::U8, a, 200)),
    )?;
    counts.native += 1;
    let before = gpu(backend.queue_profile())?;
    if !matches!(source.integer_powi_scalar(Scalar::Float(-1.75)), Err(Error::InvalidParameter)) {
        return Err(String::from("negative non-shortcut U8 power accepted"));
    }
    no_launch(backend, &before)?;
    counts.rejected += 1;
    Ok(())
}

fn public_power_errors(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("integer power device: {e}"))?;
    for (dtype, format) in [(DType::U64, Format::U64), (DType::I64, Format::I64)] {
        for length in [0, 3] {
            for direct in [true, false] {
                let input = Input::flat(dtype, format, vec![3; length])?;
                let source = Tensor::<1, Int>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
                let native_source = Native::int_from_data(input.data(), &NativeDevice);
                let before = gpu(backend.queue_profile())?;
                // 2^31 passes public Scalar::new for signed and unsigned bases.
                // It must reach Native's fallible I32 exponent probe.
                let failed = if direct {
                    Tensor::<1, Int>::from_primitive::<Native>(Native::int_powi_scalar(native_source, Scalar::UInt(1u64 << 31)))
                } else {
                    source.clone().powi_scalar(1u64 << 31)
                };
                no_launch(backend, &before)?;
                if !device.sync().is_err_and(|e| !e.is_device_poisoned())
                    || !failed.try_into_data().is_err_and(|e| !e.is_device_poisoned())
                    || device.sync().is_err()
                {
                    return Err(format!("integer large scalar power error direct={direct} dtype={dtype:?}"));
                }
                no_launch(backend, &before)?;
                counts.rejected += 1;
                let valid = source.powi_scalar(3);
                check_data(
                    &valid.try_into_data().map_err(|e| format!("integer power recovery: {e:?}"))?,
                    dtype,
                    &[length],
                    &vec![27; length],
                )?;
                counts.public += 1;
            }
        }
    }
    Ok(())
}

fn public_extensions(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("integer extension device: {e}"))?;
    for (dtype, format) in FORMATS.into_iter().filter(|(dtype, _)| dtype.is_uint()) {
        let input = Input::flat(dtype, format, edges(dtype))?;
        let source = Tensor::<1, Int>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = source.clone().neg();
        resident(backend, &before, &profile, true)?;
        check_data(
            &output.try_into_data().map_err(|e| format!("public unsigned negation: {e:?}"))?,
            dtype,
            &[input.logical.len()],
            &expected(&input, &input, Operation::Neg, None),
        )?;
        check_data(
            &source.try_into_data().map_err(|e| format!("public unsigned source: {e:?}"))?,
            dtype,
            &[input.logical.len()],
            &input.storage,
        )?;
        counts.public += 1;
    }
    // Wrapper smoke checks complement the 8x8 Native/raw matrix. Include an
    // I32 base with a wide exponent so the legacy matching-I32 route cannot win.
    for ((dtype, format), (rhs_dtype, rhs_format)) in [
        ((DType::I32, Format::I32), (DType::U64, Format::U64)),
        ((DType::U64, Format::U64), (DType::I8, Format::I8)),
        ((DType::U8, Format::U8), (DType::U64, Format::U64)),
        ((DType::I64, Format::I64), (DType::U32, Format::U32)),
    ] {
        let input = Input::flat(dtype, format, vec![3, 5, 7])?;
        let exponents = Input::flat(
            rhs_dtype,
            rhs_format,
            [oracle::mask(rhs_dtype), 256, (1u64 << 32) + 1].map(|v| v & oracle::mask(rhs_dtype)).to_vec(),
        )?;
        let source = Tensor::<1, Int>::from_data(input.data(), TensorCreationOptions::new(device.clone()).with_dtype(dtype));
        let rhs = Tensor::<1, Int>::from_data(exponents.data(), TensorCreationOptions::new(device.clone()).with_dtype(rhs_dtype));
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = source.clone().powi(rhs.clone());
        resident(backend, &before, &profile, true)?;
        check_data(
            &output.try_into_data().map_err(|e| format!("public mixed power: {e:?}"))?,
            dtype,
            &[3],
            &expected(&input, &exponents, Operation::Powi, None),
        )?;
        check_data(
            &source.try_into_data().map_err(|e| format!("public mixed base: {e:?}"))?,
            dtype,
            &[3],
            &input.storage,
        )?;
        check_data(
            &rhs.try_into_data().map_err(|e| format!("public mixed exponent: {e:?}"))?,
            rhs_dtype,
            &[3],
            &exponents.storage,
        )?;
        counts.public += 1;
    }
    Ok(())
}

fn empty_and_lifetimes(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let empty = gpu(NativeTensor::empty(Shape::new([2, 0, 3]), dtype))?;
        let before = gpu(backend.queue_profile())?;
        for op in Operation::ALL {
            let output = if op.unary() {
                empty.integer_unary(word(op))
            } else {
                empty.integer_binary(&empty, word(op))
            };
            check(&gpu(output)?, dtype, &[2, 0, 3], &[])?;
            counts.native += 1;
            if !op.unary() {
                check(&gpu(empty.integer_scalar(word(op), Scalar::Int(1)))?, dtype, &[2, 0, 3], &[])?;
                counts.native += 1;
            }
        }
        no_launch(backend, &before)?;

        let input = Input::flat(dtype, format, edges(dtype))?;
        let expected = expected(&input, &input, Operation::Mul, None);
        let a = input.tensor()?;
        let b = input.tensor()?;
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| {
            let result = a.integer_binary(&b, WordOp::Multiply)?;
            drop(a);
            drop(b);
            Ok::<_, Error>(result)
        }))?;
        resident(backend, &before, &profile, true)?;
        check(&output, dtype, &[expected.len()], &expected)?;
        counts.native += 1;

        let words = packed(dtype, &input.storage);
        let a = gpu(backend.upload_words(&words))?;
        let b = gpu(backend.upload_words(&words))?;
        let p = parameters(&input, &input, Operation::Mul, None)?;
        let before = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| {
            let result = backend.produce_words(words.len(), Kernel::IntegerArithmetic, &[(&a, 0), (&b, 0)], &p)?;
            drop(a);
            drop(b);
            Ok::<_, Error>(result)
        }))?;
        resident(backend, &before, &profile, true)?;
        unchanged(backend, &output, &packed(dtype, &expected))?;
        counts.fresh += 1;
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<()> {
    let mut counts = Counts::default();
    operation_matrix(backend, &mut counts)?;
    layouts(backend, &mut counts)?;
    raw_rejections(backend, &mut counts)?;
    native_metadata_and_broadcast(backend, &mut counts)?;
    scalar_rejections(backend, &mut counts)?;
    scalar_boundaries(backend, &mut counts)?;
    public_power_errors(backend, &mut counts)?;
    public_extensions(backend, &mut counts)?;
    empty_and_lifetimes(backend, &mut counts)?;
    crate::report(format_args!(
        "NVIDIA_BURN_INTEGER_ARITHMETIC_OK operations=15 formats=8 mixed_power=64 raw={} fresh={} native={} rejected={} public={} oracle=i128_u128 guards=ok tails=zero resident=true queued_lifetime=verified errors=recoverable backend=nvidia",
        counts.raw, counts.fresh, counts.native, counts.rejected, counts.public,
    ));
    Ok(())
}
