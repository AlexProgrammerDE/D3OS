//! Guarded predicates with independent IEEE and exact integer references.
use alloc::{format, string::String, vec, vec::Vec};
use core::cmp::Ordering;
use d3os_ai::{
    burn::backend::{BoolStore, DType, Scalar, Shape, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
    validation::storage::DTYPES,
};
use d3os_compute::{
    Backend as _, Error,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{
    self as abi, Kernel,
    cast::Format,
    predicate::{PACKED_BOOL, Predicate, SCALAR_RHS},
    words::WordOp,
};

const GUARD: u32 = 0x6bad_beef;
const COUNT: usize = 411;

// Keep the expected operation matrix independent of the ABI's supports check.
fn operations(format: Format) -> &'static [Predicate] {
    use Predicate::*;
    match format {
        Format::F64 | Format::F32 | Format::Flex32 | Format::F16 | Format::BF16 => {
            &[Equal, Greater, GreaterEqual, Less, LessEqual, IsNan, IsInfinite, IsFinite]
        }
        Format::Bool8 | Format::Bool32 => &[Equal, And, Or, Xor, Not],
        _ => &[Equal, Greater, GreaterEqual, Less, LessEqual],
    }
}

fn mask(format: Format) -> u64 {
    u64::MAX >> (64 - format.width().bytes() * 8)
}

fn samples(format: Format) -> Vec<u64> {
    if matches!(format, Format::Bool8 | Format::Bool32) {
        return vec![0, 1, 2, mask(format), 0, 0x80, 1, 0];
    }
    if (format as u32) <= Format::BF16 as u32 {
        let (sign, infinity, one, fraction) = match format {
            Format::F64 => (1 << 63, 0x7ff0_0000_0000_0000, 0x3ff0_0000_0000_0000, 52),
            Format::F32 | Format::Flex32 => (1 << 31, 0x7f80_0000, 0x3f80_0000, 23),
            Format::F16 => (1 << 15, 0x7c00, 0x3c00, 10),
            Format::BF16 => (1 << 15, 0x7f80, 0x3f80, 7),
            _ => unreachable!(),
        };
        return vec![
            0,
            sign,
            1,
            sign | 1,
            (1 << fraction) - 1,
            1 << fraction,
            one,
            one + 1,
            sign | one,
            sign | (one + 1),
            infinity - 1,
            sign | (infinity - 1),
            infinity,
            sign | infinity,
            infinity | 1,
            sign | infinity | 1,
            infinity | (1 << (fraction - 1)),
            sign | infinity | (1 << (fraction - 1)),
        ];
    }
    let maximum = mask(format);
    let sign = (maximum >> 1) + 1;
    [
        0,
        1,
        2,
        sign - 2,
        sign - 1,
        sign,
        sign + 1,
        maximum - 1,
        maximum,
        (1 << 24) - 1,
        1 << 24,
        (1 << 24) + 1,
        (1 << 53) - 1,
        1 << 53,
        (1 << 53) + 1,
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
    ]
    .into_iter()
    .map(|bits| bits & maximum)
    .collect()
}

// Decode to host IEEE values, rather than duplicating the device's sortable keys.
fn float(format: Format, bits: u64) -> f64 {
    match format {
        Format::F64 => f64::from_bits(bits),
        Format::F32 | Format::Flex32 => f64::from(f32::from_bits(bits as u32)),
        Format::BF16 => f64::from(f32::from_bits((bits as u32) << 16)),
        Format::F16 => {
            let exponent = ((bits >> 10) & 31) as i32;
            let fraction = (bits & 1023) as f64;
            let magnitude = match exponent {
                0 => libm::scalbn(fraction, -24),
                31 if fraction == 0.0 => f64::INFINITY,
                31 => f64::NAN,
                _ => libm::scalbn(1024.0 + fraction, exponent - 25),
            };
            if bits & 0x8000 == 0 { magnitude } else { -magnitude }
        }
        _ => unreachable!(),
    }
}

fn reference(format: Format, op: Predicate, a: u64, b: u64) -> bool {
    match op {
        Predicate::And => return a != 0 && b != 0,
        Predicate::Or => return a != 0 || b != 0,
        Predicate::Xor => return (a != 0) != (b != 0),
        Predicate::Not => return a == 0,
        Predicate::IsNan => return float(format, a).is_nan(),
        Predicate::IsInfinite => return float(format, a).is_infinite(),
        Predicate::IsFinite => return float(format, a).is_finite(),
        _ => {}
    }
    let order = match format {
        Format::F64 | Format::F32 | Format::Flex32 | Format::F16 | Format::BF16 => float(format, a).partial_cmp(&float(format, b)),
        Format::I64 => Some((a as i64).cmp(&(b as i64))),
        Format::I32 => Some((a as i32).cmp(&(b as i32))),
        Format::I16 => Some((a as i16).cmp(&(b as i16))),
        Format::I8 => Some((a as i8).cmp(&(b as i8))),
        Format::Bool8 | Format::Bool32 => Some((a != 0).cmp(&(b != 0))),
        _ => Some(a.cmp(&b)),
    };
    match op {
        Predicate::Equal => order == Some(Ordering::Equal),
        Predicate::Greater => order == Some(Ordering::Greater),
        Predicate::GreaterEqual => matches!(order, Some(Ordering::Greater | Ordering::Equal)),
        Predicate::Less => order == Some(Ordering::Less),
        Predicate::LessEqual => matches!(order, Some(Ordering::Less | Ordering::Equal)),
        _ => unreachable!(),
    }
}

fn bytes(format: Format, values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|bits| bits.to_le_bytes()[..format.width().bytes() as usize].to_vec())
        .collect()
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

fn guarded(format: Format, values: &[u64]) -> Vec<u32> {
    let raw = bytes(format, values);
    let mut packed = words(&raw);
    if !raw.len().is_multiple_of(4) {
        *packed.last_mut().unwrap() |= 0xa5a5_a5a5u32 << ((raw.len() % 4) * 8);
    }
    let mut result = vec![GUARD];
    result.extend(packed);
    result.push(GUARD);
    result
}

fn expected_words(values: &[bool], packed: bool) -> Vec<u32> {
    if packed {
        words(&values.iter().copied().map(u8::from).collect::<Vec<_>>())
    } else {
        values.iter().copied().map(u32::from).collect()
    }
}

fn full_writer(backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS]) -> Result<(), String> {
    super::dispatch::resident(backend, before, 0)?;
    let after = backend.service_profile().map_err(|e| format!("{e}"))?;
    if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
        || after[abi::LAUNCH as usize].calls != before[abi::LAUNCH as usize].calls
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
    {
        return Err(String::from("predicate full writer needed a separate allocation or clear"));
    }
    Ok(())
}

struct Case<'a> {
    format: Format,
    op: Predicate,
    packed: bool,
    scalar: Option<u64>,
    left: &'a Layout,
    right: &'a Layout,
    expected: &'a [bool],
}

impl Case<'_> {
    fn parameters(&self) -> Result<[usize; 32], String> {
        let mut p = self.left.word_parameters(self.right, WordOp::Equal, 0, false).map_err(|e| format!("{e}"))?;
        p[2] = self.op as usize | ((self.format as usize) << 8) | if self.packed { PACKED_BOOL as usize } else { 0 };
        if let Some(bits) = self.scalar {
            p[2] |= SCALAR_RHS as usize;
            p[3] = bits as u32 as usize;
            p[5] = (bits >> 32) as usize;
        }
        if self.scalar.is_some() || !self.op.binary() {
            if self.scalar.is_none() {
                p[5] = 0;
            }
            p[7] = 0;
            p[24..32].fill(0);
        }
        Ok(p)
    }

    fn raw(&self, backend: &Nvidia, left: &GpuBuffer<u32>, right: &GpuBuffer<u32>) -> Result<(), String> {
        let p = self.parameters()?;
        let expected = expected_words(self.expected, self.packed);
        let output = backend.upload_words(&vec![GUARD; expected.len() + 2]).map_err(|e| format!("{e}"))?;
        // Unused RHS still needs the ABI's one-word dummy allocation.
        let right = if self.scalar.is_some() || !self.op.binary() { left } else { right };
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        backend
            .with_commands(|| backend.dispatch_words(Kernel::TensorPredicate, &[(left, 1), (right, 1), (&output, 1)], &p))
            .map_err(|e| format!("raw {:?} {:?} packed={} scalar={:?}: {e}", self.format, self.op, self.packed, self.scalar))?;
        super::dispatch::resident(backend, &before, 0)?;
        let mut actual = vec![0; expected.len() + 2];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual[0] != GUARD || actual[expected.len() + 1] != GUARD || actual[1..expected.len() + 1] != expected {
            return Err(format!(
                "predicate raw guards/reference/tail {:?} {:?} packed={} scalar={:?}",
                self.format, self.op, self.packed, self.scalar
            ));
        }
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let output = backend
            .with_commands(|| backend.produce_words(expected.len(), Kernel::TensorPredicate, &[(left, 1), (right, 1)], &p))
            .map_err(|e| format!("{e}"))?;
        full_writer(backend, &before)?;
        let mut actual = vec![0; expected.len()];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual != expected {
            return Err(format!("predicate raw full writer {:?} {:?}", self.format, self.op));
        }
        Ok(())
    }

    fn native(&self, backend: &Nvidia, left: &NativeTensor, right: &NativeTensor) -> Result<(), String> {
        let dtype = DType::Bool(if self.packed { BoolStore::U8 } else { BoolStore::U32 });
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let output = backend
            .with_commands(|| {
                if let Some(bits) = self.scalar {
                    left.compare_scalar(scalar(self.format, bits), self.op, dtype)
                } else if self.op.binary() {
                    left.compare(right, self.op, dtype)
                } else {
                    left.classify(self.op, dtype)
                }
            })
            .map_err(|e| format!("native {:?} {:?} packed={} scalar={:?}: {e}", self.format, self.op, self.packed, self.scalar))?;
        full_writer(backend, &before)?;
        let actual = output.read().map_err(|e| format!("{e}"))?;
        let expected: Vec<_> = if self.packed {
            self.expected.iter().copied().map(u8::from).collect()
        } else {
            self.expected.iter().copied().flat_map(|v| u32::from(v).to_le_bytes()).collect()
        };
        if actual.dtype() != dtype || actual.shape() != self.left.shape() || actual.as_bytes() != expected {
            return Err(format!(
                "predicate native metadata/reference {:?} {:?} packed={} scalar={:?}",
                self.format, self.op, self.packed, self.scalar
            ));
        }
        Ok(())
    }
}

fn scalar(format: Format, bits: u64) -> Scalar {
    match format {
        Format::F64 | Format::F32 | Format::Flex32 | Format::F16 | Format::BF16 => Scalar::from(float(format, bits)),
        Format::I64 => Scalar::from(bits as i64),
        Format::I32 => Scalar::from(bits as i32),
        Format::I16 => Scalar::from(bits as i16),
        Format::I8 => Scalar::from(bits as i8),
        Format::Bool8 | Format::Bool32 => Scalar::from(bits != 0),
        _ => Scalar::from(bits),
    }
}

fn tensor(dtype: DType, format: Format, values: &[u64], layout: &Layout) -> Result<NativeTensor, String> {
    NativeTensor::from_data(TensorData::from_bytes_vec(bytes(format, values), [values.len()], dtype))
        .and_then(|t| t.with_layout(layout.clone()))
        .map_err(|e| format!("{e}"))
}

fn formats(backend: &Nvidia) -> Result<(usize, usize), String> {
    let shape = Shape::new([137, 3]);
    let left_layout = Layout::from_parts(shape.clone(), vec![-1, -139], 415, 417).map_err(|e| format!("{e}"))?;
    let right_layout = Layout::from_parts(shape.clone(), vec![3, -1], 3, 417).map_err(|e| format!("{e}"))?;
    let broadcast_layout = Layout::from_parts(Shape::new([3]), vec![-1], 3, 5).map_err(|e| format!("{e}"))?;
    let expanded = broadcast_layout.expand(shape.clone()).map_err(|e| format!("{e}"))?;
    let (mut raw_cases, mut native_cases) = (0, 0);
    for (dtype, format) in DTYPES.into_iter().zip(Format::ALL) {
        let pool = samples(format);
        let left: Vec<_> = (0..417).map(|i| pool[(i * 7) % pool.len()]).collect();
        let logical: Vec<_> = (0..COUNT).map(|i| left[415 - i / 3 - (i % 3) * 139]).collect();
        let mut right = left.clone();
        for i in 0..COUNT {
            right[3 + (i / 3) * 3 - i % 3] = if i % 3 == 0 { logical[i] } else { pool[(i / 3) % pool.len()] };
        }
        let logical_right: Vec<_> = (0..COUNT).map(|i| right[3 + (i / 3) * 3 - i % 3]).collect();
        let broadcast: Vec<_> = (0..5).map(|i| pool[(i * 5 + 1) % pool.len()]).collect();
        let ga = guarded(format, &left);
        let gb = guarded(format, &right);
        let gc = guarded(format, &broadcast);
        let a = backend.upload_words(&ga).map_err(|e| format!("{e}"))?;
        let b = backend.upload_words(&gb).map_err(|e| format!("{e}"))?;
        let c = backend.upload_words(&gc).map_err(|e| format!("{e}"))?;
        let na = tensor(dtype, format, &left, &left_layout)?;
        let nb = tensor(dtype, format, &right, &right_layout)?;
        let nc = tensor(dtype, format, &broadcast, &broadcast_layout)?;
        let scalar_bits = match format {
            Format::Bool8 | Format::Bool32 => 1,
            Format::F64 | Format::F32 | Format::Flex32 | Format::F16 | Format::BF16 => pool[9],
            _ => 0xfedc_ba98_7654_3210 & mask(format),
        };
        for packed in [true, false] {
            for &op in operations(format) {
                for scalar_bits in [None, Some(scalar_bits)].into_iter().take(if op.binary() { 2 } else { 1 }) {
                    let expected: Vec<_> = logical
                        .iter()
                        .enumerate()
                        .map(|(i, &a)| reference(format, op, a, scalar_bits.unwrap_or(logical_right[i])))
                        .collect();
                    let case = Case {
                        format,
                        op,
                        packed,
                        scalar: scalar_bits,
                        left: &left_layout,
                        right: &right_layout,
                        expected: &expected,
                    };
                    case.raw(backend, &a, &b)?;
                    raw_cases += 1;
                    case.native(backend, &na, &nb)?;
                    native_cases += 1;
                }
            }
            let expected: Vec<_> = logical
                .iter()
                .enumerate()
                .map(|(i, &a)| reference(format, Predicate::Equal, a, broadcast[3 - i % 3]))
                .collect();
            let case = Case {
                format,
                op: Predicate::Equal,
                packed,
                scalar: None,
                left: &left_layout,
                right: &expanded,
                expected: &expected,
            };
            case.raw(backend, &a, &c)?;
            case.native(backend, &na, &nc)?;
            raw_cases += 1;
            native_cases += 1;
        }
        for (input, original) in [(&a, &ga), (&b, &gb), (&c, &gc)] {
            let mut preserved = vec![0; original.len()];
            backend.read_words_into(input, 0, &mut preserved).map_err(|e| format!("{e}"))?;
            if &preserved != original {
                return Err(format!("predicate changed input, padding, or guard {format:?}"));
            }
        }
        if na.read().map_err(|e| format!("{e}"))?.as_bytes() != bytes(format, &logical)
            || nb.read().map_err(|e| format!("{e}"))?.as_bytes() != bytes(format, &logical_right)
            || nc.read().map_err(|e| format!("{e}"))?.as_bytes() != bytes(format, &[broadcast[3], broadcast[2], broadcast[1]])
        {
            return Err(format!("predicate changed native input {format:?}"));
        }
    }
    Ok((raw_cases, native_cases))
}

fn ranks_and_lifetimes(backend: &Nvidia) -> Result<(usize, usize), String> {
    let (mut raw_cases, mut native_cases) = (0, 0);
    for shape in [Shape::new([]), Shape::new([2; 8])] {
        let rank8 = shape.num_dims() == 8;
        let layout = Layout::contiguous(shape.clone())
            .and_then(|layout| if rank8 { layout.flip(&[0, 2, 7]) } else { Ok(layout) })
            .map_err(|e| format!("{e}"))?;
        let values: Vec<_> = (0..layout.len()).map(|i| (1 << 53) + i as u64).collect();
        let threshold = if rank8 { (1 << 53) + 85 } else { (1 << 53) - 1 };
        let expected: Vec<_> = (0..layout.len()).map(|i| values[if rank8 { i ^ 161 } else { i }] > threshold).collect();
        for packed in [true, false] {
            let a = backend.upload_words(&guarded(Format::U64, &values)).map_err(|e| format!("{e}"))?;
            let na = tensor(DType::U64, Format::U64, &values, &layout)?;
            let case = Case {
                format: Format::U64,
                op: Predicate::Greater,
                packed,
                scalar: Some(threshold),
                left: &layout,
                right: &layout,
                expected: &expected,
            };
            case.raw(backend, &a, &a)?;
            case.native(backend, &na, &na)?;
            raw_cases += 1;
            native_cases += 1;
        }
    }
    // Every packed tail length and both sides of a 128-output-word tile.
    for count in [1, 2, 3, 4, 127, 128, 129, 511, 512, 513] {
        let layout = Layout::contiguous(Shape::new([count])).map_err(|e| format!("{e}"))?;
        let source: Vec<_> = (0..count).map(|i| u64::from(i % 3 != 0)).collect();
        let expected: Vec<_> = source.iter().map(|&bits| bits == 0).collect();
        let input = backend.upload_words(&guarded(Format::Bool8, &source)).map_err(|e| format!("{e}"))?;
        let native = tensor(DType::Bool(BoolStore::U8), Format::Bool8, &source, &layout)?;
        for packed in [true, false] {
            let case = Case {
                format: Format::Bool8,
                op: Predicate::Not,
                packed,
                scalar: None,
                left: &layout,
                right: &layout,
                expected: &expected,
            };
            case.raw(backend, &input, &input)?;
            case.native(backend, &native, &native)?;
            raw_cases += 1;
            native_cases += 1;
        }
    }
    for packed in [true, false] {
        let layout = Layout::contiguous(Shape::new([129])).map_err(|e| format!("{e}"))?;
        let left: Vec<_> = (0..129).map(|i| (1 << 53) + i).collect();
        let right: Vec<_> = left.iter().map(|&v| v + 1).collect();
        let expected = vec![true; 129];
        let case = Case {
            format: Format::U64,
            op: Predicate::Less,
            packed,
            scalar: None,
            left: &layout,
            right: &layout,
            expected: &expected,
        };
        let p = case.parameters()?;
        let a = backend.upload_bytes(&bytes(Format::U64, &left)).map_err(|e| format!("{e}"))?;
        let b = backend.upload_bytes(&bytes(Format::U64, &right)).map_err(|e| format!("{e}"))?;
        let expected = expected_words(&expected, packed);
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let output = backend
            .with_commands(|| {
                let output = backend.produce_words(expected.len(), Kernel::TensorPredicate, &[(&a, 0), (&b, 0)], &p)?;
                drop(a);
                drop(b);
                Ok::<_, Error>(output)
            })
            .map_err(|e| format!("{e}"))?;
        full_writer(backend, &before)?;
        let mut actual = vec![0; expected.len()];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual != expected {
            return Err(String::from("predicate queued raw sources dropped too early"));
        }
        raw_cases += 1;
        let a = tensor(DType::U64, Format::U64, &left, &layout)?;
        let b = tensor(DType::U64, Format::U64, &right, &layout)?;
        let dtype = DType::Bool(if packed { BoolStore::U8 } else { BoolStore::U32 });
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let output = backend
            .with_commands(|| {
                let output = a.compare(&b, Predicate::Less, dtype)?;
                drop(a);
                drop(b);
                Ok::<_, Error>(output)
            })
            .map_err(|e| format!("{e}"))?;
        full_writer(backend, &before)?;
        let actual = output.read().map_err(|e| format!("{e}"))?;
        let expected: Vec<_> = expected.iter().flat_map(|v| v.to_le_bytes()).take(129 * dtype.size()).collect();
        if actual.as_bytes() != expected || actual.dtype() != dtype || actual.shape().as_slice() != [129] {
            return Err(String::from("predicate queued native sources dropped too early"));
        }
        native_cases += 1;
    }
    Ok((raw_cases, native_cases))
}

fn invalid_and_empty(backend: &Nvidia) -> Result<usize, String> {
    let layout = Layout::contiguous(Shape::new([3])).map_err(|e| format!("{e}"))?;
    let p = Case {
        format: Format::U8,
        op: Predicate::Equal,
        packed: true,
        scalar: None,
        left: &layout,
        right: &layout,
        expected: &[],
    }
    .parameters()?;
    let source = backend.upload_words(&[GUARD; 8]).map_err(|e| format!("{e}"))?;
    let rhs = backend.upload_words(&[GUARD; 8]).map_err(|e| format!("{e}"))?;
    let output = backend.upload_words(&[GUARD; 8]).map_err(|e| format!("{e}"))?;
    let mut rejected = 0;
    for (slot, value) in [
        (0, 0),
        (0, 2),
        (1, 0),
        (1, 9),
        (2, p[2] | (1 << 18)),
        (2, 12 | (Format::U8 as usize) << 8),
        (2, 15 << 8),
        (2, Predicate::IsFinite as usize | ((Format::U8 as usize) << 8)),
        (2, Predicate::And as usize | ((Format::F32 as usize) << 8)),
        (2, Predicate::Greater as usize | ((Format::Bool8 as usize) << 8)),
        (3, 1),
        (4, 3),
        (4, u32::MAX as usize),
        (5, 3),
        (6, 0),
        (6, 2),
        (7, 0),
        (8, 2),
        (9, 2),
        (16, u32::MAX as usize),
        (17, 1),
        (25, 1),
    ] {
        let mut invalid = p;
        invalid[slot] = value;
        let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
        if backend
            .dispatch_words(Kernel::TensorPredicate, &[(&source, 0), (&rhs, 0), (&output, 0)], &invalid)
            .is_ok()
            || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
        {
            return Err(format!("predicate invalid metadata accepted/launched slot={slot} value={value}"));
        }
        rejected += 1;
    }
    let mut scalar = p;
    scalar[2] |= SCALAR_RHS as usize;
    scalar[5] = 0;
    scalar[7] = 0;
    scalar[24..32].fill(0);
    let mut unary = scalar;
    unary[2] = Predicate::Not as usize | ((Format::Bool8 as usize) << 8) | PACKED_BOOL as usize;
    for (base, slot, value) in [
        (scalar, 3, 256),
        (scalar, 5, 1),
        (scalar, 7, 1),
        (scalar, 24, 1),
        (unary, 3, 1),
        (unary, 5, 1),
        (unary, 7, 1),
        (unary, 24, 1),
        (unary, 2, unary[2] | SCALAR_RHS as usize),
        (scalar, 2, Predicate::Equal as usize | ((Format::Bool8 as usize) << 8) | SCALAR_RHS as usize),
    ] {
        let mut invalid = base;
        invalid[slot] = value;
        // Boolean scalar case is otherwise valid; reject its noncanonical bits.
        if slot == 2 && base == scalar {
            invalid[3] = 2;
        }
        let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
        if backend
            .dispatch_words(Kernel::TensorPredicate, &[(&source, 0), (&rhs, 0), (&output, 0)], &invalid)
            .is_ok()
            || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
        {
            return Err(format!("predicate invalid scalar/unary geometry accepted slot={slot}"));
        }
        rejected += 1;
    }
    // Buffer ranges and every output/input overlap must fail before enqueue.
    for buffers in [
        [(&source, 0), (&rhs, 0), (&source, 0)],
        [(&source, 0), (&rhs, 0), (&rhs, 0)],
        [(&source, 0), (&rhs, 0), (&output, 8)],
        [(&source, 8), (&rhs, 0), (&output, 0)],
        [(&source, 0), (&rhs, 8), (&output, 0)],
    ] {
        let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
        if backend.dispatch_words(Kernel::TensorPredicate, &buffers, &p).is_ok()
            || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
        {
            return Err(String::from("predicate invalid alias/range accepted or launched"));
        }
        rejected += 1;
    }
    let mut wide = p;
    wide[2] = Predicate::Equal as usize | ((Format::U64 as usize) << 8) | PACKED_BOOL as usize;
    for aliased in [&source, &rhs] {
        let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
        if backend
            .dispatch_words(Kernel::TensorPredicate, &[(&source, 0), (&rhs, 0), (aliased, 1)], &wide)
            .is_ok()
            || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
        {
            return Err(String::from("predicate partial alias accepted or launched"));
        }
        rejected += 1;
    }
    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
    if backend
        .dispatch_words(Kernel::TensorPredicate, &[(&source, 0), (&rhs, 0), (&output, 0)], &p[..31])
        .is_ok()
        || backend.dispatch_words(Kernel::TensorPredicate, &[(&source, 0), (&output, 0)], &p).is_ok()
        || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
    {
        return Err(String::from("predicate malformed launch accepted"));
    }
    rejected += 2;
    for buffer in [&source, &rhs, &output] {
        let mut actual = [0; 8];
        backend.read_words_into(buffer, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual != [GUARD; 8] {
            return Err(String::from("rejected predicate changed a buffer"));
        }
    }
    for (dtype, format) in DTYPES.into_iter().zip(Format::ALL) {
        let empty = NativeTensor::empty(Shape::new([0, 3]), dtype).map_err(|e| format!("{e}"))?;
        let wrong_shape = NativeTensor::full(Shape::new([2]), dtype, 0).map_err(|e| format!("{e}"))?;
        let wrong_dtype = NativeTensor::full(Shape::new([0, 3]), if dtype == DType::U8 { DType::I8 } else { DType::U8 }, 0).map_err(|e| format!("{e}"))?;
        for store in [BoolStore::U8, BoolStore::U32] {
            let destination = DType::Bool(store);
            for &op in operations(format) {
                let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
                let output = if op.binary() {
                    empty.compare(&empty, op, destination)
                } else {
                    empty.classify(op, destination)
                }
                .map_err(|e| format!("{e}"))?;
                if !output.layout().is_empty() || output.layout().shape().as_slice() != [0, 3] || output.dtype() != destination {
                    return Err(String::from("empty predicate metadata"));
                }
                if op.binary() {
                    let output = empty.compare_scalar(Scalar::from(1u8), op, destination).map_err(|e| format!("{e}"))?;
                    if !output.layout().is_empty() || output.dtype() != destination || output.layout().shape().as_slice() != [0, 3] {
                        return Err(String::from("empty scalar predicate metadata"));
                    }
                }
                if backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches {
                    return Err(String::from("empty predicate launched work"));
                }
            }
            let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
            if empty.compare(&wrong_dtype, Predicate::Equal, destination).is_ok()
                || empty.compare(&wrong_shape, Predicate::Equal, destination).is_ok()
                || empty.compare(&empty, Predicate::Not, destination).is_ok()
                || empty.compare_scalar(Scalar::from(1u8), Predicate::IsFinite, destination).is_ok()
                || empty.classify(Predicate::Equal, destination).is_ok()
                || empty.compare(&empty, Predicate::Equal, DType::U8).is_ok()
                || empty
                    .compare_scalar(Scalar::from(1u8), Predicate::Equal, DType::Bool(BoolStore::Native))
                    .is_ok()
                || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
            {
                return Err(format!("predicate invalid native type/shape/op accepted {dtype:?}"));
            }
            rejected += 7;
            for op in Predicate::ALL.into_iter().filter(|op| !operations(format).contains(op)) {
                let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
                let result = if op.binary() {
                    empty.compare(&empty, op, destination)
                } else {
                    empty.classify(op, destination)
                };
                if result.is_ok() || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches {
                    return Err(format!("predicate unsupported operation accepted {format:?} {op:?}"));
                }
                rejected += 1;
            }
        }
    }
    Ok(rejected)
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let (raw_cases, native_cases) = formats(backend)?;
    let (extra_raw, extra_native) = ranks_and_lifetimes(backend)?;
    let raw_cases = raw_cases + extra_raw;
    let native_cases = native_cases + extra_native;
    let rejected = invalid_and_empty(backend)?;
    crate::report(format_args!(
        "NVIDIA_BURN_PREDICATE_OK formats=15 output_widths=1,4 raw_cases={raw_cases} native_cases={native_cases} rejected={rejected} oracle=independent guards=ok tails=zero broadcast=verified signed_offsets=verified full_writer=verified resident=true queued_lifetime=verified rank0=verified rank8=verified empty=no_launch invalid_inputs=no_launch"
    ));
    Ok(())
}
