//! Exact integer and boolean references, including wrapping and conversion bounds.
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Shape, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{
    self as abi,
    words::{ValueType, WordOp},
};

fn reference(op: WordOp, a: u32, b: u32) -> u32 {
    let (x, y) = (a as i32, b as i32);
    match op {
        WordOp::Add => x.wrapping_add(y) as u32,
        WordOp::Subtract => x.wrapping_sub(y) as u32,
        WordOp::Multiply => x.wrapping_mul(y) as u32,
        WordOp::Power => d3os_ai::validation::integer::power_reference(x, y) as u32,
        WordOp::Divide => {
            if y == 0 {
                u32::MAX
            } else {
                (i64::from(x) / i64::from(y)) as u32
            }
        }
        WordOp::Remainder => {
            if y == 0 {
                return a;
            }
            let remainder = i64::from(x) % i64::from(y);
            (if remainder != 0 && (remainder < 0) != (y < 0) {
                remainder + i64::from(y)
            } else {
                remainder
            }) as u32
        }
        WordOp::Equal => u32::from(x == y),
        WordOp::Greater => u32::from(x > y),
        WordOp::GreaterEqual => u32::from(x >= y),
        WordOp::Less => u32::from(x < y),
        WordOp::LessEqual => u32::from(x <= y),
        WordOp::Negate => x.wrapping_neg() as u32,
        WordOp::Absolute => x.wrapping_abs() as u32,
        WordOp::Sign => x.signum() as u32,
        WordOp::BitAnd => a & b,
        WordOp::BitOr => a | b,
        WordOp::BitXor => a ^ b,
        WordOp::BitNot => !a,
        WordOp::ShiftLeft => x.wrapping_shl(b) as u32,
        WordOp::ShiftRight => x.wrapping_shr(b) as u32,
        WordOp::BoolEqual => u32::from((a != 0) == (b != 0)),
        WordOp::BoolAnd => u32::from(a != 0 && b != 0),
        WordOp::BoolOr => u32::from(a != 0 || b != 0),
        WordOp::BoolXor => u32::from((a != 0) != (b != 0)),
        WordOp::BoolNot => u32::from(a == 0),
        WordOp::IntToFloat => (x as f32).to_bits(),
        WordOp::FloatToInt => (f32::from_bits(a) as i32) as u32,
        WordOp::IntToBool => u32::from(x != 0),
        WordOp::FloatToBool => u32::from(f32::from_bits(a) != 0.0),
        WordOp::BoolToFloat => (if a == 0 { 0.0f32 } else { 1.0 }).to_bits(),
        _ => unreachable!(),
    }
}

fn dtype(kind: ValueType) -> DType {
    match kind {
        ValueType::F32 => DType::F32,
        ValueType::I32 => DType::I32,
        ValueType::Bool => DType::Bool(BoolStore::U32),
    }
}

fn check(tensor: &NativeTensor, expected: &[u32], expected_dtype: DType) -> Result<(), String> {
    let data = tensor.read().map_err(|e| e.to_string())?;
    let bytes: Vec<_> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
    if tensor.dtype() != expected_dtype || data.as_bytes() != bytes {
        return Err(format!("exact word mismatch for {expected_dtype:?}"));
    }
    Ok(())
}

fn resident(before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], after: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS]) -> Result<(), String> {
    if [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(String::from("typed math transferred or cleared storage"));
    }
    Ok(())
}

fn integers(backend: &Nvidia) -> Result<(), String> {
    let edges = [i32::MIN, i32::MIN + 1, -16_777_217, -1, 0, 1, 16_777_216, 16_777_217, i32::MAX - 1, i32::MAX];
    let values: Vec<i32> = (0..771)
        .map(|i| {
            if i % 3 == 0 {
                edges[(i / 3) % edges.len()]
            } else {
                (i as u32).wrapping_mul(0x9e37_79b9).rotate_left((i % 32) as u32) as i32
            }
        })
        .collect();
    let original = NativeTensor::from_data(TensorData::new(values.clone(), [3, 257])).map_err(|e| e.to_string())?;
    let input = original.clone().permute(&[1, 0]).and_then(|t| t.flip(&[0])).map_err(|e| e.to_string())?;
    let ordered: Vec<_> = (0..771).map(|i| values[(i % 3) * 257 + 256 - i / 3] as u32).collect();
    let factors = [i32::MAX, -31, 33];
    let rhs = NativeTensor::from_data(TensorData::new(factors.to_vec(), [3])).map_err(|e| e.to_string())?;
    let before = backend.service_profile().map_err(|e| e.to_string())?;
    let mut outputs = Vec::new();
    for op in WordOp::ALL.into_iter().filter(|op| op.input() == ValueType::I32 && !op.mask_fill()) {
        if op.binary() {
            outputs.push((op, false, input.word_binary(&rhs, op).map_err(|e| e.to_string())?));
            outputs.push((op, true, input.word_scalar(op, 0x8000_0001).map_err(|e| e.to_string())?));
        } else {
            outputs.push((op, false, input.word_unary(op).map_err(|e| e.to_string())?));
        }
    }
    let mask = input.word_scalar(WordOp::Greater, 0).map_err(|e| e.to_string())?;
    let masked = input.mask_fill_i32(&mask, -123_456_789).map_err(|e| e.to_string())?;
    resident(&before, &backend.service_profile().map_err(|e| e.to_string())?)?;
    for (op, scalar, output) in outputs {
        let expected: Vec<_> = ordered
            .iter()
            .enumerate()
            .map(|(i, &a)| reference(op, a, if scalar { 0x8000_0001 } else { factors[i % 3] as u32 }))
            .collect();
        check(&output, &expected, dtype(op.output())).map_err(|e| format!("{op:?} scalar={scalar}: {e}"))?;
        if output.shape().as_slice() != [257, 3] {
            return Err(String::from("integer output shape"));
        }
    }
    let expected: Vec<_> = ordered.iter().map(|&a| if (a as i32) > 0 { (-123_456_789i32) as u32 } else { a }).collect();
    check(&masked, &expected, DType::I32)?;
    for op in [WordOp::Divide, WordOp::Remainder] {
        for divisor in [0, u32::MAX, i32::MIN as u32, 1, 2, 7] {
            let result = input.word_scalar(op, divisor).map_err(|e| e.to_string())?;
            let expected: Vec<_> = ordered.iter().map(|&a| reference(op, a, divisor)).collect();
            check(&result, &expected, DType::I32)?;
        }
    }
    for op in [WordOp::ShiftLeft, WordOp::ShiftRight] {
        for count in [0, 31, 32, 63, 64, u32::MAX] {
            let result = input.word_scalar(op, count).map_err(|e| e.to_string())?;
            let expected: Vec<_> = ordered.iter().map(|&a| reference(op, a, count)).collect();
            check(&result, &expected, DType::I32)?;
        }
    }
    let small_mask = NativeTensor::from_data(TensorData::new(vec![false, true, false], [3])).map_err(|e| e.to_string())?;
    let expected: Vec<_> = ordered.iter().enumerate().map(|(i, &a)| if i % 3 == 1 { i32::MIN as u32 } else { a }).collect();
    check(&input.mask_fill_i32(&small_mask, i32::MIN).map_err(|e| e.to_string())?, &expected, DType::I32)?;
    check(&original, &values.iter().map(|&v| v as u32).collect::<Vec<_>>(), DType::I32)?;
    Ok(())
}

fn booleans(backend: &Nvidia) -> Result<(), String> {
    let values: Vec<bool> = (0..771).map(|i| i % 5 < 2).collect();
    let original = NativeTensor::from_data(TensorData::new(values.clone(), [3, 257])).map_err(|e| e.to_string())?;
    let input = original.clone().permute(&[1, 0]).and_then(|t| t.flip(&[0])).map_err(|e| e.to_string())?;
    let ordered: Vec<_> = (0..771).map(|i| u32::from(values[(i % 3) * 257 + 256 - i / 3])).collect();
    let factors = [false, true, true];
    let rhs = NativeTensor::from_data(TensorData::new(factors.to_vec(), [3])).map_err(|e| e.to_string())?;
    let before = backend.service_profile().map_err(|e| e.to_string())?;
    let mut outputs = Vec::new();
    for op in WordOp::ALL.into_iter().filter(|op| op.input() == ValueType::Bool && !op.mask_fill()) {
        if op.binary() {
            outputs.push((op, false, input.word_binary(&rhs, op).map_err(|e| e.to_string())?));
            outputs.push((op, true, input.word_scalar(op, 1).map_err(|e| e.to_string())?));
        } else {
            outputs.push((op, false, input.word_unary(op).map_err(|e| e.to_string())?));
        }
    }
    let masked = input.mask_fill_bool(&rhs, true).map_err(|e| e.to_string())?;
    resident(&before, &backend.service_profile().map_err(|e| e.to_string())?)?;
    for (op, scalar, output) in outputs {
        let expected: Vec<_> = ordered
            .iter()
            .enumerate()
            .map(|(i, &a)| reference(op, a, if scalar { 1 } else { u32::from(factors[i % 3]) }))
            .collect();
        check(&output, &expected, dtype(op.output())).map_err(|e| format!("{op:?}: {e}"))?;
    }
    let expected: Vec<_> = ordered.iter().enumerate().map(|(i, &a)| if factors[i % 3] { 1 } else { a }).collect();
    check(&masked, &expected, DType::Bool(BoolStore::U32))?;
    let before = backend.service_profile().map_err(|e| e.to_string())?;
    let integers = original.cast(DType::I32).map_err(|e| e.to_string())?;
    let after = backend.service_profile().map_err(|e| e.to_string())?;
    resident(&before, &after)?;
    if after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1 || original.can_mut() || !integers.can_mut() {
        return Err(String::from("boolean-to-int cast lost resident full-writer or ownership guarantees"));
    }
    check(&integers, &values.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(), DType::I32)?;
    check(
        &original,
        &values.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(),
        DType::Bool(BoolStore::U32),
    )?;
    drop(input);
    drop(original);
    if !integers.can_mut() {
        return Err(String::from("cast retained a lost or aliased owner"));
    }
    check(&integers, &values.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(), DType::I32)?;
    Ok(())
}

fn conversions(backend: &Nvidia) -> Result<(), String> {
    let mut values = Vec::new();
    for sign in [0, 0x8000_0000] {
        for exponent in 0..256 {
            for fraction in [0, 1, 0x3f_ffff, 0x7f_ffff] {
                values.push(f32::from_bits(sign | exponent << 23 | fraction));
            }
        }
    }
    for center in [-2147483648.0f32, 2147483648.0f32] {
        for delta in -3i64..=3 {
            values.push(f32::from_bits((i64::from(center.to_bits()) + delta) as u32));
        }
    }
    let input = NativeTensor::from_data(TensorData::new(values.clone(), [values.len()]))
        .and_then(|t| t.flip(&[0]))
        .map_err(|e| e.to_string())?;
    let before = backend.service_profile().map_err(|e| e.to_string())?;
    let ints = input.cast(DType::I32).map_err(|e| e.to_string())?;
    let bools = input.cast(DType::Bool(BoolStore::U32)).map_err(|e| e.to_string())?;
    let floats = ints.cast(DType::F32).map_err(|e| e.to_string())?;
    let roundtrip = floats.cast(DType::I32).map_err(|e| e.to_string())?;
    resident(&before, &backend.service_profile().map_err(|e| e.to_string())?)?;
    check(&ints, &values.iter().rev().map(|&v| (v as i32) as u32).collect::<Vec<_>>(), DType::I32)?;
    check(
        &bools,
        &values.iter().rev().map(|&v| u32::from(v != 0.0)).collect::<Vec<_>>(),
        DType::Bool(BoolStore::U32),
    )?;
    check(
        &roundtrip,
        &values.iter().rev().map(|&v| ((v as i32) as f32 as i32) as u32).collect::<Vec<_>>(),
        DType::I32,
    )?;
    Ok(())
}

fn guards(backend: &Nvidia) -> Result<(), Error> {
    let count = 257;
    let values: Vec<u32> = (0..count).map(|i| (i as u32).wrapping_mul(0x9e37_79b9)).collect();
    let mut words = vec![0x6bad_beef];
    words.extend(&values);
    words.push(0x6bad_beef);
    let input = backend.upload_words(&words)?;
    let output = backend.upload_words(&vec![0x6bad_beef; count + 2])?;
    let layout = Layout::contiguous(Shape::new([count]))?.flip(&[0])?;
    let mut actual = vec![0; count + 2];
    for op in [WordOp::Add, WordOp::Divide, WordOp::Remainder, WordOp::Power] {
        let parameters = layout.word_parameters(&layout, op, (-7i32) as u32, true)?;
        backend.dispatch_words(abi::Kernel::TensorWords, &[(&input, 1), (&input, 1), (&output, 1)], &parameters)?;
        let mut expected = vec![0x6bad_beef];
        expected.extend(values.iter().rev().map(|&v| reference(op, v, (-7i32) as u32)));
        expected.push(0x6bad_beef);
        backend.read_words_into(&output, 0, &mut actual)?;
        if actual != expected {
            return Err(Error::DeviceFault);
        }
        let mut invalid = parameters;
        invalid[4] += count;
        if backend.dispatch_words(abi::Kernel::TensorWords, &[(&input, 1), (&input, 1), (&output, 1)], &invalid) != Err(Error::InvalidParameter) {
            return Err(Error::DeviceFault);
        }
        backend.read_words_into(&output, 0, &mut actual)?;
        if actual != expected {
            return Err(Error::DeviceFault);
        }
    }
    backend.read_words_into(&input, 0, &mut actual)?;
    if actual != words {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn power_contract(backend: &Nvidia) -> Result<(), String> {
    let raw = || -> Result<(), Error> {
        let guard = 0x6bad_beef;
        for count in [1, 127, 128, 129, 257] {
            let mut a = vec![guard];
            a.extend((0..count).map(|i| (i as u32).wrapping_mul(0x9e37_79b9)));
            a.push(guard);
            let mut b = vec![guard];
            b.extend((0..count).map(|i| [0, 1, 2, 31, 32, 33, i32::MAX as u32, i32::MIN as u32, u32::MAX][i % 9]));
            b.push(guard);
            let left = backend.upload_words(&a)?;
            let right = backend.upload_words(&b)?;
            let output = backend.upload_words(&vec![guard; count + 2])?;
            let layout = Layout::contiguous(Shape::new([count]))?.flip(&[0])?;
            for scalar in [false, true] {
                let parameters = layout.word_parameters(&layout, WordOp::Power, if scalar { u32::MAX } else { 0 }, scalar)?;
                backend.dispatch_words(abi::Kernel::TensorWords, &[(&left, 1), (&right, 1), (&output, 1)], &parameters)?;
                let mut actual = vec![0; count + 2];
                backend.read_words_into(&output, 0, &mut actual)?;
                let expected: Vec<_> = (1..=count)
                    .rev()
                    .map(|i| reference(WordOp::Power, a[i], if scalar { u32::MAX } else { b[i] }))
                    .collect();
                if actual[0] != guard || actual[count + 1] != guard || actual[1..=count] != expected {
                    return Err(Error::DeviceFault);
                }
                let before = backend.queue_profile()?;
                if backend.dispatch_words(abi::Kernel::TensorWords, &[(&left, 1), (&right, 1), (&left, 1)], &parameters) != Err(Error::InvalidParameter)
                    || backend.queue_profile()?.launches != before.launches
                {
                    return Err(Error::DeviceFault);
                }
            }
            for (buffer, expected) in [(&left, &a), (&right, &b)] {
                let mut actual = vec![0; count + 2];
                backend.read_words_into(buffer, 0, &mut actual)?;
                if &actual != expected {
                    return Err(Error::DeviceFault);
                }
            }
        }
        Ok(())
    };
    raw().map_err(|e| e.to_string())?;
    let values: Vec<_> = (0i32..129).map(|i| i.wrapping_mul(16_777_217)).collect();
    let source = NativeTensor::from_data(TensorData::new(values.clone(), [129])).map_err(|e| e.to_string())?;
    let powers = NativeTensor::from_data(TensorData::new(vec![-1i32], [1])).map_err(|e| e.to_string())?;
    let before = backend.service_profile().map_err(|e| e.to_string())?;
    let output = backend
        .with_commands(|| {
            let output = source.word_binary(&powers, WordOp::Power)?;
            drop(source);
            drop(powers);
            Ok::<_, Error>(output)
        })
        .map_err(|e| e.to_string())?;
    let after = backend.service_profile().map_err(|e| e.to_string())?;
    resident(&before, &after)?;
    if after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1 {
        return Err(String::from("integer power did not use a single full writer"));
    }
    check(
        &output,
        &values.iter().map(|&v| reference(WordOp::Power, v as u32, u32::MAX)).collect::<Vec<_>>(),
        DType::I32,
    )?;
    let scalar = NativeTensor::from_data(TensorData::new(vec![16_777_217i32], [1]))
        .and_then(|v| v.reshape(Shape::new([])))
        .map_err(|e| e.to_string())?;
    let output = scalar.word_scalar(WordOp::Power, 3).map_err(|e| e.to_string())?;
    if !output.shape().as_slice().is_empty() {
        return Err(String::from("integer power lost scalar rank"));
    }
    check(&output, &[50_331_649], DType::I32)?;
    let empty = NativeTensor::full(Shape::new([0, 3]), DType::I32, 0).map_err(|e| e.to_string())?;
    let wrong_type = NativeTensor::full(Shape::new([0, 3]), DType::F32, 0).map_err(|e| e.to_string())?;
    let wrong_shape = NativeTensor::full(Shape::new([2, 2]), DType::I32, 0).map_err(|e| e.to_string())?;
    let before = backend.queue_profile().map_err(|e| e.to_string())?;
    if !empty.word_scalar(WordOp::Power, u32::MAX).map_err(|e| e.to_string())?.layout().is_empty()
        || !empty.word_binary(&empty, WordOp::Power).map_err(|e| e.to_string())?.layout().is_empty()
        || !matches!(empty.word_binary(&wrong_type, WordOp::Power), Err(Error::InvalidParameter))
        || !matches!(empty.word_binary(&wrong_shape, WordOp::Power), Err(Error::ShapeMismatch))
        || backend.queue_profile().map_err(|e| e.to_string())?.launches != before.launches
    {
        return Err(String::from("integer power invalid or empty work launched"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_INTEGER_POWER_OK raw=10 guards=ok reference=i128 full_writer=verified resident=true queued_lifetime=verified scalar=true invalid_inputs=no_launch empty=no_launch"
    ));
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    power_contract(backend)?;
    integers(backend)?;
    booleans(backend)?;
    conversions(backend)?;
    guards(backend).map_err(|e| e.to_string())?;
    let empty = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [0, 3])).map_err(|e| e.to_string())?;
    let before = backend.queue_profile().map_err(|e| e.to_string())?;
    let empty_float = empty.cast(DType::F32).map_err(|e| e.to_string())?;
    let empty_bool = empty.cast(DType::Bool(BoolStore::U32)).map_err(|e| e.to_string())?;
    if !empty_float.layout().is_empty()
        || !empty_bool.layout().is_empty()
        || !matches!(empty.word_scalar(WordOp::BoolAnd, 2), Err(Error::InvalidParameter))
        || !matches!(empty.word_binary(&empty, WordOp::BoolAnd), Err(Error::InvalidParameter))
        || before.launches != backend.queue_profile().map_err(|e| e.to_string())?.launches
    {
        return Err(String::from("empty or ill-typed word operation"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_WORDS_OK operations=32 integers=exact booleans=canonical casts=ok aliasing=ok guards=ok resident=ok backend=nvidia"
    ));
    Ok(())
}
