//! Byte references for full writers, initialized field updates, and indexed OR.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Scalar, Shape, Slice, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{
    Backend as _, Error,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{
    self as abi, Kernel,
    elementwise::Elementwise,
    packed_updates::{MASK_U8, REMAINDER_SHIFT, SCALAR},
    storage::Width,
};

const GUARD: u32 = 0x6bad_beef;
const BOUNDARIES: [usize; 5] = [1, 127, 128, 129, 257];

fn require(condition: bool) -> Result<(), Error> {
    if condition { Ok(()) } else { Err(Error::DeviceFault) }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut bytes = [0; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect()
}

fn pattern(index: usize) -> u64 {
    match index % 8 {
        0 => 0x7ff8_1234_89ab_cdef,
        1 => 0xfff0_4567_7654_3210,
        2 => 0x8000_0000_0000_0000,
        3 => 0x0000_0001_8000_0001,
        4 => 0xffff_ffff_0000_0000,
        _ => (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
    }
}

fn bytes(count: usize, size: usize, seed: usize) -> Vec<u8> {
    (0..count).flat_map(|i| pattern(seed + i).to_le_bytes()[..size].to_vec()).collect()
}

fn append_value(output: &mut Vec<u8>, source: &[u8], index: usize, size: usize) {
    output.extend_from_slice(&source[index * size..(index + 1) * size]);
}

fn guarded(bytes: &[u8]) -> Vec<u32> {
    let mut result = vec![GUARD];
    result.extend(words(bytes));
    result.push(GUARD);
    result
}

fn read_matches(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32]) -> Result<(), Error> {
    let mut actual = vec![0; expected.len()];
    backend.read_words_into(buffer, 0, &mut actual)?;
    if let Some((index, (&actual_word, &expected_word))) = actual.iter().zip(expected).enumerate().find(|(_, (a, b))| a != b) {
        crate::report(format_args!(
            "PACKED_MOVEMENT_WORD_MISMATCH index={index} actual={actual_word:#010x} expected={expected_word:#010x} words={}",
            expected.len()
        ));
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn services(backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], reads: u64, fresh: Option<u64>) -> Result<(), Error> {
    let after = backend.service_profile()?;
    require(
        after[abi::UPLOAD as usize].calls == before[abi::UPLOAD as usize].calls
            && after[abi::DOWNLOAD as usize].calls - before[abi::DOWNLOAD as usize].calls == reads,
    )?;
    if let Some(fresh) = fresh {
        require(
            after[abi::ALLOCATE as usize].calls == before[abi::ALLOCATE as usize].calls
                && after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls == fresh,
        )?;
    }
    Ok(())
}

fn raw_gather(backend: &Nvidia, width: Width, count: usize) -> Result<usize, Error> {
    let size = width.bytes() as usize;
    let raw = bytes(23, size, 0);
    let source_words = guarded(&raw);
    let source = backend.upload_words(&source_words)?;
    let mut ids = vec![GUARD; 6 * count + 1];
    for row in 0..3 {
        for col in 0..count {
            ids[2 * (row * count + col) + 1] = ((row + col * 3) % 5) as u32;
        }
    }
    let index_words = guarded(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let indices = backend.upload_words(&index_words)?;
    let status = backend.upload_words(&[GUARD, 0, GUARD])?;
    let a = Layout::from_parts(Shape::new([3, 5]), vec![-7, -1], 20, 23)?;
    let b = Layout::from_parts(Shape::new([3, count]), vec![(2 * count) as i64, -2], (2 * count - 1) as i64, ids.len())?;
    let mut p = a.gather_parameters(&b, 1)?;
    p[0] |= (width as usize) << 8;
    let mut expected_bytes = Vec::new();
    for row in 0..3 {
        for col in 0..count {
            let selected = ids[2 * (row * count + count - 1 - col) + 1] as usize;
            append_value(&mut expected_bytes, &raw, 20 - row * 7 - selected, size);
        }
    }
    let expected = guarded(&expected_bytes);
    let output = backend.upload_words(&vec![GUARD; expected.len()])?;
    let before = backend.service_profile()?;
    backend.with_commands(|| backend.dispatch_words(Kernel::PackedGather, &[(&source, 1), (&indices, 1), (&status, 1), (&output, 1)], &p))?;
    services(backend, &before, 0, Some(0))?;
    read_matches(backend, &output, &expected)?;
    read_matches(backend, &status, &[GUARD, 0, GUARD])?;
    let before = backend.service_profile()?;
    let fresh = backend.with_commands(|| backend.produce_words(expected.len() - 2, Kernel::PackedGather, &[(&source, 1), (&indices, 1), (&status, 1)], &p))?;
    services(backend, &before, 0, Some(1))?;
    read_matches(backend, &fresh, &expected[1..expected.len() - 1])?;

    // Invalid device values initialize their complete output fields to zero.
    ids[2 * count - 1] = u32::MAX;
    ids[4 * count - 1] = 5;
    backend.write_words(&indices, 1, &ids)?;
    let mut invalid_expected = expected_bytes;
    invalid_expected[..size].fill(0);
    invalid_expected[count * size..(count + 1) * size].fill(0);
    backend.dispatch_words(Kernel::PackedGather, &[(&source, 1), (&indices, 1), (&status, 1), (&output, 1)], &p)?;
    read_matches(backend, &output, &guarded(&invalid_expected))?;
    read_matches(backend, &status, &[GUARD, 1, GUARD])?;
    let mut mutated_indices = index_words;
    mutated_indices[1..1 + ids.len()].copy_from_slice(&ids);
    read_matches(backend, &indices, &mutated_indices)?;
    read_matches(backend, &source, &source_words)?;
    let launches = backend.queue_profile()?.launches;
    let mut invalid = p;
    invalid[0] |= 1 << 12;
    require(
        backend.dispatch_words(Kernel::PackedGather, &[(&source, 1), (&indices, 1), (&status, 1), (&output, 1)], &invalid) == Err(Error::InvalidParameter),
    )?;
    require(backend.dispatch_words(Kernel::PackedGather, &[(&source, 1), (&indices, 1), (&status, 1), (&indices, 1)], &p) == Err(Error::InvalidParameter))?;
    require(backend.dispatch_words(Kernel::PackedGather, &[(&source, 1), (&indices, 1), (&output, 1), (&output, 1)], &p) == Err(Error::InvalidParameter))?;
    require(backend.queue_profile()?.launches == launches)?;
    read_matches(backend, &output, &guarded(&invalid_expected))?;
    Ok(3)
}

fn raw_mask(backend: &Nvidia, width: Width, count: usize, narrow: bool, remainder: usize, scalar: bool) -> Result<usize, Error> {
    let size = width.bytes() as usize;
    let raw = bytes(3 * count + 3, size, 9);
    let rhs_raw = bytes(3 * count + 3, size, 25);
    let source_words = guarded(&raw);
    let right_words = guarded(&rhs_raw);
    let left = backend.upload_words(&source_words)?;
    let right = backend.upload_words(&right_words)?;
    let a = Layout::from_parts(Shape::new([3, count]), vec![count as i64, -1], count as i64, 3 * count + 3)?;
    let b = Layout::from_parts(Shape::new([3, count]), vec![-1, 3], 3, 3 * count + 3)?;
    let mut p = a.elementwise_parameters(&b, Elementwise::Copy, 0, false)?;
    p[2] = width as usize | if narrow { MASK_U8 as usize } else { 0 } | remainder << REMAINDER_SHIFT;
    let scalar_bits = pattern(0) & if size == 8 { u64::MAX } else { (1u64 << (8 * size)) - 1 };
    if scalar {
        p[2] |= SCALAR as usize;
        p[3] = scalar_bits as u32 as usize;
        p[5] = (scalar_bits >> 32) as usize;
        p[7] = 0;
        p[24..32].fill(0);
    }
    let mask_size = if narrow { 1 } else { 4 };
    let truth: Vec<_> = (0..3 * count).map(|i| if i % 5 < 2 { 0 } else { [1, 0x80, 0xff][i % 3] }).collect();
    let mut mask_bytes = vec![0xa5; remainder * mask_size];
    mask_bytes.extend(truth.iter().flat_map(|&v| (v as u32).to_le_bytes()[..mask_size].to_vec()));
    mask_bytes.resize(mask_bytes.len().div_ceil(4) * 4, 0xa5);
    let mask_words = guarded(&mask_bytes);
    let mask = backend.upload_words(&mask_words)?;
    let mut expected_bytes = Vec::new();
    for row in 0..3 {
        for col in 0..count {
            if truth[row * count + col] != 0 {
                if scalar {
                    expected_bytes.extend_from_slice(&scalar_bits.to_le_bytes()[..size]);
                } else {
                    append_value(&mut expected_bytes, &rhs_raw, 3 - row + 3 * col, size);
                }
            } else {
                append_value(&mut expected_bytes, &raw, count + row * count - col, size);
            }
        }
    }
    let expected = guarded(&expected_bytes);
    let output = backend.upload_words(&vec![GUARD; expected.len()])?;
    let before = backend.service_profile()?;
    backend.with_commands(|| backend.dispatch_words(Kernel::PackedMaskWhere, &[(&left, 1), (&right, 1), (&mask, 1), (&output, 1)], &p))?;
    services(backend, &before, 0, Some(0))?;
    read_matches(backend, &output, &expected)?;
    let before = backend.service_profile()?;
    let fresh = backend.with_commands(|| backend.produce_words(expected.len() - 2, Kernel::PackedMaskWhere, &[(&left, 1), (&right, 1), (&mask, 1)], &p))?;
    services(backend, &before, 0, Some(1))?;
    read_matches(backend, &fresh, &expected[1..expected.len() - 1])?;
    let launches = backend.queue_profile()?.launches;
    let mut invalid = p;
    invalid[2] |= 1 << 12;
    require(backend.dispatch_words(Kernel::PackedMaskWhere, &[(&left, 1), (&right, 1), (&mask, 1), (&output, 1)], &invalid) == Err(Error::InvalidParameter))?;
    require(backend.dispatch_words(Kernel::PackedMaskWhere, &[(&left, 1), (&right, 1), (&mask, 1), (&left, 1)], &p) == Err(Error::InvalidParameter))?;
    require(
        backend.dispatch_words(Kernel::PackedMaskWhere, &[(&left, 1), (&right, 1), (&mask, mask_words.len()), (&output, 1)], &p) == Err(Error::ShapeMismatch),
    )?;
    require(backend.queue_profile()?.launches == launches)?;
    read_matches(backend, &output, &expected)?;
    for (buffer, expected) in [(&left, &source_words), (&right, &right_words), (&mask, &mask_words)] {
        read_matches(backend, buffer, expected)?;
    }
    Ok(2)
}

fn raw_assign(backend: &Nvidia, width: Width, count: usize, step: usize) -> Result<usize, Error> {
    let size = width.bytes() as usize;
    let extent = 3 * count * step + 3;
    let raw = bytes(3 * count + 3, size, 15);
    let source_words = guarded(&raw);
    let source = backend.upload_words(&source_words)?;
    let mut expected_bytes = bytes(extent, size, 30);
    // Nonzero physical padding belongs to initialized storage and must survive.
    expected_bytes.resize(expected_bytes.len().div_ceil(4) * 4, 0xa5);
    let initial = guarded(&expected_bytes);
    let target = backend.upload_words(&initial)?;
    let a = Layout::from_parts(
        Shape::new([3, count]),
        vec![(count * step) as i64, -(step as i64)],
        (count * step) as i64,
        extent,
    )?;
    let b = Layout::from_parts(Shape::new([3, count]), vec![-1, 3], 3, 3 * count + 3)?;
    let mut p = a.assignment_parameters(&b)?;
    p[2] = width as usize;
    for row in 0..3 {
        for col in 0..count {
            let destination = (count + row * count - col) * step;
            let source = 3 - row + 3 * col;
            expected_bytes[destination * size..(destination + 1) * size].copy_from_slice(&raw[source * size..(source + 1) * size]);
        }
    }
    // Repeat from the same initialized state to expose CAS contention, without
    // ever treating this partial writer as an uninitialized-output producer.
    for _ in 0..3 {
        backend.write_words(&target, 0, &initial)?;
        let before = backend.service_profile()?;
        backend.with_commands(|| backend.dispatch_words(Kernel::PackedAssign, &[(&target, 1), (&source, 1)], &p))?;
        services(backend, &before, 0, Some(0))?;
        read_matches(backend, &target, &guarded(&expected_bytes))?;
    }
    let launches = backend.queue_profile()?.launches;
    let mut invalid = p;
    invalid[16] = 0;
    require(backend.dispatch_words(Kernel::PackedAssign, &[(&target, 1), (&source, 1)], &invalid) == Err(Error::InvalidParameter))?;
    require(backend.dispatch_words(Kernel::PackedAssign, &[(&target, 1), (&target, 1)], &p) == Err(Error::InvalidParameter))?;
    require(backend.dispatch_words(Kernel::PackedAssign, &[(&target, 1), (&source, source_words.len())], &p) == Err(Error::ShapeMismatch))?;
    require(backend.queue_profile()?.launches == launches)?;
    read_matches(backend, &target, &guarded(&expected_bytes))?;
    read_matches(backend, &source, &source_words)?;
    Ok(3)
}

fn raw_bool_scatter(backend: &Nvidia, count: usize, width: Width, remainder: usize) -> Result<usize, Error> {
    let size = width.bytes() as usize;
    let mut p = [0usize; 32];
    p[..8].copy_from_slice(&[
        8 | (width as usize) << 8 | remainder << REMAINDER_SHIFT,
        4,
        2 * count,
        14,
        6,
        2 * count,
        count - 1,
        7,
    ]);
    p[8..16].copy_from_slice(&[1, 2, 1, 1, count, 1, 1, 1]);
    p[17] = 7;
    p[20] = (-1i32) as u32 as usize;
    p[25] = count;
    p[28] = (-1i32) as u32 as usize;
    let mut values_bytes = vec![0xa5; remainder * size];
    let truth: Vec<u32> = (0..2 * count).map(|i| if i % 5 < 2 { 0 } else { 0x80 }).collect();
    values_bytes.extend(truth.iter().flat_map(|v| v.to_le_bytes()[..size].to_vec()));
    let source_words = guarded(&values_bytes);
    let source = backend.upload_words(&source_words)?;
    let mut ids: Vec<u32> = (0..2 * count).map(|i| (i % 7) as u32).collect();
    let indices = backend.upload_words(&guarded(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()))?;
    let initial_bytes: Vec<_> = (0..14).flat_map(|i| u32::from(i == 3 || i == 12).to_le_bytes()[..size].to_vec()).collect();
    let mut initial_bytes = initial_bytes;
    initial_bytes.resize(initial_bytes.len().div_ceil(4) * 4, 0xa5);
    let initial = guarded(&initial_bytes);
    let output = backend.upload_words(&initial)?;
    let status = backend.upload_words(&[GUARD, 0, GUARD])?;
    for invalid in [false, true] {
        if invalid {
            ids[count - 1] = u32::MAX;
            ids[2 * count - 1] = 7;
            backend.write_words(&indices, 1, &ids)?;
        }
        backend.write_words(&output, 0, &initial)?;
        backend.write_words(&status, 1, &[0])?;
        let mut expected_bytes = initial_bytes.clone();
        for row in 0..2 {
            for col in 0..count {
                let selected = ids[row * count + count - 1 - col];
                if selected < 7 && truth[row * count + col] != 0 {
                    expected_bytes[(row * 7 + 6 - selected as usize) * size] = 1;
                }
            }
        }
        let before = backend.service_profile()?;
        backend.with_commands(|| backend.dispatch_words(Kernel::PackedBoolScatter, &[(&source, 1), (&indices, 1), (&status, 1), (&output, 1)], &p))?;
        services(backend, &before, 0, Some(0))?;
        read_matches(backend, &output, &guarded(&expected_bytes))?;
        read_matches(backend, &status, &[GUARD, u32::from(invalid), GUARD])?;
    }
    let alias = backend.upload_words(&vec![GUARD; width.words((2 * count + remainder).max(14) as u32) as usize + 2])?;
    let launches = backend.queue_profile()?.launches;
    let mut invalid = p;
    invalid[0] |= 1 << 12;
    require(
        backend.dispatch_words(Kernel::PackedBoolScatter, &[(&source, 1), (&indices, 1), (&status, 1), (&output, 1)], &invalid) == Err(Error::InvalidParameter),
    )?;
    require(backend.dispatch_words(Kernel::PackedBoolScatter, &[(&alias, 1), (&indices, 1), (&status, 1), (&alias, 1)], &p) == Err(Error::InvalidParameter))?;
    require(backend.dispatch_words(Kernel::PackedBoolScatter, &[(&source, 1), (&indices, 1), (&output, 1), (&output, 1)], &p) == Err(Error::InvalidParameter))?;
    require(backend.queue_profile()?.launches == launches)?;
    read_matches(backend, &source, &source_words)?;
    read_matches(backend, &indices, &guarded(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()))?;
    Ok(2)
}

fn upload(raw: &[u8], shape: &[usize], dtype: DType) -> Result<NativeTensor, Error> {
    NativeTensor::from_data(TensorData::from_bytes_vec(raw.to_vec(), Shape::from(shape.to_vec()), dtype))
}

fn native_matches(tensor: &NativeTensor, shape: &[usize], dtype: DType, expected: &[u8]) -> Result<(), Error> {
    if tensor.shape().as_slice() != shape || tensor.dtype() != dtype {
        crate::report(format_args!(
            "PACKED_MOVEMENT_METADATA_MISMATCH actual_shape={:?} expected_shape={shape:?} actual_dtype={:?} expected_dtype={dtype:?}",
            tensor.shape(),
            tensor.dtype()
        ));
        return Err(Error::DeviceFault);
    }
    let data = tensor.read()?;
    let actual = data.as_bytes();
    if let Some((index, (&actual, &expected))) = actual.iter().zip(expected).enumerate().find(|(_, (a, b))| a != b) {
        crate::report(format_args!(
            "PACKED_MOVEMENT_BYTE_MISMATCH dtype={dtype:?} shape={shape:?} index={index} actual={actual:#04x} expected={expected:#04x}"
        ));
        return Err(Error::DeviceFault);
    }
    require(actual.len() == expected.len())
}

fn native_bytes(count: usize, dtype: DType, seed: usize) -> Vec<u8> {
    if dtype.is_bool() {
        (0..count)
            .flat_map(|i| u32::from((i + seed) % 5 < 2).to_le_bytes()[..dtype.size()].to_vec())
            .collect()
    } else {
        bytes(count, dtype.size(), seed)
    }
}

fn scalar(dtype: DType) -> (Scalar, u64) {
    match dtype {
        DType::F64 => ((-2f64).into(), (-2f64).to_bits()),
        DType::F32 | DType::Flex32 => ((-2f32).into(), u64::from((-2f32).to_bits())),
        DType::F16 | DType::BF16 => ((-2f32).into(), 0xc000),
        DType::I64 => ((i64::MIN + 0x1234_5678).into(), (i64::MIN + 0x1234_5678) as u64),
        DType::U64 => (0xfedc_ba98_7654_3210u64.into(), 0xfedc_ba98_7654_3210),
        DType::I32 | DType::I16 | DType::I8 => ((-2i32).into(), (-2i64) as u64),
        DType::Bool(_) => (true.into(), 1),
        _ => (254u32.into(), 254),
    }
}

fn native_movement(backend: &Nvidia, dtype: DType) -> Result<usize, Error> {
    let count = 129;
    let size = dtype.size();
    let raw = native_bytes(3 * (count + 2), dtype, 0);
    let original = upload(&raw, &[3, count + 2], dtype)?;
    let input = original
        .clone()
        .slice(&[Slice::full(), Slice::new(1, Some((count + 1) as isize), 1)])?
        .permute(&[1, 0])?
        .flip(&[0, 1])?;
    let mut ordered = Vec::new();
    for row in 0..count {
        for col in 0..3 {
            append_value(&mut ordered, &raw, (2 - col) * (count + 2) + count - row, size);
        }
    }
    let rhs_raw = native_bytes(3, dtype, 17);
    let rhs = upload(&rhs_raw, &[1, 3], dtype)?;
    for store in [BoolStore::U8, BoolStore::U32] {
        let mask_dtype = DType::Bool(store);
        // Offset seven exercises both the word base and its byte remainder.
        let truth: Vec<_> = (0..3 * count).map(|i| i % 7 < 3).collect();
        let mut mask_bytes = vec![0; 7 * mask_dtype.size()];
        mask_bytes.extend(truth.iter().flat_map(|&v| u32::from(v).to_le_bytes()[..mask_dtype.size()].to_vec()));
        mask_bytes.extend(vec![1; mask_dtype.size()]);
        let mask_original = upload(&mask_bytes, &[3 * count + 8], mask_dtype)?;
        let mask = mask_original
            .clone()
            .slice(&[Slice::new(7, Some((3 * count + 7) as isize), 1)])?
            .reshape(Shape::new([count, 3]))?;
        let before = backend.service_profile()?;
        let kernels_before = backend.profile()?;
        let selected = input.mask_where(&mask, &rhs)?;
        services(backend, &before, 0, Some(1))?;
        let kernels_after = backend.profile()?;
        let kernel = if size == 4 && store == BoolStore::U32 {
            Kernel::MaskWhere
        } else {
            Kernel::PackedMaskWhere
        };
        require(kernels_after[kernel as usize].calls - kernels_before[kernel as usize].calls == 1)?;
        let mut expected = ordered.clone();
        for i in 0..3 * count {
            if truth[i] {
                expected[i * size..(i + 1) * size].copy_from_slice(&rhs_raw[i % 3 * size..(i % 3 + 1) * size]);
            }
        }
        native_matches(&selected, &[count, 3], dtype, &expected)?;
        let (value, bits) = scalar(dtype);
        let before = backend.service_profile()?;
        let kernels_before = backend.profile()?;
        let filled = input.mask_fill_scalar(&mask, value)?;
        services(backend, &before, 0, Some(1))?;
        let kernel = match (dtype, store) {
            (DType::F32, BoolStore::U32) => Kernel::TensorElementwise,
            (DType::I32 | DType::Bool(BoolStore::U32), BoolStore::U32) => Kernel::TensorWords,
            _ => Kernel::PackedMaskWhere,
        };
        require(backend.profile()?[kernel as usize].calls - kernels_before[kernel as usize].calls == 1)?;
        for i in 0..3 * count {
            if truth[i] {
                expected[i * size..(i + 1) * size].copy_from_slice(&bits.to_le_bytes()[..size]);
            }
        }
        native_matches(&filled, &[count, 3], dtype, &expected)?;
        let reversed_mask = mask.clone().flip(&[0, 1])?;
        let before = backend.service_profile()?;
        let reversed = input.mask_where(&reversed_mask, &rhs)?;
        services(backend, &before, 0, Some(2))?;
        let mut expected = ordered.clone();
        for i in 0..3 * count {
            if truth[3 * count - 1 - i] {
                expected[i * size..(i + 1) * size].copy_from_slice(&rhs_raw[i % 3 * size..(i % 3 + 1) * size]);
            }
        }
        native_matches(&reversed, &[count, 3], dtype, &expected)?;
        native_matches(&mask_original, &[3 * count + 8], mask_dtype, &mask_bytes)?;
    }

    let ids: Vec<i32> = (0..3 * count).map(|i| (i % 3) as i32).collect();
    let ids_tensor = NativeTensor::from_data(TensorData::new(ids.clone(), [count, 3]))?.flip(&[0, 1])?;
    let selection = NativeTensor::from_data(TensorData::new(vec![2i32, 0, 2, 1], [4]))?.flip(&[0])?;
    let before = backend.service_profile()?;
    let kernels_before = backend.profile()?;
    let gathered = input.gather(1, &ids_tensor)?;
    let selected = input.select(1, &selection)?;
    services(backend, &before, 2, Some(4))?;
    let kernel = if size == 4 { Kernel::Gather } else { Kernel::PackedGather };
    require(backend.profile()?[kernel as usize].calls - kernels_before[kernel as usize].calls == 2)?;
    let mut expected = Vec::new();
    let mut expected_select = Vec::new();
    for row in 0..count {
        for col in 0..3 {
            append_value(&mut expected, &ordered, row * 3 + ids[(count - 1 - row) * 3 + 2 - col] as usize, size);
        }
        for col in [1, 2, 0, 2] {
            append_value(&mut expected_select, &ordered, row * 3 + col, size);
        }
    }
    native_matches(&gathered, &[count, 3], dtype, &expected)?;
    native_matches(&selected, &[count, 4], dtype, &expected_select)?;

    // The assignment source aliases the original allocation. Snapshot reads
    // must remain valid while the copied destination changes adjacent fields.
    let replacement = input.clone().slice(&[Slice::new(0, None, 2)])?.flip(&[0, 1])?;
    let before = backend.service_profile()?;
    let kernels_before = backend.profile()?;
    let assigned = input.slice_assign(&[Slice::new(0, None, 2)], &replacement)?;
    services(backend, &before, 0, Some(1))?;
    let kernel = if size == 4 { Kernel::ViewAssign } else { Kernel::PackedAssign };
    require(backend.profile()?[kernel as usize].calls - kernels_before[kernel as usize].calls == 1)?;
    let mut expected = ordered.clone();
    for row in (0..count).step_by(2) {
        for col in 0..3 {
            let source = (count - 1 - row) * 3 + 2 - col;
            let destination = row * 3 + col;
            expected[destination * size..(destination + 1) * size].copy_from_slice(&ordered[source * size..(source + 1) * size]);
        }
    }
    native_matches(&assigned, &[count, 3], dtype, &expected)?;
    let parts = [
        input.clone().slice(&[Slice::new(0, Some(64), 1)])?,
        input.clone().slice(&[Slice::new(64, Some(64), 1)])?,
        input.clone().slice(&[Slice::new(64, None, 1)])?,
    ];
    let before = backend.service_profile()?;
    let joined = NativeTensor::cat(&parts, 0)?;
    services(backend, &before, 0, None)?;
    let after = backend.service_profile()?;
    require(
        after[abi::ALLOCATE as usize].calls - before[abi::ALLOCATE as usize].calls == 1
            && after[abi::LAUNCH_ALLOCATE as usize].calls == before[abi::LAUNCH_ALLOCATE as usize].calls
            && after[abi::LAUNCH as usize].calls - before[abi::LAUNCH as usize].calls == 2,
    )?;
    native_matches(&joined, &[count, 3], dtype, &ordered)?;
    let (_, bits) = scalar(dtype);
    let before = backend.service_profile()?;
    let padded = input.pad(&[(1, 2), (1, 1)], bits)?;
    services(backend, &before, 0, Some(1))?;
    let mut expected = Vec::new();
    for row in 0..count + 3 {
        for col in 0..5 {
            if (1..count + 1).contains(&row) && (1..4).contains(&col) {
                append_value(&mut expected, &ordered, (row - 1) * 3 + col - 1, size);
            } else {
                expected.extend_from_slice(&bits.to_le_bytes()[..size]);
            }
        }
    }
    native_matches(&padded, &[count + 3, 5], dtype, &expected)?;
    native_matches(&input, &[count, 3], dtype, &ordered)?;
    native_matches(&original, &[3, count + 2], dtype, &raw)?;
    native_matches(&rhs, &[1, 3], dtype, &rhs_raw)?;
    Ok(11)
}

fn native_bool(backend: &Nvidia, count: usize, remainder: usize) -> Result<usize, Error> {
    let dtype = DType::Bool(BoolStore::U8);
    let shape = [1, 2, 1, 1, count, 1, 1, 1];
    let destination = upload(&[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], &[1, 2, 1, 1, 7, 1, 1, 1], dtype)?;
    let ids: Vec<i32> = (0..2 * count).map(|i| (i % 7) as i32).collect();
    let indices = NativeTensor::from_data(TensorData::new(ids.clone(), shape))?.flip(&[4])?;
    let truth: Vec<u8> = (0..2 * count).map(|i| u8::from(i % 5 < 2)).collect();
    let mut raw = vec![1; 4 + remainder];
    raw.extend_from_slice(&truth);
    raw.push(1);
    let original = upload(&raw, &[raw.len()], dtype)?;
    let values = original
        .clone()
        .slice(&[Slice::new((4 + remainder) as isize, Some((4 + remainder + truth.len()) as isize), 1)])?
        .reshape(Shape::new(shape))?;
    let before = backend.service_profile()?;
    let kernels_before = backend.profile()?;
    let output = backend.with_commands(|| {
        let output = destination.scatter_or(4, &indices, &values)?;
        drop(indices);
        drop(values);
        Ok::<_, Error>(output)
    })?;
    services(backend, &before, 1, Some(2))?;
    let after = backend.profile()?;
    require(after[Kernel::PackedBoolScatter as usize].calls - kernels_before[Kernel::PackedBoolScatter as usize].calls == 1)?;
    let mut expected = vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    for row in 0..2 {
        for col in 0..count {
            if truth[row * count + col] != 0 {
                expected[row * 7 + ids[row * count + count - 1 - col] as usize] = 1;
            }
        }
    }
    native_matches(&output, &[1, 2, 1, 1, 7, 1, 1, 1], dtype, &expected)?;
    native_matches(&original, &[raw.len()], dtype, &raw)?;
    let mask = original
        .clone()
        .slice(&[Slice::new((4 + remainder) as isize, Some((4 + remainder + truth.len()) as isize), 1)])?
        .reshape(Shape::new(shape))?
        .flip(&[1, 4])?;
    let before = backend.service_profile()?;
    let coordinates = backend.with_commands(|| {
        let coordinates = mask.argwhere()?;
        drop(mask);
        drop(original);
        Ok::<_, Error>(coordinates)
    })?;
    services(backend, &before, 1, Some(3))?;
    let mut expected = Vec::new();
    let mut selected = 0;
    for row in 0..2 {
        for col in 0..count {
            if truth[(1 - row) * count + count - 1 - col] != 0 {
                selected += 1;
                expected.extend([0i32, row as i32, 0, 0, col as i32, 0, 0, 0].iter().flat_map(|v| v.to_le_bytes()));
            }
        }
    }
    native_matches(&coordinates, &[selected, 8], DType::I32, &expected)?;
    Ok(2)
}

fn native_invalid(backend: &Nvidia, dtype: DType) -> Result<usize, Error> {
    let source = upload(&native_bytes(6, dtype, 0), &[2, 3], dtype)?;
    let empty = upload(&[], &[0, 3], dtype)?;
    let value = upload(&native_bytes(3, dtype, 10), &[1, 3], dtype)?;
    let mask = upload(&[1], &[1], DType::Bool(BoolStore::U8))?;
    let bad_mask = upload(&[1], &[1], DType::U8)?;
    let wrong_shape = upload(&native_bytes(8, dtype, 0), &[2, 4], dtype)?;
    let wrong_dtype = NativeTensor::from_data(TensorData::new(vec![0i32; 6], [2, 3]))?;
    let empty_ids = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [0, 3]))?;
    let before = backend.service_profile()?;
    let launches = backend.queue_profile()?.launches;
    let outputs = [
        empty.gather(1, &empty_ids)?,
        empty.mask_where(&mask, &value)?,
        empty.mask_fill_scalar(&mask, scalar(dtype).0)?,
        empty.slice_assign(&[], &value)?,
        NativeTensor::cat(&[empty.clone(), empty.clone()], 0)?,
        empty.pad(&[(0, 0), (0, 0)], 0)?,
    ];
    for result in [
        source.gather(2, &wrong_dtype),
        source.gather(1, &source),
        source.gather(1, &empty_ids),
        source.select(1, &wrong_dtype),
        source.mask_where(&bad_mask, &value),
        source.mask_where(&mask, &wrong_shape),
        source.mask_where(&mask, &wrong_dtype),
        source.slice_assign(&[], &wrong_shape),
        source.slice_assign(&[Slice { start: 0, end: None, step: 0 }], &value),
        NativeTensor::cat(&[], 0),
        NativeTensor::cat(&[source.clone(), wrong_shape], 0),
        source.pad(&[(0, 0)], 0),
    ] {
        require(result.is_err())?;
    }
    services(backend, &before, 0, Some(0))?;
    require(backend.queue_profile()?.launches == launches)?;
    for output in outputs {
        native_matches(&output, &[0, 3], dtype, &[])?;
    }
    // Runtime-invalid IDs are checked on-device and report only one word.
    for invalid in [-1i32, 3] {
        let indices = NativeTensor::from_data(TensorData::new(vec![invalid; 6], [2, 3]))?;
        let before = backend.service_profile()?;
        require(matches!(source.gather(1, &indices), Err(Error::InvalidParameter)))?;
        services(backend, &before, 1, Some(2))?;
    }
    native_matches(&source, &[2, 3], dtype, &native_bytes(6, dtype, 0))?;
    Ok(8)
}

fn queued_lifetime(backend: &Nvidia, width: Width) -> Result<usize, Error> {
    let size = width.bytes() as usize;
    let count = 257;
    let raw = bytes(count, size, 0);
    let source = backend.upload_words(&words(&raw))?;
    let ids = backend.upload_words(&(0..count as u32).rev().collect::<Vec<_>>())?;
    let mask = backend.upload_words(&vec![0x0100_0100; count.div_ceil(4)])?;
    let status = backend.upload_words(&[0])?;
    let assigned = backend.upload_words(&vec![GUARD; width.words(count as u32) as usize + 2])?;
    let layout = Layout::contiguous(Shape::new([count]))?;
    let mut gather = layout.gather_parameters(&layout, 0)?;
    gather[0] |= (width as usize) << 8;
    let mut selection = layout.elementwise_parameters(&layout, Elementwise::Copy, 0, false)?;
    selection[2] = width as usize | MASK_U8 as usize;
    let mut assignment = layout.assignment_parameters(&layout)?;
    assignment[2] = width as usize;
    let before = backend.service_profile()?;
    let (gathered, selected) = backend.with_commands(|| {
        let gathered = backend.produce_words(
            width.words(count as u32) as usize,
            Kernel::PackedGather,
            &[(&source, 0), (&ids, 0), (&status, 0)],
            &gather,
        )?;
        let selected = backend.produce_words(
            width.words(count as u32) as usize,
            Kernel::PackedMaskWhere,
            &[(&source, 0), (&gathered, 0), (&mask, 0)],
            &selection,
        )?;
        backend.dispatch_words(Kernel::PackedAssign, &[(&assigned, 1), (&source, 0)], &assignment)?;
        drop(source);
        drop(ids);
        drop(mask);
        drop(status);
        Ok::<_, Error>((gathered, selected))
    })?;
    services(backend, &before, 0, Some(2))?;
    let mut reverse = Vec::new();
    let mut expected = Vec::new();
    for i in 0..count {
        append_value(&mut reverse, &raw, count - 1 - i, size);
        append_value(&mut expected, &raw, if i % 2 == 1 { count - 1 - i } else { i }, size);
    }
    read_matches(backend, &gathered, &words(&reverse))?;
    read_matches(backend, &selected, &words(&expected))?;
    let mut expected = guarded(&raw);
    // Assignment leaves the initialized bytes after its final logical field.
    let tail = raw.len() % 4;
    if tail != 0 {
        let last = expected.len() - 2;
        expected[last] |= GUARD & (u32::MAX << (8 * tail));
    }
    read_matches(backend, &assigned, &expected)?;
    Ok(3)
}

fn queued_bool_lifetime(backend: &Nvidia, width: Width) -> Result<usize, Error> {
    let count = 257;
    let size = width.bytes() as usize;
    let remainder = if width == Width::One { 3 } else { 0 };
    let truth: Vec<u32> = (0..count).map(|i| u32::from(i % 3 == 1)).collect();
    let mut raw = vec![0xa5; remainder];
    raw.extend(truth.iter().flat_map(|v| v.to_le_bytes()[..size].to_vec()));
    let values = backend.upload_words(&guarded(&raw))?;
    let ids: Vec<u32> = (0..count).map(|i| (i % 7) as u32).collect();
    let indices = backend.upload_words(&ids)?;
    let status = backend.upload_words(&[0])?;
    let mut initial = vec![0; 7 * size];
    initial.resize(initial.len().div_ceil(4) * 4, 0xa5);
    let output = backend.upload_words(&guarded(&initial))?;
    let mut p = Layout::contiguous(Shape::new([7]))?.gather_parameters(&Layout::contiguous(Shape::new([count]))?, 0)?;
    p[0] |= (width as usize) << 8 | remainder << REMAINDER_SHIFT;
    let before = backend.service_profile()?;
    backend.with_commands(|| {
        backend.dispatch_words(Kernel::PackedBoolScatter, &[(&values, 1), (&indices, 0), (&status, 0), (&output, 1)], &p)?;
        drop(values);
        drop(indices);
        drop(status);
        Ok::<_, Error>(())
    })?;
    services(backend, &before, 0, Some(0))?;
    for i in 0..count {
        if truth[i] != 0 {
            initial[ids[i] as usize * size] = 1;
        }
    }
    read_matches(backend, &output, &guarded(&initial))?;
    Ok(1)
}

fn scalar_compaction(backend: &Nvidia) -> Result<(), Error> {
    for dtype in [DType::Bool(BoolStore::U8), DType::Bool(BoolStore::U32)] {
        for value in [0u32, 1] {
            let input = NativeTensor::full(Shape::new([]), dtype, value)?;
            let before = backend.service_profile()?;
            let coordinates = input.argwhere()?;
            services(backend, &before, 1, Some(2))?;
            native_matches(&coordinates, &[value as usize, 0], DType::I32, &[])?;
        }
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let mut gathers = 0;
    let mut masks = 0;
    let mut assignments = 0;
    let mut scatter = 0;
    let mut queued = 0;
    for width in Width::ALL {
        for count in BOUNDARIES {
            gathers += raw_gather(backend, width, count).map_err(|e| format!("packed gather {width:?} count={count}: {e}"))?;
            for narrow in [false, true] {
                for remainder in 0..if narrow { 4 } else { 1 } {
                    for scalar in [false, true] {
                        masks += raw_mask(backend, width, count, narrow, remainder, scalar)
                            .map_err(|e| format!("packed mask {width:?} count={count} u8={narrow} remainder={remainder} scalar={scalar}: {e}"))?;
                    }
                }
            }
            for step in [1, 2] {
                assignments += raw_assign(backend, width, count, step).map_err(|e| format!("packed assign {width:?} count={count} step={step}: {e}"))?;
            }
        }
        queued += queued_lifetime(backend, width).map_err(|e| format!("packed queued lifetime {width:?}: {e}"))?;
    }
    for count in BOUNDARIES {
        for width in [Width::One, Width::Four] {
            for remainder in 0..if width == Width::One { 4 } else { 1 } {
                scatter += raw_bool_scatter(backend, count, width, remainder)
                    .map_err(|e| format!("packed boolean scatter {width:?} count={count} remainder={remainder}: {e}"))?;
            }
        }
    }
    for width in [Width::One, Width::Four] {
        queued += queued_bool_lifetime(backend, width).map_err(|e| format!("packed boolean queued lifetime {width:?}: {e}"))?;
    }
    if (gathers, masks, assignments, scatter, queued) != (60, 400, 120, 50, 14) {
        return Err(String::from("packed movement corpus totals"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_PACKED_MOVEMENT_RAW_OK gather={gathers} mask={masks} assign={assignments} bool_scatter={scatter} queued={queued} widths=1,2,4,8 boundaries=1,127,128,129,257 guards=verified full_writer=gather,mask initialized_partial=assign,bool_scatter tails=verified scalar64=exact rank8=true invalid_metadata=no_launch aliases=no_launch"
    ));
    let mut native = 0;
    for dtype in [
        DType::F64,
        DType::F32,
        DType::Flex32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool(BoolStore::U8),
        DType::Bool(BoolStore::U32),
    ] {
        native += native_movement(backend, dtype).map_err(|e| format!("native movement {dtype:?}: {e}"))?;
    }
    let mut boolean = 0;
    for remainder in 0..4 {
        boolean += native_bool(backend, 129, remainder).map_err(|e| format!("native boolean movement remainder={remainder}: {e}"))?;
    }
    let mut invalid = 0;
    for dtype in [DType::U8, DType::F16, DType::U64, DType::Bool(BoolStore::U8)] {
        invalid += native_invalid(backend, dtype).map_err(|e| format!("native invalid movement {dtype:?}: {e}"))?;
    }
    if (native, boolean, invalid) != (165, 8, 32) {
        return Err(String::from("native movement corpus totals"));
    }
    scalar_compaction(backend).map_err(|e| format!("rank-zero boolean compaction: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_PACKED_MOVEMENT_OK native={native} bool_scatter_argwhere={boolean} empty_invalid={invalid} scalar_argwhere=4 dtypes=15 bytes=exact nan_payloads=preserved source_alias=snapshot mask_offsets=4,5,6,7 scalar64=exact compaction_metadata_bytes=4 indexed_status_bytes=4 host_activation_reads=0 f32_i32_fastpaths=verified counters=verified"
    ));
    Ok(())
}
