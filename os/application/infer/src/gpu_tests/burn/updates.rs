//! Independent logical-coordinate references for exact resident tensor updates.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Shape, Slice, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Backend, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, elementwise::Elementwise};

fn upload(words: &[u32], shape: Shape, dtype: DType) -> Result<NativeTensor, Error> {
    NativeTensor::from_data(TensorData::from_bytes_vec(words.iter().flat_map(|v| v.to_le_bytes()).collect(), shape, dtype))
}

fn check(tensor: &NativeTensor, shape: &[usize], words: &[u32]) -> Result<(), Error> {
    let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    if tensor.shape().as_slice() != shape || tensor.read()?.as_bytes() != bytes {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn word(i: usize, dtype: DType) -> u32 {
    match dtype {
        DType::F32 => [
            0,
            0x8000_0000,
            0x7fc0_1234,
            0xffc0_5678,
            0x7fa0_4321,
            0x7f80_0000,
            0xff80_0000,
            1,
            0x8000_0001,
            (i as f32 * 0.25 - 14.0).to_bits(),
        ][i % 10],
        DType::I32 => (i as u32).wrapping_mul(0x9e37_79b9),
        _ => u32::from(i.is_multiple_of(3)),
    }
}

fn coordinate(mut i: usize, shape: [usize; 3]) -> [usize; 3] {
    let mut result = [0; 3];
    for axis in (0..3).rev() {
        result[axis] = i % shape[axis];
        i /= shape[axis];
    }
    result
}

fn linear(c: [usize; 3], shape: [usize; 3]) -> usize {
    (c[0] * shape[1] + c[1]) * shape[2] + c[2]
}

fn transfers_unchanged(before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], after: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS]) -> Result<(), Error> {
    if [abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn strided(backend: &Nvidia, dtype: DType) -> Result<(), Error> {
    let shape = [129, 3, 3];
    let words: Vec<_> = (0..5 * 3 * 129).map(|i| word(i, dtype)).collect();
    let original = upload(&words, Shape::new([5, 3, 129]), dtype)?;
    let input = original.clone().slice(&[Slice::new(1, Some(4), 1)])?.permute(&[2, 0, 1])?.flip(&[0, 2])?;
    let ordered: Vec<_> = (0..129 * 3 * 3)
        .map(|i| {
            let [x, n, c] = coordinate(i, shape);
            words[((n + 1) * 3 + 2 - c) * 129 + 128 - x]
        })
        .collect();
    let replacement: Vec<_> = (0..3).map(|i| word(94 + i, dtype)).collect();
    let value = upload(&replacement, Shape::new([3, 1]), dtype)?;
    let mask_words: Vec<_> = (0..131 * 3).map(|i| u32::from(i % 5 < 2)).collect();
    let mask_original = upload(&mask_words, Shape::new([131, 1, 3]), DType::Bool(BoolStore::U32))?;
    let mask = mask_original.clone().slice(&[Slice::new(1, Some(130), 1)])?.flip(&[0, 2])?;
    let before = backend.service_profile()?;
    let selected = input.mask_where(&mask, &value)?;
    let after = backend.service_profile()?;
    transfers_unchanged(&before, &after)?;
    let expected: Vec<_> = (0..ordered.len())
        .map(|i| {
            let [x, n, c] = coordinate(i, shape);
            if mask_words[(129 - x) * 3 + 2 - c] != 0 { replacement[n] } else { ordered[i] }
        })
        .collect();
    check(&selected, &shape, &expected)?;
    if selected.dtype() != dtype {
        return Err(Error::DeviceFault);
    }

    // The source overlaps the original destination allocation. Every read
    // must observe the original snapshot rather than another thread's write.
    let source = input.clone().slice(&[Slice::new(0, Some(65), 1)])?.flip(&[0, 2])?;
    let before = backend.service_profile()?;
    let assigned = input.slice_assign(&[Slice::new(0, None, 2)], &source)?;
    transfers_unchanged(&before, &backend.service_profile()?)?;
    let expected: Vec<_> = (0..ordered.len())
        .map(|i| {
            let [x, n, c] = coordinate(i, shape);
            if x % 2 == 0 {
                ordered[linear([64 - x / 2, n, 2 - c], shape)]
            } else {
                ordered[i]
            }
        })
        .collect();
    check(&assigned, &shape, &expected)?;

    let broadcast_values = upload(&replacement[..2], Shape::new([2, 1]), dtype)?;
    let slices = [Slice::new(1, Some(128), 3), Slice::new(1, Some(3), -1), Slice::new(0, Some(3), 2)];
    let before = backend.service_profile()?;
    let assigned = input.slice_assign(&slices, &broadcast_values)?;
    transfers_unchanged(&before, &backend.service_profile()?)?;
    let mut expected = ordered.clone();
    for x in (1..128).step_by(3) {
        for n in 1..3 {
            for c in [0, 2] {
                expected[linear([x, n, c], shape)] = replacement[2 - n];
            }
        }
    }
    check(&assigned, &shape, &expected)?;

    for axis in 0..3 {
        let split = shape[axis] / 2;
        let mut left = vec![Slice::full(); axis + 1];
        left[axis] = Slice::new(0, Some(split as isize), 1);
        let mut right = left.clone();
        right[axis] = Slice::new(split as isize, None, 1);
        let mut empty = left.clone();
        empty[axis] = Slice::new(split as isize, Some(split as isize), 1);
        let parts = [
            input.clone().slice(&left)?,
            input.clone().slice(&empty)?,
            input.clone().slice(&right)?,
            input.clone(),
        ];
        let before = backend.service_profile()?;
        let output = NativeTensor::cat(&parts, axis)?;
        transfers_unchanged(&before, &backend.service_profile()?)?;
        let mut out_shape = shape;
        out_shape[axis] *= 2;
        let expected: Vec<_> = (0..ordered.len() * 2)
            .map(|i| {
                let mut c = coordinate(i, out_shape);
                c[axis] %= shape[axis];
                ordered[linear(c, shape)]
            })
            .collect();
        check(&output, &out_shape, &expected)?;
    }
    let fill = match dtype {
        DType::F32 => 0x7fa0_6578,
        DType::I32 => 0xffff_ffff,
        _ => 1,
    };
    let before = backend.service_profile()?;
    let padded = input.pad(&[(1, 2), (2, 0), (1, 1)], fill)?;
    transfers_unchanged(&before, &backend.service_profile()?)?;
    let out_shape = [132, 5, 5];
    let expected: Vec<_> = (0..132 * 5 * 5)
        .map(|i| {
            let [x, n, c] = coordinate(i, out_shape);
            if (1..130).contains(&x) && (2..5).contains(&n) && (1..4).contains(&c) {
                ordered[linear([x - 1, n - 2, c - 1], shape)]
            } else {
                fill
            }
        })
        .collect();
    check(&padded, &out_shape, &expected)?;
    check(&input, &shape, &ordered)?;
    check(&original, &[5, 3, 129], &words)?;
    check(&value, &[3, 1], &replacement)?;
    check(&mask_original, &[131, 1, 3], &mask_words)?;
    Ok(())
}

fn empty_and_invalid(backend: &Nvidia) -> Result<(), Error> {
    let empty = NativeTensor::empty(Shape::new([0, 3, 3]), DType::F32)?;
    let value = NativeTensor::full(Shape::new([1, 3, 1]), DType::F32, 1.0f32.to_bits())?;
    let mask = NativeTensor::full(Shape::new([1]), DType::Bool(BoolStore::U32), 1)?;
    let before = backend.service_profile()?;
    let outputs = [
        empty.mask_where(&mask, &value)?,
        empty.slice_assign(&[], &value)?,
        NativeTensor::cat(&[empty.clone(), empty.clone()], 0)?,
        empty.pad(&[(0, 0); 3], 0)?,
    ];
    let after = backend.service_profile()?;
    for op in [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
        if before[op as usize].calls != after[op as usize].calls {
            return Err(Error::DeviceFault);
        }
    }
    for output in outputs {
        check(&output, &[0, 3, 3], &[])?;
    }
    let padded = empty.pad(&[(1, 1), (0, 0), (0, 0)], 0x7fc0_1234)?;
    check(&padded, &[2, 3, 3], &[0x7fc0_1234; 18])?;
    let wrong = NativeTensor::full(Shape::new([2, 4]), DType::I32, 0)?;
    let before = backend.service_profile()?;
    for result in [
        value.mask_where(&value, &value),
        value.mask_where(&mask, &wrong),
        value.slice_assign(&[], &wrong),
        value.slice_assign(&[Slice { start: 0, end: None, step: 0 }], &value),
        NativeTensor::cat(&[], 0),
        NativeTensor::cat(core::slice::from_ref(&value), 3),
        NativeTensor::cat(&[value.clone(), wrong], 0),
        value.pad(&[(0, 0)], 0),
        value.pad(&[(usize::MAX, 1), (0, 0), (0, 0)], 0),
        mask.pad(&[(0, 0)], 2),
    ] {
        if result.is_ok() {
            return Err(Error::DeviceFault);
        }
    }
    let after = backend.service_profile()?;
    for op in [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
        if before[op as usize].calls != after[op as usize].calls {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

fn guards(backend: &Nvidia) -> Result<(), Error> {
    let guard = 0x6bad_beefu32;
    let count = 7 * 19;
    let mut a = vec![guard; count + 5];
    let mut b = a.clone();
    let mut mask = a.clone();
    for i in 0..count {
        a[2 + i] = word(i, DType::F32);
        b[2 + i] = word(i + 13, DType::F32);
        mask[2 + i] = u32::from(i % 3 == 1);
    }
    let left = backend.upload_words(&a)?;
    let right = backend.upload_words(&b)?;
    let condition = backend.upload_words(&mask)?;
    let output = backend.upload_words(&vec![guard; count + 5])?;
    let la = Layout::from_parts(Shape::new([7, 19]), vec![-1, 7], 8, a.len())?;
    let lb = Layout::from_parts(Shape::new([7, 19]), vec![19, -1], 20, b.len())?;
    let p = la.elementwise_parameters(&lb, Elementwise::Copy, 0, false)?;
    backend.dispatch_words(Kernel::MaskWhere, &[(&left, 0), (&right, 0), (&condition, 2), (&output, 2)], &p)?;
    let mut expected = vec![guard; count + 5];
    for row in 0..7 {
        for col in 0..19 {
            let i = row * 19 + col;
            expected[2 + i] = if mask[2 + i] == 0 { a[8 - row + 7 * col] } else { b[20 + 19 * row - col] };
        }
    }
    let mut actual = vec![0; expected.len()];
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual != expected {
        return Err(Error::DeviceFault);
    }

    let mut assigned = vec![guard; 299];
    let target = backend.upload_words(&assigned)?;
    let layout = Layout::from_parts(Shape::new([7, 19]), vec![-43, 2], 260, assigned.len())?;
    let parameters = layout.assignment_parameters(&lb)?;
    backend.dispatch_words(Kernel::ViewAssign, &[(&target, 0), (&right, 0)], &parameters)?;
    for row in 0..7 {
        for col in 0..19 {
            assigned[260 - 43 * row + 2 * col] = b[20 + 19 * row - col];
        }
    }
    let mut actual = vec![0; assigned.len()];
    backend.read_words_into(&target, 0, &mut actual)?;
    if actual != assigned {
        return Err(Error::DeviceFault);
    }
    backend.with_commands(|| {
        for (index, value) in [(2, 1), (3, 1), (6, 296), (7, 134), (16, 0)] {
            let mut invalid = parameters;
            invalid[index] = value;
            if backend.dispatch_words(Kernel::ViewAssign, &[(&target, 0), (&right, 0)], &invalid).is_ok() {
                return Err(Error::DeviceFault);
            }
        }
        if backend.dispatch_words(Kernel::ViewAssign, &[(&target, 0), (&target, 0)], &parameters).is_ok()
            || backend
                .dispatch_words(Kernel::MaskWhere, &[(&left, 0), (&right, 0), (&condition, 2), (&left, 0)], &p)
                .is_ok()
            || backend
                .dispatch_words(Kernel::MaskWhere, &[(&left, 0), (&right, 0), (&condition, 6), (&output, 2)], &p)
                .is_ok()
            || backend.produce_words(299, Kernel::ViewAssign, &[(&right, 0)], &parameters).is_ok()
        {
            return Err(Error::DeviceFault);
        }
        Ok(())
    })?;
    backend.read_words_into(&target, 0, &mut actual)?;
    if actual != assigned {
        return Err(Error::DeviceFault);
    }
    for (buffer, expected) in [(&left, &a), (&right, &b), (&condition, &mask)] {
        let mut actual = vec![0; expected.len()];
        backend.read_words_into(buffer, 0, &mut actual)?;
        if &actual != expected {
            return Err(Error::DeviceFault);
        }
    }
    // A valid request still succeeds after rejected launches in a command scope.
    backend.dispatch_words(Kernel::MaskWhere, &[(&left, 0), (&right, 0), (&condition, 2), (&output, 2)], &p)?;
    let mut actual = vec![0; expected.len()];
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    for dtype in [DType::F32, DType::I32, DType::Bool(BoolStore::U32)] {
        strided(backend, dtype).map_err(|e| format!("strided updates {dtype:?}: {e}"))?;
    }
    empty_and_invalid(backend).map_err(|e| format!("empty or invalid updates: {e}"))?;
    guards(backend).map_err(|e| format!("update guards: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_UPDATES_OK mask_where=exact slice_assign=snapshot cat=all_axes pad=exact dtypes=3 views=strided guards=ok resident=ok backend=nvidia"
    ));
    Ok(())
}
