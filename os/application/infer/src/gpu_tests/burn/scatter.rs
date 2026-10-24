//! Independent scatter references and the adjoint needed by gather backward.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Shape, Slice, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Backend, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, elementwise::Elementwise, indexing::Scatter, reduction::Reduction};

fn upload(words: &[u32], shape: Shape, dtype: DType) -> Result<NativeTensor, Error> {
    NativeTensor::from_data(TensorData::from_bytes_vec(words.iter().flat_map(|v| v.to_le_bytes()).collect(), shape, dtype))
}

fn read(tensor: &NativeTensor) -> Result<Vec<u32>, Error> {
    Ok(tensor
        .read()?
        .as_bytes()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn coordinates(mut index: usize, shape: [usize; 3]) -> [usize; 3] {
    let mut c = [0; 3];
    for a in (0..3).rev() {
        c[a] = index % shape[a];
        index /= shape[a];
    }
    c
}

fn physical(c: [usize; 3]) -> usize {
    (c[1] * 5 + 4 - c[2]) * 7 + 6 - c[0]
}

fn update(old: u32, value: u32, dtype: DType) -> u32 {
    match dtype {
        DType::F32 => (f32::from_bits(old) + f32::from_bits(value)).to_bits(),
        DType::I32 => old.wrapping_add(value),
        _ => old | value,
    }
}

fn scatter(source: &NativeTensor, axis: usize, indices: &NativeTensor, values: &NativeTensor, dtype: DType) -> Result<NativeTensor, Error> {
    if matches!(dtype, DType::Bool(_)) {
        source.scatter_or(axis, indices, values)
    } else {
        source.scatter_add(axis, indices, values)
    }
}

fn strided(backend: &Nvidia, dtype: DType) -> Result<(), Error> {
    let words: Vec<_> = (0..105u32)
        .map(|i| match dtype {
            DType::F32 => (i as f32 / 8.0 - 3.0).to_bits(),
            DType::I32 => i.wrapping_mul(0x9e37_79b9),
            _ => u32::from(i % 3 == 0),
        })
        .collect();
    let original = upload(&words, Shape::new([3, 5, 7]), dtype)?;
    let source = original.clone().permute(&[2, 0, 1])?.flip(&[0, 2])?;
    for axis in 0..3 {
        let mut shape = [7, 3, 5];
        let length = shape[axis];
        shape[axis] = 131;
        let count: usize = shape.into_iter().product();
        let ids: Vec<i32> = (0..count).map(|i| ((i * 13 + 2) % length) as i32).collect();
        let index_data = NativeTensor::from_data(TensorData::new(ids.clone(), [shape[1], shape[0], shape[2]]))?;
        let indices = index_data.clone().permute(&[1, 0, 2])?.flip(&[2])?;
        let values: Vec<_> = (0..count)
            .map(|i| match dtype {
                DType::F32 => ((i as i32 % 17 - 8) as f32 / 16.0).to_bits(),
                DType::I32 => (i as u32).wrapping_mul(0x6a09_e667),
                _ => u32::from(i % 5 == 0),
            })
            .collect();
        let value_data = upload(&values, Shape::new([shape[1], shape[0], shape[2]]), dtype)?;
        let value_view = value_data.clone().permute(&[1, 0, 2])?.flip(&[0, 2])?;
        let mut expected: Vec<_> = (0..105).map(|i| words[physical(coordinates(i, [7, 3, 5]))]).collect();
        for i in 0..count {
            let mut c = coordinates(i, shape);
            let index = (c[1] * shape[0] + c[0]) * shape[2] + shape[2] - 1 - c[2];
            let value = (c[1] * shape[0] + shape[0] - 1 - c[0]) * shape[2] + shape[2] - 1 - c[2];
            c[axis] = ids[index] as usize;
            let destination = (c[0] * 3 + c[1]) * 5 + c[2];
            expected[destination] = update(expected[destination], values[value], dtype);
        }
        let before = backend.service_profile()?;
        let output = scatter(&source, axis, &indices, &value_view, dtype)?;
        let after = backend.service_profile()?;
        if [abi::ALLOCATE, abi::UPLOAD]
            .iter()
            .any(|&op| before[op as usize].calls != after[op as usize].calls)
            || after[abi::DOWNLOAD as usize].calls - before[abi::DOWNLOAD as usize].calls != 1
            || output.shape().as_slice() != [7, 3, 5]
            || output.dtype() != dtype
            || read(&output)? != expected
            || read(&value_data)? != values
            || read(&index_data)? != ids.iter().map(|&v| v as u32).collect::<Vec<_>>()
        {
            return Err(Error::DeviceFault);
        }
    }
    let ids = NativeTensor::from_data(TensorData::new(vec![0i32, 4, 0, 4, 1, 0], [6]))?
        .slice(&[Slice::new(1, Some(5), 1)])?
        .flip(&[0])?;
    let values = NativeTensor::full(Shape::new([7, 3, 4]), dtype, if dtype == DType::F32 { 0.5f32.to_bits() } else { 1 })?;
    let output = if matches!(dtype, DType::Bool(_)) {
        source.select_or(2, &ids, &values)?
    } else {
        source.select_add(2, &ids, &values)?
    };
    let mut expected: Vec<_> = (0..105).map(|i| words[physical(coordinates(i, [7, 3, 5]))]).collect();
    for row in 0..21 {
        for index in [1, 4, 0, 4] {
            let at = row * 5 + index;
            expected[at] = update(expected[at], if dtype == DType::F32 { 0.5f32.to_bits() } else { 1 }, dtype);
        }
    }
    if read(&output)? != expected || read(&original)? != words {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn float_edges() -> Result<(), Error> {
    let a = [1, 0x8000_0001, 0x0080_0000, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234, 0x7f7f_ffff];
    let b = [
        1,
        0x8000_0001,
        0x807f_ffff,
        0x8000_0000,
        0xff80_0000,
        1.0f32.to_bits(),
        1.0f32.to_bits(),
        0x7f7f_ffff,
    ];
    let source = upload(&a, Shape::new([a.len()]), DType::F32)?;
    let values = upload(&b, Shape::new([b.len()]), DType::F32)?;
    let ids = NativeTensor::from_data(TensorData::new((0..a.len() as i32).collect::<Vec<_>>(), [a.len()]))?;
    for ((actual, a), b) in read(&source.scatter_add(0, &ids, &values)?)?.into_iter().zip(a).zip(b) {
        let expected = f32::from_bits(a) + f32::from_bits(b);
        if if expected.is_nan() {
            !f32::from_bits(actual).is_nan()
        } else {
            actual != expected.to_bits()
        } {
            return Err(Error::DeviceFault);
        }
    }
    let ids = NativeTensor::from_data(TensorData::new(vec![0i32; 257], [257]))?;
    let values = NativeTensor::full(Shape::new([257]), DType::F32, 1)?;
    for initial in [0, 0x7fc0_1234] {
        let source = NativeTensor::full(Shape::new([1]), DType::F32, initial)?;
        let result = read(&source.scatter_add(0, &ids, &values)?)?[0];
        if if initial == 0 { result != 257 } else { !f32::from_bits(result).is_nan() } {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

fn subviews() -> Result<(), Error> {
    let source = NativeTensor::from_data(TensorData::new((0..25i32).collect::<Vec<_>>(), [5, 5]))?
        .slice(&[Slice::new(1, Some(4), 1)])?
        .flip(&[1])?;
    let ids: Vec<i32> = (0..35).map(|i| (i * 3) % 5).collect();
    let indices = NativeTensor::from_data(TensorData::new(ids.clone(), [5, 7]))?
        .slice(&[Slice::new(1, Some(4), 1)])?
        .flip(&[1])?;
    // This dense subview retains a nonzero base and needs no value copy.
    let values = NativeTensor::from_data(TensorData::new((0..35i32).collect::<Vec<_>>(), [5, 7]))?.slice(&[Slice::new(1, Some(4), 1)])?;
    let mut expected: Vec<_> = (0..15).map(|i| ((i / 5 + 1) * 5 + 4 - i % 5) as u32).collect();
    for row in 0..3 {
        for col in 0..7 {
            let at = row * 5 + ids[(row + 1) * 7 + 6 - col] as usize;
            expected[at] = expected[at].wrapping_add(((row + 1) * 7 + col) as u32);
        }
    }
    if read(&source.scatter_add(1, &indices, &values)?)? != expected {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn recovery(backend: &Nvidia) -> Result<(), Error> {
    let source = NativeTensor::from_data(TensorData::new(vec![3i32, 5, 7], [3]))?;
    let values = NativeTensor::from_data(TensorData::new(vec![1i32; 257], [257]))?;
    let valid = NativeTensor::from_data(TensorData::new(vec![1i32; 257], [257]))?;
    for bad in [-1, 3, i32::MIN, i32::MAX] {
        let mut indices = vec![1i32; 257];
        indices[256] = bad;
        let indices = NativeTensor::from_data(TensorData::new(indices, [257]))?;
        let before = backend.service_profile()?;
        if !matches!(backend.with_commands(|| source.scatter_add(0, &indices, &values)), Err(Error::InvalidParameter)) {
            return Err(Error::DeviceFault);
        }
        let after = backend.service_profile()?;
        if after[abi::FREE as usize].calls - before[abi::FREE as usize].calls
            != after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls
            || read(&backend.with_commands(|| source.scatter_add(0, &valid, &values))?)? != [3, 262, 7]
            || read(&source)? != [3, 5, 7]
        {
            return Err(Error::DeviceFault);
        }
    }
    let alias_ids = NativeTensor::from_data(TensorData::new(vec![0i32; 3], [3]))?;
    if read(&source.scatter_add(0, &alias_ids, &source)?)? != [18, 5, 7] || read(&source)? != [3, 5, 7] {
        return Err(Error::DeviceFault);
    }
    let empty_ids = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [0]))?;
    let empty_values = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [0]))?;
    let wrong_values = NativeTensor::full(Shape::new([257]), DType::F32, 0)?;
    let zero_axis = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [0]))?;
    let before = backend.queue_profile()?;
    if read(&source.scatter_add(0, &empty_ids, &empty_values)?)? != [3, 5, 7]
        || !matches!(source.scatter_add(1, &valid, &values), Err(Error::ShapeMismatch))
        || !matches!(source.scatter_add(0, &valid, &source), Err(Error::ShapeMismatch))
        || !matches!(source.scatter_add(0, &valid, &wrong_values), Err(Error::InvalidParameter))
        || !matches!(source.scatter_or(0, &valid, &values), Err(Error::InvalidParameter))
        || !matches!(zero_axis.scatter_add(0, &valid, &values), Err(Error::InvalidParameter))
        || before.launches != backend.queue_profile()?.launches
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn adjoint() -> Result<(), Error> {
    let x: Vec<f32> = (0..15).map(|i| i as f32 / 8.0 - 0.5).collect();
    let ids: Vec<i32> = (0..21).map(|i| (i * 3 + 1) % 5).collect();
    let weights: Vec<f32> = (0..21).map(|i| (i as f32 - 10.0) / 16.0).collect();
    let source = NativeTensor::from_data(TensorData::new(x.clone(), [3, 5]))?;
    let indices = NativeTensor::from_data(TensorData::new(ids.clone(), [3, 7]))?;
    let values = NativeTensor::from_data(TensorData::new(weights.clone(), [3, 7]))?;
    let loss = source
        .gather(1, &indices)?
        .binary(&values, Elementwise::Multiply)?
        .reduce(1, Reduction::Sum)?
        .reduce(0, Reduction::Sum)?;
    let objective = |input: &[f32]| -> f64 { (0..21).map(|i| f64::from(input[i / 7 * 5 + ids[i] as usize]) * f64::from(weights[i])).sum() };
    if (f64::from(f32::from_bits(read(&loss)?[0])) - objective(&x)).abs() > 1e-6 {
        return Err(Error::DeviceFault);
    }
    let zeros = NativeTensor::full(Shape::new([3, 5]), DType::F32, 0)?;
    let gradient = read(&zeros.scatter_add(1, &indices, &values)?)?;
    for (i, actual) in gradient.into_iter().enumerate() {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[i] += 1.0 / 1024.0;
        minus[i] -= 1.0 / 1024.0;
        let derivative = (objective(&plus) - objective(&minus)) * 512.0;
        if (f64::from(f32::from_bits(actual)) - derivative).abs() > 1e-6 {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

fn guards(backend: &Nvidia) -> Result<(), Error> {
    let guard = 0x6bad_beef;
    let shape = [7, 3, 131];
    let count = 7 * 3 * 131;
    let mut values = vec![guard];
    values.extend((0..count).map(|i| (i as u32).wrapping_mul(0x9e37_79b9)));
    values.push(guard);
    let mut ids = vec![guard];
    ids.extend((0..count).map(|i| (i % 5) as u32));
    ids[count] = 5;
    ids.push(guard);
    let values_buffer = backend.upload_words(&values)?;
    let ids_buffer = backend.upload_words(&ids)?;
    let mut expected = vec![guard];
    expected.extend(0..105u32);
    expected.push(guard);
    let output = backend.upload_words(&expected)?;
    let status = backend.upload_words(&[guard, 0, guard])?;
    let layout = Layout::contiguous(Shape::new([3, 5, 7]))?.permute(&[2, 0, 1])?.flip(&[0, 2])?;
    let indices = Layout::contiguous(Shape::new(shape))?;
    let p = layout.scatter_parameters(&indices, 2, Scatter::IntAdd)?;
    backend.dispatch_words(abi::Kernel::Scatter, &[(&values_buffer, 1), (&ids_buffer, 1), (&status, 1), (&output, 1)], &p)?;
    for i in 0..count - 1 {
        let mut c = coordinates(i, shape);
        c[2] = ids[i + 1] as usize;
        let at = physical(c) + 1;
        expected[at] = expected[at].wrapping_add(values[i + 1]);
    }
    let mut actual = vec![0; expected.len()];
    backend.read_words_into(&output, 0, &mut actual)?;
    let mut error = [0; 3];
    backend.read_words_into(&status, 0, &mut error)?;
    if actual != expected || error != [guard, 1, guard] {
        return Err(Error::DeviceFault);
    }
    if backend.dispatch_words(
        abi::Kernel::Scatter,
        &[(&values_buffer, 1), (&ids_buffer, 1), (&status, 1), (&values_buffer, 1)],
        &p,
    ) != Err(Error::InvalidParameter)
    {
        return Err(Error::DeviceFault);
    }
    let mut malformed = p;
    malformed[3] = 104;
    if backend.dispatch_words(
        abi::Kernel::Scatter,
        &[(&values_buffer, 1), (&ids_buffer, 1), (&status, 1), (&output, 1)],
        &malformed,
    ) != Err(Error::InvalidParameter)
    {
        return Err(Error::DeviceFault);
    }
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    for (buffer, expected) in [(&values_buffer, &values), (&ids_buffer, &ids)] {
        let mut actual = vec![0; expected.len()];
        backend.read_words_into(buffer, 0, &mut actual)?;
        if &actual != expected {
            return Err(Error::DeviceFault);
        }
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    for dtype in [DType::F32, DType::I32, DType::Bool(BoolStore::U32)] {
        strided(backend, dtype).map_err(|e| format!("scatter {dtype:?}: {e}"))?;
    }
    float_edges().map_err(|e| format!("scatter FP32 edges: {e}"))?;
    subviews().map_err(|e| format!("scatter subviews: {e}"))?;
    recovery(backend).map_err(|e| format!("scatter recovery: {e}"))?;
    adjoint().map_err(|e| format!("gather adjoint: {e}"))?;
    guards(backend).map_err(|e| format!("scatter guards: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_SCATTER_OK add=f32,i32 or=bool select=ok adjoint=finite-difference subnormals=preserved bounds=recoverable guards=ok backend=nvidia"
    ));
    Ok(())
}
