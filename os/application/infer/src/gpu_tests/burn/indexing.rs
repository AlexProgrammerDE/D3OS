//! Gather references use independent coordinates, not the device layout decoder.
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Shape, Slice, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Backend, Error, nvidia::Nvidia};
use d3os_nvidia::compute as abi;

fn upload(words: &[u32], shape: Shape, dtype: DType) -> Result<NativeTensor, Error> {
    NativeTensor::from_data(TensorData::from_bytes_vec(words.iter().flat_map(|v| v.to_le_bytes()).collect(), shape, dtype))
}

fn check(tensor: &NativeTensor, words: &[u32]) -> Result<(), Error> {
    let expected: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    if tensor.read()?.as_bytes() != expected {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn source_words(dtype: DType) -> Vec<u32> {
    (0..105)
        .map(|i| match dtype {
            DType::F32 => match i % 9 {
                0 => 0x7fc0_1234,
                1 => 0x8000_0000,
                2 => 0x7f80_0000,
                _ => (i as f32 / 8.0 - 3.0).to_bits(),
            },
            DType::I32 => (i as u32).wrapping_mul(0x9e37_79b9),
            _ => u32::from(i % 3 == 0),
        })
        .collect()
}

fn coordinates(mut index: usize, shape: [usize; 3]) -> [usize; 3] {
    let mut c = [0; 3];
    for axis in (0..3).rev() {
        c[axis] = index % shape[axis];
        index /= shape[axis];
    }
    c
}

fn physical(c: [usize; 3]) -> usize {
    (c[1] * 5 + 4 - c[2]) * 7 + 6 - c[0]
}

fn strided(backend: &Nvidia, dtype: DType) -> Result<(), Error> {
    let words = source_words(dtype);
    let original = upload(&words, Shape::new([3, 5, 7]), dtype)?;
    let source = original.clone().permute(&[2, 0, 1])?.flip(&[0, 2])?;
    for axis in 0..3 {
        let mut shape = [7, 3, 5];
        let length = shape[axis];
        shape[axis] = 131;
        let count: usize = shape.into_iter().product();
        let values: Vec<i32> = (0..count).map(|i| ((i * 13 + 2) % length) as i32).collect();
        let original_indices = NativeTensor::from_data(TensorData::new(values.clone(), [shape[1], shape[0], shape[2]]))?;
        let indices = original_indices.clone().permute(&[1, 0, 2])?.flip(&[2])?;
        let before = backend.service_profile()?;
        let output = source.gather(axis, &indices)?;
        let after = backend.service_profile()?;
        if [abi::ALLOCATE, abi::UPLOAD]
            .iter()
            .any(|&op| before[op as usize].calls != after[op as usize].calls)
            || after[abi::DOWNLOAD as usize].calls - before[abi::DOWNLOAD as usize].calls != 1
            || output.dtype() != dtype
            || output.shape().as_slice() != shape
        {
            return Err(Error::DeviceFault);
        }
        let expected: Vec<_> = (0..count)
            .map(|i| {
                let mut c = coordinates(i, shape);
                let index = (c[1] * shape[0] + c[0]) * shape[2] + shape[2] - 1 - c[2];
                c[axis] = values[index] as usize;
                words[physical(c)]
            })
            .collect();
        check(&output, &expected)?;
        check(&original_indices, &values.iter().map(|&v| v as u32).collect::<Vec<_>>())?;
    }
    let ids = [4i32, 0, 4, 1];
    let mut values = vec![0i32; 20];
    values[10..14].copy_from_slice(&ids);
    let indices = NativeTensor::from_data(TensorData::new(values, [20]))?
        .slice(&[Slice::new(10, Some(14), 1)])?
        .flip(&[0])?;
    let selected = source.select(2, &indices)?;
    let expected: Vec<_> = (0..7 * 3 * 4)
        .map(|i| {
            let mut c = coordinates(i, [7, 3, 4]);
            c[2] = ids[3 - c[2]] as usize;
            words[physical(c)]
        })
        .collect();
    check(&selected, &expected)?;
    if selected.shape().as_slice() != [7, 3, 4] {
        return Err(Error::DeviceFault);
    }
    check(&original, &words)?;
    Ok(())
}

fn subviews() -> Result<(), Error> {
    let source = NativeTensor::from_data(TensorData::new((0..20i32).collect::<Vec<_>>(), [4, 5]))?.slice(&[Slice::new(1, Some(4), 1)])?;
    let ids: Vec<i32> = (0..15).map(|i| i % 5).collect();
    let indices = NativeTensor::from_data(TensorData::new(ids.clone(), [5, 3]))?
        .slice(&[Slice::new(1, Some(4), 1)])?
        .flip(&[1])?;
    let expected: Vec<_> = (0..9).map(|i| (5 + (i / 3) * 5 + ids[3 + (i / 3) * 3 + 2 - i % 3] as usize) as u32).collect();
    check(&source.gather(1, &indices)?, &expected)
}

fn rejected_indices(backend: &Nvidia) -> Result<(), Error> {
    let source = NativeTensor::from_data(TensorData::new((0..15i32).collect::<Vec<_>>(), [3, 5]))?;
    let valid = NativeTensor::from_data(TensorData::new(vec![4i32, 2, 0], [3, 1]))?;
    for bad in [-1, 5, i32::MIN, i32::MAX] {
        let indices = NativeTensor::from_data(TensorData::new(vec![0i32, bad, 4], [3, 1]))?;
        let before = backend.service_profile()?;
        if !matches!(source.gather(1, &indices), Err(Error::InvalidParameter)) {
            return Err(Error::DeviceFault);
        }
        let after = backend.service_profile()?;
        if after[abi::FREE as usize].calls - before[abi::FREE as usize].calls
            != after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls
        {
            return Err(Error::DeviceFault);
        }
        check(&source.gather(1, &valid)?, &[4, 7, 10])?;
    }
    let empty_axis = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [3, 0]))?;
    if !matches!(empty_axis.gather(1, &valid), Err(Error::InvalidParameter)) {
        return Err(Error::DeviceFault);
    }
    let invalid = NativeTensor::from_data(TensorData::new(vec![0i32, -1, 4], [3, 1]))?;
    if !matches!(backend.with_commands(|| source.gather(1, &invalid)), Err(Error::InvalidParameter)) {
        return Err(Error::DeviceFault);
    }
    check(&backend.with_commands(|| source.gather(1, &valid))?, &[4, 7, 10])?;
    let empty_indices = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [3, 0]))?;
    let empty_vector = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [0]))?;
    let wrong_dtype = NativeTensor::from_data(TensorData::new(vec![0f32; 3], [3, 1]))?;
    let wrong_shape = NativeTensor::from_data(TensorData::new(vec![0i32; 2], [2, 1]))?;
    let before = backend.queue_profile()?;
    if !source.gather(1, &empty_indices)?.layout().is_empty()
        || !source.select(1, &empty_vector)?.layout().is_empty()
        || !matches!(source.gather(2, &valid), Err(Error::ShapeMismatch))
        || !matches!(source.gather(1, &wrong_dtype), Err(Error::InvalidParameter))
        || !matches!(source.gather(1, &wrong_shape), Err(Error::ShapeMismatch))
        || before.launches != backend.queue_profile()?.launches
    {
        return Err(Error::DeviceFault);
    }
    check(&source, &(0..15u32).collect::<Vec<_>>())
}

fn guards(backend: &Nvidia) -> Result<(), Error> {
    let guard = 0x6bad_beefu32;
    let mut source = vec![guard];
    source.extend(0..105u32);
    source.push(guard);
    let mut indices = vec![guard];
    indices.extend((0..257).map(|i| (i % 105) as u32));
    indices[257] = 105;
    indices.push(guard);
    let source_buffer = backend.upload_words(&source)?;
    let index_buffer = backend.upload_words(&indices)?;
    let error_buffer = backend.upload_words(&[guard, 0, guard])?;
    let output = backend.upload_words(&vec![guard; 259])?;
    let layout = Layout::contiguous(Shape::new([1, 105]))?;
    let index_layout = Layout::contiguous(Shape::new([1, 257]))?;
    let parameters = layout.gather_parameters(&index_layout, 1)?;
    backend.dispatch_words(
        abi::Kernel::Gather,
        &[(&source_buffer, 1), (&index_buffer, 1), (&error_buffer, 1), (&output, 1)],
        &parameters,
    )?;
    let mut actual = vec![0; 259];
    backend.read_words_into(&output, 0, &mut actual)?;
    let mut expected = vec![guard];
    expected.extend((0..256).map(|i| (i % 105) as u32));
    expected.extend([0, guard]);
    let mut status = [0; 3];
    backend.read_words_into(&error_buffer, 0, &mut status)?;
    if actual != expected || status != [guard, 1, guard] {
        return Err(Error::DeviceFault);
    }
    let mut invalid = parameters;
    invalid[3] = 104;
    if backend.dispatch_words(
        abi::Kernel::Gather,
        &[(&source_buffer, 1), (&index_buffer, 1), (&error_buffer, 1), (&output, 1)],
        &invalid,
    ) != Err(Error::InvalidParameter)
    {
        return Err(Error::DeviceFault);
    }
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    let mut unchanged = vec![0; source.len()];
    backend.read_words_into(&source_buffer, 0, &mut unchanged)?;
    if unchanged != source {
        return Err(Error::DeviceFault);
    }
    let mut unchanged = vec![0; indices.len()];
    backend.read_words_into(&index_buffer, 0, &mut unchanged)?;
    if unchanged != indices {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    for dtype in [DType::F32, DType::I32, DType::Bool(BoolStore::U32)] {
        strided(backend, dtype).map_err(|e| format!("gather {dtype:?}: {e}"))?;
    }
    subviews().map_err(|e| format!("gather subviews: {e}"))?;
    rejected_indices(backend).map_err(|e| format!("gather rejected indices: {e}"))?;
    guards(backend).map_err(|e| e.to_string())?;
    crate::report(format_args!(
        "NVIDIA_BURN_INDEXING_OK gather=ok select=ok dtypes=3 views=strided bounds=recoverable guards=ok status_bytes=4 backend=nvidia"
    ));
    Ok(())
}
