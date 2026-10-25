//! Prefix payloads cannot alter writer coverage or escape the output allocation.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Shape, TensorData},
        tensor::{Bool, Tensor},
    },
    native::NativeTensor,
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel};

fn parameters(shape: &[usize], selected: usize) -> [usize; 16] {
    let count = shape.iter().product::<usize>();
    let mut p = [0; 16];
    p[..4].copy_from_slice(&[count, shape.len(), selected, count.ilog2() as usize + 1]);
    p[4..12].fill(1);
    p[4..4 + shape.len()].copy_from_slice(shape);
    p
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    for shape in [vec![257], vec![3, 129, 5], vec![1, 2, 1, 1, 129, 1, 1, 1], vec![1; 8]] {
        let count: usize = shape.iter().product();
        let mut prefix = Vec::with_capacity(count);
        let mut expected = Vec::new();
        let mut selected = 0;
        for i in 0..count {
            if i % 7 == 0 || i == count - 1 {
                selected += 1;
                let mut index = i;
                let mut coordinates = vec![0u32; shape.len()];
                for axis in (0..shape.len()).rev() {
                    coordinates[axis] = (index % shape[axis]) as u32;
                    index /= shape[axis];
                }
                expected.extend(coordinates);
            }
            prefix.push(selected as u32);
        }
        let mut input = vec![0x55aa_1122; count + 8];
        input[3..3 + count].copy_from_slice(&prefix);
        let source = backend.upload_words(&input).map_err(|e| format!("{e}"))?;
        let output = backend.upload_words(&vec![0x5a5a_9988; expected.len() + 7]).map_err(|e| format!("{e}"))?;
        let p = parameters(&shape, selected);
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        backend
            .with_commands(|| backend.dispatch_words(Kernel::ArgWhere, &[(&source, 3), (&output, 2)], &p))
            .map_err(|e| format!("{e}"))?;
        super::dispatch::resident(backend, &before, 0)?;
        let mut actual = vec![0; expected.len() + 7];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        let mut preserved = vec![0; input.len()];
        backend.read_words_into(&source, 0, &mut preserved).map_err(|e| format!("{e}"))?;
        if actual[2..2 + expected.len()] != expected || actual[..2].iter().chain(&actual[2 + expected.len()..]).any(|&v| v != 0x5a5a_9988) || preserved != input
        {
            return Err(String::from("argwhere coordinate, source, or output guard mismatch"));
        }
    }
    for word in [0, u32::MAX] {
        let source = backend.upload_words(&vec![word; 129]).map_err(|e| format!("{e}"))?;
        let p = parameters(&[3, 43], 127);
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let output = backend
            .with_commands(|| backend.produce_words(254, Kernel::ArgWhere, &[(&source, 0)], &p))
            .map_err(|e| format!("{e}"))?;
        let after = backend.service_profile().map_err(|e| format!("{e}"))?;
        super::dispatch::resident(backend, &before, 0)?;
        if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
            || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
        {
            return Err(String::from("argwhere full writer needed an output clear"));
        }
        let mut actual = vec![0; 254];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual.iter().any(|&v| v != u32::MAX) {
            return Err(String::from("malformed prefix exposed an unwritten coordinate"));
        }
    }
    // One selected position above FP32's consecutive integer range. Build the
    // large prefix on the GPU instead of staging a large host reference array.
    let count = 16_777_220;
    let prefix = backend.produce_words(count, Kernel::Fill, &[], &[count, 0]).map_err(|e| format!("{e}"))?;
    backend.write_words(&prefix, count - 1, &[1]).map_err(|e| format!("{e}"))?;
    let output = backend
        .produce_words(1, Kernel::ArgWhere, &[(&prefix, 0)], &parameters(&[count], 1))
        .map_err(|e| format!("{e}"))?;
    let mut coordinate = [0];
    backend.read_words_into(&output, 0, &mut coordinate).map_err(|e| format!("{e}"))?;
    if coordinate != [(count - 1) as u32] {
        return Err(String::from("argwhere lost integer coordinate precision"));
    }
    drop(prefix);
    drop(output);
    let mask_values: Vec<_> = (0..257).map(|i| i % 7 == 0 || i == 256).collect();
    let mut guarded = vec![true];
    guarded.extend(&mask_values);
    guarded.push(true);
    let mask = NativeTensor::from_data(TensorData::new(guarded, [259]))
        .and_then(|t| t.slice(&[d3os_ai::burn::tensor::Slice::new(1, Some(258), 1)]))
        .and_then(|t| t.flip(&[0]))
        .map_err(|e| format!("{e}"))?;
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let output = backend
        .with_commands(|| {
            let output = mask.argwhere()?;
            drop(mask);
            Ok::<_, Error>(output)
        })
        .map_err(|e| format!("{e}"))?;
    super::dispatch::resident(backend, &before, 1)?;
    let after = backend.service_profile().map_err(|e| format!("{e}"))?;
    if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 3
    {
        return Err(String::from("strided argwhere did not use three full-writer allocations"));
    }
    let expected: Vec<_> = mask_values.iter().rev().enumerate().filter_map(|(i, &keep)| keep.then_some(i as i32)).collect();
    let data = output.read().map_err(|e| format!("{e}"))?;
    if output.layout().shape().as_slice() != [expected.len(), 1] || data.as_slice::<i32>().map_err(|e| format!("{e:?}"))? != expected {
        return Err(String::from("strided argwhere or queued prefix lifetime"));
    }
    let device = d3os_ai::native_device().map_err(|e| format!("{e}"))?;
    let source = Tensor::<1>::from_data(TensorData::new((0..257).map(|i| i as f32).collect::<Vec<_>>(), [257]), &device);
    let mask = Tensor::<1, Bool>::from_data(TensorData::new(mask_values.clone(), [257]), &device);
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let result = source.mask_select(mask);
    super::dispatch::resident(backend, &before, 2)?;
    let expected: Vec<_> = mask_values.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i as f32)).collect();
    if result.try_into_vec_as::<f32>().map_err(|e| format!("{e:?}"))? != expected {
        return Err(String::from("inherited resident mask selection"));
    }
    let empty = NativeTensor::from_data(TensorData::new(Vec::<bool>::new(), [2, 0, 3])).map_err(|e| format!("{e}"))?;
    let invalid = NativeTensor::full(Shape::new([0]), DType::F32, 0).map_err(|e| format!("{e}"))?;
    let queue = backend.queue_profile().map_err(|e| format!("{e}"))?;
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let result = empty.argwhere().map_err(|e| format!("{e}"))?;
    if result.layout().shape().as_slice() != [0, 3]
        || !matches!(invalid.argwhere(), Err(Error::InvalidParameter))
        || queue.launches != backend.queue_profile().map_err(|e| format!("{e}"))?.launches
    {
        return Err(String::from("empty argwhere launch or dtype validation"));
    }
    super::dispatch::resident(backend, &before, 0)?;
    crate::report(format_args!(
        "NVIDIA_BURN_COMPACTION_OK guards=ok malformed_prefix=full_writer queued_lifetime=verified argwhere_reads=1 argwhere_read_bytes=4 mask_select_reads=2 empty_input=no_launch resident=true"
    ));
    Ok(())
}
