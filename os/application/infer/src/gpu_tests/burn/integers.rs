//! Independent exact references for resident reductions and index generation.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Shape, TensorData, TensorMetadata},
        tensor::{Int, Tensor},
    },
    native::{NativeTensor, layout::Layout, range::RangePlan},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, reduction::Reduction};

type Result<T> = core::result::Result<T, String>;

fn reference(values: &[i32], shape: &[usize], axis: usize, operation: Reduction) -> Vec<i32> {
    let width = shape[axis];
    let suffix: usize = shape[axis + 1..].iter().product();
    let count: usize = shape.iter().enumerate().filter(|&(a, _)| a != axis).map(|(_, d)| d).product();
    (0..count)
        .map(|row| {
            let base = row / suffix * width * suffix + row % suffix;
            let value = |k| values[base + k * suffix];
            match operation {
                Reduction::Sum | Reduction::Mean => {
                    let sum = (0..width).fold(0i32, |sum, k| sum.wrapping_add(value(k)));
                    if operation == Reduction::Mean { sum / width as i32 } else { sum }
                }
                Reduction::Product => (0..width).fold(1i32, |product, k| product.wrapping_mul(value(k))),
                _ => {
                    let minimum = matches!(operation, Reduction::Min | Reduction::ArgMin);
                    let mut winner = 0;
                    for k in 1..width {
                        if (minimum && value(k) < value(winner)) || (!minimum && value(k) > value(winner)) {
                            winner = k;
                        }
                    }
                    if operation.index() { winner as i32 } else { value(winner) }
                }
            }
        })
        .collect()
}

fn check(output: NativeTensor, expected: &[i32]) -> Result<()> {
    let data = output.read().map_err(|e| format!("I32 read: {e}"))?;
    if output.dtype() != DType::I32 || data.as_slice::<i32>().map_err(|e| format!("I32 data: {e:?}"))? != expected {
        return Err(format!("I32 reference mismatch, shape={:?}", output.shape()));
    }
    Ok(())
}

fn resident(backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS]) -> Result<()> {
    let after = backend.service_profile().map_err(|e| format!("I32 profile: {e}"))?;
    if [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(String::from("I32 operation transferred or cleared tensor data"));
    }
    Ok(())
}

fn reductions(backend: &Nvidia) -> Result<()> {
    let edges = [i32::MIN, i32::MAX, -16_777_217, 16_777_217, 0, 1, -1, 3, -7];
    let values: Vec<_> = (0..3 * 259 * 5)
        .map(|i| {
            if i % 3 == 0 {
                edges[i / 3 % edges.len()]
            } else {
                (i as i32).wrapping_mul(0x1357_9bdf)
            }
        })
        .collect();
    let source = NativeTensor::from_data(TensorData::new(values.clone(), [3, 259, 5])).map_err(|e| format!("I32 upload: {e}"))?;
    let input = source
        .clone()
        .flip(&[0, 1, 2])
        .and_then(|t| t.permute(&[2, 0, 1]))
        .map_err(|e| format!("I32 view: {e}"))?;
    let ordered: Vec<_> = (0..5)
        .flat_map(|i| (0..3).flat_map(move |j| (0..259).map(move |k| ((2 - j) * 259 + 258 - k) * 5 + 4 - i)))
        .map(|index| values[index])
        .collect();
    let before = backend.service_profile().map_err(|e| format!("I32 profile: {e}"))?;
    let mut outputs = Vec::new();
    backend
        .with_commands(|| {
            for axis in 0..3 {
                for operation in Reduction::ALL {
                    outputs.push((axis, operation, input.reduce(axis, operation)?));
                }
            }
            Ok::<_, Error>(())
        })
        .map_err(|e| format!("I32 reduction: {e}"))?;
    resident(backend, &before)?;
    for (axis, operation, output) in outputs {
        let mut shape = [5, 3, 259];
        shape[axis] = 1;
        if output.shape().as_slice() != shape {
            return Err(String::from("I32 reduction shape"));
        }
        check(output, &reference(&ordered, &[5, 3, 259], axis, operation))?;
    }
    check(source, &values)?;

    // More than two lane groups contain equal extrema. First logical indices
    // must win even when every value equals the initial reduction identity.
    let values: Vec<_> = [i32::MIN, i32::MAX, -1, 16_777_217].into_iter().flat_map(|v| vec![v; 259]).collect();
    let source = NativeTensor::from_data(TensorData::new(values.clone(), [4, 259])).map_err(|e| format!("I32 ties: {e}"))?;
    for operation in Reduction::ALL {
        check(
            source.reduce(1, operation).map_err(|e| format!("I32 ties: {e}"))?,
            &reference(&values, &[4, 259], 1, operation),
        )?;
    }
    let broadcast = NativeTensor::from_data(TensorData::new(vec![i32::MIN, i32::MAX], [2, 1]))
        .and_then(|t| t.expand(Shape::new([2, 259])))
        .map_err(|e| format!("I32 broadcast: {e}"))?;
    let values: Vec<_> = vec![i32::MIN; 259].into_iter().chain(vec![i32::MAX; 259]).collect();
    for operation in Reduction::ALL {
        check(
            broadcast.reduce(1, operation).map_err(|e| format!("I32 broadcast: {e}"))?,
            &reference(&values, &[2, 259], 1, operation),
        )?;
    }
    let empty = NativeTensor::empty(Shape::new([2, 0, 3]), DType::I32).map_err(|e| format!("I32 empty: {e}"))?;
    check(empty.reduce(1, Reduction::Sum).map_err(|e| format!("I32 empty sum: {e}"))?, &[0; 6])?;
    check(empty.reduce(1, Reduction::Product).map_err(|e| format!("I32 empty product: {e}"))?, &[1; 6])?;
    let before = backend.queue_profile().map_err(|e| format!("I32 queue: {e}"))?;
    for operation in [Reduction::Mean, Reduction::Min, Reduction::Max, Reduction::ArgMin, Reduction::ArgMax] {
        if !matches!(empty.reduce(1, operation), Err(Error::InvalidParameter)) {
            return Err(String::from("I32 empty reduction accepted"));
        }
    }
    check(empty.reduce(0, Reduction::Sum).map_err(|e| format!("I32 empty output: {e}"))?, &[])?;
    if backend.queue_profile().map_err(|e| format!("I32 queue: {e}"))?.launches != before.launches {
        return Err(String::from("I32 empty work launched"));
    }
    Ok(())
}

fn guarded(backend: &Nvidia) -> Result<()> {
    let verify = || -> d3os_compute::Result<()> {
        let guard = 0x6bad_beefu32;
        let mut words = vec![guard];
        words.extend((0i32..3 * 259).map(|i| i.wrapping_mul(16_777_217) as u32));
        words.push(guard);
        let input = backend.upload_words(&words)?;
        let output = backend.upload_words(&[guard; 5])?;
        let layout = Layout::contiguous(Shape::new([3, 259]))?.flip(&[1])?;
        for operation in Reduction::INTEGERS {
            let parameters = layout.reduction_parameters(1, operation)?;
            backend.dispatch_words(abi::Kernel::TensorReduction, &[(&input, 1), (&output, 1)], &parameters)?;
            let mut actual = [0; 5];
            backend.read_words_into(&output, 0, &mut actual)?;
            let values: Vec<_> = words[1..words.len() - 1]
                .chunks_exact(259)
                .flat_map(|row| row.iter().rev().map(|v| *v as i32))
                .collect();
            let float_operation = Reduction::ALL[operation as usize - 7];
            let expected = reference(&values, &[3, 259], 1, float_operation);
            if actual[0] != guard || actual[4] != guard || actual[1..4].iter().map(|&v| v as i32).ne(expected) {
                return Err(Error::DeviceFault);
            }
        }
        let mut untouched = vec![0; words.len()];
        backend.read_words_into(&input, 0, &mut untouched)?;
        if untouched != words {
            return Err(Error::DeviceFault);
        }
        let output = backend.upload_words(&vec![guard; 261])?;
        backend.dispatch_words(abi::Kernel::Range, &[(&output, 1)], &[259, (-31i32) as u32 as usize, 3])?;
        let mut actual = vec![0; 261];
        backend.read_words_into(&output, 0, &mut actual)?;
        if actual[0] != guard || actual[260] != guard || actual[1..260].iter().enumerate().any(|(i, &v)| v as i32 != -31 + i as i32 * 3) {
            return Err(Error::DeviceFault);
        }
        if backend.dispatch_words(abi::Kernel::Range, &[(&output, 1)], &[259, i32::MAX as usize, 1]) != Err(Error::InvalidParameter) {
            return Err(Error::DeviceFault);
        }
        let mut unchanged = vec![0; 261];
        backend.read_words_into(&output, 0, &mut unchanged)?;
        if unchanged != actual {
            return Err(Error::DeviceFault);
        }
        Ok(())
    };
    verify().map_err(|e| format!("I32 guards: {e}"))
}

fn public(backend: &Nvidia) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("I32 device: {e}"))?;
    let source = Tensor::<2, Int>::from_ints([[1, 2], [3, 4]], &device).transpose();
    let truncation = Tensor::<2, Int>::from_ints([[0, 0], [1, 3]], &device).transpose();
    let before = backend.service_profile().map_err(|e| format!("I32 profile: {e}"))?;
    let truncated_mean = truncation.mean();
    let indices = Tensor::<1, Int>::arange_step(-31..746, 3, &device);
    let mean = source.clone().mean();
    let sum = source.clone().sum();
    let min = source.clone().min_dim(1);
    let max = source.clone().max_dim(1);
    let argmin = source.clone().argmin(1);
    let argmax = source.clone().argmax(1);
    resident(backend, &before)?;
    let read = |tensor: Tensor<1, Int>, expected: &[i32]| -> Result<()> {
        let actual = tensor.try_into_vec_as::<i32>().map_err(|e| format!("I32 public read: {e:?}"))?;
        if actual != expected {
            return Err(format!("I32 public reference: {actual:?} != {expected:?}"));
        }
        Ok(())
    };
    read(indices, &(0..259).map(|i| -31 + i * 3).collect::<Vec<i32>>())?;
    read(mean, &[2])?;
    read(truncated_mean, &[1])?;
    read(sum, &[10])?;
    read(min.reshape([2]), &[1, 2])?;
    read(max.reshape([2]), &[3, 4])?;
    read(argmin.reshape([2]), &[0, 0])?;
    read(argmax.reshape([2]), &[1, 1])?;
    for (range, step) in [
        (-16_777_217..-16_776_430, 3),
        (i64::from(i32::MIN)..i64::from(i32::MAX) + 1, u32::MAX as usize),
        (17..18, usize::MAX),
        (core::ops::Range { start: 5, end: -2 }, 1),
    ] {
        let expected: Vec<i32> = range.clone().step_by(step).map(|v| v as i32).collect();
        let plan = RangePlan::new(range, step).map_err(|e| format!("I32 range: {e}"))?;
        check(NativeTensor::arange(plan).map_err(|e| format!("I32 range: {e}"))?, &expected)?;
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<()> {
    reductions(backend)?;
    guarded(backend)?;
    public(backend)?;
    crate::report(format_args!(
        "NVIDIA_BURN_INTEGER_REDUCTION_OK operations=7 views=strided_broadcast wrapping=exact ties=first range=resident guards=ok backend=nvidia"
    ));
    Ok(())
}
