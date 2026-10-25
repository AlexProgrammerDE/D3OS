//! Raw guarded FP32 floor remainder, residency and queued ownership checks.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{DType, Shape, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
    validation::remainder::{self as reference, Domain},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, elementwise::Elementwise};

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    const GUARD: u32 = 0x6bad_beef;
    let (all_a, all_b) = reference::pairs(Domain::Ieee);
    for count in [1, 127, 128, 129, 257, 411, all_a.len()] {
        let a: Vec<_> = (0..count).map(|i| all_a[(i * 131) % all_a.len()]).collect();
        let b: Vec<_> = (0..count).map(|i| all_b[(i * 131) % all_b.len()]).collect();
        let guarded = |values: &[f32]| {
            let mut words = vec![GUARD];
            words.extend(values.iter().map(|v| v.to_bits()));
            words.push(GUARD);
            words
        };
        let ga = guarded(&a);
        let gb = guarded(&b);
        let lhs = backend.upload_words(&ga).map_err(|e| format!("{e}"))?;
        let rhs = backend.upload_words(&gb).map_err(|e| format!("{e}"))?;
        let output = backend.upload_words(&vec![GUARD; count + 2]).map_err(|e| format!("{e}"))?;
        let layout = Layout::contiguous(Shape::new([count])).and_then(|l| l.flip(&[0])).map_err(|e| format!("{e}"))?;
        let parameters = layout
            .elementwise_parameters(&layout, Elementwise::Remainder, 0, false)
            .map_err(|e| format!("{e}"))?;
        backend
            .with_commands(|| backend.dispatch_words(Kernel::TensorElementwise, &[(&lhs, 1), (&rhs, 1), (&output, 1)], &parameters))
            .map_err(|e| format!("{e}"))?;
        let mut actual = vec![0; count + 2];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual[0] != GUARD || actual[count + 1] != GUARD {
            return Err(String::from("remainder output guard"));
        }
        let ea: Vec<_> = a.iter().copied().rev().collect();
        let eb: Vec<_> = b.iter().copied().rev().collect();
        reference::check(&actual[1..count + 1].iter().copied().map(f32::from_bits).collect::<Vec<_>>(), &ea, &eb, true)?;
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let result = backend
            .with_commands(|| backend.produce_words(count, Kernel::TensorElementwise, &[(&lhs, 1), (&rhs, 1)], &parameters))
            .map_err(|e| format!("{e}"))?;
        super::dispatch::resident(backend, &before, 0)?;
        let after = backend.service_profile().map_err(|e| format!("{e}"))?;
        if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
            || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
        {
            return Err(String::from("remainder full writer allocated a separate clear"));
        }
        let mut result_words = vec![0; count];
        backend.read_words_into(&result, 0, &mut result_words).map_err(|e| format!("{e}"))?;
        reference::check(&result_words.into_iter().map(f32::from_bits).collect::<Vec<_>>(), &ea, &eb, true)?;
        for (input, original) in [(&lhs, &ga), (&rhs, &gb)] {
            let mut words = vec![0; count + 2];
            backend.read_words_into(input, 0, &mut words).map_err(|e| format!("{e}"))?;
            if &words != original {
                return Err(String::from("remainder changed guarded input"));
            }
        }
    }
    let a = NativeTensor::from_data(TensorData::new(vec![-5.375f32, 3.125], [2])).map_err(|e| format!("{e}"))?;
    let b = NativeTensor::from_data(TensorData::new(vec![1.5f32, -2.5], [2])).map_err(|e| format!("{e}"))?;
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let survivor = backend
        .with_commands(|| {
            let output = a.binary(&b, Elementwise::Remainder)?;
            drop(a);
            drop(b);
            Ok::<_, Error>(output)
        })
        .map_err(|e| format!("{e}"))?;
    super::dispatch::resident(backend, &before, 0)?;
    reference::check(
        survivor.read().map_err(|e| format!("{e}"))?.as_slice::<f32>().map_err(|e| format!("{e:?}"))?,
        &[-5.375, 3.125],
        &[1.5, -2.5],
        true,
    )?;
    let scalar = NativeTensor::from_data(TensorData::new(vec![-5.375f32], Shape::new([]))).map_err(|e| format!("{e}"))?;
    let scalar = scalar.scalar(Elementwise::Remainder, 1.5).map_err(|e| format!("{e}"))?;
    if scalar.shape().num_dims() != 0 {
        return Err(String::from("remainder scalar rank"));
    }
    reference::check(
        scalar.read().map_err(|e| format!("{e}"))?.as_slice::<f32>().map_err(|e| format!("{e:?}"))?,
        &[-5.375],
        &[1.5],
        true,
    )?;
    let empty = NativeTensor::full(Shape::new([0, 3]), DType::F32, 0).map_err(|e| format!("{e}"))?;
    let wrong_type = NativeTensor::from_data(TensorData::new(vec![1i32], [1])).map_err(|e| format!("{e}"))?;
    let wrong_shape = NativeTensor::from_data(TensorData::new(vec![1f32, 2.0], [2])).map_err(|e| format!("{e}"))?;
    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
    if !empty.binary(&empty, Elementwise::Remainder).map_err(|e| format!("{e}"))?.layout().is_empty()
        || !empty.scalar(Elementwise::Remainder, 0.0).map_err(|e| format!("{e}"))?.layout().is_empty()
        || !matches!(empty.binary(&wrong_type, Elementwise::Remainder), Err(Error::InvalidParameter))
        || !matches!(empty.binary(&wrong_shape, Elementwise::Remainder), Err(Error::ShapeMismatch))
        || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
    {
        return Err(String::from("remainder empty or invalid input launched work"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_REMAINDER_OK pairs={} guards=ok finite_bits=exact zero_sign=dividend full_writer=verified resident=true queued_lifetime=verified scalar=true invalid_inputs=no_launch empty=no_launch",
        all_a.len()
    ));
    Ok(())
}
