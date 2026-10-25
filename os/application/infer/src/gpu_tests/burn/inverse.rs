//! Guarded atan2 execution. The common math suite checks all six unary entries.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{DType, Shape, TensorData, TensorMetadata},
    native::{NativeTensor, layout::Layout},
    validation::inverse::{self as reference, Domain},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, elementwise::Elementwise};
fn check(actual: &[f32], y: &[f32], x: &[f32]) -> Result<(), String> {
    reference::check(
        actual,
        &y.iter().zip(x).map(|(&y, &x)| reference::atan2_reference(y, x)).collect::<Vec<_>>(),
        "guarded atan2",
    )
}
pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    const GUARD: u32 = 0x6bad_beef;
    let (ys, xs) = reference::atan2_pairs(Domain::Ieee);
    for count in [1, 127, 128, 129, 257, 411, ys.len()] {
        let y: Vec<_> = (0..count).map(|i| ys[(i * 131) % ys.len()]).collect();
        let x: Vec<_> = (0..count).map(|i| xs[(i * 131) % xs.len()]).collect();
        let guarded = |v: &[f32]| {
            let mut words = vec![GUARD];
            words.extend(v.iter().map(|v| v.to_bits()));
            words.push(GUARD);
            words
        };
        let gy = guarded(&y);
        let gx = guarded(&x);
        let a = backend.upload_words(&gy).map_err(|e| format!("{e}"))?;
        let b = backend.upload_words(&gx).map_err(|e| format!("{e}"))?;
        let output = backend.upload_words(&vec![GUARD; count + 2]).map_err(|e| format!("{e}"))?;
        let layout = Layout::contiguous(Shape::new([count])).and_then(|l| l.flip(&[0])).map_err(|e| format!("{e}"))?;
        let p = layout
            .elementwise_parameters(&layout, Elementwise::Atan2, 0, false)
            .map_err(|e| format!("{e}"))?;
        backend
            .with_commands(|| backend.dispatch_words(Kernel::TensorElementwise, &[(&a, 1), (&b, 1), (&output, 1)], &p))
            .map_err(|e| format!("{e}"))?;
        let mut actual = vec![0; count + 2];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual[0] != GUARD || actual[count + 1] != GUARD {
            return Err(String::from("atan2 output guard"));
        }
        let ey: Vec<_> = y.iter().copied().rev().collect();
        let ex: Vec<_> = x.iter().copied().rev().collect();
        check(&actual[1..count + 1].iter().copied().map(f32::from_bits).collect::<Vec<_>>(), &ey, &ex)?;
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let result = backend
            .with_commands(|| backend.produce_words(count, Kernel::TensorElementwise, &[(&a, 1), (&b, 1)], &p))
            .map_err(|e| format!("{e}"))?;
        super::dispatch::resident(backend, &before, 0)?;
        let after = backend.service_profile().map_err(|e| format!("{e}"))?;
        if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
            || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
        {
            return Err(String::from("atan2 full writer cleared output separately"));
        }
        let mut words = vec![0; count];
        backend.read_words_into(&result, 0, &mut words).map_err(|e| format!("{e}"))?;
        check(&words.into_iter().map(f32::from_bits).collect::<Vec<_>>(), &ey, &ex)?;
        for (input, expected) in [(&a, &gy), (&b, &gx)] {
            let mut words = vec![0; count + 2];
            backend.read_words_into(input, 0, &mut words).map_err(|e| format!("{e}"))?;
            if &words != expected {
                return Err(String::from("atan2 changed guarded source"));
            }
        }
    }
    let a = NativeTensor::from_data(TensorData::new(vec![-0.5f32, 2.0], [2])).map_err(|e| format!("{e}"))?;
    let b = NativeTensor::from_data(TensorData::new(vec![-3f32, 0.75], [2])).map_err(|e| format!("{e}"))?;
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let output = backend
        .with_commands(|| {
            let result = a.binary(&b, Elementwise::Atan2)?;
            drop(a);
            drop(b);
            Ok::<_, Error>(result)
        })
        .map_err(|e| format!("{e}"))?;
    super::dispatch::resident(backend, &before, 0)?;
    check(
        output.read().map_err(|e| format!("{e}"))?.as_slice::<f32>().map_err(|e| format!("{e:?}"))?,
        &[-0.5, 2.0],
        &[-3.0, 0.75],
    )?;
    let a = NativeTensor::from_data(TensorData::new(vec![-0.5f32], Shape::new([]))).map_err(|e| format!("{e}"))?;
    let b = NativeTensor::from_data(TensorData::new(vec![-3f32], Shape::new([]))).map_err(|e| format!("{e}"))?;
    let out = a.binary(&b, Elementwise::Atan2).map_err(|e| format!("{e}"))?;
    if out.shape().num_dims() != 0 {
        return Err(String::from("atan2 scalar rank"));
    }
    check(
        out.read().map_err(|e| format!("{e}"))?.as_slice::<f32>().map_err(|e| format!("{e:?}"))?,
        &[-0.5],
        &[-3.0],
    )?;
    let empty = NativeTensor::full(Shape::new([0, 3]), DType::F32, 0).map_err(|e| format!("{e}"))?;
    let wrong = NativeTensor::from_data(TensorData::new(vec![1i32], [1])).map_err(|e| format!("{e}"))?;
    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
    if !empty.binary(&empty, Elementwise::Atan2).map_err(|e| format!("{e}"))?.layout().is_empty()
        || !matches!(empty.binary(&wrong, Elementwise::Atan2), Err(Error::InvalidParameter))
        || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
    {
        return Err(String::from("empty or invalid atan2 launched"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_INVERSE_OK atan2_pairs={} guards=ok ulp=8 full_writer=verified resident=true queued_lifetime=verified scalar=true invalid_inputs=no_launch empty=no_launch",
        ys.len()
    ));
    Ok(())
}
