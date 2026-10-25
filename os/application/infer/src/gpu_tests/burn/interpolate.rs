//! Guarded separable axes, transpose filters, broadcast reads and residency.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{DType, Shape, TensorData, TensorMetadata},
    native::{NativeTensor, interpolate::Interpolation, layout::Layout},
    validation::interpolate::{self as reference, MODES},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel};
fn check(actual: &[u32], expected: &[f32], exact: bool) -> Result<(), String> {
    reference::check(
        &actual.iter().copied().map(f32::from_bits).collect::<Vec<_>>(),
        expected,
        exact,
        "guarded interpolation",
    )
}
pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    const GUARD: u32 = 0x6bad_beef;
    let shape = [2, 3, 3, 137];
    let mut cases = 0;
    for mode in MODES {
        for axis in [2, 3] {
            for backward in [false, true] {
                for broadcast in [false, true] {
                    let size = if axis == 2 { [5, 137] } else { [3, 129] };
                    let plan = Interpolation::new(Shape::new(shape), size, &mode.options(size, false)).map_err(|e| format!("{e}"))?;
                    let logical_shape = if backward { [2, 3, size[0], size[1]] } else { shape };
                    let stored_shape = if broadcast {
                        [1, 1, logical_shape[2], logical_shape[3]]
                    } else {
                        logical_shape
                    };
                    let values = if backward {
                        reference::gradient(stored_shape)
                    } else {
                        reference::source(stored_shape)
                    };
                    let mut guarded = vec![GUARD];
                    guarded.extend(values.iter().map(|v| v.to_bits()));
                    guarded.push(GUARD);
                    let source = backend.upload_words(&guarded).map_err(|e| format!("{e}"))?;
                    let layout = Layout::contiguous(Shape::new(stored_shape))
                        .and_then(|l| l.expand(Shape::new(logical_shape)))
                        .and_then(|l| l.flip(&[0, 1, 2, 3]))
                        .map_err(|e| format!("{e}"))?;
                    let p = plan.parameters(&layout, axis, backward).map_err(|e| format!("{e}"))?;
                    let logical: Vec<_> = if broadcast { values.repeat(6) } else { values.clone() }.into_iter().rev().collect();
                    let expected = if backward {
                        reference::backward(&logical, shape, size, mode, false).into_iter().map(|v| v as f32).collect()
                    } else {
                        reference::forward(&logical, shape, size, mode, false)
                    };
                    let output = backend.upload_words(&vec![GUARD; expected.len() + 2]).map_err(|e| format!("{e}"))?;
                    backend
                        .with_commands(|| backend.dispatch_words(Kernel::Interpolate, &[(&source, 1), (&output, 1)], &p))
                        .map_err(|e| format!("{e}"))?;
                    let mut actual = vec![0; expected.len() + 2];
                    backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
                    if actual[0] != GUARD || actual[expected.len() + 1] != GUARD {
                        return Err(String::from("interpolation output guard"));
                    }
                    check(&actual[1..expected.len() + 1], &expected, mode.nearest())?;
                    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
                    let full = backend
                        .with_commands(|| backend.produce_words(expected.len(), Kernel::Interpolate, &[(&source, 1)], &p))
                        .map_err(|e| format!("{e}"))?;
                    super::dispatch::resident(backend, &before, 0)?;
                    let after = backend.service_profile().map_err(|e| format!("{e}"))?;
                    if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
                        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
                    {
                        return Err(String::from("interpolation full writer cleared separately"));
                    }
                    let mut words = vec![0; expected.len()];
                    backend.read_words_into(&full, 0, &mut words).map_err(|e| format!("{e}"))?;
                    check(&words, &expected, mode.nearest())?;
                    let mut preserved = vec![0; guarded.len()];
                    backend.read_words_into(&source, 0, &mut preserved).map_err(|e| format!("{e}"))?;
                    if preserved != guarded {
                        return Err(String::from("interpolation changed guarded source"));
                    }
                    let mut invalid = p;
                    invalid[15] ^= 1;
                    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
                    if backend.dispatch_words(Kernel::Interpolate, &[(&source, 1), (&output, 1)], &invalid) != Err(Error::InvalidParameter)
                        || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
                    {
                        return Err(String::from("interpolation accepted malformed rational transform"));
                    }
                    cases += 1;
                }
            }
        }
    }
    for mode in MODES {
        let options = mode.options([7, 9], true);
        let source = NativeTensor::from_data(TensorData::new(reference::source([1, 1, 3, 5]), [1, 1, 3, 5])).map_err(|e| format!("{e}"))?;
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let survivor = backend
            .with_commands(|| {
                let result = source.interpolate([7, 9], &options)?;
                drop(source);
                Ok::<_, Error>(result)
            })
            .map_err(|e| format!("{e}"))?;
        super::dispatch::resident(backend, &before, 0)?;
        reference::check(
            survivor.read().map_err(|e| format!("{e}"))?.as_slice::<f32>().map_err(|e| format!("{e:?}"))?,
            &reference::forward(&reference::source([1, 1, 3, 5]), [1, 1, 3, 5], [7, 9], mode, true),
            mode.nearest(),
            "interpolation queued lifetime",
        )?;
        for shape in [[0, 2, 3, 5], [2, 0, 3, 5]] {
            let empty = NativeTensor::full(Shape::new(shape), DType::F32, 0).map_err(|e| format!("{e}"))?;
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let out = empty.interpolate([7, 9], &options).map_err(|e| format!("{e}"))?;
            let back = empty.interpolate_backward(&out, [7, 9], &options).map_err(|e| format!("{e}"))?;
            let after = backend.service_profile().map_err(|e| format!("{e}"))?;
            if out.shape().as_slice() != [shape[0], shape[1], 7, 9]
                || back.shape().as_slice() != shape
                || [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD, abi::LAUNCH, abi::LAUNCH_ALLOCATE]
                    .iter()
                    .any(|&op| before[op as usize].calls != after[op as usize].calls)
            {
                return Err(String::from("empty interpolation shape, allocation or launch"));
            }
        }
        let a = NativeTensor::from_data(TensorData::new(reference::source([1, 1, 3, 5]), [1, 1, 3, 5])).map_err(|e| format!("{e}"))?;
        let wrong = NativeTensor::from_data(TensorData::new(vec![1i32; 15], [1, 1, 3, 5])).map_err(|e| format!("{e}"))?;
        let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
        if !matches!(a.interpolate([0, 9], &options), Err(Error::ShapeMismatch))
            || !matches!(wrong.interpolate([7, 9], &options), Err(Error::InvalidParameter))
            || !matches!(a.interpolate_backward(&a, [7, 9], &options), Err(Error::ShapeMismatch))
            || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
        {
            return Err(String::from("invalid interpolation launched work"));
        }
    }
    crate::report(format_args!(
        "NVIDIA_BURN_INTERPOLATE_OK cases={} modes=5 directions=2 axes=2 guards=ok broadcast_reads=true resident=true full_writer=verified queued_lifetime=verified invalid_inputs=no_launch empty=no_launch",
        cases
    ));
    Ok(())
}
