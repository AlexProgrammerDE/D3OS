//! Guard spatial windows/folds independently of the matrix products using them.
use alloc::sync::Arc;
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{
        Shape,
        ops::{ConvOptions, ConvTransposeOptions},
    },
    native::{layout::Layout, spatial_convolution::SpatialConvolution},
    validation::spatial_convolution::{self as reference, Case},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, spatial_convolution as spatial};
fn plan(case: &Case) -> Result<SpatialConvolution, Error> {
    if !case.transpose {
        SpatialConvolution::conv3d(
            Shape::from(case.input.clone()),
            Shape::from(case.weights.clone()),
            &ConvOptions::new(
                case.stride.clone().try_into().unwrap(),
                case.padding.clone().try_into().unwrap(),
                case.dilation.clone().try_into().unwrap(),
                case.groups,
            ),
        )
    } else if case.input.len() == 3 {
        SpatialConvolution::transpose(
            Shape::from(case.input.clone()),
            Shape::from(case.weights.clone()),
            &ConvTransposeOptions::<1>::new(
                case.stride.clone().try_into().unwrap(),
                case.padding.clone().try_into().unwrap(),
                case.extra.clone().try_into().unwrap(),
                case.dilation.clone().try_into().unwrap(),
                case.groups,
            ),
        )
    } else if case.input.len() == 4 {
        SpatialConvolution::transpose(
            Shape::from(case.input.clone()),
            Shape::from(case.weights.clone()),
            &ConvTransposeOptions::<2>::new(
                case.stride.clone().try_into().unwrap(),
                case.padding.clone().try_into().unwrap(),
                case.extra.clone().try_into().unwrap(),
                case.dilation.clone().try_into().unwrap(),
                case.groups,
            ),
        )
    } else {
        SpatialConvolution::transpose(
            Shape::from(case.input.clone()),
            Shape::from(case.weights.clone()),
            &ConvTransposeOptions::<3>::new(
                case.stride.clone().try_into().unwrap(),
                case.padding.clone().try_into().unwrap(),
                case.extra.clone().try_into().unwrap(),
                case.dilation.clone().try_into().unwrap(),
                case.groups,
            ),
        )
    }
}
pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    const GUARD: u32 = 0x6bad_beef;
    let mut count = 0;
    for case in reference::cases() {
        for broadcast in [false, true] {
            let plan = plan(&case).map_err(|e| format!("{e}"))?;
            let image = if case.transpose { case.output() } else { case.input.clone() };
            let mut stored = image.clone();
            if broadcast {
                stored[0] = 1;
                stored[1] = 1;
            }
            let physical = reference::values(&stored, 11);
            let mut guarded = vec![GUARD];
            guarded.extend(physical.iter().map(|v| v.to_bits()));
            guarded.push(GUARD);
            let source = Arc::new(backend.upload_words(&guarded).map_err(|e| format!("{e}"))?);
            let layout = Layout::contiguous(Shape::from(stored))
                .and_then(|l| l.expand(Shape::from(image.clone())))
                .and_then(|l| l.flip(&(0..image.len()).collect::<Vec<_>>()))
                .map_err(|e| format!("{e}"))?;
            let p = plan.windows_parameters(&layout).map_err(|e| format!("{e}"))?;
            let p32 = p.map(|v| v as u32);
            let geometry = spatial::geometry(&p32[..25]).map_err(|e| format!("{e:?}"))?;
            let mut columns = vec![0f32; geometry.columns as usize];
            let input_columns = reference::values(&[geometry.columns as usize], 13);
            let mut folded = vec![0f64; geometry.images as usize];
            let dim = [p[0], p[1], p[2], p[3], p[4]];
            let groups = p[21] & 0x7fffffff;
            let ci = p[1] / groups;
            let inner = ci * p[6] * p[7] * p[8];
            let spatial = p[9] * p[10] * p[11];
            for (i, column) in columns.iter_mut().enumerate() {
                let s = i % spatial;
                let k = i / spatial % inner;
                let group = i / (spatial * inner) % groups;
                let n = i / (spatial * inner * groups);
                let c = group * ci + k / (p[6] * p[7] * p[8]);
                let coords = [s / (p[10] * p[11]), s / p[11] % p[10], s % p[11]];
                let taps = [k / (p[7] * p[8]) % p[6], k / p[8] % p[7], k % p[8]];
                let mut at = p[26] as i64 + n as i64 * (p[27] as u32 as i32 as i64) + c as i64 * (p[28] as u32 as i32 as i64);
                let mut dense = (n * dim[1] + c) * dim[2] * dim[3] * dim[4];
                let mut valid = true;
                for axis in 0..3 {
                    let v = (coords[axis] * p[12 + axis] + taps[axis] * p[18 + axis]) as i64 - p[15 + axis] as i64;
                    if v < 0 || v >= dim[2 + axis] as i64 {
                        valid = false;
                        break;
                    }
                    at += v * (p[29 + axis] as u32 as i32 as i64);
                    dense += v as usize * dim[3 + axis..].iter().product::<usize>();
                }
                if valid {
                    *column = physical[at as usize];
                    folded[dense] += f64::from(input_columns[i]);
                }
            }
            for (kernel, values, expected, parameters, offset) in [
                (
                    Kernel::SpatialWindows,
                    guarded.clone(),
                    columns.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                    p.to_vec(),
                    1,
                ),
                (
                    Kernel::SpatialFold,
                    {
                        let mut v = vec![GUARD];
                        v.extend(input_columns.iter().map(|v| v.to_bits()));
                        v.push(GUARD);
                        v
                    },
                    folded,
                    plan.fold_parameters().to_vec(),
                    1,
                ),
            ] {
                let source = if kernel == Kernel::SpatialWindows {
                    source.clone()
                } else {
                    Arc::new(backend.upload_words(&values).map_err(|e| format!("{e}"))?)
                };
                let output = backend.upload_words(&vec![GUARD; expected.len() + 2]).map_err(|e| format!("{e}"))?;
                backend
                    .with_commands(|| backend.dispatch_words(kernel, &[(&source, offset), (&output, 1)], &parameters))
                    .map_err(|e| format!("{e}"))?;
                let mut actual = vec![0; expected.len() + 2];
                backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
                if actual[0] != GUARD || actual[expected.len() + 1] != GUARD {
                    return Err(String::from("spatial output guard"));
                }
                reference::compare(
                    &actual[1..expected.len() + 1].iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
                    &expected,
                    "guarded spatial",
                )?;
                let before = backend.service_profile().map_err(|e| format!("{e}"))?;
                let full = backend
                    .with_commands(|| backend.produce_words(expected.len(), kernel, &[(&source, offset)], &parameters))
                    .map_err(|e| format!("{e}"))?;
                super::dispatch::resident(backend, &before, 0)?;
                let after = backend.service_profile().map_err(|e| format!("{e}"))?;
                if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
                    || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
                {
                    return Err(String::from("spatial writer cleared separately"));
                }
                let mut words = vec![0; expected.len()];
                backend.read_words_into(&full, 0, &mut words).map_err(|e| format!("{e}"))?;
                reference::compare(&words.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(), &expected, "full spatial")?;
                let mut preserved = vec![0; values.len()];
                backend.read_words_into(&source, 0, &mut preserved).map_err(|e| format!("{e}"))?;
                if preserved != values {
                    return Err(String::from("spatial source modified"));
                }
                let temporary = backend.upload_words(&values).map_err(|e| format!("{e}"))?;
                let before = backend.service_profile().map_err(|e| format!("{e}"))?;
                let queued = backend
                    .with_commands(|| {
                        let result = backend.produce_words(expected.len(), kernel, &[(&temporary, offset)], &parameters)?;
                        drop(temporary);
                        Ok::<_, Error>(result)
                    })
                    .map_err(|e| format!("{e}"))?;
                super::dispatch::resident(backend, &before, 0)?;
                let mut queued_words = vec![0; expected.len()];
                backend.read_words_into(&queued, 0, &mut queued_words).map_err(|e| format!("{e}"))?;
                reference::compare(
                    &queued_words.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
                    &expected,
                    "queued spatial",
                )?;
                let mut invalid = parameters.clone();
                invalid[9] += 1;
                let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
                if backend.dispatch_words(kernel, &[(&source, offset), (&output, 1)], &invalid) != Err(Error::InvalidParameter)
                    || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
                {
                    return Err(String::from("spatial invalid geometry launched"));
                }
                count += 1;
            }
        }
    }
    crate::report(format_args!(
        "NVIDIA_BURN_SPATIAL_CONVOLUTION_OK raw={count} guards=ok broadcast=true signed_views=true full_writer=verified resident=true queued_lifetime=verified invalid_geometry=no_launch"
    ));
    Ok(())
}
