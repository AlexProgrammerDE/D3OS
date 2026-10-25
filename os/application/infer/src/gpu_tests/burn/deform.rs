//! Guarded dense deformable primitives, including nonfinite coordinates.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::Shape,
    native::deform::DeformConv2d,
    validation::deform::{self as reference, Case},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, deform};
fn guarded(values: &[f32]) -> Vec<u32> {
    let mut words = vec![0x6bad_beef];
    words.extend(values.iter().map(|v| v.to_bits()));
    words.push(0x6bad_beef);
    words
}
fn plan(case: &Case, masked: bool) -> Result<DeformConv2d, Error> {
    DeformConv2d::new(
        Shape::new(case.input),
        Shape::new(case.offset_shape()),
        Shape::new(case.weights),
        masked.then(|| Shape::new(case.mask_shape())).as_ref(),
        None,
        &case.options(),
    )
}
fn primitive_reference(case: &Case, x: &[f32], offset: &[f32], mask: Option<&[f32]>, columns: &[f32]) -> [Vec<f64>; 3] {
    let [n, c, _, _] = case.input;
    let out = case.output();
    let spatial = out[2] * out[3];
    let kernel = case.weights[2] * case.weights[3];
    let mcount = case.mask_shape().iter().product::<usize>();
    let mut window = vec![0.0; columns.len()];
    let mut coordinate = vec![0.0; if mask.is_some() { 3 * mcount } else { 2 * mcount }];
    let mut dx = vec![0.0; x.len()];
    for channel in 0..c {
        for k in 0..kernel {
            for batch in 0..n {
                for s in 0..spatial {
                    let at = ((channel * kernel + k) * n + batch) * spatial + s;
                    let og = channel / (c / case.offset_groups);
                    let mi = ((batch * case.offset_groups + og) * kernel + k) * spatial + s;
                    let oi = (mi / spatial) * 2 * spatial + s;
                    let y = (s / out[3] * case.stride[0] + k / case.weights[3] * case.dilation[0]) as f64 - case.padding[0] as f64 + f64::from(offset[oi]);
                    let xx = (s % out[3] * case.stride[1] + k % case.weights[3] * case.dilation[1]) as f64 - case.padding[1] as f64
                        + f64::from(offset[oi + spatial]);
                    let (v, gy, gx, pixels) = reference::sample(x, case.input, batch, channel, y, xx);
                    let m = mask.map_or(1.0, |m| f64::from(m[mi]));
                    let g = f64::from(columns[at]);
                    window[at] = v * m;
                    coordinate[oi] += g * gy * m;
                    coordinate[oi + spatial] += g * gx * m;
                    if mask.is_some() {
                        coordinate[2 * mcount + mi] += g * v;
                    }
                    for (i, w) in pixels {
                        dx[i] += g * w * m;
                    }
                }
            }
        }
    }
    [window, coordinate, dx]
}
pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let mut count = 0;
    for case in reference::cases() {
        for masked in [false, true] {
            for nonfinite in [false, true] {
                let plan = plan(&case, masked).map_err(|e| format!("{e}"))?;
                let p = plan.parameters();
                let g = deform::geometry(&p.map(|v| v as u32)).map_err(|e| format!("{e:?}"))?;
                let x = reference::values(&case.input, 1, true);
                let mut offset = case.offsets();
                if nonfinite {
                    for (at, value) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY].into_iter().enumerate() {
                        offset[at] = value;
                    }
                }
                let mask = reference::masks(&case.mask_shape());
                let columns = reference::values(&[g.columns as usize], 7, true);
                let expected = primitive_reference(&case, &x, &offset, masked.then_some(mask.as_slice()), &columns);
                for (index, kernel) in [Kernel::DeformWindows, Kernel::DeformCoordinates, Kernel::DeformInput].into_iter().enumerate() {
                    let values = match index {
                        0 => vec![guarded(&x), guarded(&offset), guarded(&mask)],
                        1 => vec![guarded(&x), guarded(&offset), guarded(&mask), guarded(&columns)],
                        _ => vec![guarded(&offset), guarded(&mask), guarded(&columns)],
                    };
                    let sources = values
                        .iter()
                        .map(|v| backend.upload_words(v))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| format!("{e}"))?;
                    let mut words = vec![0; expected[index].len() + 2];
                    words[0] = 0x6bad_beef;
                    let last = words.len() - 1;
                    words[last] = 0x6bad_beef;
                    let output = backend.upload_words(&words).map_err(|e| format!("{e}"))?;
                    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
                    backend
                        .with_commands(|| {
                            let mut pointers = sources.iter().map(|s| (s, 1)).collect::<Vec<_>>();
                            pointers.push((&output, 1));
                            backend.dispatch_words(kernel, &pointers, &p)
                        })
                        .map_err(|e| format!("{e}"))?;
                    super::dispatch::resident(backend, &before, 0)?;
                    backend.read_words_into(&output, 0, &mut words).map_err(|e| format!("{e}"))?;
                    if words[0] != 0x6bad_beef || words[last] != 0x6bad_beef {
                        return Err(String::from("deform output guard"));
                    }
                    reference::compare(
                        &words[1..last].iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
                        &expected[index],
                        &format!("guarded deform {kernel:?} masked={masked} nonfinite={nonfinite}"),
                    )?;
                    for (source, value) in sources.iter().zip(&values) {
                        let mut preserved = vec![0; value.len()];
                        backend.read_words_into(source, 0, &mut preserved).map_err(|e| format!("{e}"))?;
                        if &preserved != value {
                            return Err(String::from("deform source modified"));
                        }
                    }
                    if kernel.allocates_output() {
                        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
                        let full = backend
                            .with_commands(|| backend.produce_words(expected[index].len(), kernel, &sources.iter().map(|s| (s, 1)).collect::<Vec<_>>(), &p))
                            .map_err(|e| format!("{e}"))?;
                        super::dispatch::resident(backend, &before, 0)?;
                        let after = backend.service_profile().map_err(|e| format!("{e}"))?;
                        if after[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
                            || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 1
                        {
                            return Err(String::from("deform full writer cleared separately"));
                        }
                        let mut result = vec![0; expected[index].len()];
                        backend.read_words_into(&full, 0, &mut result).map_err(|e| format!("{e}"))?;
                        reference::compare(&result.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(), &expected[index], "full deform")?;
                    }
                    let temporary = values
                        .iter()
                        .map(|v| backend.upload_words(v))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| format!("{e}"))?;
                    let zero = backend.upload_words(&vec![0; expected[index].len()]).map_err(|e| format!("{e}"))?;
                    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
                    let queued = backend
                        .with_commands(|| {
                            let mut pointers = temporary.iter().map(|s| (s, 1)).collect::<Vec<_>>();
                            let result = if kernel.allocates_output() {
                                backend.produce_words(expected[index].len(), kernel, &pointers, &p)?
                            } else {
                                pointers.push((&zero, 0));
                                backend.dispatch_words(kernel, &pointers, &p)?;
                                zero
                            };
                            drop(temporary);
                            Ok::<_, Error>(result)
                        })
                        .map_err(|e| format!("{e}"))?;
                    super::dispatch::resident(backend, &before, 0)?;
                    let mut result = vec![0; expected[index].len()];
                    backend.read_words_into(&queued, 0, &mut result).map_err(|e| format!("{e}"))?;
                    reference::compare(
                        &result.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(),
                        &expected[index],
                        "queued deform",
                    )?;
                    let mut bad = p;
                    bad[16] = 0;
                    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
                    let mut pointers = sources.iter().map(|s| (s, 1)).collect::<Vec<_>>();
                    pointers.push((&output, 1));
                    if backend.dispatch_words(kernel, &pointers, &bad) != Err(Error::InvalidParameter)
                        || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
                    {
                        return Err(String::from("deform invalid geometry launched"));
                    }
                    count += 1;
                }
            }
        }
    }
    let p = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1];
    let offset = backend.upload_words(&[0, 0]).map_err(|e| format!("{e}"))?;
    let mask = backend.upload_words(&[1f32.to_bits()]).map_err(|e| format!("{e}"))?;
    let columns = backend.upload_words(&[1]).map_err(|e| format!("{e}"))?;
    let output = backend.allocate_words(1).map_err(|e| format!("{e}"))?;
    backend
        .with_commands(|| backend.dispatch_words(Kernel::DeformInput, &[(&offset, 0), (&mask, 0), (&columns, 0), (&output, 0)], &p))
        .map_err(|e| format!("{e}"))?;
    let mut tiny = [0];
    backend.read_words_into(&output, 0, &mut tiny).map_err(|e| format!("{e}"))?;
    if tiny != [1] {
        return Err(format!("deform input gradient flushed subnormal: {tiny:?}"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_DEFORM_OK raw={count} guards=ok full_writers=verified initialized_atomic_output=true subnormal=preserved nonfinite_offsets=zero resident=true queued_lifetime=verified invalid_geometry=no_launch"
    ));
    Ok(())
}
