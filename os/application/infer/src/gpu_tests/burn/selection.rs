//! One-read greedy selection across GPU partitions and logical tensor views.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::tensor::{Tensor, TensorData},
    qwen::{Error, greedy},
};
use d3os_compute::nvidia::Nvidia;
use d3os_nvidia::compute as abi;

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let device = d3os_ai::native_device().map_err(|e| format!("{e}"))?;
    for width in [1, 127, 128, 129, 4096, 4097, 8193, 248320] {
        let mut values: Vec<_> = (0..width + 2).map(|i| (i % 17) as f32 - 19.).collect();
        values[0] = 1e20;
        values[width + 1] = 1e20;
        let first = if width > 4096 { 128 } else { width / 3 };
        let last = if width > 4096 { 4096 } else { width - 1 };
        values[first + 1] = 7.;
        values[last + 1] = 7.;
        let backing = Tensor::<2>::from_data(TensorData::new(values.clone(), [1, width + 2]), &device);
        let input = backing.clone().slice([0..1, 1..width + 1]);
        for reversed in [false, true] {
            let input = if reversed { input.clone().flip([1]) } else { input.clone() };
            let expected = if reversed { width - 1 - last } else { first };
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let kernels = backend.profile().map_err(|e| format!("{e}"))?;
            let selected = greedy(&input).map_err(|e| format!("{e}"))?;
            let after = backend.profile().map_err(|e| format!("{e}"))?;
            super::dispatch::resident(backend, &before, 1)?;
            if selected as usize != expected
                || after[abi::Kernel::ArgMax as usize].calls - kernels[abi::Kernel::ArgMax as usize].calls != 1
                || after[abi::Kernel::TensorReduction as usize].calls != kernels[abi::Kernel::TensorReduction as usize].calls
            {
                return Err(format!(
                    "greedy selection width={width} reversed={reversed} actual={selected} expected={expected}"
                ));
            }
        }
        let preserved = backing.try_into_vec_as::<f32>().map_err(|e| format!("{e:?}"))?;
        if preserved != values {
            return Err(String::from("greedy selection mutated its source or neighboring guards"));
        }
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = vec![-2.; 8193];
        values[128] = 7.;
        values[8192] = invalid;
        let input = Tensor::<2>::from_data(TensorData::new(values, [1, 8193]), &device);
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        if !matches!(greedy(&input), Err(Error::NonFinite)) {
            return Err(String::from("greedy selection accepted a nonfinite losing partition"));
        }
        super::dispatch::resident(backend, &before, 1)?;
        device.sync().map_err(|e| format!("nonfinite selection poisoned the device: {e:?}"))?;
    }
    let failed = super::dispatch::invalid_gather(Tensor::<2>::from_floats([[1., 2., 3.]], &device), &device);
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let kernels = backend.profile().map_err(|e| format!("{e}"))?;
    if !matches!(greedy(&failed), Err(Error::Execution(_))) {
        return Err(String::from("greedy selection lost an input execution error"));
    }
    super::dispatch::resident(backend, &before, 0)?;
    let after = backend.profile().map_err(|e| format!("{e}"))?;
    if kernels.iter().zip(after).any(|(a, b)| a.calls != b.calls) {
        return Err(String::from("failed greedy selection launched work"));
    }
    device
        .sync()
        .map_err(|e| format!("greedy selection did not acknowledge its root error: {e:?}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_SELECTION_OK views=offset,signed ties=first_global nonfinite=rejected source=preserved errors=recoverable read_calls=1 read_bytes=256 backend=nvidia"
    ));
    Ok(())
}
