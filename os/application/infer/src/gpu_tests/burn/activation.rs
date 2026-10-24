//! Public ReLU ownership, IEEE values, signed views, and autodiff.
use alloc::{format, string::String, vec::Vec};
use d3os_ai::burn::tensor::{Device, Slice, Tensor, TensorData, activation::relu};
use d3os_compute::nvidia::Nvidia;
use d3os_nvidia::compute as abi;

fn check<const D: usize>(tensor: Tensor<D>, expected: &[u32]) -> Result<(), String> {
    let values = tensor.try_into_vec_as::<f32>().map_err(|e| format!("ReLU read: {e:?}"))?;
    if values.len() != expected.len() || values.iter().zip(expected).any(|(a, &b)| a.to_bits() != b) {
        let bits: Vec<_> = values.iter().map(|v| v.to_bits()).collect();
        return Err(format!("ReLU bits differ: {bits:x?} != {expected:x?}"));
    }
    Ok(())
}

fn forward(backend: &Nvidia, device: &Device) -> Result<(), String> {
    let bits = [
        0,
        0x8000_0000,
        0x7fc0_1234,
        0xffc0_5678,
        0x7f80_0000,
        0xff80_0000,
        1,
        0x8000_0001,
        0xc040_0000,
        0x4040_0000,
        0x7fa0_1234,
        0xffa0_5678,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    let values: Vec<_> = bits.iter().copied().map(f32::from_bits).collect();
    let expected: Vec<_> = values.iter().map(|&v| if v <= 0. { 0 } else { v.to_bits() }).collect();
    for shared in [false, true] {
        let input = Tensor::<1>::from_data(TensorData::new(values.clone(), [values.len()]), device);
        let retained = shared.then(|| input.clone());
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let kernels = backend.profile().map_err(|e| format!("{e}"))?;
        let output = relu(input);
        let after = backend.profile().map_err(|e| format!("{e}"))?;
        let services = backend.service_profile().map_err(|e| format!("{e}"))?;
        super::dispatch::resident(backend, &before, 0)?;
        if after[abi::Kernel::Relu as usize].calls - kernels[abi::Kernel::Relu as usize].calls != 1
            || services[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != u64::from(shared)
            || services[abi::ALLOCATE as usize].calls != before[abi::ALLOCATE as usize].calls
        {
            return Err(format!("ReLU ownership dispatch: shared={shared}"));
        }
        check(output, &expected)?;
        if let Some(retained) = retained {
            check(retained, &bits)?;
        }
    }
    let backing = Tensor::<2>::from_data(TensorData::new(values.repeat(3), [3, values.len()]), device);
    let view = backing.clone().slice([1..3, 1..9]).transpose().flip([0, 1]);
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let output = relu(view);
    super::dispatch::resident(backend, &before, 0)?;
    let strided: Vec<_> = (1..9).rev().flat_map(|i| [expected[i]; 2]).collect();
    check(output, &strided)?;
    check(backing, &bits.repeat(3))?;

    // A unique contiguous slice can reuse storage at its positive offset.
    let input = Tensor::<1>::from_data(TensorData::new(values, [bits.len()]), device).slice([Slice::new(1, Some(9), 1)]);
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let output = relu(input);
    let after = backend.service_profile().map_err(|e| format!("{e}"))?;
    if [abi::ALLOCATE, abi::LAUNCH_ALLOCATE, abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(String::from("ReLU unique slice allocated or transferred storage"));
    }
    check(output, &expected[1..9])?;
    let input = Tensor::<2>::empty([3, 0], device);
    let before = backend.profile().map_err(|e| format!("{e}"))?;
    let output = relu(input);
    let after = backend.profile().map_err(|e| format!("{e}"))?;
    if output.dims() != [3, 0] || before.iter().zip(after).any(|(a, b)| a.calls != b.calls) {
        return Err(String::from("ReLU empty tensor launched work"));
    }
    check(output, &[])
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let device = d3os_ai::native_device().map_err(|e| format!("{e}"))?;
    forward(backend, &device)?;
    let device = device.autodiff();
    let input = Tensor::<1>::from_floats([-3., -0., 0., 1., 2.], &device).require_grad();
    let weights = Tensor::<1>::from_floats([1., 2., 3., 4., 5.], &device);
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let output = relu(input.clone());
    let gradients = (output.clone() * weights).sum().backward();
    let dx = input.grad(&gradients).ok_or("ReLU gradient is missing")?;
    super::dispatch::resident(backend, &before, 0)?;
    check(output, &[0, 0, 0, 1f32.to_bits(), 2f32.to_bits()])?;
    check(dx, &[0, 0, 0, 4f32.to_bits(), 5f32.to_bits()])?;
    check(input, &[(-3f32).to_bits(), 0x8000_0000, 0, 1f32.to_bits(), 2f32.to_bits()])?;
    crate::report(format_args!(
        "NVIDIA_BURN_RELU_OK ieee_bits=ok ownership=ok signed_views=ok empty=ok autodiff=ok backend=nvidia"
    ));
    Ok(())
}
