//! Public linear dispatch, flattened leading dimensions, and immutable sources.
use alloc::{format, string::String, vec::Vec};
use d3os_ai::burn::tensor::{Device, Tensor, TensorData, module::linear};
use d3os_compute::nvidia::Nvidia;
use d3os_nvidia::compute as abi;

fn check<const D: usize>(tensor: Tensor<D>, expected: &[f32]) -> Result<(), String> {
    let actual = tensor.try_into_vec_as::<f32>().map_err(|e| format!("linear read: {e:?}"))?;
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 2e-5 * (1. + b.abs()))
    {
        return Err(format!("linear reference differs: {actual:?} != {expected:?}"));
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let device = d3os_ai::native_device().map_err(|e| format!("{e}"))?;
    let input: Vec<_> = (0..45).map(|i| (i as f32 - 23.) / 16.).collect();
    let weights: Vec<_> = (0..35).map(|i| (i as f32 - 17.) / 32.).collect();
    let biases: Vec<_> = (0..7).map(|i| (i as f32 - 3.) / 8.).collect();
    let x = Tensor::<3>::from_data(TensorData::new(input.clone(), [3, 3, 5]), &device);
    let w = Tensor::<2>::from_data(TensorData::new(weights.clone(), [5, 7]), &device);
    let bias = Tensor::<1>::from_data(TensorData::new(biases.clone(), [7]), &device);
    let expected: Vec<_> = (0..42)
        .map(|i| {
            let (batch, row, output) = (i / 21, i / 7 % 3, i % 7);
            ((0..5)
                .map(|k| f64::from(input[(2 - batch) * 15 + row * 5 + 4 - k]) * f64::from(weights[(4 - k) * 7 + 6 - output]))
                .sum::<f64>()
                + f64::from(biases[6 - output])) as f32
        })
        .collect();
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let kernels = backend.profile().map_err(|e| format!("{e}"))?;
    let output = linear(
        x.clone().slice([1..3, 0..3, 0..5]).flip([0, 2]),
        w.clone().flip([0, 1]),
        Some(bias.clone().flip([0])),
    );
    let after = backend.profile().map_err(|e| format!("{e}"))?;
    super::dispatch::resident(backend, &before, 0)?;
    if output.dims() != [2, 3, 7]
        || after[abi::Kernel::Matmul as usize].calls - kernels[abi::Kernel::Matmul as usize].calls != 1
        || after[abi::Kernel::BiasRow as usize].calls - kernels[abi::Kernel::BiasRow as usize].calls != 1
    {
        return Err(format!(
            "linear dispatch: shape={:?} matmul_calls={} bias_row_calls={}",
            output.dims(),
            after[abi::Kernel::Matmul as usize].calls - kernels[abi::Kernel::Matmul as usize].calls,
            after[abi::Kernel::BiasRow as usize].calls - kernels[abi::Kernel::BiasRow as usize].calls
        ));
    }
    check(output, &expected)?;
    check(x, &input)?;
    check(w, &weights)?;
    check(bias.clone(), &biases)?;
    empty(backend, &device, bias, &biases)?;
    crate::report(format_args!(
        "NVIDIA_BURN_LINEAR_OK leading_dims=folded signed_views=ok sources=preserved empty=ok zero_inner=ok backend=nvidia"
    ));
    Ok(())
}

fn empty(backend: &Nvidia, device: &Device, bias: Tensor<1>, values: &[f32]) -> Result<(), String> {
    let x = Tensor::<3>::empty([0, 3, 5], device);
    let w = Tensor::<2>::zeros([5, 7], device);
    let before = backend.profile().map_err(|e| format!("{e}"))?;
    let output = linear(x, w, Some(bias.clone()));
    let after = backend.profile().map_err(|e| format!("{e}"))?;
    if output.dims() != [0, 3, 7] || before.iter().zip(after).any(|(a, b)| a.calls != b.calls) {
        return Err(String::from("empty linear launched element work"));
    }
    check(output, &[])?;
    let x = Tensor::<3>::empty([2, 3, 0], device);
    let w = Tensor::<2>::empty([0, 7], device);
    check(linear(x, w, Some(bias)), &values.repeat(6))
}
