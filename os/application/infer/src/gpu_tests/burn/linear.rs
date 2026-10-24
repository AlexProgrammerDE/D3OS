//! Public linear dispatch, flattened leading dimensions, and immutable sources.
use alloc::{format, string::String, vec::Vec};
use d3os_ai::burn::tensor::{Device, Slice, Tensor, TensorData, module::linear};
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
    projections(backend, &device)?;
    gradients(backend, &device)?;
    empty(backend, &device, bias, &biases)?;
    crate::report(format_args!(
        "NVIDIA_BURN_LINEAR_OK leading_dims=folded signed_views=ok column_weights=ok autodiff=ok sources=preserved empty=ok zero_inner=ok backend=nvidia"
    ));
    Ok(())
}

fn gradients(backend: &Nvidia, device: &Device) -> Result<(), String> {
    let device = device.clone().autodiff();
    let x = Tensor::<2>::from_floats([[1., 2., 3.], [-1., 4., 2.]], &device).require_grad();
    let w = Tensor::<2>::from_floats([[1., -1., 0.5], [2., 0., -2.]], &device).require_grad();
    let b = Tensor::<1>::from_floats([0.25, -0.5], &device).require_grad();
    let dy = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let output = linear(x.clone(), w.clone().transpose(), Some(b.clone()));
    let gradients = (output.clone() * dy).sum().backward();
    let dx = x.grad(&gradients).ok_or("linear input gradient is missing")?;
    let dw = w.grad(&gradients).ok_or("linear weight gradient is missing")?;
    let db = b.grad(&gradients).ok_or("linear bias gradient is missing")?;
    super::dispatch::resident(backend, &before, 0)?;
    check(output, &[0.75, -4.5, -3.75, -6.5])?;
    check(dx, &[5., -1., -3.5, 11., -3., -6.5])?;
    check(dw, &[-2., 14., 9., -2., 20., 14.])?;
    check(db, &[4., 6.])?;
    check(x, &[1., 2., 3., -1., 4., 2.])?;
    check(w, &[1., -1., 0.5, 2., 0., -2.])?;
    check(b, &[0.25, -0.5])
}

fn projections(backend: &Nvidia, device: &Device) -> Result<(), String> {
    for rows in [1, 6] {
        for biased in [false, true] {
            let input: Vec<_> = (0..(rows + 2) * 5).map(|i| (i as f32 - 11.) / 16.).collect();
            let weights: Vec<_> = (0..45).map(|i| (i as f32 - 23.) / 32.).collect();
            let biases: Vec<_> = (0..9).map(|i| (i as f32 - 4.) / 8.).collect();
            let x = Tensor::<2>::from_data(TensorData::new(input.clone(), [rows + 2, 5]), device);
            let w = Tensor::<2>::from_data(TensorData::new(weights.clone(), [9, 5]), device);
            let bias = Tensor::<1>::from_data(TensorData::new(biases.clone(), [9]), device);
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let kernels = backend.profile().map_err(|e| format!("{e}"))?;
            let output = linear(
                x.clone().slice([1..rows + 1, 0..5]),
                w.clone().slice([1..8, 0..5]).transpose(),
                biased.then(|| bias.clone().slice([Slice::new(1, Some(8), 1)])),
            );
            let after = backend.profile().map_err(|e| format!("{e}"))?;
            super::dispatch::resident(backend, &before, 0)?;
            if output.dims() != [rows, 7]
                || after[abi::Kernel::Project as usize].calls - kernels[abi::Kernel::Project as usize].calls != 1
                || after[abi::Kernel::Matmul as usize].calls != kernels[abi::Kernel::Matmul as usize].calls
                || after[abi::Kernel::BiasRow as usize].calls - kernels[abi::Kernel::BiasRow as usize].calls != u64::from(biased)
            {
                return Err(format!("linear projection dispatch: rows={rows} biased={biased}"));
            }
            let expected: Vec<_> = (0..rows * 7)
                .map(|i| {
                    let (row, output) = (i / 7, i % 7);
                    ((0..5)
                        .map(|k| f64::from(input[(row + 1) * 5 + k]) * f64::from(weights[(output + 1) * 5 + k]))
                        .sum::<f64>()
                        + if biased { f64::from(biases[output + 1]) } else { 0. }) as f32
                })
                .collect();
            check(output, &expected)?;
            check(x, &input)?;
            check(w, &weights)?;
            check(bias, &biases)?;
        }
    }
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
