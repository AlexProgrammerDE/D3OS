//! Exercise Burn's public tensor, module, optimizer, and autodiff APIs on D3OS.
use alloc::{format, string::String, vec::Vec};
use d3os_ai::{
    burn::{
        optim::{GradientsParams, SgdConfig},
        tensor::{Device, Int, Tensor, TensorData},
    },
    mnist::Mnist,
};
use d3os_compute::nvidia::Nvidia;
use d3os_nvidia::compute as abi;

pub(super) fn resident(backend: &Nvidia, before: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], status_reads: u64) -> Result<(), String> {
    let after = backend.service_profile().map_err(|e| format!("Burn service profile: {e}"))?;
    if [abi::UPLOAD].iter().any(|&op| before[op as usize].calls != after[op as usize].calls)
        || after[abi::DOWNLOAD as usize].calls - before[abi::DOWNLOAD as usize].calls != status_reads
    {
        return Err(format!(
            "Burn residency: uploads={} downloads={} allocations={} expected_status_reads={status_reads}",
            after[abi::UPLOAD as usize].calls - before[abi::UPLOAD as usize].calls,
            after[abi::DOWNLOAD as usize].calls - before[abi::DOWNLOAD as usize].calls,
            after[abi::ALLOCATE as usize].calls - before[abi::ALLOCATE as usize].calls
        ));
    }
    Ok(())
}

fn check<const D: usize>(tensor: Tensor<D>, expected: &[f32]) -> Result<(), String> {
    let actual = tensor.try_into_vec_as::<f32>().map_err(|e| format!("public Burn read: {e:?}"))?;
    if actual.len() != expected.len() || actual.iter().zip(expected).any(|(a, b)| !a.is_finite() || (a - b).abs() > 2e-6) {
        return Err(format!("public Burn reference: {actual:?} != {expected:?}"));
    }
    Ok(())
}

fn tensors_and_errors(backend: &Nvidia, device: &Device) -> Result<(), String> {
    let source = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], device);
    let before = backend.service_profile().map_err(|e| format!("Burn profile: {e}"))?;
    let product = source.clone().matmul(source.clone().transpose());
    let viewed = source.clone().transpose().reshape([4]).flip([0]);
    resident(backend, &before, 0)?;
    if product.device() != *device || viewed.device() != *device {
        return Err(String::from("Burn dispatch changed devices"));
    }
    check(product, &[5.0, 11.0, 11.0, 25.0])?;
    check(viewed, &[4.0, 2.0, 3.0, 1.0])?;

    let before = backend.service_profile().map_err(|e| format!("Burn profile: {e}"))?;
    let failed = source.clone().acos();
    let dependent = (failed.clone().reshape([4]) + 2.0).sum();
    resident(backend, &before, 0)?;
    let after = backend.service_profile().map_err(|e| format!("Burn failure profile: {e}"))?;
    if [abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(String::from("Burn failed nodes allocated or launched work"));
    }
    if !device.sync().is_err_and(|e| !e.is_device_poisoned()) || dependent.try_into_data().is_ok() || failed.try_into_data().is_ok() || device.sync().is_err() {
        return Err(String::from("Burn failed-node propagation or acknowledgment"));
    }
    let indices = Tensor::<2, Int>::from_ints([[0, 1], [1, 99]], device);
    let failed = source.clone().gather(1, indices);
    if !failed.try_into_data().is_err_and(|e| !e.is_device_poisoned()) || device.sync().is_err() {
        return Err(String::from("Burn indexed error did not recover"));
    }
    // An acknowledged error in an optional bias remains the original root.
    // The unsupported consumer must not create a second, unrelated error.
    let failed_bias = source.clone().sum().acos();
    if device.sync().is_ok() {
        return Err(String::from("Burn optional input failure was not reported"));
    }
    let input = source.clone().reshape([1, 1, 2, 2, 1]);
    let weight = Tensor::<5>::ones([1, 1, 1, 1, 1], device);
    let failed = d3os_ai::burn::tensor::module::conv3d(
        input,
        weight,
        Some(failed_bias),
        d3os_ai::burn::backend::ops::ConvOptions::new([1; 3], [0; 3], [1; 3], 1),
    );
    if device.sync().is_err() || failed.try_into_data().is_ok() {
        return Err(String::from("Burn optional input lost its original failure"));
    }
    check(source.clone() + 1.0, &[2.0, 3.0, 4.0, 5.0])?;
    check(source, &[1.0, 2.0, 3.0, 4.0])?;
    Ok(())
}

fn convolutional_step(backend: &Nvidia, device: &Device) -> Result<(), String> {
    let device = device.clone().autodiff();
    device.seed(42);
    let input: Vec<f32> = (0..2 * 28 * 28).map(|i| ((i * 17 % 257) as f32 - 128.0) / 128.0).collect();
    let images = Tensor::<4>::from_data(TensorData::new(input, [2, 1, 28, 28]), &device);
    let target = Tensor::<2>::from_floats([[0.0; 10]; 2], &device);
    let before = backend.service_profile().map_err(|e| format!("Burn profile: {e}"))?;
    let model = Mnist::new(&device);
    let retained = model.clone();
    let error = model.forward(images.clone()) - target.clone();
    let loss = (error.clone() * error).mean();
    let gradients = GradientsParams::from_grads(loss.clone().backward(), &model);
    if gradients.len() != 8 {
        return Err(format!("Burn MNIST gradients: expected eight parameter tensors, got {}", gradients.len()));
    }
    let mut optimizer = SgdConfig::new().init();
    let updated = optimizer.step(0.0001, model, gradients);
    let error = updated.forward(images.clone()) - target.clone();
    let after_loss = (error.clone() * error).mean();
    let error = retained.forward(images) - target;
    let retained_loss = (error.clone() * error).mean();
    // Batched gradient products use initialized storage because individual
    // matrix launches only write one batch. The two reads are bounds flags
    // from max-pool gradients; activations remain resident.
    resident(backend, &before, 2)?;
    let before = loss.try_into_scalar::<f32>().map_err(|e| format!("Burn convolution loss: {e:?}"))?;
    let after = after_loss
        .try_into_scalar::<f32>()
        .map_err(|e| format!("Burn convolution updated loss: {e:?}"))?;
    let retained = retained_loss.try_into_scalar::<f32>().map_err(|e| format!("Burn retained model loss: {e:?}"))?;
    if !before.is_finite() || !after.is_finite() || before <= 0.0 || after >= before || (before - retained).abs() > 2e-6 {
        return Err(format!("Burn convolution SGD: before={before} after={after} retained={retained}"));
    }
    Ok(())
}

fn convolution_gradients(backend: &Nvidia, device: &Device) -> Result<(), String> {
    use d3os_ai::burn::{backend::ops::ConvOptions, tensor::module::conv2d};
    let device = device.clone().autodiff();
    let input = Tensor::<4>::from_data(TensorData::new((1..=9).map(|n| n as f32).collect(), [1, 1, 3, 3]), &device).require_grad();
    let weight = Tensor::<4>::from_floats([[[[1.0, -1.0], [0.5, 2.0]]]], &device).require_grad();
    let bias = Tensor::<1>::from_floats([0.25], &device).require_grad();
    let gradient = Tensor::<4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
    let before = backend.service_profile().map_err(|e| format!("Burn profile: {e}"))?;
    let output = conv2d(input.clone(), weight.clone(), Some(bias.clone()), ConvOptions::new([1, 1], [0, 0], [1, 1], 1));
    let loss = (output.clone() * gradient).sum();
    let gradients = loss.clone().backward();
    let dx = input.grad(&gradients).ok_or("Burn convolution input gradient is missing")?;
    let dw = weight.grad(&gradients).ok_or("Burn convolution weight gradient is missing")?;
    let db = bias.grad(&gradients).ok_or("Burn convolution bias gradient is missing")?;
    resident(backend, &before, 0)?;
    check(output, &[11.25, 13.75, 18.75, 21.25])?;
    check(loss, &[180.0])?;
    check(dx, &[1.0, 1.0, -2.0, 3.5, 4.0, 0.0, 1.5, 8.0, 8.0])?;
    check(dw, &[37.0, 47.0, 67.0, 77.0])?;
    check(db, &[10.0])?;
    Ok(())
}

fn classification(backend: &Nvidia, device: &Device) -> Result<(), String> {
    use d3os_ai::burn::nn::loss::CrossEntropyLossConfig;
    let device = device.clone().autodiff();
    let logits = Tensor::<2>::from_floats(
        [[0.0, core::f32::consts::LN_2, 1.098_612_3], [1.098_612_3, core::f32::consts::LN_2, 0.0]],
        &device,
    )
    .require_grad();
    let targets = Tensor::<1, Int>::from_ints([2, 0], &device);
    let criterion = CrossEntropyLossConfig::new().init(&device);
    let before = backend.service_profile().map_err(|e| format!("Burn classification profile: {e}"))?;
    let kernels_before = backend.profile().map_err(|e| format!("Burn classification kernels: {e}"))?;
    let correct = logits.clone().argmax(1).reshape([2]).equal(targets.clone()).int().sum();
    let loss = criterion.forward(logits.clone(), targets);
    let gradients = loss.clone().backward();
    let gradient = logits.grad(&gradients).ok_or("Burn classification gradient is missing")?;
    // Burn's default maximum-with-indices and target selection each gather.
    // Their two backward scatters also validate device indices. The detached
    // maximum retains require_grad in Burn, so its separate branch runs too.
    resident(backend, &before, 4)?;
    let kernels_after = backend.profile().map_err(|e| format!("Burn classification kernels: {e}"))?;
    if [abi::Kernel::Gather, abi::Kernel::Scatter]
        .iter()
        .any(|&kernel| kernels_after[kernel as usize].calls - kernels_before[kernel as usize].calls != 2)
    {
        return Err(String::from("Burn classification bounds reads do not match the indexed operations"));
    }
    check(loss, &[core::f32::consts::LN_2])?;
    check(gradient, &[1.0 / 12.0, 1.0 / 6.0, -0.25, -0.25, 1.0 / 6.0, 1.0 / 12.0])?;
    if correct.try_into_scalar::<i32>().map_err(|e| format!("Burn classification accuracy: {e:?}"))? != 2 {
        return Err(String::from("Burn classification accuracy reference"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_CLASSIFICATION_OK loss=cross_entropy gradients=analytic accuracy=exact backend=nvidia"
    ));
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let device = d3os_ai::native_device().map_err(|e| format!("Burn native device: {e}"))?;
    tensors_and_errors(backend, &device)?;
    convolution_gradients(backend, &device)?;
    classification(backend, &device)?;
    d3os_ai::validation::training(&device.clone().autodiff())?;
    crate::report(format_args!("NVIDIA_BURN_PUBLIC_OPTIMIZER_OK sgd=analytic adam_checkpoint=ok backend=nvidia"));
    convolutional_step(backend, &device)?;
    d3os_ai::validation::mnist_checkpoint(&device.clone().autodiff())?;
    crate::report(format_args!(
        "NVIDIA_BURN_MNIST_CHECKPOINT_OK optimizer=adam parameters=8 partial_batch=ok epoch_boundary=ok random_state=exact inference_export=verified backend=nvidia"
    ));
    device.sync().map_err(|e| format!("Burn native synchronization: {e:?}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_DISPATCH_OK api=public backend=nvidia autodiff=ok sgd=analytic adam_checkpoint=ok convolutional_sgd=ok shared_model=preserved errors=recoverable"
    ));
    Ok(())
}
