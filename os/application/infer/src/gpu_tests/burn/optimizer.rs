//! Public Burn Adam fusion, shared state, and independent CPU execution.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::burn::{
    optim::{AdamConfig, AdamState, AdaptiveMomentumState, Optimizer, decay::WeightDecayConfig},
    tensor::{Device, Tensor, TensorData},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, optimizer};

fn values(tensor: Tensor<2>) -> Result<Vec<f32>, String> {
    tensor.try_into_vec_as::<f32>().map_err(|e| format!("Adam read: {e:?}"))
}

fn compare(actual: Tensor<2>, expected: Tensor<2>) -> Result<(), String> {
    let actual = values(actual)?;
    let expected = values(expected)?;
    compare_values(&actual, &expected)
}

fn compare_values(actual: &[f32], expected: &[f32]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(String::from("Adam reference length differs"));
    }
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 2e-6 * (1. + b.abs()) {
            return Err(format!("Adam reference [{index}]: {a} != {b}"));
        }
    }
    Ok(())
}

fn state(actual: &AdamState<2>, expected: &AdamState<2>) -> Result<(), String> {
    let (actual, expected) = (&actual.momentum, &expected.momentum);
    if actual.time != expected.time || actual.max_moment_2.is_some() != expected.max_moment_2.is_some() {
        return Err(String::from("Adam reference state differs"));
    }
    compare(actual.moment_1.clone(), expected.moment_1.clone())?;
    compare(actual.moment_2.clone(), expected.moment_2.clone())?;
    if let (Some(a), Some(b)) = (&actual.max_moment_2, &expected.max_moment_2) {
        compare(a.clone(), b.clone())?;
    }
    Ok(())
}

fn guards(backend: &Nvidia) -> Result<(), String> {
    const COUNT: usize = 129;
    let guard = 0x7fc0deadu32;
    let cpu = Device::flex();
    let data: Vec<Vec<_>> = (0..5)
        .map(|input| (0..COUNT).map(|i| ((i * (input + 3) % 67) as f32 + 1.) / 71.).collect())
        .collect();
    let mut buffers = Vec::new();
    for input in &data {
        let mut words = vec![guard; 3];
        words.extend(input.iter().map(|value| value.to_bits()));
        words.push(guard);
        buffers.push(backend.upload_words(&words).map_err(|e| format!("Adam guard upload: {e}"))?);
    }
    let tensor = |index: usize| Tensor::<2>::from_data(TensorData::new(data[index].clone(), [1, COUNT]), &cpu);
    for flags in [
        0,
        optimizer::STATE,
        optimizer::AMSGRAD,
        optimizer::STATE | optimizer::AMSGRAD,
        optimizer::STATE | optimizer::AMSGRAD | optimizer::MAXIMUM,
    ] {
        let components = if flags & optimizer::AMSGRAD == 0 { 3 } else { 4 };
        let count = COUNT * components;
        let output = backend.upload_words(&vec![guard; count + 2]).map_err(|e| format!("Adam guard output: {e}"))?;
        let pointers = [
            (&buffers[0], 3),
            (&buffers[1], 3),
            (&buffers[2], 3),
            (&buffers[3], 3),
            (&buffers[4], 3),
            (&output, 1),
        ];
        let coefficients = [0.8f32, 1. - 0.8f32, 0.95f32, 1. - 0.95f32, 1.2, 0.00001, 0.003];
        let mut parameters = [0; optimizer::PARAMETERS];
        parameters[..2].copy_from_slice(&[COUNT, flags as usize]);
        for (p, value) in parameters[2..].iter_mut().zip(coefficients) {
            *p = value.to_bits() as usize;
        }
        backend
            .dispatch_words(Kernel::Adam, &pointers, &parameters)
            .map_err(|e| format!("Adam guarded launch: {e}"))?;
        let mut actual = vec![0; count + 2];
        backend
            .read_words_into(&output, 0, &mut actual)
            .map_err(|e| format!("Adam guarded read: {e}"))?;
        if actual[0] != guard || actual[count + 1] != guard {
            return Err(format!("Adam output guard changed: flags={flags}"));
        }
        let gradient = tensor(1);
        let first = gradient.clone() * coefficients[1];
        let second = gradient.square() * coefficients[3];
        let (first, second) = if flags & optimizer::STATE == 0 {
            (first, second)
        } else {
            (tensor(2) * coefficients[0] + first, tensor(3) * coefficients[2] + second)
        };
        let variance = if flags & optimizer::MAXIMUM == 0 {
            second.clone()
        } else {
            tensor(4).max_pair(second.clone())
        };
        let next = tensor(0) - ((first.clone() * coefficients[4] / (variance.clone().sqrt() + coefficients[5])) * coefficients[6]);
        let mut expected = values(next)?;
        expected.extend(values(first)?);
        expected.extend(values(second)?);
        if components == 4 {
            expected.extend(values(variance)?);
        }
        let actual: Vec<_> = actual[1..count + 1].iter().map(|&word| f32::from_bits(word)).collect();
        compare_values(&actual, &expected)?;
        if actual[COUNT..].iter().zip(&expected[COUNT..]).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(format!("Adam moment rounding differs from separate FP32 operations: flags={flags}"));
        }
    }
    for (input, buffer) in data.iter().zip(&buffers) {
        let mut actual = vec![0; COUNT + 4];
        backend.read_words_into(buffer, 0, &mut actual).map_err(|e| format!("Adam source read: {e}"))?;
        if actual[..3] != [guard; 3] || actual[COUNT + 3] != guard || actual[3..COUNT + 3].iter().zip(input).any(|(&a, b)| a != b.to_bits()) {
            return Err(String::from("Adam source or source guard changed"));
        }
    }
    Ok(())
}

fn contexts(backend: &Nvidia, native: &Device, cpu: &Device) -> Result<(), String> {
    for balanced in [false, true] {
        let native = if balanced {
            native.clone().autodiff().gradient_checkpointing()
        } else {
            native.clone().autodiff()
        };
        let cpu = if balanced {
            cpu.clone().autodiff().gradient_checkpointing()
        } else {
            cpu.clone().autodiff()
        };
        let optimizer = AdamConfig::new().with_amsgrad(true).build();
        let mut parameter = Tensor::<2>::from_floats([[0.5, -0.25, 1.]], &native);
        let mut reference = Tensor::<2>::from_floats([[0.5, -0.25, 1.]], &cpu);
        let restored = |device: &Device| AdamState {
            momentum: AdaptiveMomentumState {
                time: 7,
                moment_1: Tensor::<2>::from_floats([[0.1, -0.1, 0.05]], device),
                moment_2: Tensor::<2>::from_floats([[0.02, 0.03, 0.01]], device),
                max_moment_2: Some(Tensor::<2>::from_floats([[0.03, 0.04, 0.02]], device)),
            },
        };
        let mut actual_state = Some(restored(&native));
        let mut expected_state = Some(restored(&cpu));
        let before = backend.profile().map_err(|e| format!("Adam context profile: {e}"))?;
        for _ in 0..2 {
            // Restored state can retain autodiff while the parameter is inner.
            let grad = Tensor::<2>::from_floats([[0.125, -0.5, 0.25]], &native.clone().inner());
            let cpu_grad = Tensor::<2>::from_floats([[0.125, -0.5, 0.25]], &cpu.clone().inner());
            (parameter, actual_state) = backend
                .with_commands(|| Ok::<_, Error>(optimizer.step(0.003, parameter.inner(), grad, actual_state)))
                .map_err(|e| format!("Adam context scope: {e}"))?;
            (reference, expected_state) = optimizer.step(0.003, reference.inner(), cpu_grad, expected_state);
            let current = actual_state.as_ref().ok_or("Adam context state is missing")?;
            let expected = expected_state.as_ref().ok_or("Adam context CPU state is missing")?;
            if current.momentum.moment_1.device().gradient_checkpointing_strategy() != native.gradient_checkpointing_strategy()
                || parameter.device().gradient_checkpointing_strategy() != native.gradient_checkpointing_strategy()
            {
                return Err(String::from("Adam restored-state checkpointing context changed"));
            }
            compare(parameter.clone(), reference.clone())?;
            state(current, expected)?;
        }
        let after = backend.profile().map_err(|e| format!("Adam context profile: {e}"))?;
        if after[Kernel::Adam as usize].calls - before[Kernel::Adam as usize].calls != 2 {
            return Err(String::from("Adam restored-state updates did not remain fused"));
        }
    }
    // A tracked gradient keeps the differentiable tensor implementation.
    let optimizer = AdamConfig::new().build();
    let parameter = Tensor::<2>::from_floats([[0.5, -0.25, 1.]], native);
    let gradient = Tensor::<2>::from_floats([[0.001, -0.002, 0.02]], &native.clone().autodiff()).require_grad();
    let cpu_parameter = Tensor::<2>::from_floats([[0.5, -0.25, 1.]], cpu);
    let cpu_gradient = Tensor::<2>::from_floats([[0.001, -0.002, 0.02]], &cpu.clone().autodiff()).require_grad();
    let before = backend.profile().map_err(|e| format!("Adam gradient profile: {e}"))?;
    let (result, _) = optimizer.step(0.003, parameter, gradient.clone(), None);
    let grads = result.clone().sum().backward();
    let derivative = gradient.grad(&grads).ok_or("Adam dropped the tracked gradient")?;
    let after = backend.profile().map_err(|e| format!("Adam gradient profile: {e}"))?;
    if after[Kernel::Adam as usize].calls != before[Kernel::Adam as usize].calls {
        return Err(String::from("Adam fused a tracked optimizer input"));
    }
    let (cpu_result, _) = optimizer.step(0.003, cpu_parameter, cpu_gradient.clone(), None);
    let cpu_grads = cpu_result.clone().sum().backward();
    compare(result, cpu_result)?;
    compare(derivative, cpu_gradient.grad(&cpu_grads).ok_or("Adam CPU gradient is missing")?)?;
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    guards(backend)?;
    crate::report(format_args!(
        "NVIDIA_BURN_ADAM_GUARDS_OK flags=5 output_guards=ok source_guards=ok moment_rounding=exact"
    ));
    let native = d3os_ai::native_device().map_err(|e| format!("Adam device: {e}"))?;
    let cpu = Device::flex();
    contexts(backend, &native, &cpu)?;
    for (rows, cols, amsgrad, decay, toggle) in [
        (3, 57, false, false, false),
        (3, 57, true, false, false),
        (3, 57, true, true, false),
        (3, 57, true, false, true),
        (0, 3, false, false, false),
    ] {
        let initial: Vec<_> = (0..rows * cols).map(|i| ((i * 13 % 97) as f32 - 48.) / 64.).collect();
        // Construct the oracle from logical values independently of native views.
        let mut canonical = initial.clone();
        for row in canonical.chunks_mut(cols) {
            row.reverse();
        }
        let mut parameter = Tensor::<2>::from_data(TensorData::new(initial.clone(), [rows, cols]), &native).flip([1]);
        let mut reference = Tensor::<2>::from_data(TensorData::new(canonical, [rows, cols]), &cpu);
        let mut native_state = None;
        let mut reference_state = None;
        for (step, scale) in [1f32, 0.01, -0.05, 0., 0.25, -0.001].into_iter().enumerate() {
            let enabled = if toggle { step < 2 || step == 5 } else { amsgrad };
            let optimizer = AdamConfig::new()
                .with_beta_1(0.8)
                .with_beta_2(0.95)
                .with_amsgrad(enabled)
                .with_weight_decay(decay.then(|| WeightDecayConfig::new(0.01)))
                .build();
            let gradient: Vec<_> = (0..rows * cols).map(|i| ((i * 7 % 59) as f32 - 29.) / 37. * scale).collect();
            let canonical: Vec<_> = (0..rows * cols).map(|index| gradient[(index % cols) * rows + index / cols]).collect();
            let grad = Tensor::<2>::from_data(TensorData::new(gradient.clone(), [cols, rows]), &native).transpose();
            let cpu_grad = Tensor::<2>::from_data(TensorData::new(canonical, [rows, cols]), &cpu);
            let saved = parameter.clone();
            let expected_saved = reference.clone();
            let saved_state = native_state.clone();
            let expected_saved_state = reference_state.clone();
            let before = backend.profile().map_err(|e| format!("Adam profile: {e}"))?;
            let services = backend.service_profile().map_err(|e| format!("Adam services: {e}"))?;
            (parameter, native_state) = backend
                .with_commands(|| Ok::<_, Error>(optimizer.step(0.003, parameter, grad, native_state)))
                .map_err(|e| format!("Adam scope: {e}"))?;
            let after = backend.profile().map_err(|e| format!("Adam profile: {e}"))?;
            super::dispatch::resident(backend, &services, 0)?;
            if after[Kernel::Adam as usize].calls - before[Kernel::Adam as usize].calls != u64::from(rows * cols != 0) {
                return Err(format!("Adam fusion launch count: step={step} amsgrad={amsgrad} decay={decay}"));
            }
            (reference, reference_state) = optimizer.step(0.003, reference, cpu_grad, reference_state);
            compare(parameter.clone(), reference.clone())?;
            state(
                native_state.as_ref().ok_or("Adam state is missing")?,
                reference_state.as_ref().ok_or("Adam CPU state is missing")?,
            )?;
            compare(saved, expected_saved)?;
            if let (Some(a), Some(b)) = (&saved_state, &expected_saved_state) {
                state(a, b)?;
            }
        }
        crate::report(format_args!(
            "NVIDIA_BURN_ADAM_CASE_OK rows={rows} cols={cols} amsgrad={amsgrad} decay={decay} toggle={toggle} steps=6"
        ));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_ADAM_OK reference=flex_cpu steps=30 amsgrad=ok amsgrad_reconfigure=ok weight_decay=ok shared_state=preserved signed_views=ok guards=ok moment_rounding=exact empty=ok resident=ok fused_launches=one_per_parameter"
    ));
    Ok(())
}
