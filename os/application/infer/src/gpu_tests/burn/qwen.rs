//! Public Burn transformer extensions on the passed-through NVIDIA device.
use alloc::{format, string::String, vec::Vec};
use d3os_ai::{
    burn::tensor::Tensor,
    qwen::{MatrixFormat, PackedMatrix, residual_norm, rms_norm},
};
use d3os_compute::nvidia::Nvidia;
use d3os_nvidia::compute as abi;

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let device = d3os_ai::native_device().map_err(|e| format!("Qwen Burn device: {e}"))?;
    d3os_ai::validation::qwen::run(&device)?;
    let before = backend.profile().map_err(|e| format!("Qwen state kernel profile: {e}"))?;
    d3os_ai::validation::qwen_state::run(&device)?;
    let after = backend.profile().map_err(|e| format!("Qwen state kernel profile: {e}"))?;
    for kernel in [
        abi::Kernel::AttentionFusedBatch,
        abi::Kernel::AttentionOutput,
        abi::Kernel::AttentionScores,
        abi::Kernel::GroupedProjectBf16,
        abi::Kernel::DeltaFused,
        abi::Kernel::DeltaConvolution,
        abi::Kernel::DeltaPrepare,
        abi::Kernel::DeltaUpdate,
        abi::Kernel::DeltaOutput,
    ] {
        if after[kernel as usize].calls <= before[kernel as usize].calls {
            return Err(format!("Qwen state path did not execute {kernel:?}"));
        }
    }
    state_residency(backend, &device)?;
    crate::report(format_args!(
        "NVIDIA_BURN_QWEN_STATE_OK attention_cases=8 delta_cases=16 reference=binary64 split_attention=verified grouped_recurrence=verified projected_recurrence=verified cache_snapshots=preserved failed_outputs=one_root activation_uploads=0 activation_downloads=0 backend=nvidia"
    ));
    let data: Vec<_> = [1f32, 2., -1., 0.5, -0.5, 1., 0.25, -2., 0., 1., 2., 3., -1., 0., 1., 0.]
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let matrix = PackedMatrix::from_bytes([4, 4], MatrixFormat::Bf16, &data, &device)?;
    let input = Tensor::<2>::from_floats([[1., 2., 3., 4.]], &device);
    let residual = Tensor::<2>::from_floats([[2., 1., 0., -1.]], &device);
    let weight = Tensor::<1>::from_floats([0., 0.5, -0.5, 1.], &device);
    let before = backend.service_profile().map_err(|e| format!("Qwen Burn profile: {e}"))?;
    let kernels = backend.profile().map_err(|e| format!("Qwen Burn kernel profile: {e}"))?;
    let projection = matrix.project(input.clone())?;
    let added = matrix.project_add(input.clone(), residual.clone())?;
    let gated = matrix.gated_project(&matrix, input.clone())?;
    let norm = rms_norm(input.clone(), weight.clone(), 1e-6)?;
    let (sum, fused) = residual_norm(input.clone(), residual.clone(), weight, 1e-6)?;
    device.sync().map_err(|e| format!("Qwen Burn sync: {e:?}"))?;
    super::dispatch::resident(backend, &before, 0)?;
    let after = backend.profile().map_err(|e| format!("Qwen Burn kernel profile: {e}"))?;
    for kernel in [
        abi::Kernel::ProjectBf16,
        abi::Kernel::ProjectResidualBf16,
        abi::Kernel::GatedProjectBf16,
        abi::Kernel::RmsNorm,
        abi::Kernel::ResidualNorm,
    ] {
        if after[kernel as usize].calls <= kernels[kernel as usize].calls {
            return Err(format!("Qwen Burn did not dispatch {kernel:?}"));
        }
    }
    for tensor in [projection, added, gated, norm, sum, fused] {
        if tensor.device() != device
            || tensor
                .try_into_vec_as::<f32>()
                .map_err(|e| format!("Qwen Burn result: {e:?}"))?
                .iter()
                .any(|v| !v.is_finite())
        {
            return Err(String::from("Qwen Burn produced an invalid result"));
        }
    }
    // Invalid archives must fail before any GPU allocation or upload.
    let before = backend.service_profile().map_err(|e| format!("Qwen Burn profile: {e}"))?;
    if PackedMatrix::from_bytes([1, 1], MatrixFormat::Bf16, &0x7f80u16.to_le_bytes(), &device).is_ok()
        || PackedMatrix::from_bytes([4, 4], MatrixFormat::Bf16, &data[..31], &device).is_ok()
        || matrix.embedding(&[4]).is_ok()
    {
        return Err(String::from("Qwen Burn accepted malformed input"));
    }
    let after = backend.service_profile().map_err(|e| format!("Qwen Burn profile: {e}"))?;
    if [abi::ALLOCATE, abi::UPLOAD, abi::LAUNCH_ALLOCATE, abi::LAUNCH]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(String::from("Qwen Burn malformed input touched GPU storage"));
    }
    // A failed Burn input propagates through the custom operation without work.
    let failed = input.acos();
    let before = backend.service_profile().map_err(|e| format!("Qwen Burn profile: {e}"))?;
    let dependent = matrix.project(failed)?;
    let after = backend.service_profile().map_err(|e| format!("Qwen Burn profile: {e}"))?;
    if [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD, abi::LAUNCH_ALLOCATE, abi::LAUNCH]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || !device.sync().is_err_and(|e| !e.is_device_poisoned())
        || dependent.try_into_data().is_ok()
        || device.sync().is_err()
    {
        return Err(String::from("Qwen Burn failed input did not preserve its root"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_QWEN_PACKED_OK formats=f32,bf16,q8_0,q3k,q4k,q5k,q6k projection=scalar_reference split=verified embedding=gguf_reference gate=verified rms=verified residual_alias=preserved malformed=no_allocation errors=recoverable activation_uploads=0 activation_downloads=0 backend=nvidia"
    ));
    Ok(())
}

fn state_residency(backend: &Nvidia, device: &d3os_ai::burn::tensor::Device) -> Result<(), String> {
    use d3os_ai::qwen::{AttentionConfig, AttentionInputs, AttentionState, DeltaConfig, DeltaInputs, DeltaParameters, DeltaState};
    let attention_config = AttentionConfig {
        heads: 2,
        kv_heads: 1,
        head_dim: 4,
        rotary_dim: 2,
        rope_theta: 10000.,
        epsilon: 1e-6,
    };
    let attention = AttentionState::new(attention_config, 4, device)?;
    let attention_snapshot = attention.clone();
    let query = Tensor::<2>::ones([2, 16], device) * 0.25;
    let key = Tensor::<2>::ones([2, 4], device) * 0.5;
    let value = Tensor::<2>::ones([2, 4], device) * 0.75;
    let norm = Tensor::<1>::zeros([4], device);
    let delta_config = DeltaConfig {
        key_heads: 1,
        heads: 2,
        key_dim: 4,
        value_dim: 3,
        kernel: 3,
        epsilon: 1e-6,
    };
    let delta = DeltaState::new(delta_config, device)?;
    let delta_snapshot = delta.clone();
    let parameters = DeltaParameters {
        convolution: Tensor::ones([14, 1, 3], device) * 0.25,
        log_decay: Tensor::zeros([2], device),
        time_bias: Tensor::zeros([2], device),
        norm: Tensor::ones([3], device),
    };
    let input = DeltaInputs {
        qkv: Tensor::ones([2, 14], device) * 0.25,
        gate: Tensor::ones([2, 6], device) * 0.5,
        beta: Tensor::zeros([2, 2], device),
        alpha: Tensor::zeros([2, 2], device),
    };
    let before = backend.service_profile().map_err(|e| format!("Qwen state profile: {e}"))?;
    let (attention, attention_output) = attention.forward(
        AttentionInputs {
            query_gate: query.clone(),
            key: key.clone(),
            value: value.clone(),
            query_norm: norm.clone(),
            key_norm: norm.clone(),
        },
        attention_config,
    )?;
    let (delta, delta_output) = delta.forward(input, parameters, delta_config)?;
    device.sync().map_err(|e| format!("Qwen state sync: {e:?}"))?;
    super::dispatch::resident(backend, &before, 0)?;
    for output in [attention_output, delta_output] {
        if output.device() != *device
            || output
                .try_into_vec_as::<f32>()
                .map_err(|e| format!("Qwen state output: {e:?}"))?
                .iter()
                .any(|v| !v.is_finite() || *v == 0.)
        {
            return Err(String::from("Qwen state output is invalid"));
        }
    }
    for original in [
        attention_snapshot.keys().clone().reshape::<1, _>([16]),
        attention_snapshot.values().clone().reshape([16]),
        delta_snapshot.convolution().clone().reshape([42]),
        delta_snapshot.recurrent().clone().reshape([24]),
    ] {
        if original
            .try_into_vec_as::<f32>()
            .map_err(|e| format!("Qwen cache snapshot: {e:?}"))?
            .iter()
            .any(|&v| v != 0.)
        {
            return Err(String::from("Qwen cache clone changed"));
        }
    }
    if attention.position() != 2 || delta.recurrent().dims() != [2, 4, 3] {
        return Err(String::from("Qwen state metadata differs"));
    }
    let failed = query.acos();
    let before = backend.service_profile().map_err(|e| format!("Qwen state failure profile: {e}"))?;
    let (failed, output) = attention.forward(
        AttentionInputs {
            query_gate: failed,
            key,
            value,
            query_norm: norm.clone(),
            key_norm: norm,
        },
        attention_config,
    )?;
    let after = backend.service_profile().map_err(|e| format!("Qwen state failure profile: {e}"))?;
    if [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD, abi::LAUNCH_ALLOCATE, abi::LAUNCH]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || !device.sync().is_err_and(|e| !e.is_device_poisoned())
        || device.sync().is_err()
        || output.try_into_data().is_ok()
        || failed.keys().clone().try_into_data().is_ok()
        || failed.values().clone().try_into_data().is_ok()
        || device.sync().is_err()
    {
        return Err(String::from("Qwen state failure did not preserve one error across all outputs"));
    }
    Ok(())
}
