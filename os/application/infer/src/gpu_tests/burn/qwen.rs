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
