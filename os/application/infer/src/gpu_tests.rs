//! Hardware conformance checks, invoked only by the explicit GPU test command.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_compute::{Backend, Conv2d, Pool2d, Tensor, cpu::Cpu, nvidia::Nvidia};

mod bf16;
mod raster;
mod raster_bytes;
mod selection;
mod transformer;

fn compare(actual: &[f32], expected: &[f32], operation: &str) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{operation}: output length differs"));
    }
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > 2e-4 + 2e-4 * expected.abs() {
            return Err(format!("{operation}[{i}]: actual={actual}, expected={expected}"));
        }
    }
    Ok(())
}

fn verify_buffer_reuse(backend: &Nvidia) -> Result<(), d3os_compute::Error> {
    // Allocate, drop, and reuse storage before the matrix tests. The driver
    // must not retain references to a completed dispatch or a dropped tensor.
    for width in [17, 257, 33] {
        let tensor = Tensor::from_vec(backend, &[1, width], vec![-1.; width])?;
        let output = backend.relu(tensor)?;
        if backend.download(&output)?.iter().any(|&value| value != 0.) {
            return Err(d3os_compute::Error::DeviceFault);
        }
    }
    Ok(())
}

fn verify_syscall_direction_flag(backend: &Nvidia) -> Result<(), d3os_compute::Error> {
    use d3os_nvidia::compute::{self as abi, ProfileEntry, Request};

    let expected = backend.profile()?;
    let mut actual = vec![ProfileEntry::default(); abi::KERNEL_COUNT];
    let request = Request {
        version: abi::VERSION,
        operation: abi::PROFILE,
        pointer: actual.as_mut_ptr() as u64,
        elements: actual.len() as u64,
        ..Request::default()
    };
    let status: isize;
    let returned_flags: u64;
    // Keep DF manipulation inside one assembly block. Rust sees DF clear on
    // both sides, while the kernel must clear it before copying the profile.
    unsafe {
        core::arch::asm!(
            "std",
            "syscall",
            "pushfq",
            "pop {returned_flags}",
            "cld",
            inlateout("rax") syscall::SystemCall::GpuCompute as usize => status,
            in("rdi") &request as *const Request,
            lateout("rcx") _,
            lateout("r11") _,
            returned_flags = lateout(reg) returned_flags,
        );
    }
    if status != 0
        || returned_flags & (1 << 10) == 0
        || expected
            .iter()
            .zip(&actual)
            .any(|(a, b)| (a.calls, a.cycles, a.wait_cycles, a.device_ns) != (b.calls, b.cycles, b.wait_cycles, b.device_ns))
    {
        return Err(d3os_compute::Error::DeviceFault);
    }
    super::report(format_args!("NVIDIA_SYSCALL_DIRECTION_OK backend=nvidia"));
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    verify_syscall_direction_flag(backend).map_err(|error| format!("{error}"))?;
    verify_command_scopes(backend).map_err(|error| format!("{error}"))?;
    verify_buffer_reuse(backend).map_err(|error| format!("{error}"))?;
    bf16::run(backend)?;
    selection::run(backend)?;
    for (m, k, n) in [(17, 33, 9), (37, 65, 257), (1, 9216, 17)] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 73 % 251) as f32 - 125.) / 127.).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 37 % 127) as f32 - 63.) / 71.).collect();
        let mut expected = vec![0.; m * n];
        for row in 0..m {
            for col in 0..n {
                expected[row * n + col] = (0..k).map(|inner| a[row * k + inner] as f64 * b[inner * n + col] as f64).sum::<f64>() as f32;
            }
        }
        let a = Tensor::from_vec(backend, &[m, k], a).map_err(|error| format!("{error}"))?;
        let matrix = Tensor::from_vec(backend, &[k, n], b.clone()).map_err(|error| format!("{error}"))?;
        let output = backend
            .matmul(&a, &matrix)
            .and_then(|output| backend.download(&output))
            .map_err(|error| format!("{error}"))?;
        compare(&output, &expected, "tensor matmul")?;
        let mut transposed = vec![0.; b.len()];
        for row in 0..k {
            for col in 0..n {
                transposed[col * k + row] = b[row * n + col];
            }
        }
        let weights = Tensor::from_vec(backend, &[n, k], transposed).map_err(|error| format!("{error}"))?;
        let bias = Tensor::from_vec(backend, &[n], vec![0.; n]).map_err(|error| format!("{error}"))?;
        let output = backend
            .linear(&a, &weights, &bias)
            .and_then(|output| backend.download(&output))
            .map_err(|error| format!("{error}"))?;
        compare(&output, &expected, "tensor projection")?;
    }
    // This case exercises batch offsets, rectangular kernels, padding, stride,
    // per-channel bias, and the pooling plane index independently of AlexNet.
    let input: Vec<f32> = (0..2 * 2 * 5 * 7).map(|i| (i % 19) as f32 / 7. - 1.).collect();
    let weights: Vec<f32> = (0..3 * 2 * 3 * 2).map(|i| (i % 11) as f32 / 13. - 0.4).collect();
    let gpu = convolution(backend, input.clone(), weights.clone()).map_err(|error| format!("{error}"))?;
    let cpu = convolution(&Cpu, input, weights).map_err(|error| format!("{error}"))?;
    compare(&gpu, &cpu, "batched convolution and pooling")?;
    let logits = vec![1000., 1001., 999., -1000., -1001., -999., 0., 0., 0.];
    let gpu = Tensor::from_vec(backend, &[3, 3], logits.clone()).map_err(|error| format!("{error}"))?;
    let cpu = Tensor::from_vec(&Cpu, &[3, 3], logits).map_err(|error| format!("{error}"))?;
    let gpu = backend
        .softmax(&gpu)
        .and_then(|output| backend.download(&output))
        .map_err(|error| format!("{error}"))?;
    let cpu = Cpu.softmax(&cpu).and_then(|output| Cpu.download(&output)).map_err(|error| format!("{error}"))?;
    compare(&gpu, &cpu, "stable softmax")?;
    transformer::run(backend)?;
    raster::run(backend)?;
    raster_bytes::run(backend)?;
    super::report(format_args!(
        "NVIDIA_INFERENCE_OPERATORS_OK matrix_cases=6 convolution_cases=1 softmax_cases=1 backend=nvidia"
    ));
    Ok(())
}

fn verify_command_scopes(backend: &Nvidia) -> d3os_compute::Result<()> {
    use d3os_nvidia::compute::Kernel;
    const COUNT: usize = 257;
    const GUARD: u32 = 0x4b123456;
    let mut values = vec![0; COUNT + 2];
    values[0] = GUARD;
    values[COUNT + 1] = GUARD;
    let counter = backend.upload_words(&values)?;
    let increment = backend.upload_words(&vec![0.25f32.to_bits(); COUNT])?;
    let verify = |steps: usize| -> d3os_compute::Result<()> {
        let mut values = vec![0; COUNT + 2];
        backend.read_words_into(&counter, 0, &mut values)?;
        if values[0] != GUARD || values[COUNT + 1] != GUARD || values[1..COUNT + 1].iter().any(|&v| v != (steps as f32 * 0.25).to_bits()) {
            return Err(d3os_compute::Error::DeviceFault);
        }
        Ok(())
    };
    let before = backend.queue_profile()?;
    backend.with_commands(|| -> d3os_compute::Result<()> {
        for step in 0..97 {
            let temporary = backend.allocate_words(COUNT)?;
            backend.dispatch_words(Kernel::Copy, &[(&counter, 1), (&temporary, 0)], &[COUNT])?;
            backend.dispatch_words(Kernel::Add, &[(&temporary, 0), (&increment, 0), (&counter, 1)], &[COUNT])?;
            drop(temporary);
            if step == 47 {
                verify(step + 1)?; // Readback must drain the queue inside a scope.
            }
            if step == 49 {
                backend.with_commands(|| -> d3os_compute::Result<()> { backend.dispatch_words(Kernel::Relu, &[(&counter, 1), (&counter, 1)], &[COUNT]) })?;
            }
        }
        Ok(())
    })?;
    verify(97)?;
    let after = backend.queue_profile()?;
    if after.launches.wrapping_sub(before.launches) <= after.batches.wrapping_sub(before.batches) {
        return Err(d3os_compute::Error::DeviceFault);
    }
    let failed: d3os_compute::Result<()> = backend.with_commands(|| {
        backend.dispatch_words(Kernel::Add, &[(&counter, 1), (&increment, 0), (&counter, 1)], &[COUNT])?;
        Err(d3os_compute::Error::InvalidParameter)
    });
    if failed != Err(d3os_compute::Error::InvalidParameter) {
        return Err(d3os_compute::Error::DeviceFault);
    }
    verify(98)?;
    let before = backend.queue_profile()?;
    for _ in 0..2 {
        backend.dispatch_words(Kernel::Relu, &[(&counter, 1), (&counter, 1)], &[COUNT])?;
    }
    let after = backend.queue_profile()?;
    if after.batches.wrapping_sub(before.batches) != 2 || after.launches.wrapping_sub(before.launches) != 2 {
        return Err(d3os_compute::Error::DeviceFault);
    }
    super::report(format_args!("NVIDIA_COMMAND_SCOPES_OK steps=98 count={COUNT} backend=nvidia"));
    Ok(())
}

fn convolution<B: Backend>(backend: &B, input: Vec<f32>, weights: Vec<f32>) -> d3os_compute::Result<Vec<f32>> {
    let input = Tensor::from_vec(backend, &[2, 2, 5, 7], input)?;
    let weights = Tensor::from_vec(backend, &[3, 2, 3, 2], weights)?;
    let bias = Tensor::from_vec(backend, &[3], vec![0.25, -0.75, 1.25])?;
    let output = backend.conv2d(
        &input,
        &weights,
        &bias,
        Conv2d {
            stride: [2, 1],
            padding: [1, 2],
        },
    )?;
    let output = backend.max_pool2d(
        &output,
        Pool2d {
            kernel: [2, 3],
            stride: [1, 2],
        },
    )?;
    backend.download(&output)
}
