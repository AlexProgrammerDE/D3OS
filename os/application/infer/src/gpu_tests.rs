//! Hardware conformance checks, invoked only by the explicit GPU test command.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_compute::{Backend, Conv2d, Pool2d, Tensor, cpu::Cpu, nvidia::Nvidia};

mod transformer;
mod raster;

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

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    verify_buffer_reuse(backend).map_err(|error| format!("{error}"))?;
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
    super::report(format_args!(
        "NVIDIA_INFERENCE_OPERATORS_OK matrix_cases=6 convolution_cases=1 softmax_cases=1 backend=nvidia"
    ));
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
