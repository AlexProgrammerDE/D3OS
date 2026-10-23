//! Resident-model measurements. Reference comparisons remain outside timed work.
use alloc::{format, string::String, vec::Vec};
use d3os_compute::transformer::TransformerBackend;
use d3os_compute::{Backend, Buffer, Tensor, nvidia::Nvidia};
use d3os_inference::{
    alexnet::AlexNet,
    archive::Archive,
    qwen::{self, Qwen},
    tokenizer::Tokenizer,
};
use d3os_nvidia::compute::Kernel;

use super::{read_file, report};

fn now() -> i64 {
    time::systime().num_milliseconds()
}

fn profile(before: &[d3os_nvidia::compute::ProfileEntry], after: &[d3os_nvidia::compute::ProfileEntry], phase: &str) {
    let total: u64 = before.iter().zip(after).map(|(a, b)| b.cycles.wrapping_sub(a.cycles)).sum();
    for (index, (a, b)) in before.iter().zip(after).enumerate() {
        let calls = b.calls.wrapping_sub(a.calls);
        if calls == 0 {
            continue;
        }
        let cycles = b.cycles.wrapping_sub(a.cycles);
        let wait = b.wait_cycles.wrapping_sub(a.wait_cycles);
        let kernel = Kernel::from_raw(index as u32).expect("profile index is a built-in kernel");
        report(format_args!(
            "NATIVE_KERNEL_PROFILE symbol={} calls={calls} cycles={cycles} wait_cycles={wait} dispatch_share_pct={:.2} wait_pct={:.2} phase={phase}",
            kernel.symbol(),
            cycles as f64 * 100. / total.max(1) as f64,
            wait as f64 * 100. / cycles.max(1) as f64
        ));
    }
}

pub(super) fn kernels(backend: &Nvidia) -> Result<(), String> {
    for (rows, inner, cols) in [
        (1usize, 9216usize, 4096usize),
        (1, 1024, 3584),
        (1, 3584, 1024),
        (1, 1024, 248320),
        (3025, 363, 64),
    ] {
        let values = |length: usize| -> Vec<u32> { (0..length).map(|i| (((i % 31) as i32 - 15) as f32 / 32.).to_bits()).collect() };
        let a = backend.upload_words(&values(rows * inner)).map_err(|error| format!("{error}"))?;
        let b = backend.upload_words(&values(inner * cols)).map_err(|error| format!("{error}"))?;
        let output = backend.allocate_words(rows * cols).map_err(|error| format!("{error}"))?;
        let dispatch = || backend.dispatch_words(Kernel::Project, &[(&a, 0), (&b, 0), (&output, 0)], &[rows, inner, cols]);
        dispatch().map_err(|error| format!("{error}"))?;
        const REPEATS: usize = 100;
        let start = now();
        for _ in 0..REPEATS {
            dispatch().map_err(|error| format!("{error}"))?;
        }
        let elapsed = (now() - start).max(1);
        backend.set_device_timing(true).map_err(|error| format!("{error}"))?;
        let timed_start = now();
        let measured = (|| {
            let before = backend.profile()?;
            for _ in 0..REPEATS {
                dispatch()?;
            }
            let after = backend.profile()?;
            Ok::<_, d3os_compute::Error>(
                after[Kernel::Project as usize]
                    .device_ns
                    .wrapping_sub(before[Kernel::Project as usize].device_ns),
            )
        })();
        let timed_elapsed = (now() - timed_start).max(1);
        let stopped = backend.set_device_timing(false);
        let device_ns = measured.map_err(|error| format!("{error}"))?;
        stopped.map_err(|error| format!("{error}"))?;
        let mut result = alloc::vec![0u32; rows * cols];
        backend.read_words_into(&output, 0, &mut result).map_err(|error| format!("{error}"))?;
        for (i, actual) in result.iter().enumerate() {
            let row = i / cols;
            let col = i % cols;
            let mut expected = 0.;
            for k in 0..inner {
                expected += (((row * inner + k) % 31) as i32 - 15) as f32 / 32. * (((col * inner + k) % 31) as i32 - 15) as f32 / 32.;
            }
            if !f32::from_bits(*actual).is_finite() || (f32::from_bits(*actual) - expected).abs() > 0.0002 + 0.0002 * expected.abs() {
                return Err(format!("kernel benchmark mismatch at {i}"));
            }
        }
        report(format_args!(
            "NATIVE_KERNEL_BENCH rows={rows} inner={inner} cols={cols} repeats={REPEATS} mean_us={:.3} effective_gb_s={:.3} device_mean_us={:.3} device_gb_s={:.3} instrumented_mean_us={:.3}",
            elapsed as f64 * 1000. / REPEATS as f64,
            inner as f64 * cols as f64 * 4. * REPEATS as f64 / (elapsed as f64 * 1e6),
            device_ns as f64 / (REPEATS as f64 * 1000.),
            inner as f64 * cols as f64 * 4. * REPEATS as f64 / device_ns.max(1) as f64,
            timed_elapsed as f64 * 1000. / REPEATS as f64
        ));
    }
    gated_projection(backend).map_err(|error| format!("{error}"))?;
    report(format_args!("INFERENCE_BENCH_OK kernels backend=nvidia"));
    Ok(())
}

fn gated_projection(backend: &Nvidia) -> d3os_compute::Result<()> {
    const INNER: usize = 1024;
    const OUTPUTS: usize = 3584;
    const REPEATS: usize = 1000;
    let input = Tensor::from_vec(backend, &[1, INNER], (0..INNER).map(|i| ((i % 31) as f32 - 15.) / 32.).collect())?;
    let weights = |salt| {
        Tensor::from_buffer(
            &[OUTPUTS, INNER],
            backend.upload_bf16((0..INNER * OUTPUTS).map(|i| (((((i + salt) % 127) as f32 - 63.) / 128.).to_bits() >> 16) as u16))?,
        )
    };
    let gate = weights(3)?;
    let up = weights(7)?;
    let run = |fused| {
        if fused {
            backend.gated_project(&input, &gate, &up)
        } else {
            let gate = backend.project(&input, &gate)?;
            let up = backend.project(&input, &up)?;
            backend.silu_mul(gate, &up)
        }
    };
    let expected = backend.download(&run(false)?)?;
    for round in 0..3 {
        for fused in if round % 2 == 0 { [false, true] } else { [true, false] } {
            drop(run(fused)?);
            let start = now();
            for _ in 0..REPEATS {
                drop(run(fused)?);
            }
            let elapsed = now() - start;
            let actual = backend.download(&run(fused)?)?;
            if actual
                .iter()
                .zip(&expected)
                .any(|(a, e)| !a.is_finite() || (a - e).abs() > 0.0002 + 0.0002 * e.abs())
            {
                return Err(d3os_compute::Error::DeviceFault);
            }
            report(format_args!(
                "NATIVE_GATED_BENCH round={round} fused={fused} inner={INNER} outputs={OUTPUTS} repeats={REPEATS} mean_us={:.3}",
                elapsed as f64 * 1000. / REPEATS as f64
            ));
        }
    }
    // Evict the small weight set between calls. Otherwise repeated calls can
    // remain in L2 and hide the streaming behavior of a complete model.
    let eviction = backend.allocate_words(128 * 1024 * 1024 / 4)?;
    const COLD_REPEATS: usize = 200;
    backend.set_device_timing(true)?;
    let measured = (|| {
        for round in 0..3 {
            for fused in if round % 2 == 0 { [false, true] } else { [true, false] } {
                let before = backend.profile()?;
                for _ in 0..COLD_REPEATS {
                    backend.dispatch_words(Kernel::Zero, &[(&eviction, 0)], &[eviction.len()])?;
                    drop(run(fused)?);
                }
                let after = backend.profile()?;
                let kernels: &[Kernel] = if fused {
                    &[Kernel::GatedProjectBf16]
                } else {
                    &[Kernel::ProjectBf16, Kernel::SiluMul]
                };
                let device_ns: u64 = kernels
                    .iter()
                    .map(|&kernel| after[kernel as usize].device_ns.wrapping_sub(before[kernel as usize].device_ns))
                    .sum();
                report(format_args!(
                    "NATIVE_GATED_COLD_BENCH round={round} fused={fused} inner={INNER} outputs={OUTPUTS} repeats={COLD_REPEATS} eviction_bytes=134217728 instrumented_device_mean_us={:.3}",
                    device_ns as f64 / (COLD_REPEATS as f64 * 1000.)
                ));
            }
        }
        Ok(())
    })();
    let stopped = backend.set_device_timing(false);
    measured?;
    stopped?;
    Ok(())
}

pub(super) fn alexnet(backend: &Nvidia) -> Result<(), String> {
    let start = now();
    let weights = read_file("/usr/share/inference/alexnet.d3t")?;
    let fixture = read_file("/usr/share/inference/dog.d3t")?;
    let file_ms = now() - start;
    let start = now();
    let model = AlexNet::load(backend, &weights).map_err(|error| format!("{error}"))?;
    let load_ms = now() - start;
    let mut archive = Archive::parse(&fixture).map_err(|error| format!("{error}"))?;
    let input = archive.take("input").map_err(|error| format!("{error}"))?;
    let values = input.to_vec().map_err(|error| format!("{error}"))?;
    let mut elapsed = 0;
    let mut final_logits = Vec::new();
    const REPEATS: usize = 20;
    for run in 0..=REPEATS {
        let input = Tensor::from_vec(backend, input.shape(), values.clone()).map_err(|error| format!("{error}"))?;
        let start = now();
        let logits = model.forward(backend, input, |_, _| Ok(())).map_err(|error| format!("{error}"))?;
        let duration = now() - start;
        if run != 0 {
            elapsed += duration;
        }
        final_logits = backend.download(&logits).map_err(|error| format!("{error}"))?;
    }
    let top = final_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .ok_or("empty logits")?;
    report(format_args!(
        "INFERENCE_BENCH alexnet backend=nvidia file_ms={file_ms} load_ms={load_ms} repeats={REPEATS} forward_total_ms={elapsed} forward_mean_ms={:.3} class={top}",
        elapsed as f64 / REPEATS as f64
    ));
    drop(model);
    d3os_inference::verify_alexnet(backend, &weights, &fixture).map_err(|error| format!("{error}"))?;
    report(format_args!("INFERENCE_BENCH_OK alexnet backend=nvidia reference_layers=8"));
    Ok(())
}

pub(super) fn qwen(backend: &Nvidia) -> Result<(), String> {
    let start = now();
    let tokenizer = read_file("/usr/share/inference/qwen3.5.bpe")?;
    let tokenizer = Tokenizer::parse(&tokenizer).map_err(|error| format!("{error}"))?;
    let prompt = tokenizer
        .encode_chat(include_str!("../../../../tools/inference/qwen-benchmark.prompt"))
        .map_err(|error| format!("{error}"))?;
    let weights = read_file("/usr/share/inference/qwen3.5-0.8b.d3t")?;
    let file_ms = now() - start;
    let start = now();
    let model = Qwen::load(backend, &weights).map_err(|error| format!("{error}"))?;
    let load_ms = now() - start;
    drop(weights);
    report(format_args!("INFERENCE_BENCH_TOKENS phase=prompt ids={prompt:?}"));
    const DECODE: usize = 64;
    const CAPACITY: usize = 256;
    for run in 0..3 {
        let start = now();
        let mut session = model.session(backend, CAPACITY).map_err(|error| format!("{error}"))?;
        let state_ms = now() - start;
        let prefill_profile = if run == 1 {
            Some(backend.profile().map_err(|error| format!("{error}"))?)
        } else {
            None
        };
        let start = now();
        let mut logits = session.prefill(&prompt).map_err(|error| format!("{error}"))?;
        let prefill_ms = now() - start;
        let before = backend.profile().map_err(|error| format!("{error}"))?;
        let mut tokens = Vec::with_capacity(DECODE);
        let start = now();
        let mut checksum = 0u64;
        // Fixed-length synthetic decode measures work after EOS too, like a
        // throughput benchmark. This is not a generated conversational answer.
        for _ in 0..DECODE {
            let token = qwen::greedy(backend, &logits).map_err(|error| format!("{error}"))?;
            checksum = checksum.wrapping_mul(31).wrapping_add(u64::from(token));
            tokens.push(token);
            logits = session.step(token).map_err(|error| format!("{error}"))?;
        }
        let decode_ms = now() - start;
        if run == 1 {
            let after = backend.profile().map_err(|error| format!("{error}"))?;
            if let Some(prefill_before) = prefill_profile {
                profile(&prefill_before, &before, "prefill");
            }
            profile(&before, &after, "decode");
        }
        report(format_args!(
            "INFERENCE_BENCH qwen backend=nvidia run={run} warm={} file_ms={file_ms} load_ms={load_ms} state_ms={state_ms} session_capacity={CAPACITY} prompt_tokens={} prefill_ms={prefill_ms} prefill_tps={:.3} decode_tokens={DECODE} decode_ms={decode_ms} decode_tps={:.3} checksum={checksum}",
            run != 0,
            prompt.len(),
            prompt.len() as f64 * 1000. / prefill_ms.max(1) as f64,
            DECODE as f64 * 1000. / decode_ms.max(1) as f64
        ));
        report(format_args!("INFERENCE_BENCH_TOKENS phase=decode run={run} ids={tokens:?}"));
    }
    let fixture = read_file("/usr/share/inference/qwen-fixture.d3t")?;
    let positions = qwen::verify_fixture(&model, backend, &fixture).map_err(|error| format!("{error}"))?;
    report(format_args!(
        "INFERENCE_BENCH_OK qwen backend=nvidia reference_positions={positions} reference_layers=24 batch_reference_positions={positions} cache_comparison=ok"
    ));
    Ok(())
}
