//! Resident-model measurements. Reference comparisons remain outside timed work.
use alloc::{format, string::String, vec::Vec};
use d3os_ai::qwen::{self, Qwen, tokenizer::Tokenizer};
use d3os_compute::transformer::TransformerBackend;
use d3os_compute::{Backend, Buffer, Tensor, nvidia::Nvidia};
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
            "NATIVE_KERNEL_PROFILE symbol={} calls={calls} prepare_cycles={cycles} serial_wait_cycles={wait} prepare_share_pct={:.2} device_ns={} phase={phase}",
            kernel.symbol(),
            cycles as f64 * 100. / total.max(1) as f64,
            b.device_ns.wrapping_sub(a.device_ns)
        ));
    }
}

fn queue_profile(before: d3os_nvidia::compute::QueueProfile, after: d3os_nvidia::compute::QueueProfile, phase: &str) {
    report(format_args!(
        "NATIVE_QUEUE_PROFILE phase={phase} batches={} launches={} submit_cycles={} wait_cycles={}",
        after.batches.wrapping_sub(before.batches),
        after.launches.wrapping_sub(before.launches),
        after.cycles.wrapping_sub(before.cycles),
        after.wait_cycles.wrapping_sub(before.wait_cycles)
    ));
}

fn service_profile(before: &[d3os_nvidia::compute::ServiceEntry], after: &[d3os_nvidia::compute::ServiceEntry], phase: &str) {
    for (operation, (before, after)) in before.iter().zip(after).enumerate() {
        let calls = after.calls.wrapping_sub(before.calls);
        if calls == 0 {
            continue;
        }
        let cycles = after.cycles.wrapping_sub(before.cycles);
        let queue_cycles = after.queue_cycles.wrapping_sub(before.queue_cycles);
        report(format_args!(
            "NATIVE_SERVICE_PROFILE operation={operation} calls={calls} total_cycles={cycles} queue_cycles={queue_cycles} cpu_cycles={} phase={phase}",
            cycles.saturating_sub(queue_cycles)
        ));
    }
}

pub(super) fn kernels(backend: &Nvidia) -> Result<(), String> {
    for (packed, rows, inner, cols) in [
        (false, 1usize, 9216usize, 4096usize),
        (false, 1, 1024, 3584),
        (false, 1, 3584, 1024),
        (false, 1, 1024, 248320),
        (false, 3025, 363, 64),
        (true, 1, 1024, 1024),
        (true, 1, 1024, 3584),
        (true, 1, 3584, 1024),
        (true, 1, 1024, 248320),
    ] {
        let value = |i: usize| (((i % 31) as i32 - 15) as f32 / 32.).to_bits();
        let values = |length: usize| -> Vec<u32> { (0..length).map(value).collect() };
        let a = backend.upload_words(&values(rows * inner)).map_err(|error| format!("{error}"))?;
        let weights = if packed {
            (0..inner * cols / 2).map(|i| (value(i * 2) >> 16) | (value(i * 2 + 1) & 0xffff0000)).collect()
        } else {
            values(inner * cols)
        };
        let b = backend.upload_words(&weights).map_err(|error| format!("{error}"))?;
        let output = backend.allocate_words(rows * cols).map_err(|error| format!("{error}"))?;
        let kernel = if packed { Kernel::ProjectBf16 } else { Kernel::Project };
        let bytes_per_weight = if packed { 2. } else { 4. };
        let dispatch = || backend.dispatch_words(kernel, &[(&a, 0), (&b, 0), (&output, 0)], &[rows, inner, cols]);
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
            Ok::<_, d3os_compute::Error>(after[kernel as usize].device_ns.wrapping_sub(before[kernel as usize].device_ns))
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
            "NATIVE_KERNEL_BENCH rows={rows} inner={inner} cols={cols} packed_bf16={packed} repeats={REPEATS} mean_us={:.3} effective_gb_s={:.3} device_mean_us={:.3} device_gb_s={:.3} instrumented_mean_us={:.3}",
            elapsed as f64 * 1000. / REPEATS as f64,
            inner as f64 * cols as f64 * bytes_per_weight * REPEATS as f64 / (elapsed as f64 * 1e6),
            device_ns as f64 / (REPEATS as f64 * 1000.),
            inner as f64 * cols as f64 * bytes_per_weight * REPEATS as f64 / device_ns.max(1) as f64,
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
    let measured: d3os_compute::Result<()> = (|| {
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
    use d3os_ai::{
        alexnet::{AlexNet, Observer},
        burn::tensor::{Tensor as BurnTensor, TensorData},
    };
    use d3os_tensor_file::Archive;
    let device = d3os_ai::native_device().map_err(|e| format!("{e}"))?;
    let start = now();
    let weights = read_file("/usr/share/inference/alexnet.d3t")?;
    let fixture = read_file("/usr/share/inference/dog.d3t")?;
    let file_ms = now() - start;
    let start = now();
    let model = AlexNet::load(&device, &weights)?;
    let load_ms = now() - start;
    let mut archive = Archive::parse(&fixture).map_err(|e| format!("{e}"))?;
    let record = archive.take("input").map_err(|e| format!("{e}"))?;
    let values = record.to_vec().map_err(|e| format!("{e}"))?;
    let mut elapsed = 0;
    let mut final_logits = Vec::new();
    const REPEATS: usize = 100;
    for run in 0..=REPEATS {
        let input = BurnTensor::<4>::from_data(TensorData::new(values.clone(), record.shape()), &device);
        device.sync().map_err(|e| format!("AlexNet input: {e:?}"))?;
        let before = if run == REPEATS {
            Some((
                backend.profile().map_err(|e| format!("{e}"))?,
                backend.queue_profile().map_err(|e| format!("{e}"))?,
                backend.service_profile().map_err(|e| format!("{e}"))?,
            ))
        } else {
            None
        };
        let start = now();
        let logits = backend
            .with_commands(|| Ok::<_, d3os_compute::Error>(model.forward(input)))
            .map_err(|e| format!("{e}"))??;
        device.sync().map_err(|e| format!("AlexNet forward: {e:?}"))?;
        let duration = now() - start;
        if run != 0 {
            elapsed += duration;
        }
        if let Some((before, queue_before, service_before)) = before {
            let service_after = backend.service_profile().map_err(|e| format!("{e}"))?;
            if [d3os_nvidia::compute::UPLOAD, d3os_nvidia::compute::DOWNLOAD]
                .iter()
                .any(|&op| service_after[op as usize].calls != service_before[op as usize].calls)
            {
                return Err(String::from("Burn AlexNet forward transferred activation data"));
            }
            report(format_args!("INFERENCE_ALEXNET_RESIDENT_OK framework=burn os=d3os uploads=0 downloads=0"));
            profile(&before, &backend.profile().map_err(|e| format!("{e}"))?, "alexnet_forward");
            queue_profile(queue_before, backend.queue_profile().map_err(|e| format!("{e}"))?, "alexnet_forward");
        }
        final_logits = logits.try_into_vec_as::<f32>().map_err(|e| format!("AlexNet output: {e:?}"))?;
    }
    let top = final_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .ok_or("empty logits")?;
    report(format_args!(
        "INFERENCE_BENCH alexnet backend=nvidia file_ms={file_ms} load_ms={load_ms} repeats={REPEATS} forward_total_ms={elapsed} forward_mean_ms={:.3} class={top} framework=burn os=d3os",
        elapsed as f64 / REPEATS as f64
    ));

    struct LayerTiming<'a> {
        backend: &'a Nvidia,
        before: [d3os_nvidia::compute::ProfileEntry; d3os_nvidia::compute::KERNEL_COUNT],
    }
    impl Observer for LayerTiming<'_> {
        fn observe<const D: usize>(&mut self, name: &'static str, _: &BurnTensor<D>) -> Result<(), String> {
            let after = self.backend.profile().map_err(|e| format!("{e}"))?;
            for kernel in [Kernel::Conv2d, Kernel::Matmul, Kernel::Project] {
                let previous = self.before[kernel as usize];
                let current = after[kernel as usize];
                if current.calls != previous.calls {
                    report(format_args!(
                        "BURN_ALEXNET_LAYER_DEVICE layer={name} kernel={} calls={} device_ns={} os=d3os",
                        kernel as u32,
                        current.calls - previous.calls,
                        current.device_ns - previous.device_ns
                    ));
                }
            }
            self.before = after;
            Ok(())
        }
    }
    backend.set_device_timing(true).map_err(|e| format!("{e}"))?;
    let timing = DeviceTiming(Some(backend));
    let input = BurnTensor::<4>::from_data(TensorData::new(values, record.shape()), &device);
    let mut layers = LayerTiming {
        backend,
        before: backend.profile().map_err(|e| format!("{e}"))?,
    };
    let instrumented_before = layers.before;
    drop(model.forward_observed(input, &mut layers)?);
    device.sync().map_err(|e| format!("AlexNet timing: {e:?}"))?;
    profile(&instrumented_before, &backend.profile().map_err(|e| format!("{e}"))?, "alexnet_device");
    timing.finish().map_err(|e| format!("{e}"))?;
    d3os_ai::alexnet::verify_model(&model, &device, &fixture)?;
    report(format_args!(
        "INFERENCE_BENCH_OK alexnet backend=nvidia reference_layers=8 framework=burn os=d3os"
    ));
    Ok(())
}

struct DeviceTiming<'a>(Option<&'a Nvidia>);

impl DeviceTiming<'_> {
    fn finish(mut self) -> d3os_compute::Result<()> {
        if let Some(backend) = self.0 {
            backend.set_device_timing(false)?;
            self.0 = None;
        }
        Ok(())
    }
}

impl Drop for DeviceTiming<'_> {
    fn drop(&mut self) {
        if let Some(backend) = self.0 {
            let _ = backend.set_device_timing(false);
        }
    }
}

pub(super) fn qwen<const DECODE: usize>(backend: &Nvidia, device_timing: bool) -> Result<(), String> {
    let start = now();
    let tokenizer = read_file("/usr/share/inference/qwen3.5.bpe")?;
    let tokenizer = Tokenizer::parse(&tokenizer).map_err(|error| format!("{error}"))?;
    let prompt = tokenizer
        .encode_chat(include_str!("../../../../tools/inference/qwen-benchmark.prompt"))
        .map_err(|error| format!("{error}"))?;
    let weights = read_file("/usr/share/inference/qwen3.5-0.8b.d3t")?;
    let file_ms = now() - start;
    let load_profile = backend.service_profile().map_err(|error| format!("{error}"))?;
    let start = now();
    let load_cycles = unsafe { core::arch::x86_64::_rdtsc() };
    let device = d3os_ai::native_device().map_err(|e| format!("{e}"))?;
    let model = Qwen::load(&device, &weights).map_err(|error| format!("{error}"))?;
    let load_cycles = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(load_cycles);
    let load_ms = now() - start;
    let loaded_profile = backend.service_profile().map_err(|error| format!("{error}"))?;
    drop(weights);
    service_profile(&load_profile, &loaded_profile, "load");
    let service_cycles: u64 = load_profile.iter().zip(loaded_profile).map(|(a, b)| b.cycles.wrapping_sub(a.cycles)).sum();
    report(format_args!(
        "NATIVE_LOAD_PROFILE total_cycles={load_cycles} service_cycles={service_cycles} application_cycles={}",
        load_cycles.saturating_sub(service_cycles)
    ));
    report(format_args!("INFERENCE_BENCH_TOKENS phase=prompt ids={prompt:?}"));
    let capacity = (prompt.len() + DECODE).div_ceil(256) * 256;
    for run in 0..3 {
        let instrumented = device_timing && run == 1;
        let timing = if instrumented {
            backend.set_device_timing(true).map_err(|error| format!("{error}"))?;
            Some(DeviceTiming(Some(backend)))
        } else {
            None
        };
        let start = now();
        let mut session = model.session(capacity).map_err(|error| format!("{error}"))?;
        let state_ms = now() - start;
        let prefill_profile = if run == 1 {
            Some((
                backend.profile().map_err(|error| format!("{error}"))?,
                backend.queue_profile().map_err(|error| format!("{error}"))?,
                backend.service_profile().map_err(|error| format!("{error}"))?,
            ))
        } else {
            None
        };
        let start = now();
        let mut logits = session.prefill(&prompt).map_err(|error| format!("{error}"))?;
        let prefill_ms = now() - start;
        let before = backend.profile().map_err(|error| format!("{error}"))?;
        let queue_before = backend.queue_profile().map_err(|error| format!("{error}"))?;
        let service_before = if run == 1 {
            Some(backend.service_profile().map_err(|error| format!("{error}"))?)
        } else {
            None
        };
        let mut tokens = Vec::with_capacity(DECODE);
        let start = now();
        let mut checksum = 0u64;
        // Fixed-length synthetic decode measures work after EOS too, like a
        // throughput benchmark. This is not a generated conversational answer.
        for _ in 0..DECODE {
            let token = qwen::greedy(&logits).map_err(|error| format!("{error}"))?;
            checksum = checksum.wrapping_mul(31).wrapping_add(u64::from(token));
            tokens.push(token);
            logits = session.step(token).map_err(|error| format!("{error}"))?;
        }
        let decode_ms = now() - start;
        if run == 1 {
            let after = backend.profile().map_err(|error| format!("{error}"))?;
            let queue_after = backend.queue_profile().map_err(|error| format!("{error}"))?;
            let service_after = backend.service_profile().map_err(|error| format!("{error}"))?;
            if let Some((prefill_before, queue_prefill, service_prefill)) = prefill_profile {
                let prefill_after = service_before.as_ref().expect("warm profile");
                let upload = d3os_nvidia::compute::UPLOAD as usize;
                let download = d3os_nvidia::compute::DOWNLOAD as usize;
                let prompt_uploads = prompt.len().div_ceil(32) as u64;
                if prefill_after[upload].calls - service_prefill[upload].calls != prompt_uploads
                    || prefill_after[download].calls != service_prefill[download].calls
                    || service_after[upload].calls - prefill_after[upload].calls != DECODE as u64
                    || service_after[download].calls - prefill_after[download].calls != 2 * DECODE as u64
                {
                    return Err(String::from("Burn Qwen performed an unexpected host tensor transfer"));
                }
                report(format_args!(
                    "NVIDIA_BURN_QWEN_RESIDENT_OK prompt_id_uploads={prompt_uploads} decode_id_uploads={DECODE} selection_scalar_reads={} activation_uploads=0 activation_downloads=0 backend=nvidia",
                    2 * DECODE
                ));
                profile(&prefill_before, &before, "prefill");
                queue_profile(queue_prefill, queue_before, "prefill");
                service_profile(&service_prefill, service_before.as_ref().expect("warm profile"), "prefill");
            }
            profile(&before, &after, "decode");
            queue_profile(queue_before, queue_after, "decode");
            service_profile(service_before.as_ref().expect("warm profile"), &service_after, "decode");
        }
        if let Some(timing) = timing {
            timing.finish().map_err(|error| format!("{error}"))?;
        }
        report(format_args!(
            "INFERENCE_BENCH qwen backend=nvidia run={run} warm={} instrumented={instrumented} file_ms={file_ms} load_ms={load_ms} state_ms={state_ms} session_capacity={capacity} prompt_tokens={} prefill_ms={prefill_ms} prefill_tps={:.3} decode_tokens={DECODE} decode_ms={decode_ms} decode_tps={:.3} checksum={checksum} framework=burn os=d3os",
            run != 0,
            prompt.len(),
            prompt.len() as f64 * 1000. / prefill_ms.max(1) as f64,
            DECODE as f64 * 1000. / decode_ms.max(1) as f64
        ));
        // Long traces are benchmark evidence. Sending thousands of token IDs
        // through the interactive terminal can fill its output queue.
        log::info!("INFERENCE_BENCH_TOKENS phase=decode run={run} ids={tokens:?}");
    }
    let tokenizer_fixture = read_file("/usr/share/inference/tokenizer-fixture.bin")?;
    let tokenizer_cases = d3os_ai::qwen::tokenizer::verify_fixture(&tokenizer, &tokenizer_fixture).map_err(|e| format!("{e}"))?;
    report(format_args!("QWEN_TOKENIZER_OK cases={tokenizer_cases}"));
    let fixture = read_file("/usr/share/inference/qwen-fixture.d3t")?;
    let positions = qwen::verify_fixture(&model, &fixture).map_err(|error| format!("{error}"))?;
    report(format_args!(
        "INFERENCE_BENCH_OK qwen backend=nvidia reference_positions={positions} reference_layers=24 batch_reference_positions={positions} cache_comparison=ok framework=burn"
    ));
    Ok(())
}
