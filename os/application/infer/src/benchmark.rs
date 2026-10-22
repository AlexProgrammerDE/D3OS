//! Resident-model measurements. Reference comparisons remain outside timed work.
use alloc::{format, string::String, vec::Vec};
use d3os_compute::{Backend, Tensor, nvidia::Nvidia};
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

fn profile(before: &[d3os_nvidia::compute::ProfileEntry], after: &[d3os_nvidia::compute::ProfileEntry]) {
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
            "NATIVE_KERNEL_PROFILE symbol={} calls={calls} cycles={cycles} wait_cycles={wait} dispatch_share_pct={:.2} wait_pct={:.2}",
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
            "NATIVE_KERNEL_BENCH rows={rows} inner={inner} cols={cols} repeats={REPEATS} mean_us={:.3} effective_gb_s={:.3}",
            elapsed as f64 * 1000. / REPEATS as f64,
            inner as f64 * cols as f64 * 4. * REPEATS as f64 / (elapsed as f64 * 1e6)
        ));
    }
    report(format_args!("INFERENCE_BENCH_OK kernels backend=nvidia"));
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
        .encode_chat("What country is largest by size? Answer briefly.")
        .map_err(|error| format!("{error}"))?;
    let weights = read_file("/usr/share/inference/qwen3.5-0.8b.d3t")?;
    let file_ms = now() - start;
    let start = now();
    let model = Qwen::load(backend, &weights).map_err(|error| format!("{error}"))?;
    let load_ms = now() - start;
    drop(weights);
    const DECODE: usize = 64;
    for run in 0..3 {
        let start = now();
        let mut session = model.session(backend, prompt.len() + DECODE).map_err(|error| format!("{error}"))?;
        let state_ms = now() - start;
        let start = now();
        let mut logits = session.prefill(&prompt).map_err(|error| format!("{error}"))?;
        let prefill_ms = now() - start;
        let before = backend.profile().map_err(|error| format!("{error}"))?;
        let start = now();
        let mut checksum = 0u64;
        // Fixed-length synthetic decode measures work after EOS too, like a
        // throughput benchmark. This is not a generated conversational answer.
        for _ in 0..DECODE {
            let token = qwen::greedy(backend, &logits).map_err(|error| format!("{error}"))?;
            checksum = checksum.wrapping_mul(31).wrapping_add(u64::from(token));
            logits = session.step(token).map_err(|error| format!("{error}"))?;
        }
        let decode_ms = now() - start;
        if run == 1 {
            let after = backend.profile().map_err(|error| format!("{error}"))?;
            profile(&before, &after);
        }
        report(format_args!(
            "INFERENCE_BENCH qwen backend=nvidia run={run} warm={} file_ms={file_ms} load_ms={load_ms} state_ms={state_ms} prompt_tokens={} prefill_ms={prefill_ms} prefill_tps={:.3} decode_tokens={DECODE} decode_ms={decode_ms} decode_tps={:.3} checksum={checksum}",
            run != 0,
            prompt.len(),
            prompt.len() as f64 * 1000. / prefill_ms.max(1) as f64,
            DECODE as f64 * 1000. / decode_ms.max(1) as f64
        ));
    }
    let fixture = read_file("/usr/share/inference/qwen-fixture.d3t")?;
    let positions = qwen::verify_fixture(&model, backend, &fixture).map_err(|error| format!("{error}"))?;
    report(format_args!(
        "INFERENCE_BENCH_OK qwen backend=nvidia reference_positions={positions} reference_layers=24"
    ));
    Ok(())
}
