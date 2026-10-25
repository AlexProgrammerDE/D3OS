//! Train, checkpoint, and reload a Burn classifier inside D3OS.
use super::{read_file, report};
use alloc::{format, string::String, vec::Vec};
use core::fmt::Write;
use d3os_ai::{
    burn::module::Module,
    mnist::{
        Mnist, artifact,
        data::Dataset,
        training::{Config, Trainer, evaluate},
    },
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};

fn export(name: &str, bytes: &[u8]) -> Result<(), String> {
    log::info!("MNIST_ARTIFACT_BEGIN name={name} bytes={} crc32={:08x}", bytes.len(), crc32fast::hash(bytes));
    for (chunk, data) in bytes.chunks(256).enumerate() {
        let mut hex = String::with_capacity(data.len() * 2);
        for byte in data {
            write!(hex, "{byte:02x}").map_err(|e| format!("MNIST artifact encoding: {e}"))?;
        }
        log::info!("MNIST_ARTIFACT_DATA name={name} offset={} hex={hex}", chunk * 256);
    }
    log::info!("MNIST_ARTIFACT_END name={name}");
    Ok(())
}

fn dataset<'a>(images: &'a [u8], labels: &'a [u8], expected: usize) -> Result<Dataset<'a>, String> {
    let data = Dataset::parse(images, labels)?;
    if data.len() != expected {
        return Err(format!("MNIST split has {} items, expected {expected}", data.len()));
    }
    Ok(data)
}

pub(super) fn train() -> Result<(), String> {
    let config_bytes = read_file("/fwcfg/mnist.config")?;
    if config_bytes.len() != 40 {
        return Err(String::from("invalid MNIST run configuration"));
    }
    let word = |offset| u32::from_le_bytes(config_bytes[offset..offset + 4].try_into().expect("checked run configuration"));
    let wide = |offset| u64::from_le_bytes(config_bytes[offset..offset + 8].try_into().expect("checked run configuration"));
    let config = Config {
        seed: wide(0),
        batch_size: word(8) as usize,
        learning_rate: f64::from_bits(wide(16)),
    };
    let epochs = word(12);
    let max_steps = wide(24);
    let resume = word(32);
    let override_rate = word(36);
    if epochs == 0 || epochs > 100 || resume > 1 || override_rate > 1 || (override_rate == 1 && resume == 0) {
        return Err(String::from("invalid MNIST epochs or resume flag"));
    }
    let images = read_file("/fwcfg/mnist.train-images")?;
    let labels = read_file("/fwcfg/mnist.train-labels")?;
    let test_images = read_file("/fwcfg/mnist.test-images")?;
    let test_labels = read_file("/fwcfg/mnist.test-labels")?;
    let training = dataset(&images, &labels, 60_000)?;
    let testing = dataset(&test_images, &test_labels, 10_000)?;
    let device = d3os_ai::native_device().map_err(|e| format!("MNIST GPU: {e}"))?.autodiff();
    let backend = Nvidia::new().map_err(|e| format!("MNIST compute: {e}"))?;
    let mut trainer = if resume == 1 {
        Trainer::restore(&read_file("/fwcfg/mnist.checkpoint")?, &training, &device)?
    } else {
        Trainer::new(&training, &device, config)?
    };
    if override_rate == 1 {
        let previous = trainer.config().learning_rate;
        trainer.set_learning_rate(config.learning_rate)?;
        report(format_args!(
            "MNIST_LR_CHANGE os=d3os backend=nvidia previous={previous} learning_rate={} step={} optimizer_state=preserved",
            config.learning_rate,
            trainer.progress().steps
        ));
    }
    if trainer.config() != config {
        return Err(String::from("MNIST resume configuration differs from checkpoint"));
    }
    report(format_args!(
        "MNIST_START os=d3os backend=nvidia precision=fp32 tensor_cores=tf32x3 model=conv16-conv32-fc64-fc10 train=60000 test=10000 seed={} batch={} optimizer=adam learning_rate={} epochs={epochs} max_steps={max_steps} resume={resume} step={} epoch={} offset={}",
        config.seed,
        config.batch_size,
        config.learning_rate,
        trainer.progress().steps,
        trainer.progress().epoch,
        trainer.progress().offset
    ));
    let start = time::systime().num_milliseconds();
    let mut processed = 0usize;
    let mut loss_sum = 0f64;
    let mut loss_samples = 0usize;
    while trainer.progress().epoch < epochs && (max_steps == 0 || trainer.progress().steps < max_steps) {
        let (next, step) = backend
            .with_commands(|| Ok::<_, Error>(trainer.step(&training)))
            .map_err(|e| format!("MNIST command scope: {e}"))??;
        trainer = next;
        processed += step.samples;
        loss_sum += f64::from(step.loss) * step.samples as f64;
        loss_samples += step.samples;
        if trainer.progress().steps.is_multiple_of(32) || step.completed_epoch.is_some() || (max_steps != 0 && trainer.progress().steps == max_steps) {
            let elapsed = time::systime().num_milliseconds() - start;
            let memory = backend.memory_profile().map_err(|e| format!("MNIST memory profile: {e}"))?;
            report(format_args!(
                "MNIST_PROGRESS os=d3os backend=nvidia step={} epoch={} offset={} segment_loss={:.6} samples={} elapsed_ms={elapsed} samples_per_second={:.3} gpu_live_bytes={} gpu_device_cached_bytes={} memory_scope=compute_buffers_at_step_boundary",
                trainer.progress().steps,
                trainer.progress().epoch,
                trainer.progress().offset,
                loss_sum / loss_samples as f64,
                processed,
                processed as f64 * 1000.0 / elapsed.max(1) as f64,
                memory.process_live_bytes,
                memory.device_cached_bytes
            ));
        }
        if step.completed_epoch.is_some() {
            let result = backend
                .with_commands(|| Ok::<_, Error>(evaluate(trainer.model(), &testing, config.batch_size)))
                .map_err(|e| format!("MNIST evaluation scope: {e}"))??;
            report(format_args!(
                "MNIST_EPOCH os=d3os backend=nvidia epoch={} segment_loss={:.6} test_correct={} test_samples={} accuracy={:.6} predictions_crc32={:08x}",
                trainer.progress().epoch,
                loss_sum / loss_samples as f64,
                result.correct,
                result.samples,
                result.accuracy(),
                result.predictions_crc32
            ));
            loss_sum = 0.0;
            loss_samples = 0;
        }
        if trainer.progress().steps.is_multiple_of(256) || step.completed_epoch.is_some() {
            export("mnist.checkpoint", &trainer.checkpoint()?)?;
        }
    }
    export("mnist.checkpoint", &trainer.checkpoint()?)?;
    let model_bytes = artifact::export(trainer.model())?;
    let reloaded = artifact::load(&model_bytes, &device)?;
    let expected = backend
        .with_commands(|| Ok::<_, Error>(evaluate(trainer.model(), &testing, config.batch_size)))
        .map_err(|e| format!("MNIST final scope: {e}"))??;
    let actual = backend
        .with_commands(|| Ok::<_, Error>(evaluate(&reloaded, &testing, config.batch_size)))
        .map_err(|e| format!("MNIST reload scope: {e}"))??;
    if expected.correct != actual.correct || expected.predictions_crc32 != actual.predictions_crc32 {
        return Err(String::from("MNIST exported predictions changed"));
    }
    let indices = [0, 1, 2, 3];
    let (input, _) = testing.batch(&indices, &device.clone().without_autodiff())?;
    let before = trainer
        .model()
        .valid()
        .forward(input.clone())
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("MNIST original logits: {e:?}"))?;
    let after = reloaded
        .forward(input)
        .try_into_vec_as::<f32>()
        .map_err(|e| format!("MNIST reloaded logits: {e:?}"))?;
    if before.iter().zip(after).any(|(a, b)| !b.is_finite() || (a - b).abs() > 2e-6) {
        return Err(String::from("MNIST exported logits changed"));
    }
    export("mnist.model", &model_bytes)?;
    let complete = trainer.progress().epoch >= epochs;
    report(format_args!(
        "MNIST_RESULT os=d3os backend=nvidia complete={complete} accepted={} step={} epochs={} test_correct={} test_samples={} accuracy={:.6} predictions_crc32={:08x} checkpoint=exported inference=reload_verified elapsed_ms={}",
        complete && actual.accuracy() >= 0.99,
        trainer.progress().steps,
        trainer.progress().epoch,
        actual.correct,
        actual.samples,
        actual.accuracy(),
        actual.predictions_crc32,
        time::systime().num_milliseconds() - start
    ));
    if complete && actual.accuracy() < 0.99 {
        return Err(String::from(
            "MNIST test accuracy is below 99 percent; checkpoint exported for continued training",
        ));
    }
    report(format_args!("INFERENCE_OK mnist mode=train backend=nvidia complete={complete}"));
    Ok(())
}

pub(super) fn infer() -> Result<(), String> {
    let config = read_file("/fwcfg/mnist.config")?;
    if config.len() != 40 {
        return Err(String::from("invalid MNIST inference configuration"));
    }
    let batch_size = u32::from_le_bytes(config[8..12].try_into().map_err(|_| "MNIST inference batch size")?) as usize;
    if !(1..=4096).contains(&batch_size) {
        return Err(String::from("invalid MNIST inference batch size"));
    }
    let images = read_file("/fwcfg/mnist.test-images")?;
    let labels = read_file("/fwcfg/mnist.test-labels")?;
    let testing = dataset(&images, &labels, 10_000)?;
    let device = d3os_ai::native_device().map_err(|e| format!("MNIST GPU: {e}"))?;
    let model = artifact::load(&read_file("/fwcfg/mnist.model")?, &device)?;
    let backend = Nvidia::new().map_err(|e| format!("MNIST compute: {e}"))?;
    let actual = backend
        .with_commands(|| Ok::<_, Error>(evaluate(&model, &testing, batch_size)))
        .map_err(|e| format!("MNIST inference scope: {e}"))??;
    report(format_args!(
        "INFERENCE_OK mnist mode=infer os=d3os backend=nvidia test_correct={} test_samples={} accuracy={:.6} predictions_crc32={:08x}",
        actual.correct,
        actual.samples,
        actual.accuracy(),
        actual.predictions_crc32
    ));
    Ok(())
}

fn benchmark_logits(model: &Mnist, data: &Dataset<'_>, samples: usize, batch_size: usize) -> Result<Vec<f32>, String> {
    let model = model.valid();
    let device = model.devices().into_iter().next().ok_or("MNIST model has no device")?;
    let mut output = Vec::with_capacity(samples * 10);
    for offset in (0..samples).step_by(batch_size) {
        let indices: Vec<_> = (offset..(offset + batch_size).min(samples)).collect();
        let (input, _) = data.batch(&indices, &device)?;
        let values = model.forward(input).try_into_vec_as::<f32>().map_err(|e| format!("MNIST benchmark logits: {e:?}"))?;
        output.extend(values);
    }
    Ok(output)
}

fn export_values(name: &str, values: &[f32]) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("non-finite MNIST benchmark reference: {name}"));
    }
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    export(name, &bytes)
}

/// Separate diagnostics. Device timestamps change submission behavior, so these
/// intervals never contribute to the ordinary training throughput benchmark.
pub(super) fn profile() -> Result<(), String> {
    let images = read_file("/fwcfg/mnist.train-images")?;
    let labels = read_file("/fwcfg/mnist.train-labels")?;
    let training = dataset(&images, &labels, 60_000)?;
    let device = d3os_ai::native_device().map_err(|e| format!("MNIST GPU: {e}"))?.autodiff();
    let backend = Nvidia::new().map_err(|e| format!("MNIST compute: {e}"))?;
    let config = Config {
        seed: 42,
        batch_size: 64,
        learning_rate: 0.0001,
    };
    let mut trainer = Trainer::from_artifact(&read_file("/fwcfg/mnist.model")?, &training, &device, config)?;
    for _ in 0..4 {
        trainer = backend
            .with_commands(|| Ok::<_, Error>(trainer.step(&training)))
            .map_err(|e| format!("MNIST profile warmup: {e}"))??
            .0;
    }
    for instrumented in [false, true] {
        backend.set_device_timing(instrumented).map_err(|e| format!("MNIST device timing: {e}"))?;
        let measured = (|| {
            let before = backend.profile().map_err(|e| format!("MNIST profile: {e}"))?;
            let queue = backend.queue_profile().map_err(|e| format!("MNIST queue profile: {e}"))?;
            let service = backend.service_profile().map_err(|e| format!("MNIST service profile: {e}"))?;
            let start = time::systime().num_milliseconds();
            let mut next = trainer;
            for _ in 0..8 {
                next = backend
                    .with_commands(|| Ok::<_, Error>(next.step(&training)))
                    .map_err(|e| format!("MNIST profile step: {e}"))??
                    .0;
            }
            let elapsed_ms = time::systime().num_milliseconds() - start;
            let after = backend.profile().map_err(|e| format!("MNIST profile: {e}"))?;
            let after_queue = backend.queue_profile().map_err(|e| format!("MNIST queue profile: {e}"))?;
            let after_service = backend.service_profile().map_err(|e| format!("MNIST service profile: {e}"))?;
            let phase = if instrumented { "mnist_train_device" } else { "mnist_train" };
            report(format_args!(
                "MNIST_TRAIN_PROFILE os=d3os backend=nvidia instrumented={instrumented} steps=8 samples=512 elapsed_ms={elapsed_ms} final_step={} phase={phase} performance_comparison=false",
                next.progress().steps
            ));
            super::benchmark::profile(&before, &after, phase);
            super::benchmark::queue_profile(queue, after_queue, phase);
            super::benchmark::service_profile(&service, &after_service, phase);
            Ok::<_, String>(next)
        })();
        let stopped = backend.set_device_timing(false);
        trainer = measured?;
        stopped.map_err(|e| format!("MNIST stop device timing: {e}"))?;
    }
    profile_matrices(&backend).map_err(|e| format!("MNIST matrix profile: {e}"))?;
    report(format_args!("INFERENCE_OK mnist mode=profile os=d3os backend=nvidia"));
    Ok(())
}

/// Isolate the dense products used by one batch-64 training step. Resident
/// constant inputs exclude views, allocation, upload, and optimizer work.
fn profile_matrices(backend: &Nvidia) -> d3os_compute::Result<()> {
    use d3os_nvidia::compute::Kernel;
    const REPEATS: usize = 16;
    for (rows, inner, cols, calls_per_step) in [
        (64, 512, 64, 1),
        (64, 64, 10, 2),
        (64, 10, 64, 1),
        (64, 64, 512, 1),
        (512, 64, 64, 1),
        (400, 32, 4096, 1),
        (32, 4096, 400, 1),
        (16, 36864, 25, 1),
    ] {
        let left = backend.produce_words(rows * inner, Kernel::Fill, &[], &[rows * inner, 0.125f32.to_bits() as usize])?;
        let right = backend.produce_words(inner * cols, Kernel::Fill, &[], &[inner * cols, 0.25f32.to_bits() as usize])?;
        let output = backend.allocate_words(rows * cols)?;
        let variants: &[usize] = if rows <= 32 && inner >= 4096 { &[1, 8, 16, 32, 64, 128] } else { &[1] };
        for &splits in variants {
            let partials = backend.allocate_words(rows * cols * splits)?;
            let run = || {
                if splits == 1 {
                    backend.dispatch_words(Kernel::Matmul, &[(&left, 0), (&right, 0), (&output, 0)], &[rows, inner, cols])
                } else {
                    backend.dispatch_words(Kernel::Matmul, &[(&left, 0), (&right, 0), (&partials, 0)], &[rows, inner, cols, splits])?;
                    backend.dispatch_words(Kernel::SumProjectionPartials, &[(&partials, 0), (&output, 0)], &[rows, cols, splits])
                }
            };
            run()?;
            backend.set_device_timing(true)?;
            let measured = (|| {
                let before = backend.profile()?;
                for _ in 0..REPEATS {
                    run()?;
                }
                let after = backend.profile()?;
                let elapsed = |kernel: Kernel| after[kernel as usize].device_ns.wrapping_sub(before[kernel as usize].device_ns);
                let matmul_ns = elapsed(Kernel::Matmul);
                let reduction_ns = elapsed(Kernel::SumProjectionPartials);
                let device_ns = matmul_ns + reduction_ns;
                report(format_args!(
                    "MNIST_MATMUL_PROFILE os=d3os backend=nvidia rows={rows} inner={inner} cols={cols} splits={splits} calls_per_step={calls_per_step} repeats={REPEATS} device_ns={device_ns} matmul_ns={matmul_ns} reduction_ns={reduction_ns} instrumented=true performance_comparison=false"
                ));
                Ok::<_, Error>(())
            })();
            let stopped = backend.set_device_timing(false);
            measured?;
            stopped?;
        }
    }
    Ok(())
}

/// Fixed work shared with the Linux Burn reference. No checkpoint or export work
/// enters a timed interval. Each training repetition starts with fresh Adam state.
pub(super) fn benchmark() -> Result<(), String> {
    const WARMUP_STEPS: usize = 4;
    const TIMED_STEPS: usize = 64;
    const REPEATS: usize = 3;
    const INFERENCE_BATCHES: [usize; 3] = [1, 64, 256];
    let images = read_file("/fwcfg/mnist.train-images")?;
    let labels = read_file("/fwcfg/mnist.train-labels")?;
    let test_images = read_file("/fwcfg/mnist.test-images")?;
    let test_labels = read_file("/fwcfg/mnist.test-labels")?;
    let training = dataset(&images, &labels, 60_000)?;
    let testing = dataset(&test_images, &test_labels, 10_000)?;
    let initial = read_file("/fwcfg/mnist.model")?;
    let start = time::systime().num_milliseconds();
    let device = d3os_ai::native_device().map_err(|e| format!("MNIST GPU: {e}"))?;
    let backend = Nvidia::new().map_err(|e| format!("MNIST compute: {e}"))?;
    let model = artifact::load(&initial, &device)?;
    device.sync().map_err(|e| format!("MNIST model load: {e:?}"))?;
    let load_ms = time::systime().num_milliseconds() - start;
    let config = Config { seed: 42, batch_size: 64, learning_rate: 0.0001 };
    let contract = serde_json::json!({
        "execution_os": "d3os", "framework": "burn", "burn_version": "0.22.1",
        "backend": "nvidia", "precision": "fp32", "model": "conv16-conv32-fc64-fc10",
        "cpu_isa": "x86-64-v3",
        "training_samples": training.len(), "test_samples": testing.len(),
        "training_fingerprint": training.fingerprint(), "test_fingerprint": testing.fingerprint(),
        "seed": config.seed, "batch_size": config.batch_size, "learning_rate": config.learning_rate,
        "optimizer": {"name": "adam", "beta_1": 0.9, "beta_2": 0.999, "epsilon": 0.00001, "amsgrad": false},
        "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS, "repeats": REPEATS,
        "inference_batches": INFERENCE_BATCHES, "model_load_ms": load_ms,
        "training_scope": "input_normalization_upload_forward_backward_eight_gradients_adam_sync_loss_read",
        "inference_scope": "full_test_split_normalization_upload_forward_argmax_accuracy_sync_prediction_read",
        "timing_excludes": ["model_load", "compilation", "autotuning", "warmup", "export", "reference_dump", "logging"]
    });
    report(format_args!("MNIST_BENCH_CONTRACT {contract}"));
    for batch_size in INFERENCE_BATCHES {
        let start = time::systime().num_milliseconds();
        let expected = backend
            .with_commands(|| Ok::<_, Error>(evaluate(&model, &testing, batch_size)))
            .map_err(|e| format!("MNIST warmup scope: {e}"))??;
        let warmup_ms = time::systime().num_milliseconds() - start;
        if expected.accuracy() < 0.99 {
            return Err(format!("MNIST starting model accuracy is below 99 percent: {}", expected.accuracy()));
        }
        report(format_args!("MNIST_BENCH_WARMUP os=d3os backend=nvidia mode=inference batch={batch_size} elapsed_ms={warmup_ms}"));
        for run in 0..REPEATS {
            let start = time::systime().num_milliseconds();
            let result = backend
                .with_commands(|| Ok::<_, Error>(evaluate(&model, &testing, batch_size)))
                .map_err(|e| format!("MNIST inference benchmark scope: {e}"))??;
            let elapsed_ms = time::systime().num_milliseconds() - start;
            if elapsed_ms <= 0 || result.correct != expected.correct || result.predictions_crc32 != expected.predictions_crc32 {
                return Err(String::from("MNIST inference timing or repeated predictions changed"));
            }
            report(format_args!(
                "MNIST_BENCH os=d3os framework=burn backend=nvidia precision=fp32 mode=inference run={run} batch={batch_size} samples={} correct={} predictions_crc32={:08x} elapsed_ms={elapsed_ms} images_per_second={:.6}",
                result.samples, result.correct, result.predictions_crc32, result.samples as f64 * 1000.0 / elapsed_ms as f64
            ));
        }
        let values = backend
            .with_commands(|| Ok::<_, Error>(benchmark_logits(&model, &testing, testing.len(), batch_size)))
            .map_err(|e| format!("MNIST inference reference scope: {e}"))??;
        export_values(&format!("inference-{batch_size}-logits.f32"), &values)?;
    }
    let device = device.autodiff();
    for run in 0..REPEATS {
        let mut trainer = Trainer::from_artifact(&initial, &training, &device, config)?;
        let mut losses = Vec::with_capacity(TIMED_STEPS);
        let start = time::systime().num_milliseconds();
        for _ in 0..WARMUP_STEPS {
            trainer = backend
                .with_commands(|| Ok::<_, Error>(trainer.step(&training)))
                .map_err(|e| format!("MNIST training warmup scope: {e}"))??.0;
        }
        let warmup_ms = time::systime().num_milliseconds() - start;
        let start = time::systime().num_milliseconds();
        let mut samples = 0;
        for _ in 0..TIMED_STEPS {
            let (next, step) = backend
                .with_commands(|| Ok::<_, Error>(trainer.step(&training)))
                .map_err(|e| format!("MNIST training benchmark scope: {e}"))??;
            trainer = next;
            samples += step.samples;
            losses.push(step.loss);
        }
        let elapsed_ms = time::systime().num_milliseconds() - start;
        if elapsed_ms <= 0 {
            return Err(String::from("MNIST training interval is below timer resolution"));
        }
        report(format_args!(
            "MNIST_BENCH os=d3os framework=burn backend=nvidia precision=fp32 mode=train run={run} batch={} warmup_steps={WARMUP_STEPS} timed_steps={TIMED_STEPS} samples={samples} warmup_ms={warmup_ms} elapsed_ms={elapsed_ms} images_per_second={:.6} steps_per_second={:.6} final_step={} final_epoch={} final_offset={}",
            config.batch_size, samples as f64 * 1000.0 / elapsed_ms as f64, TIMED_STEPS as f64 * 1000.0 / elapsed_ms as f64,
            trainer.progress().steps, trainer.progress().epoch, trainer.progress().offset
        ));
        export_values(&format!("train-{run}-losses.f32"), &losses)?;
        let values = backend
            .with_commands(|| Ok::<_, Error>(benchmark_logits(trainer.model(), &testing, 256, 64)))
            .map_err(|e| format!("MNIST training reference scope: {e}"))??;
        export_values(&format!("train-{run}-logits.f32"), &values)?;
        export(&format!("train-{run}.model"), &artifact::export(trainer.model())?)?;
    }
    report(format_args!("INFERENCE_OK mnist mode=bench os=d3os backend=nvidia"));
    Ok(())
}
