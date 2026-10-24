//! Train, checkpoint, and reload a Burn classifier inside D3OS.
use super::{read_file, report};
use alloc::{format, string::String};
use core::fmt::Write;
use d3os_ai::{
    burn::module::Module,
    mnist::{
        artifact,
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
