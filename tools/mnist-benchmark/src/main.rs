//! Linux reference for the shared D3OS Burn model and optimizer.
use d3os_ai::{
    burn::{module::Module, tensor::Device},
    mnist::{
        Mnist, artifact,
        data::Dataset,
        training::{Config, Trainer, evaluate},
    },
};
use serde_json::json;
use std::{env, fs, path::Path, time::Instant};

const WARMUP_STEPS: usize = 4;
const TIMED_STEPS: usize = 64;
const REPEATS: usize = 3;
const INFERENCE_BATCHES: [usize; 3] = [1, 64, 256];

#[cfg(feature = "cuda")]
const BACKEND: &str = "cuda";
#[cfg(all(feature = "cpu", not(feature = "cuda")))]
const BACKEND: &str = "flex_cpu";

fn write_values(path: &Path, values: &[f32]) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("non-finite reference values: {}", path.display()));
    }
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn logits(model: &Mnist, dataset: &Dataset<'_>, samples: usize, batch: usize) -> Result<Vec<f32>, String> {
    let model = model.valid();
    let device = model.devices().into_iter().next().ok_or("MNIST model has no device")?;
    let mut output = Vec::with_capacity(samples * 10);
    for offset in (0..samples).step_by(batch) {
        let indices: Vec<_> = (offset..(offset + batch).min(samples)).collect();
        let (input, _) = dataset.batch(&indices, &device)?;
        let values = model
            .forward(input)
            .try_into_vec_as::<f32>()
            .map_err(|error| format!("MNIST logits: {error:?}"))?;
        output.extend(values);
    }
    Ok(output)
}

fn run(data: &Path, model_path: &Path, output: &Path) -> Result<(), String> {
    let read = |name| fs::read(data.join(name)).map_err(|error| format!("{name}: {error}"));
    let images = read("train-images-idx3-ubyte")?;
    let labels = read("train-labels-idx1-ubyte")?;
    let test_images = read("t10k-images-idx3-ubyte")?;
    let test_labels = read("t10k-labels-idx1-ubyte")?;
    let training = Dataset::parse(&images, &labels)?;
    let testing = Dataset::parse(&test_images, &test_labels)?;
    if training.len() != 60_000 || testing.len() != 10_000 {
        return Err(String::from("the benchmark requires the complete MNIST training and test splits"));
    }
    let initial = fs::read(model_path).map_err(|error| format!("{}: {error}", model_path.display()))?;
    // A new output directory prevents mixing results from separate executions.
    fs::create_dir(output).map_err(|error| format!("{}: {error}", output.display()))?;
    let start = Instant::now();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let device = Device::flex();
    let model = artifact::load(&initial, &device)?;
    device.sync().map_err(|error| format!("MNIST model load: {error:?}"))?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let config = Config {
        seed: 42,
        batch_size: 64,
        learning_rate: 0.0001,
    };
    let metadata = json!({
        "execution_os": "linux", "framework": "burn", "burn_version": "0.22.1",
        "backend": BACKEND, "precision": "fp32", "model": "conv16-conv32-fc64-fc10",
        "cpu_isa": if cfg!(all(target_feature = "avx2", target_feature = "fma", target_feature = "bmi2")) { "x86-64-v3" } else { "other" },
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
    fs::write(
        output.join("contract.json"),
        serde_json::to_vec_pretty(&metadata).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    println!("MNIST_BENCH_CONTRACT {metadata}");

    for batch_size in INFERENCE_BATCHES {
        let start = Instant::now();
        let expected = evaluate(&model, &testing, batch_size)?;
        let warmup_ms = start.elapsed().as_secs_f64() * 1000.0;
        if expected.accuracy() < 0.99 {
            return Err(format!("MNIST starting model accuracy is below 99 percent: {}", expected.accuracy()));
        }
        println!("MNIST_BENCH_WARMUP os=linux backend={BACKEND} mode=inference batch={batch_size} elapsed_ms={warmup_ms:.6}");
        for run in 0..REPEATS {
            let start = Instant::now();
            let result = evaluate(&model, &testing, batch_size)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if result.correct != expected.correct || result.predictions_crc32 != expected.predictions_crc32 {
                return Err(String::from("MNIST inference results changed between repetitions"));
            }
            println!(
                "MNIST_BENCH os=linux framework=burn backend={BACKEND} precision=fp32 mode=inference run={run} batch={batch_size} samples={} correct={} predictions_crc32={:08x} elapsed_ms={elapsed_ms:.6} images_per_second={:.6}",
                result.samples,
                result.correct,
                result.predictions_crc32,
                result.samples as f64 * 1000.0 / elapsed_ms
            );
        }
        write_values(
            &output.join(format!("inference-{batch_size}-logits.f32")),
            &logits(&model, &testing, testing.len(), batch_size)?,
        )?;
    }

    let device = device.autodiff();
    for run in 0..REPEATS {
        let mut trainer = Trainer::from_artifact(&initial, &training, &device, config)?;
        let mut losses = Vec::with_capacity(TIMED_STEPS);
        let start = Instant::now();
        for _ in 0..WARMUP_STEPS {
            trainer = trainer.step(&training)?.0;
        }
        let warmup_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let mut samples = 0;
        for _ in 0..TIMED_STEPS {
            let (next, step) = trainer.step(&training)?;
            trainer = next;
            samples += step.samples;
            losses.push(step.loss);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "MNIST_BENCH os=linux framework=burn backend={BACKEND} precision=fp32 mode=train run={run} batch={} warmup_steps={WARMUP_STEPS} timed_steps={TIMED_STEPS} samples={samples} warmup_ms={warmup_ms:.6} elapsed_ms={elapsed_ms:.6} images_per_second={:.6} steps_per_second={:.6} final_step={} final_epoch={} final_offset={}",
            config.batch_size,
            samples as f64 * 1000.0 / elapsed_ms,
            TIMED_STEPS as f64 * 1000.0 / elapsed_ms,
            trainer.progress().steps,
            trainer.progress().epoch,
            trainer.progress().offset
        );
        write_values(&output.join(format!("train-{run}-losses.f32")), &losses)?;
        write_values(&output.join(format!("train-{run}-logits.f32")), &logits(trainer.model(), &testing, 256, 64)?)?;
        fs::write(output.join(format!("train-{run}.model")), artifact::export(trainer.model())?).map_err(|error| error.to_string())?;
    }
    println!("LINUX_MNIST_BENCH_OK backend={BACKEND}");
    Ok(())
}

fn main() {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    let result = match arguments.as_slice() {
        [data, model, output] => run(Path::new(data), Path::new(model), Path::new(output)),
        _ => Err(String::from("usage: d3os-mnist-benchmark DATA_DIRECTORY MODEL_ARTIFACT NEW_OUTPUT_DIRECTORY")),
    };
    if let Err(error) = result {
        eprintln!("MNIST_BENCH_FAIL {error}");
        std::process::exit(1);
    }
}
