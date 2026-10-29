#![no_std]

extern crate alloc;

use alloc::{format, string::String, vec::Vec};
use core::fmt;
use d3os_ai::burn::tensor::Device;
use d3os_ai::qwen::{
    self, Qwen,
    tokenizer::{self, Tokenizer},
};
use d3os_compute::nvidia::Nvidia;
use naming::shared_types::{OpenOptions, SeekOrigin};
use runtime::env;

static LOGGER: logger::Logger = logger::Logger::new();

mod benchmark;
mod file_tests;
mod gpu_tests;
mod mnist;
mod model_file;

#[derive(Clone, Copy)]
enum QwenSource {
    Archive,
    FirmwareGguf,
}

impl QwenSource {
    fn label(self) -> &'static str {
        match self {
            Self::Archive => "Qwen3.5-0.8B",
            Self::FirmwareGguf => "Qwen GGUF",
        }
    }
    fn load(self, device: &Device) -> Result<Qwen, String> {
        match self {
            Self::Archive => {
                let bytes = read_file("/usr/share/inference/qwen3.5-0.8b.d3t")?;
                Qwen::load(device, &bytes).map_err(|e| format!("{e}"))
            }
            Self::FirmwareGguf => model_file::load(device).map(|(model, _)| model),
        }
    }
}

fn report(message: fmt::Arguments<'_>) {
    terminal::println!("{}", message);
    log::info!("{message}");
}

struct File(usize);

impl Drop for File {
    fn drop(&mut self) {
        let _ = naming::close(self.0);
    }
}

fn read_file(path: &str) -> Result<Vec<u8>, String> {
    let file = File(naming::open(path, OpenOptions::READONLY).map_err(|error| format!("cannot open {path}: {error:?}"))?);
    let size = naming::seek(file.0, 0, SeekOrigin::End).map_err(|error| format!("cannot size {path}: {error:?}"))?;
    naming::seek(file.0, 0, SeekOrigin::Start).map_err(|error| format!("cannot seek {path}: {error:?}"))?;
    let mut data = Vec::new();
    data.try_reserve_exact(size).map_err(|_| format!("cannot allocate {size} bytes for {path}"))?;
    data.resize(size, 0);
    let mut offset = 0;
    while offset < size {
        let count = naming::read(file.0, &mut data[offset..]).map_err(|error| format!("cannot read {path}: {error:?}"))?;
        if count == 0 {
            return Err(format!("unexpected end of file: {path}"));
        }
        offset += count;
    }
    Ok(data)
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("burn-state-self-test") if args.len() == 1 => {
            d3os_ai::validation::running_state::run(&Device::flex().autodiff())?;
            report(format_args!(
                "INFERENCE_OK burn-state-self-test backend=cpu threads=3 identity_checks=192 record=ok"
            ));
        }
        Some("burn-self-test") if args.len() == 1 => {
            let device = d3os_ai::burn::tensor::Device::flex().autodiff();
            device.seed(42);
            d3os_ai::validation::training(&device)?;
            report(format_args!(
                "INFERENCE_OK burn-self-test backend=cpu gradients=ok aliasing=ok checkpoint=ok resume=ok"
            ));
        }
        Some("gpu-mnist-train") if args.len() == 1 => mnist::train()?,
        Some("gpu-mnist-infer") if args.len() == 1 => mnist::infer()?,
        Some("gpu-mnist-bench") if args.len() == 1 => mnist::benchmark()?,
        Some("gpu-mnist-profile") if args.len() == 1 => mnist::profile()?,
        Some("fwcfg-self-test") if args.len() == 1 => file_tests::firmware()?,
        None | Some("self-test") if args.len() <= 1 => {
            d3os_ai::self_test(&Device::flex())?;
            report(format_args!("INFERENCE_OK self-test backend=cpu"));
        }
        Some("gpu-self-test") if args.len() == 1 => {
            let backend = Nvidia::new().map_err(|error| format!("{error}"))?;
            d3os_ai::self_test(&d3os_ai::native_device().map_err(|e| format!("{e}"))?)?;
            gpu_tests::run(&backend)?;
            report(format_args!("INFERENCE_OK self-test backend=nvidia"));
        }
        Some("gpu-kernel-bench") if args.len() == 1 => benchmark::kernels(&Nvidia::new().map_err(|error| format!("{error}"))?)?,
        Some("gpu-alexnet-bench") if args.len() == 1 => benchmark::alexnet(&Nvidia::new().map_err(|error| format!("{error}"))?)?,
        Some("gpu-qwen-bench") => {
            let mut long = false;
            let mut device_timing = false;
            let mut firmware_model = false;
            for argument in &args[1..] {
                match argument.as_str() {
                    "long" if !long => long = true,
                    "device-timing" if !device_timing => device_timing = true,
                    "firmware-model" if !firmware_model => firmware_model = true,
                    _ => return Err(String::from("usage: infer gpu-qwen-bench [long] [device-timing] [firmware-model]")),
                }
            }
            let backend = Nvidia::new().map_err(|error| format!("{error}"))?;
            if long {
                benchmark::qwen::<4096>(&backend, device_timing, firmware_model)?;
            } else {
                benchmark::qwen::<64>(&backend, device_timing, firmware_model)?;
            }
        }
        Some("gpu-alexnet") if args.len() == 1 || args.len() == 3 => {
            alexnet(&d3os_ai::native_device().map_err(|e| format!("{e}"))?, &args[1..])?;
        }
        Some("alexnet") if args.len() == 1 || args.len() == 3 => {
            alexnet(&d3os_ai::burn::tensor::Device::flex(), &args[1..])?;
        }
        Some("qwen") if args.len() == 1 || args.len() >= 3 => generate_qwen(&Device::flex(), "cpu", QwenSource::Archive, &args[1..])?,
        Some("gpu-qwen") if args.len() == 1 || args.len() >= 3 => {
            generate_qwen(
                &d3os_ai::native_device().map_err(|e| format!("{e}"))?,
                "nvidia",
                QwenSource::Archive,
                &args[1..],
            )?;
        }
        Some("gpu-qwen-gguf") if args.len() == 1 || args.len() >= 3 => {
            generate_qwen(
                &d3os_ai::native_device().map_err(|e| format!("{e}"))?,
                "nvidia",
                QwenSource::FirmwareGguf,
                &args[1..],
            )?;
        }
        Some("gpu-qwen-gguf-bench") => {
            let mut device_timing = false;
            let mut paired_q4 = false;
            let mut quantization = qwen::PrefillQuantization::Accurate;
            for option in &args[1..] {
                match option.as_str() {
                    "device-timing" if !device_timing => device_timing = true,
                    "paired-q4" if !paired_q4 => paired_q4 = true,
                    "q8-prefill" if quantization == qwen::PrefillQuantization::Accurate => quantization = qwen::PrefillQuantization::Q8_1,
                    "q8-attention-prefill" if quantization == qwen::PrefillQuantization::Accurate => {
                        quantization = qwen::PrefillQuantization::Q8_1WithAttention;
                    }
                    _ => {
                        return Err(String::from(
                            "usage: infer gpu-qwen-gguf-bench [device-timing] [paired-q4|q8-prefill|q8-attention-prefill]",
                        ));
                    }
                }
            }
            let backend = Nvidia::new().map_err(|e| format!("{e}"))?;
            model_file::benchmark(
                &d3os_ai::native_device().map_err(|e| format!("{e}"))?,
                &backend,
                device_timing,
                quantization,
                paired_q4,
            )?;
        }
        Some("gpu-qwen-gguf-verify") if args.len() == 1 => {
            model_file::verify(&d3os_ai::native_device().map_err(|e| format!("{e}"))?)?;
        }
        Some("qwen-verify") if args.len() == 1 => verify_qwen(&Device::flex(), "cpu")?,
        Some("gpu-qwen-verify") if args.len() == 1 => {
            verify_qwen(&d3os_ai::native_device().map_err(|e| format!("{e}"))?, "nvidia")?;
        }
        _ => {
            return Err(String::from(
                "usage: infer self-test | burn-self-test | gpu-mnist-train | gpu-mnist-infer | gpu-mnist-bench | gpu-mnist-profile | fwcfg-self-test | gpu-self-test | gpu-kernel-bench | alexnet [MODEL FIXTURE] | gpu-alexnet [MODEL FIXTURE] | gpu-alexnet-bench | qwen [MAX_TOKENS PROMPT...] | qwen-verify | gpu-qwen [MAX_TOKENS PROMPT...] | gpu-qwen-verify | gpu-qwen-bench [long] [device-timing] [firmware-model] | gpu-qwen-gguf [MAX_TOKENS PROMPT...] | gpu-qwen-gguf-bench [device-timing] [paired-q4|q8-prefill|q8-attention-prefill] | gpu-qwen-gguf-verify",
            ));
        }
    }
    Ok(())
}

fn alexnet(device: &d3os_ai::burn::tensor::Device, paths: &[String]) -> Result<(), String> {
    let defaults = ["/usr/share/inference/alexnet.d3t", "/usr/share/inference/dog.d3t"];
    let paths = if paths.is_empty() { defaults } else { [paths[0].as_str(), paths[1].as_str()] };
    let start = time::systime().num_milliseconds();
    let backend = if *device == d3os_ai::burn::tensor::Device::flex() { "cpu" } else { "nvidia" };
    report(format_args!("INFERENCE_START alexnet backend={backend} framework=burn"));
    let model = read_file(paths[0])?;
    let fixture = read_file(paths[1])?;
    let predictions = d3os_ai::alexnet::verify(device, &model, &fixture)?;
    for (class, probability) in predictions {
        report(format_args!("class={class} probability={probability:.6}"));
    }
    report(format_args!(
        "INFERENCE_OK alexnet backend={backend} layers=8 framework=burn elapsed_ms={}",
        time::systime().num_milliseconds() - start
    ));
    Ok(())
}

fn generate_qwen(device: &Device, backend: &str, source: QwenSource, args: &[String]) -> Result<(), String> {
    let start = time::systime().num_milliseconds();
    let limit = if args.is_empty() {
        32
    } else {
        args[0]
            .parse::<usize>()
            .ok()
            .filter(|&limit| limit > 0)
            .ok_or_else(|| String::from("MAX_TOKENS must be a positive integer"))?
    };
    let prompt = if args.is_empty() {
        String::from("What is 2 + 2? Answer briefly.")
    } else {
        args[1..].join(" ")
    };
    let data = read_file("/usr/share/inference/qwen3.5.bpe")?;
    let tokenizer = Tokenizer::parse(&data).map_err(|error| format!("{error}"))?;
    let tokens = tokenizer.encode_chat(&prompt).map_err(|error| format!("{error}"))?;
    report(format_args!(
        "INFERENCE_START qwen backend={} prompt_tokens={} framework=burn model={}",
        backend,
        tokens.len(),
        source.label()
    ));
    let model = source.load(device)?;
    report(format_args!("{} | {} | streaming greedy generation", source.label(), backend));
    let generation_start = time::systime().num_milliseconds();
    let mut first_token = None;
    let mut count = 0usize;
    let mut decoder = tokenizer.decoder(true);
    let mut transcript = String::new();
    let mut pending = String::new();
    let mut last_flush = generation_start;
    let generated = model
        .generate_stream(&tokens, limit, |token| {
            let now = time::systime().num_milliseconds();
            let first = *first_token.get_or_insert(now);
            count += 1;
            let text = decoder.push(token)?;
            transcript.push_str(&text);
            pending.push_str(&text);
            if now - last_flush >= 50 || count == 1 {
                terminal::print!("{}", pending);
                pending.clear();
                last_flush = now;
            }
            if count.is_multiple_of(256) {
                terminal::print!("{}", pending);
                pending.clear();
                report(format_args!(
                    "\n[{} tokens | {:.1} tokens/s | {} ms since first token]",
                    count,
                    (count - 1) as f64 * 1000. / (now - first).max(1) as f64,
                    now - first
                ));
            }
            Ok(())
        })
        .map_err(|error| format!("{error}"))?;
    let generation_end = time::systime().num_milliseconds();
    let tail = decoder.finish();
    transcript.push_str(&tail);
    terminal::println!("{}{}", pending, tail);
    log::info!("QWEN_TRANSCRIPT_BEGIN\n{}\nQWEN_TRANSCRIPT_END", transcript);
    let first = first_token.unwrap_or(generation_end);
    report(format_args!(
        "QWEN_STREAM tokens={} first_token_ms={} decode_ms={} tokens_per_second={:.3}",
        count,
        first - generation_start,
        generation_end - first,
        count.saturating_sub(1) as f64 * 1000. / (generation_end - first).max(1) as f64
    ));
    report(format_args!(
        "INFERENCE_OK qwen backend={} tokens={} stop={:?} elapsed_ms={} framework=burn",
        backend,
        generated.tokens.len(),
        generated.stop,
        time::systime().num_milliseconds() - start
    ));
    Ok(())
}

fn verify_qwen(device: &Device, backend: &str) -> Result<(), String> {
    let start = time::systime().num_milliseconds();
    report(format_args!("INFERENCE_START qwen-verify backend={}", backend));
    let data = read_file("/usr/share/inference/qwen3.5.bpe")?;
    let tokenizer = Tokenizer::parse(&data).map_err(|error| format!("{error}"))?;
    let cases = tokenizer::verify_fixture(&tokenizer, &read_file("/usr/share/inference/tokenizer-fixture.bin")?).map_err(|error| format!("{error}"))?;
    report(format_args!("QWEN_TOKENIZER_OK cases={cases}"));
    let bytes = read_file("/usr/share/inference/qwen3.5-0.8b.d3t")?;
    let model = Qwen::load(device, &bytes).map_err(|error| format!("{error}"))?;
    drop(bytes);
    let fixture = read_file("/usr/share/inference/qwen-fixture.d3t")?;
    let positions = qwen::verify_fixture(&model, &fixture).map_err(|error| format!("{error}"))?;
    report(format_args!(
        "INFERENCE_OK qwen-verify backend={} positions={positions} layers={} elapsed_ms={} framework=burn",
        backend,
        model.config().layers,
        time::systime().num_milliseconds() - start
    ));
    Ok(())
}

#[unsafe(no_mangle)]
pub fn main() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    if let Err(error) = run() {
        report(format_args!("INFERENCE_FAIL {error}"));
    }
}
