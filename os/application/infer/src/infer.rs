#![no_std]

extern crate alloc;

use alloc::{format, string::String, vec::Vec};
use core::fmt;
use d3os_compute::transformer::TransformerBackend;
use d3os_compute::{Backend, cpu::Cpu, nvidia::Nvidia};
use d3os_inference::{
    qwen::{self, Qwen},
    tokenizer::{self, Tokenizer},
};
use naming::shared_types::{OpenOptions, SeekOrigin};
use runtime::env;

static LOGGER: logger::Logger = logger::Logger::new();

mod benchmark;
mod gpu_tests;

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
        None | Some("self-test") if args.len() <= 1 => {
            d3os_inference::self_test(&Cpu).map_err(|error| format!("{error}"))?;
            report(format_args!("INFERENCE_OK self-test backend=cpu"));
        }
        Some("gpu-self-test") if args.len() == 1 => {
            let backend = Nvidia::new().map_err(|error| format!("{error}"))?;
            d3os_inference::self_test(&backend).map_err(|error| format!("{error}"))?;
            gpu_tests::run(&backend)?;
            report(format_args!("INFERENCE_OK self-test backend=nvidia"));
        }
        Some("gpu-kernel-bench") if args.len() == 1 => benchmark::kernels(&Nvidia::new().map_err(|error| format!("{error}"))?)?,
        Some("gpu-alexnet-bench") if args.len() == 1 => benchmark::alexnet(&Nvidia::new().map_err(|error| format!("{error}"))?)?,
        Some("gpu-qwen-bench") if args.len() == 1 => benchmark::qwen(&Nvidia::new().map_err(|error| format!("{error}"))?)?,
        Some("gpu-alexnet") if args.len() == 1 || args.len() == 3 => {
            let backend = Nvidia::new().map_err(|error| format!("{error}"))?;
            alexnet(&backend, &args[1..])?;
        }
        Some("alexnet") if args.len() == 1 || args.len() == 3 => {
            alexnet(&Cpu, &args[1..])?;
        }
        Some("qwen") if args.len() == 1 || args.len() >= 3 => generate_qwen(&Cpu, &args[1..])?,
        Some("gpu-qwen") if args.len() == 1 || args.len() >= 3 => {
            let backend = Nvidia::new().map_err(|error| format!("{error}"))?;
            generate_qwen(&backend, &args[1..])?;
        }
        Some("qwen-verify") if args.len() == 1 => verify_qwen(&Cpu)?,
        Some("gpu-qwen-verify") if args.len() == 1 => {
            let backend = Nvidia::new().map_err(|error| format!("{error}"))?;
            verify_qwen(&backend)?;
        }
        _ => {
            return Err(String::from(
                "usage: infer self-test | gpu-self-test | gpu-kernel-bench | alexnet [MODEL FIXTURE] | gpu-alexnet [MODEL FIXTURE] | gpu-alexnet-bench | qwen [MAX_TOKENS PROMPT...] | qwen-verify | gpu-qwen [MAX_TOKENS PROMPT...] | gpu-qwen-verify | gpu-qwen-bench",
            ));
        }
    }
    Ok(())
}

fn alexnet<B: Backend>(backend: &B, paths: &[String]) -> Result<(), String> {
    let defaults = ["/usr/share/inference/alexnet.d3t", "/usr/share/inference/dog.d3t"];
    let paths = if paths.is_empty() { defaults } else { [paths[0].as_str(), paths[1].as_str()] };
    let start = time::systime().num_milliseconds();
    report(format_args!("INFERENCE_START alexnet backend={}", backend.name()));
    let model = read_file(paths[0])?;
    let fixture = read_file(paths[1])?;
    let predictions = d3os_inference::verify_alexnet(backend, &model, &fixture).map_err(|error| format!("{error}"))?;
    for (class, probability) in predictions {
        report(format_args!("class={class} probability={probability:.6}"));
    }
    report(format_args!(
        "INFERENCE_OK alexnet backend={} layers=8 elapsed_ms={}",
        backend.name(),
        time::systime().num_milliseconds() - start
    ));
    Ok(())
}

fn generate_qwen<B: TransformerBackend>(backend: &B, args: &[String]) -> Result<(), String> {
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
        "INFERENCE_START qwen3.5-0.8b backend={} prompt_tokens={}",
        backend.name(),
        tokens.len()
    ));
    let bytes = read_file("/usr/share/inference/qwen3.5-0.8b.d3t")?;
    let model = Qwen::load(backend, &bytes).map_err(|error| format!("{error}"))?;
    drop(bytes);
    let generated = model.generate(backend, &tokens, limit).map_err(|error| format!("{error}"))?;
    report(format_args!(
        "{}",
        tokenizer.decode(&generated.tokens, true).map_err(|error| format!("{error}"))?
    ));
    report(format_args!(
        "INFERENCE_OK qwen backend={} tokens={} stop={:?} elapsed_ms={}",
        backend.name(),
        generated.tokens.len(),
        generated.stop,
        time::systime().num_milliseconds() - start
    ));
    Ok(())
}

fn verify_qwen<B: TransformerBackend>(backend: &B) -> Result<(), String> {
    let start = time::systime().num_milliseconds();
    report(format_args!("INFERENCE_START qwen-verify backend={}", backend.name()));
    let data = read_file("/usr/share/inference/qwen3.5.bpe")?;
    let tokenizer = Tokenizer::parse(&data).map_err(|error| format!("{error}"))?;
    let cases = tokenizer::verify_fixture(&tokenizer, &read_file("/usr/share/inference/tokenizer-fixture.bin")?).map_err(|error| format!("{error}"))?;
    report(format_args!("QWEN_TOKENIZER_OK cases={cases}"));
    let bytes = read_file("/usr/share/inference/qwen3.5-0.8b.d3t")?;
    let model = Qwen::load(backend, &bytes).map_err(|error| format!("{error}"))?;
    drop(bytes);
    let fixture = read_file("/usr/share/inference/qwen-fixture.d3t")?;
    let positions = qwen::verify_fixture(&model, backend, &fixture).map_err(|error| format!("{error}"))?;
    report(format_args!(
        "INFERENCE_OK qwen-verify backend={} positions={positions} layers={} elapsed_ms={}",
        backend.name(),
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
