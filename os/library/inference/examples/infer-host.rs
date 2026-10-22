use d3os_inference::{
    qwen::{self, Qwen},
    tokenizer::{self, Tokenizer},
};
use std::{env, fs, path::Path, process, time::Instant};

use d3os_compute::cpu::Cpu;

fn main() {
    if let Err(error) = run() {
        eprintln!("INFERENCE_FAIL {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let start = Instant::now();
    match args.first().map(String::as_str) {
        Some("self-test") if args.len() == 1 => {
            d3os_inference::self_test(&Cpu)?;
            println!("INFERENCE_OK self-test backend=cpu");
        }
        Some("alexnet") if args.len() == 3 => {
            let model = fs::read(&args[1])?;
            let fixture = fs::read(&args[2])?;
            let predictions = d3os_inference::verify_alexnet(&Cpu, &model, &fixture)?;
            for (class, probability) in predictions {
                println!("class={class} probability={probability:.6}");
            }
            println!("INFERENCE_OK alexnet backend=cpu layers=8 elapsed_ms={}", start.elapsed().as_millis());
        }
        Some("qwen-verify") if args.len() == 2 => {
            let directory = Path::new(&args[1]);
            let data = fs::read(directory.join("qwen3.5.bpe"))?;
            let tokenizer = Tokenizer::parse(&data)?;
            let cases = tokenizer::verify_fixture(&tokenizer, &fs::read(directory.join("tokenizer-fixture.bin"))?)?;
            println!("QWEN_TOKENIZER_OK cases={cases}");
            let bytes = fs::read(directory.join("qwen3.5-0.8b.d3t"))?;
            let model = Qwen::load(&Cpu, &bytes)?;
            drop(bytes);
            let positions = qwen::verify_fixture(&model, &Cpu, &fs::read(directory.join("qwen-fixture.d3t"))?)?;
            println!(
                "INFERENCE_OK qwen-verify backend=cpu positions={positions} layers={} elapsed_ms={}",
                model.config().layers,
                start.elapsed().as_millis()
            );
        }
        Some("qwen") if args.len() >= 4 => {
            let directory = Path::new(&args[1]);
            let limit: usize = args[2].parse()?;
            let prompt = args[3..].join(" ");
            let data = fs::read(directory.join("qwen3.5.bpe"))?;
            let tokenizer = Tokenizer::parse(&data)?;
            let tokens = tokenizer.encode_chat(&prompt)?;
            let bytes = fs::read(directory.join("qwen3.5-0.8b.d3t"))?;
            let model = Qwen::load(&Cpu, &bytes)?;
            drop(bytes);
            let generated = model.generate(&Cpu, &tokens, limit)?;
            println!("{}", tokenizer.decode(&generated.tokens, true)?);
            println!(
                "INFERENCE_OK qwen backend=cpu tokens={} stop={:?} elapsed_ms={}",
                generated.tokens.len(),
                generated.stop,
                start.elapsed().as_millis()
            );
        }
        _ => return Err("usage: infer-host self-test | alexnet MODEL FIXTURE | qwen DIRECTORY MAX_TOKENS PROMPT... | qwen-verify DIRECTORY".into()),
    }
    Ok(())
}
