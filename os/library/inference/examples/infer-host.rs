use std::{env, fs, process, time::Instant};

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
        _ => return Err("usage: infer-host self-test | alexnet MODEL FIXTURE".into()),
    }
    Ok(())
}
