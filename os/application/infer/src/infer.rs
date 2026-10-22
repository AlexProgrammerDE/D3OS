#![no_std]

extern crate alloc;

use alloc::{format, string::String, vec, vec::Vec};
use core::fmt;
use d3os_compute::cpu::Cpu;
use naming::shared_types::{OpenOptions, SeekOrigin};
use runtime::env;

static LOGGER: logger::Logger = logger::Logger::new();

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
        Some("alexnet") if args.len() == 1 || args.len() == 3 => {
            let paths = if args.len() == 3 {
                vec![args[1].as_str(), args[2].as_str()]
            } else {
                vec!["/usr/share/inference/alexnet.d3t", "/usr/share/inference/dog.d3t"]
            };
            let start = time::systime().num_milliseconds();
            report(format_args!("INFERENCE_START alexnet backend=cpu"));
            let model = read_file(paths[0])?;
            let fixture = read_file(paths[1])?;
            let predictions = d3os_inference::verify_alexnet(&Cpu, &model, &fixture).map_err(|error| format!("{error}"))?;
            for (class, probability) in predictions {
                report(format_args!("class={class} probability={probability:.6}"));
            }
            report(format_args!(
                "INFERENCE_OK alexnet backend=cpu layers=8 elapsed_ms={}",
                time::systime().num_milliseconds() - start
            ));
        }
        _ => return Err(String::from("usage: infer [self-test | alexnet [MODEL FIXTURE]]")),
    }
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
