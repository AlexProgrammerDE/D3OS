//! Inventory modes and MODEL checks do not initialize a device.
#[cfg(any(feature = "cpu", feature = "cuda"))]
use d3os_ai::burn::tensor::{Device, try_read_sync};
use d3os_ai::{
    burn::backend::DType,
    validation::indexed_accumulation::{self as indexed, ExecutionScope, Options},
};
use std::process::ExitCode;

fn fail(options: &Options, message: &str) -> ExitCode {
    println!(
        "BURN_INDEXED_ACCUMULATION_REPORT {}",
        indexed::observed_json(&indexed::rejected_report(options, message))
    );
    ExitCode::FAILURE
}
#[cfg(any(feature = "cpu", feature = "cuda"))]
fn interrupted(options: &Options, message: &str) -> ExitCode {
    println!(
        "BURN_INDEXED_ACCUMULATION_REPORT {}",
        indexed::observed_json(&indexed::interrupted_report(options, message))
    );
    ExitCode::FAILURE
}
fn main() -> ExitCode {
    let mut options = Options::default();
    let mut summary = false;
    let mut population = false;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--expected-summary" => summary = true,
            "--population" | "--diagnostics" => population = true,
            "--models" => options.scope = ExecutionScope::ModelReference,
            _ => {
                if let Some(scope) = argument.strip_prefix("--scope=") {
                    options.scope = match scope {
                        "linux-common" => ExecutionScope::LinuxCommon,
                        "native-values" => ExecutionScope::NativeValues,
                        "native-errors" => ExecutionScope::NativeErrors,
                        "model-reference" => ExecutionScope::ModelReference,
                        _ => return fail(&options, &format!("unknown scope {scope}")),
                    };
                } else if let Some(dtype) = argument.strip_prefix("--dtype=") {
                    options.dtype = Some(match dtype {
                        "f64" => DType::F64,
                        "f32" => DType::F32,
                        "flex32" => DType::Flex32,
                        "f16" => DType::F16,
                        "bf16" => DType::BF16,
                        "i64" => DType::I64,
                        "i32" => DType::I32,
                        "i16" => DType::I16,
                        "i8" => DType::I8,
                        "u64" => DType::U64,
                        "u32" => DType::U32,
                        "u16" => DType::U16,
                        "u8" => DType::U8,
                        _ => return fail(&options, &format!("unknown dtype {dtype}")),
                    });
                } else {
                    return fail(&options, &format!("unknown argument {argument}"));
                }
            }
        }
    }
    if summary {
        match indexed::expected_summary(&options) {
            Ok(s) => println!("{}", indexed::summary_json(&s)),
            Err(error) => return fail(&options, &format!("selection: {error:?}")),
        };
        return ExitCode::SUCCESS;
    }
    if population {
        match indexed::population(&options) {
            Ok(p) => println!("{}", indexed::population_json(&p)),
            Err(error) => return fail(&options, &format!("selection: {error:?}")),
        };
        return ExitCode::SUCCESS;
    }
    let report = if options.scope == ExecutionScope::ModelReference {
        indexed::run_models(&options)
    } else {
        if options.scope != ExecutionScope::LinuxCommon {
            return fail(&options, "Native execution requires Root's registered Native runner");
        }
        #[cfg(feature = "cuda")]
        let device = std::panic::catch_unwind(|| Device::cuda(0));
        #[cfg(all(feature = "cpu", not(feature = "cuda")))]
        let device = std::panic::catch_unwind(Device::flex);
        #[cfg(not(any(feature = "cpu", feature = "cuda")))]
        return fail(&options, "select the cpu or cuda feature for device execution");
        #[cfg(any(feature = "cpu", feature = "cuda"))]
        {
            let device = match device {
                Ok(device) => device,
                Err(_) => return fail(&options, "device initialization panicked; retain the accompanying panic diagnostic"),
            };
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| try_read_sync(indexed::run(&device, &options)))) {
                Ok(Some(report)) => report,
                Ok(None) => return interrupted(&options, "executor did not complete synchronously"),
                Err(_) => return interrupted(&options, "executor panicked; retain the accompanying panic diagnostic"),
            }
        }
    };
    // The finalized observed report always precedes success or a failure exit.
    println!("BURN_INDEXED_ACCUMULATION_REPORT {}", indexed::observed_json(&report));
    match indexed::success_footer(&report) {
        Ok(footer) => {
            println!("{footer}");
            ExitCode::SUCCESS
        }
        Err(reason) => {
            eprintln!("INDEXED_ACCUMULATION_FAILED {reason}");
            ExitCode::FAILURE
        }
    }
}
