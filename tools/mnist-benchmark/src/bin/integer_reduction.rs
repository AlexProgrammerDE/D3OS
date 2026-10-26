//! Shared integer policy checks. Description never initializes a device.
use d3os_ai::{
    burn::{
        backend::DType,
        tensor::{Device, try_read_sync},
    },
    validation::integer_reduction::{Options, corpus, expected_summary, linux_error_diagnostic, run, success_footer, summary_json},
};

fn main() {
    let mut options = Options::default();
    let mut describe = false;
    let mut diagnostics = false;
    let mut error_case = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--expected-summary" => describe = true,
            "--diagnostics" => diagnostics = true,
            "--skip-rank-zero" => options.skip_rank_zero = true,
            "--nonempty" => options.include_empty = false,
            // Native's recoverable errors must never be called by this Linux
            // default runner. Isolated panic diagnostics are a separate path.
            "--native-error-extensions" => panic!("Native error extensions require the Native runner"),
            _ => {
                if let Some(dtype) = arg.strip_prefix("--dtype=") {
                    options.dtype = Some(match dtype {
                        "i64" => DType::I64,
                        "i32" => DType::I32,
                        "i16" => DType::I16,
                        "i8" => DType::I8,
                        "u64" => DType::U64,
                        "u32" => DType::U32,
                        "u16" => DType::U16,
                        "u8" => DType::U8,
                        _ => panic!("unknown integer reduction dtype: {dtype}"),
                    });
                } else if let Some(id) = arg.strip_prefix("--linux-error-case=") {
                    error_case = Some(String::from(id));
                } else {
                    panic!("unknown integer reduction argument: {arg}");
                }
            }
        }
    }
    // Both routes precede Device construction, runtime initialization and sync.
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    if diagnostics {
        let rows: Vec<_> = corpus(&options).into_iter().filter_map(|case| case.expected().err().map(|error| serde_json::json!({"id": case.id, "expected_native_error": format!("{error:?}"), "linux": "isolated_panic_diagnostic_not_executed"}))).collect();
        println!("{}", serde_json::json!({"diagnostics": rows}));
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend, fusion) = (Device::cuda(0), "cuda", "enabled_materialized_chain_steps");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend, fusion) = (Device::flex(), "flex_cpu", "disabled");
    if let Some(id) = error_case {
        let result = try_read_sync(linux_error_diagnostic(&device, &options, &id)).expect("isolated diagnostic executor");
        panic!("isolated Linux diagnostic {id}: {result:?}");
    }
    let report = try_read_sync(run(&device, &options)).expect("integer reduction executor");
    for failure in &report.failures {
        eprintln!("INTEGER_REDUCTION_FAILURE id={} detail={}", failure.case_id, failure.message);
    }
    println!(
        "BURN_INTEGER_REDUCTION_REPORT {}",
        serde_json::json!({
            "attempted": report.attempted, "passed": report.passed,
            "skipped_diagnostics": report.skipped_diagnostics,
            "observed_errors": report.observed_errors, "failed": report.failed,
            "failures": report.failures.iter().map(|f| serde_json::json!({"id": f.case_id, "message": f.message})).collect::<Vec<_>>(),
        })
    );
    let footer = success_footer(&options, &report).expect("integer reduction conformance");
    device.sync().expect("integer reduction synchronization");
    println!("{footer} os=linux backend={backend} fusion={fusion}");
}
