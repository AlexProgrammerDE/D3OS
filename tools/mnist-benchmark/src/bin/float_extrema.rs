//! Strict shared floating extrema checks. Pure plans never initialize a device.
use d3os_ai::{
    burn::{
        backend::DType,
        tensor::{Device, try_read_sync},
    },
    validation::float_extrema::{Options, corpus, expected_summary, linux_error_diagnostic, run, success_footer, summary_json},
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
            "--native-error-extensions" => options.native_error_extensions = true,
            _ => {
                if let Some(dtype) = arg.strip_prefix("--dtype=") {
                    options.dtype = Some(match dtype {
                        "f64" => DType::F64,
                        "f32" => DType::F32,
                        "flex32" => DType::Flex32,
                        "f16" => DType::F16,
                        "bf16" => DType::BF16,
                        _ => panic!("unknown floating extrema dtype: {dtype}"),
                    });
                } else if let Some(id) = arg.strip_prefix("--linux-error-case=") {
                    error_case = Some(String::from(id));
                } else {
                    panic!("unknown floating extrema argument: {arg}");
                }
            }
        }
    }
    // Native error scope is legal in both pure routes, in either argument order.
    // Reject Linux execution only after description, before Device construction.
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    if diagnostics {
        let rows: Vec<_> = corpus(&options)
            .into_iter()
            .filter_map(|case| {
                case.expected().err().map(|error|
            serde_json::json!({"id": case.id, "expected_native_error": format!("{error:?}"), "linux": "isolated_panic_diagnostic_not_executed"}))
            })
            .collect();
        let plan: serde_json::Value = serde_json::from_str(&summary_json(&expected_summary(&options))).expect("pure plan JSON");
        println!(
            "{}",
            serde_json::json!({"plan": plan, "diagnostics": rows,
            "native_bit_policy": "first_logical_nan_all_source_bits_and_first_numeric_tie_including_zero",
            "linux_source_guarantees": "coordinate_tracking_nan_priority_first_coordinate; value_payload_zero_sign_and_global_traversal_separate",
            "gpu_observations": "unmeasured", "physical_counters": "future_raw_owner_runtime_assertions"})
        );
        return;
    }
    assert!(
        !options.native_error_extensions,
        "Native error extensions require the Native runner; Linux supports pure --expected-summary or --diagnostics only"
    );
    #[cfg(feature = "cuda")]
    let (device, backend, fusion) = (Device::cuda(0), "cuda", "enabled_queued_dependencies");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend, fusion) = (Device::flex(), "flex_cpu", "disabled");
    if let Some(id) = error_case {
        let result = try_read_sync(linux_error_diagnostic(&device, &options, &id)).expect("isolated diagnostic executor");
        panic!("isolated Linux diagnostic {id}: {result:?}");
    }
    let report = try_read_sync(run(&device, &options)).expect("floating extrema executor");
    for failure in &report.failures {
        eprintln!("FLOAT_EXTREMA_FAILURE id={} detail={}", failure.case_id, failure.message);
    }
    println!(
        "BURN_FLOAT_EXTREMA_REPORT {}",
        serde_json::json!({"attempted": report.attempted, "passed": report.passed,
        "skipped_diagnostics": report.skipped_diagnostics, "observed_errors": report.observed_errors, "failed": report.failed,
        "checks": {"input": report.checks.input, "source": report.checks.source, "alias": report.checks.alias,
            "output": report.checks.output, "chain_index": report.checks.chain_index, "recovery": report.checks.recovery},
        "failures": report.failures.iter().map(|f| serde_json::json!({"id": f.case_id, "message": f.message})).collect::<Vec<_>>() })
    );
    let footer = success_footer(&options, &report).expect("strict floating extrema conformance");
    device.sync().expect("floating extrema synchronization");
    println!("{footer} os=linux backend={backend} fusion={fusion}");
}
