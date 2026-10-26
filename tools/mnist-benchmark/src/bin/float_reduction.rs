//! Shared floating reduction checks. Description does not initialize a device.
use d3os_ai::{
    burn::{
        backend::DType,
        tensor::{Device, try_read_sync},
    },
    validation::float_reduction::{Options, corpus, expected_summary, linux_error_diagnostic, run, success_footer, summary_json},
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
            "--native-error-extensions" => panic!("Native error extensions require the Native runner"),
            _ => {
                if let Some(dtype) = arg.strip_prefix("--dtype=") {
                    options.dtype = Some(match dtype {
                        "f64" => DType::F64,
                        "f32" => DType::F32,
                        "flex32" => DType::Flex32,
                        "f16" => DType::F16,
                        "bf16" => DType::BF16,
                        _ => panic!("unknown floating reduction dtype: {dtype}"),
                    });
                } else if let Some(id) = arg.strip_prefix("--linux-error-case=") {
                    error_case = Some(String::from(id));
                } else {
                    panic!("unknown floating reduction argument: {arg}");
                }
            }
        }
    }
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    if diagnostics {
        let rows=corpus(&options).into_iter().map(|c|serde_json::json!({
            "id":c.id,"label":c.label,"dtype":format!("{:?}",c.dtype),"evidence":format!("{:?}",c.evidence),
            "native_error":c.expected().err().map(|e|format!("{e:?}")),
            "execution":if c.model.is_some(){"symbolic_reference_only_no_large_fold"}else if c.expected().is_err(){"isolated_linux_panic_diagnostic_not_executed"}else{"selected_fixture_not_general_linux_equivalence"},
            "observation":if c.chain{"final_logical_read_only_physical_and_recording_scope_in_raw_harness"}else{"logical_values_and_metadata"},
        })).collect::<Vec<_>>();
        println!("{}", serde_json::json!({"diagnostics":rows,"device_initialized":false}));
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
    let report = try_read_sync(run(&device, &options)).expect("floating reduction executor");
    for failure in &report.failures {
        eprintln!("FLOAT_REDUCTION_FAILURE id={} detail={}", failure.case_id, failure.message);
    }
    let checks = &report.checks;
    println!(
        "BURN_FLOAT_REDUCTION_REPORT {}",
        serde_json::json!({
            "attempted":report.attempted,"passed":report.passed,"failed":report.failed,
            "model_passed":report.model_passed,"skipped_diagnostics":report.skipped_diagnostics,"observed_errors":report.observed_errors,
            "checks":{"input":checks.input,"source":checks.source,"alias":checks.alias,"output":checks.output,"error":checks.error,"recovery":checks.recovery,"model":checks.model},
            "failures":report.failures.iter().map(|f|serde_json::json!({"id":f.case_id,"message":f.message})).collect::<Vec<_>>(),
        })
    );
    let footer = success_footer(&options, &report).expect("floating reduction conformance");
    device.sync().expect("floating reduction synchronization");
    println!("{footer} os=linux backend={backend} fusion={fusion}");
}
