//! Shared mixed matrix checks. Inventory modes never initialize a device.
use d3os_ai::{
    burn::tensor::{Device, try_read_sync},
    validation::mixed_matmul::{self as mixed, DType, Options, Scope},
};
fn main() {
    let mut options = Options::default();
    let mut scope = Scope::Shared;
    let mut describe = false;
    let mut diagnostics = false;
    let mut frontend = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--expected-summary" => describe = true,
            "--diagnostics" => diagnostics = true,
            "--nonempty" => options.include_empty = false,
            "--native-error-extensions" => panic!("Native errors are separate raw-harness gates"),
            _ => {
                if let Some(dt) = arg.strip_prefix("--dtype=") {
                    options.dtype = Some(match dt {
                        "f16" => DType::F16,
                        "bf16" => DType::BF16,
                        "flex32" => DType::Flex32,
                        "f64" => DType::F64,
                        _ => panic!("unsupported mixed dtype {dt}"),
                    });
                } else if let Some(id) = arg.strip_prefix("--case=") {
                    options.case_id = Some(id.into());
                } else if let Some(s) = arg.strip_prefix("--scope=") {
                    scope = match s {
                        "shared" => Scope::Shared,
                        "native-extensions" => Scope::NativeExtensions,
                        "preservation" => Scope::Preservation,
                        "f64-diagnostics" => Scope::F64Diagnostics,
                        "all" => Scope::All,
                        _ => panic!("unknown scope {s}"),
                    };
                } else if let Some(id) = arg.strip_prefix("--frontend-diagnostic=") {
                    assert_eq!(id, mixed::FRONTEND_ZERO_LINEAR_ID, "unknown frontend diagnostic");
                    frontend = true;
                } else {
                    panic!("unknown mixed matrix argument {arg}")
                }
            }
        }
    }
    if describe {
        println!("{}", mixed::summary_json(&mixed::expected_summary(&options, scope)));
        return;
    }
    if diagnostics {
        let rows = mixed::fixtures().iter().map(|case| {
            let matrices=case.matrices.iter().map(|m|serde_json::json!({"name":m.name,"dtype":format!("{:?}",m.values.dtype),"shape":m.values.shape,"logical_bits":m.values.bits()})).collect::<Vec<_>>();
            let expected=case.expected.iter().map(|e|serde_json::json!({"name":e.name,"dtype":format!("{:?}",e.values.dtype),"shape":e.values.shape,"expected_bits":e.values.bits(),"stored_stage":e.stage})).collect::<Vec<_>>();
            let numeric=case.numeric.iter().map(|cell|serde_json::json!({"name":cell.name,"ideal_bits":cell.ideal_bits,"serial_bits":cell.serial_bits,"ideal_rational":cell.ideal_rational,"sum_abs_products":cell.sum_abs_products,"scratch_low":cell.scratch_low,"scratch_high":cell.scratch_high})).collect::<Vec<_>>();
            serde_json::json!({"id":case.id,"group":format!("{:?}",case.group),"dtype":case.dtype.map(|d|format!("{d:?}")),"operation":format!("{:?}",case.operation),"raw_matrices":case.raw_matrices.len(),"raw_expected":case.raw_expected.len(),"matrices":matrices,"expected":expected,"numeric":numeric,"status":"not_device_observed"})
        }).collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::json!({"device_initialized":false,"rows":rows,"expected_population":serde_json::from_str::<serde_json::Value>(&mixed::population_json(&options,scope)).expect("population JSON"),"frontend_diagnostic":{"id":mixed::FRONTEND_ZERO_LINEAR_ID,"label":"frontend_source_derived_not_backend_result"},"conditional_model":mixed::MODEL_LABEL,"full_public_qualification":false})
        );
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend, fusion) = {
        options.materialize_chain_steps = true;
        (Device::cuda(0), "cuda", "enabled_materialized_chain_steps")
    };
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend, fusion) = (Device::flex(), "flex_cpu", "disabled");
    #[cfg(not(any(feature = "cpu", feature = "cuda")))]
    panic!("select cuda or cpu feature");
    #[cfg(any(feature = "cpu", feature = "cuda"))]
    {
        if frontend {
            eprintln!(
                "MIXED_MATMUL_FRONTEND_DIAGNOSTIC id={} label=frontend_source_derived_not_backend_result",
                mixed::FRONTEND_ZERO_LINEAR_ID
            );
            mixed::frontend_zero_linear_diagnostic(&device);
            panic!("frontend diagnostic returned; record changed frontend behavior separately");
        }
        let report = try_read_sync(mixed::run(&device, &options, scope)).expect("mixed matrix executor");
        // Emit the complete report before asserting. Backend poison or missing
        // observations must remain visible even when later cases cannot run.
        println!("BURN_MIXED_MATMUL_REPORT {}", mixed::report_json(&report));
        for f in &report.failures {
            eprintln!("MIXED_MATMUL_FAILURE key={} detail={}", f.key.text(), f.message);
        }
        let footer = mixed::success_footer(&options, scope, &report).expect("mixed matrix logical conformance");
        device.sync().expect("mixed matrix final synchronization");
        println!("{footer} os=linux backend={backend} fusion={fusion}");
    }
}
