//! Typed Exp/Log qualification. Expected summaries never initialize a device.
use d3os_ai::{
    burn::{
        backend::DType,
        tensor::{Device, try_read_sync},
    },
    validation::typed_exp_log::{
        CORPUS_VERSION, FIXTURE_SHA256, Operation, Options, POLICY_SHA256, accuracy_json, expected_summary, run, success_footer, summary_json,
    },
};

fn main() {
    let mut options = Options::default();
    let mut describe = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--expected-summary" => describe = true,
            "--no-flex32" => options.no_flex32 = true,
            "--skip-rank-zero" => options.skip_rank_zero = true,
            "--finite" => options.finite = true,
            _ => {
                if let Some(value) = arg.strip_prefix("--dtype=") {
                    options.dtype = Some(match value {
                        "f64" => DType::F64,
                        "f32" => DType::F32,
                        "flex32" => DType::Flex32,
                        "f16" => DType::F16,
                        "bf16" => DType::BF16,
                        _ => panic!("unknown exp/log dtype: {value}"),
                    });
                } else if let Some(value) = arg.strip_prefix("--op=") {
                    options.op = Some(match value {
                        "exp" => Operation::Exp,
                        "log" => Operation::Log,
                        _ => panic!("unknown exp/log operation: {value}"),
                    });
                } else {
                    panic!("unknown exp/log argument: {arg}");
                }
            }
        }
    }
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend, fusion) = (Device::cuda(0), "cuda", "enabled_materialized_chain_steps");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend, fusion) = (Device::flex(), "flex_cpu", "disabled");
    let report = try_read_sync(run(&device, &options)).expect("exp/log executor").expect("exp/log runner");
    device.sync().expect("exp/log synchronization");
    println!(
        "BURN_TYPED_EXP_LOG_ACCURACY os=linux backend={backend} fusion={fusion} corpus={CORPUS_VERSION} policy_sha256={POLICY_SHA256} fixture_sha256={FIXTURE_SHA256} qualification_passed={} totals={} accuracy={}",
        report.is_success(),
        summary_json(&report.summary),
        accuracy_json(&report.accuracy)
    );
    if !report.is_success() {
        for example in &report.failures.examples {
            eprintln!("BURN_TYPED_EXP_LOG_FAILURE {example}");
        }
        panic!("typed exp/log qualification failed: {} failures", report.failures.count);
    }
    println!(
        "BURN_TYPED_EXP_LOG_OK os=linux backend={backend} fusion={fusion} corpus={CORPUS_VERSION} {}",
        success_footer(&options, &report)
    );
}
