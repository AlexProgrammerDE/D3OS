//! Bounded exact typed float arithmetic. Description never initializes a device.
use d3os_ai::{
    burn::{
        backend::DType,
        tensor::{Device, try_read_sync},
    },
    validation::float_arithmetic::{CORPUS_VERSION, FIXTURE_SHA256, Operation, Options, Summary, expected_summary, run},
};

fn summary_json(s: &Summary) -> serde_json::Value {
    serde_json::json!({
        "dtypes": s.dtypes, "operations": s.operations, "cases": s.cases(),
        "tensor_cases": s.tensor_cases, "scalar_cases": s.scalar_cases,
        "unary_cases": s.unary_cases, "high_level_cases": s.high_level_cases,
        "chain_cases": s.chain_cases, "chain_steps": s.chain_steps,
        "source_checks": s.source_checks, "alias_checks": s.alias_checks,
        "poison_checks": s.poison_checks, "values": s.values,
        "exact_values": s.exact_values, "nan_values": s.nan_values,
    })
}
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
                        _ => panic!("unknown float dtype: {value}"),
                    });
                } else if let Some(value) = arg.strip_prefix("--op=") {
                    options.operation = Some(match value {
                        "add" => Operation::Add,
                        "sub" => Operation::Sub,
                        "mul" => Operation::Mul,
                        "div" => Operation::Div,
                        "neg" => Operation::Neg,
                        "abs" => Operation::Abs,
                        "recip" => Operation::Recip,
                        "sign" => Operation::Sign,
                        _ => panic!("unknown float operation: {value}"),
                    });
                } else {
                    panic!("unknown float arithmetic argument: {arg}");
                }
            }
        }
    }
    // Keep this ahead of Device construction, runtime initialization and sync.
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend, fusion) = (Device::cuda(0), "cuda", "enabled_materialized_chain_steps");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend, fusion) = (Device::flex(), "flex_cpu", "disabled");
    let summary = try_read_sync(run(&device, &options))
        .expect("float arithmetic executor")
        .expect("exact float arithmetic");
    device.sync().expect("float arithmetic synchronization");
    println!(
        "BURN_FLOAT_ARITHMETIC_OK os=linux backend={backend} fusion={fusion} corpus={CORPUS_VERSION} fixture_sha256={FIXTURE_SHA256} totals={} reference=python_fraction_ieee_rne finite_inf_signed_zero=exact_bits arithmetic_nan=classification scalar=f64_then_f32_then_half high_level_dtype=explicit chain=materialized_each_primitive poison=retained_allocation physical_padding=not_public no_flex32={} skip_rank_zero={} finite={}",
        summary_json(&summary),
        options.no_flex32,
        options.skip_rank_zero,
        options.finite,
    );
}
