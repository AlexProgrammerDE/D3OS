//! Bounded exact typed ReLU. Summary enumeration never initializes a device.
use d3os_ai::{
    burn::{
        backend::DType,
        tensor::{Device, try_read_sync},
    },
    validation::relu::{CORPUS_VERSION, Options, Summary, expected_summary, run},
};

fn summary_json(s: &Summary) -> serde_json::Value {
    serde_json::json!({
        "dtypes": s.dtypes, "cases": s.cases(), "forward_cases": s.forward_cases,
        "high_level_cases": s.high_level_cases, "unique_cases": s.unique_cases,
        "backward_cases": s.backward_cases, "idempotence_cases": s.idempotence_cases,
        "mask_cases": s.mask_cases, "chain_steps": s.chain_steps,
        "source_checks": s.source_checks, "alias_checks": s.alias_checks,
        "gradient_checks": s.gradient_checks, "output_checks": s.output_checks,
        "poison_checks": s.poison_checks, "values": s.values,
        "nan_values": s.nan_values, "bool_values": s.bool_values,
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
            _ => {
                if let Some(value) = arg.strip_prefix("--dtype=") {
                    options.dtype = Some(match value {
                        "f64" => DType::F64,
                        "f32" => DType::F32,
                        "flex32" => DType::Flex32,
                        "f16" => DType::F16,
                        "bf16" => DType::BF16,
                        _ => panic!("unknown ReLU dtype: {value}"),
                    });
                } else {
                    panic!("unknown ReLU argument: {arg}");
                }
            }
        }
    }
    // Keep this branch before runtime initialization, device creation and sync.
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend, fusion) = (Device::cuda(0), "cuda", "enabled_materialized_chain_steps");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend, fusion) = (Device::flex(), "flex_cpu", "disabled");
    let summary = try_read_sync(run(&device, &options)).expect("ReLU executor").expect("exact typed ReLU");
    device.sync().expect("ReLU synchronization");
    println!(
        "BURN_TYPED_RELU_OK os=linux backend={backend} fusion={fusion} totals={} corpus={CORPUS_VERSION} exact_bits=true reference=integer_ieee_fields backward=stored_output mask_stores=u8,u32 materialized=true physical_padding=not_public no_flex32={} skip_rank_zero={} scope=bounded_public_tensor_dispatch_forward_dispatch_activation_backward_and_maskfill",
        summary_json(&summary),
        options.no_flex32,
        options.skip_rank_zero,
    );
}
