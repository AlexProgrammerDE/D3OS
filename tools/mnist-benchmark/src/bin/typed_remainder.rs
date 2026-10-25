//! Strict Native remainder corpus. Description never initializes a device.
use d3os_ai::{
    burn::{
        backend::DType,
        tensor::{Device, try_read_sync},
    },
    validation::typed_remainder::{CORPUS_VERSION, FIXTURE_SHA256, GROUPS, Options, Summary, expected_summary, run},
};
fn summary_json(s: &Summary) -> serde_json::Value {
    serde_json::json!({
        "dtypes": s.dtypes, "cases": s.cases(), "tensor_cases": s.tensor_cases,
        "scalar_cases": s.scalar_cases, "high_level_cases": s.high_level_cases,
        "chain_cases": s.chain_cases, "chain_steps": s.chain_steps,
        "tensor_rows": s.tensor_rows, "scalar_specs": s.scalar_specs, "scalar_results": s.scalar_results,
        "source_checks": s.source_checks, "alias_checks": s.alias_checks, "poison_checks": s.poison_checks,
        "values": s.values, "exact_values": s.exact_values, "nan_values": s.nan_values,
    })
}
fn main() {
    let mut options = Options::default();
    let mut describe = false;
    let mut dtype_label = "all";
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--expected-summary" => describe = true,
            "--no-flex32" => options.no_flex32 = true,
            "--skip-rank-zero" => options.skip_rank_zero = true,
            "--finite" => options.finite = true,
            _ => {
                if let Some(value) = arg.strip_prefix("--dtype=") {
                    let (label, dtype) = match value {
                        "f64" => ("f64", DType::F64),
                        "f32" => ("f32", DType::F32),
                        "flex32" => ("flex32", DType::Flex32),
                        "f16" => ("f16", DType::F16),
                        "bf16" => ("bf16", DType::BF16),
                        _ => panic!("unknown remainder dtype: {value}"),
                    };
                    dtype_label = label;
                    options.dtype = Some(dtype);
                } else if let Some(value) = arg.strip_prefix("--group=") {
                    options.group = Some(
                        *GROUPS
                            .iter()
                            .find(|&&g| g == value)
                            .unwrap_or_else(|| panic!("unknown remainder group: {value}")),
                    );
                } else {
                    panic!("unknown remainder argument: {arg}");
                }
            }
        }
    }
    // Keep this before Device creation, runtime initialization, and sync.
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend, fusion) = (Device::cuda(0), "cuda", "enabled_materialized_chain_steps");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend, fusion) = (Device::flex(), "flex_cpu", "disabled");
    let summary = try_read_sync(run(&device, &options))
        .expect("remainder executor")
        .expect("strict Native typed remainder");
    device.sync().expect("remainder synchronization");
    println!(
        "BURN_TYPED_REMAINDER_OK os=linux backend={backend} fusion={fusion} corpus={CORPUS_VERSION} fixture_sha256={FIXTURE_SHA256} totals={} reference=python_fraction_truncated_quotient_integer_rne schedule=f64_f64_f32_f32_half_stored_f32_remf_correction_one_destination_rne scalar=canonical_f64_f32_storage finite_inf_signed_zero=exact_bits zero_sign=dividend arithmetic_nan=classification preserved_inputs=exact_bytes public_scalar=burn_float_conversion dispatch_scalar=original_variants chain=materialized_each_primitive poison=retained_allocation physical_padding=not_public scope=strict_native no_flex32={} skip_rank_zero={} finite={} dtype={dtype_label} group={}",
        summary_json(&summary),
        options.no_flex32,
        options.skip_rank_zero,
        options.finite,
        options.group.unwrap_or("all"),
    );
}
