//! Exact public predicate checks on Linux CUDA, with explicit diagnostic scopes.
use d3os_ai::{
    burn::tensor::Device,
    validation::predicate::{Options, run},
};

fn main() {
    let mut options = Options::default();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--no-flex32" => options.include_flex32 = false,
            "--finite" => options.finite_only = true,
            _ => panic!("unknown predicate argument: {arg}"),
        }
    }
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let result = run(&device, &options).expect("exact public predicates");
    device.sync().expect("predicate synchronization");
    let flex32 = if options.include_flex32 { "direct" } else { "excluded_upstream_failure" };
    let ieee = if options.finite_only { "finite_only" } else { "full" };
    println!(
        "BURN_PREDICATE_CONFORMANCE_OK os=linux backend={backend} numeric_dtypes={} bool_dtypes=2 cases={} comparisons={} scalars={} classifications={} boolean={} source_checks={} half_patterns={} values={} bool_storage=explicit_u8_u32 canonical_bytes=true flex32={flex32} ieee={ieee} views=reversed,transposed,offset,broadcast tails=1,127,128,129,257 empty=true reference=host_types",
        result.numeric_dtypes,
        result.cases(),
        result.comparison_cases,
        result.scalar_cases,
        result.classification_cases,
        result.boolean_cases,
        result.source_checks,
        result.half_patterns,
        result.values,
    );
}
