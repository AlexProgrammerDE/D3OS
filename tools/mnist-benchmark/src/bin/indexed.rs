//! Exact stored-value indexing through public Burn backend APIs.
use d3os_ai::{
    burn::tensor::{Device, try_read_sync},
    validation::indexed::{Options, Summary, expected_summary, run},
};

fn summary_json(s: &Summary) -> serde_json::Value {
    serde_json::json!({
        "dtypes": s.dtypes, "index_dtypes": s.index_dtypes, "cases": s.cases(),
        "unsupported_bool_nd": s.unsupported_bool_nd,
        "gather": s.gather, "select": s.select, "gather_nd": s.gather_nd,
        "scatter_assign": s.scatter_assign, "select_assign": s.select_assign,
        "scatter_nd_assign": s.scatter_nd_assign, "argmin": s.argmin,
        "argmax": s.argmax, "sort_pairs": s.sort_pairs, "topk_pairs": s.topk_pairs,
        "argsort": s.argsort, "argtopk": s.argtopk, "argwhere": s.argwhere,
        "source_checks": s.source_checks, "values": s.values,
    })
}
fn main() {
    let mut options = Options::default();
    let mut describe = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--no-flex32" => options.include_flex32 = false,
            "--skip-scalar-nd" => options.include_scalar_nd = false,
            "--expected-summary" => describe = true,
            _ => panic!("unknown indexed argument: {arg}"),
        }
    }
    // Return before even constructing a device. This path is valid on a host
    // without a CUDA driver and reports the same generators as the runner.
    if describe {
        println!("{}", summary_json(&expected_summary(&options)));
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let summary = try_read_sync(run(&device, &options))
        .expect("synchronous indexed executor")
        .expect("exact indexed byte records");
    device.sync().expect("indexed synchronization");
    println!(
        "BURN_INDEXED_CONFORMANCE_OK os=linux backend={backend} totals={} exact_bytes=true reference=byte_records ranks=1..8 views=offset,step,reversed,transposed overlap=true unique_assign=true tails=0,1,127,128,129,257 nd_indices=nonempty nd_empty=trailing_data input_index_dtypes=all8 requested_output_dtypes=all8 bool_storage=explicit_u8_u32 bool_nd=unsupported_shared_api invalid_indices=raw_only flex32={} scalar_nd={}",
        summary_json(&summary),
        if options.include_flex32 { "direct" } else { "excluded_upstream_upload" },
        if options.include_scalar_nd { "checked" } else { "excluded_upstream_failure" },
    );
}
