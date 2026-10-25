//! Exact ordinary storage movement on Linux through public Burn Dispatch.
use d3os_ai::{
    burn::tensor::{Device, try_read_sync},
    validation::movement::{Options, expected_summary, run},
};

fn main() {
    let mut options = Options::default();
    let mut describe = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--no-flex32" => options.include_flex32 = false,
            "--expected-summary" => describe = true,
            "--skip-empty-bool-scatter" => options.skip_empty_bool_scatter = true,
            "--matching-bool-masks" => options.matching_bool_masks_only = true,
            _ => panic!("unknown movement argument: {arg}"),
        }
    }
    if describe {
        let s = expected_summary(&options);
        println!(
            "{}",
            serde_json::json!({"dtypes":s.dtypes,"cases":s.cases(),"gather":s.gather,"select":s.select,"mask_where":s.mask_where,"mask_fill":s.mask_fill,"slice_assign":s.slice_assign,"cat":s.cat,"pad":s.pad,"mask_select":s.mask_select,"argwhere":s.argwhere,"bool_scatter_or":s.bool_scatter_or,"source_checks":s.source_checks,"values":s.values})
        );
        return;
    }
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let summary = try_read_sync(run(&device, &options))
        .expect("synchronous movement executor")
        .expect("exact stored-value movement");
    device.sync().expect("movement synchronization");
    let bool_masks = if options.matching_bool_masks_only { "matching_only" } else { "all_stores" };
    let empty_scatter = if options.skip_empty_bool_scatter {
        "excluded_upstream_failure"
    } else {
        "checked"
    };
    let flex32 = if options.include_flex32 { "direct" } else { "excluded_upstream_upload" };
    println!(
        "BURN_MOVEMENT_CONFORMANCE_OK os=linux backend={backend} dtypes={} cases={} gather={} select={} mask_where={} mask_fill={} slice_assign={} cat={} pad={} mask_select={} argwhere={} bool_scatter_or={} source_checks={} values={} bool_storage=explicit_u8_u32 exact_bytes=true flex32={flex32} empty_bool_scatter={empty_scatter} bool_masks={bool_masks} views=offset,step,reversed,transposed,broadcast overlap=true tails=1,127,128,129,257 empty=true reference=byte_records",
        summary.dtypes,
        summary.cases(),
        summary.gather,
        summary.select,
        summary.mask_where,
        summary.mask_fill,
        summary.slice_assign,
        summary.cat,
        summary.pad,
        summary.mask_select,
        summary.argwhere,
        summary.bool_scatter_or,
        summary.source_checks,
        summary.values,
    );
}
