//! Public dynamic-shape selection checks on the explicitly selected backend.
use d3os_ai::burn::tensor::{Device, try_read_sync};
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let include_empty = !std::env::args().any(|arg| arg == "--nonempty");
    let include_compact_q8 = !std::env::args().any(|arg| arg == "--packed-aligned");
    let quantized_domain = if include_compact_q8 { "compact_tails" } else { "packed_aligned" };
    try_read_sync(d3os_ai::validation::compaction::run(&device, include_empty, include_compact_q8))
        .expect("synchronous compaction probe")
        .expect("public compaction conformance");
    device.sync().expect("compaction synchronization");
    println!(
        "BURN_COMPACTION_CONFORMANCE_OK os=linux backend={backend} dtypes=f32,i32,bool,q8 argwhere=true nonzero=true mask_select=true gradients=4 ranks=1,3,8 quantized_domain={quantized_domain} empty={include_empty} shared_views=true"
    );
}
