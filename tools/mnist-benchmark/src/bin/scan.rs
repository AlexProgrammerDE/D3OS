//! Matched public cumulative checks on the Linux CUDA device.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let include_empty = std::env::args().nth(1).as_deref() != Some("--nonempty");
    d3os_ai::validation::scan::run(&device, include_empty).expect("public cumulative conformance");
    device.sync().expect("cumulative synchronization");
    println!(
        "BURN_SCAN_CONFORMANCE_OK os=linux backend={backend} operations=8 reference=i128,f64 lengths=1,127,128,129,257,531 gradients=4 empty={include_empty} shared_views=true"
    );
}
