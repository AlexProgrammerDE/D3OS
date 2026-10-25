//! Matched ordered scatter API checks on the explicitly selected device.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let include_empty = !std::env::args().any(|a| a == "--nonempty");
    let include_ieee = !std::env::args().any(|a| a == "--finite");
    d3os_ai::validation::scatter_updates::run(&device, include_empty, include_ieee).expect("ordered scatter conformance");
    device.sync().expect("scatter synchronization");
    println!(
        "BURN_SCATTER_UPDATES_OK os=linux backend={backend} dtypes=f32,i32 modes=4 axes=all unique_indices=true gradient_cases=12 ieee={include_ieee} empty={include_empty} shared_views=true"
    );
}
