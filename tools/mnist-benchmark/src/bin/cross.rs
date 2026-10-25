//! Matched public vector-product checks on the explicitly selected backend.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let include_empty = !std::env::args().any(|arg| arg == "--nonempty");
    d3os_ai::validation::cross::run(&device, include_empty).expect("public cross conformance");
    device.sync().expect("cross synchronization");
    println!("BURN_CROSS_CONFORMANCE_OK os=linux backend={backend} dtype=f32 axes=all signed_views=true broadcasting=true gradients=6 empty={include_empty}");
}
