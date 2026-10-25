//! Matched Linux CUDA execution of the public integer arithmetic checks.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    d3os_ai::validation::integer::run(&device).expect("public integer division and remainder conformance");
    device.sync().expect("integer synchronization");
    println!("BURN_INTEGER_CONFORMANCE_OK os=linux backend={backend} operations=div,remainder tensor=true scalar=true shared_views=true");
}
