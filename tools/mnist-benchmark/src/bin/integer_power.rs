//! Matched public I32 integer power checks on Linux CUDA.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let cases = d3os_ai::validation::integer::run_power(&device).expect("exact public integer powers");
    device.sync().expect("integer power synchronization");
    println!(
        "BURN_INTEGER_POWER_OK os=linux backend={backend} cases={cases} reference=i128 wrapping=exact exponent=u32_bits signed_views=true broadcast=true shared=true empty=true"
    );
}
