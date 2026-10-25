//! Public ordinary numeric conversions on an explicitly selected backend.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let flex32 = !std::env::args().any(|a| a == "--no-flex32");
    let pairs = d3os_ai::validation::conversions::run(&device, flex32).expect("ordinary conversion conformance");
    device.sync().expect("conversion synchronization");
    let gradient_cases = if flex32 { 8 } else { 6 };
    println!(
        "BURN_CONVERSIONS_OK os=linux backend={backend} pairs={pairs} flex32={flex32} float_to_int=finite_in_range rounding=nearest_even integer_narrowing=low_bits half_float_codes=exhaustive bf16_midpoints=all_intervals gradients={gradient_cases} signed_views=true empty=true"
    );
}
