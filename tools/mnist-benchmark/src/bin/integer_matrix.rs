//! Matched Linux CUDA execution of the public I32 matrix product checks.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let (include_empty, cases, domain) = if std::env::args().nth(1).as_deref() == Some("--nonempty") {
        (false, 5, "nonempty")
    } else {
        (true, 8, "all")
    };
    d3os_ai::validation::integer_matrix::run(&device, include_empty).expect("public I32 matrix product conformance");
    device.sync().expect("I32 matrix synchronization");
    println!(
        "BURN_INTEGER_MATMUL_CONFORMANCE_OK os=linux backend={backend} cases={cases} domain={domain} reference=i128 wrapping=exact shared_views=true empty={include_empty}"
    );
}
