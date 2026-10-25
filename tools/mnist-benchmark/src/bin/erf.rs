//! Linux execution of the public erf/GELU checks shared with D3OS.
use d3os_ai::burn::tensor::Device;
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    #[cfg(feature = "cuda")]
    let gradient = d3os_ai::validation::erf::GeluGradient::BurnDefault;
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let gradient = d3os_ai::validation::erf::GeluGradient::Analytic;
    if std::env::args().nth(1).as_deref() == Some("--profile") {
        let profile = d3os_ai::validation::erf::profile(&device).expect("erf profile read");
        d3os_ai::validation::erf::gelu_conformance(&device, gradient).expect("GELU forward/backward conformance");
        println!(
            "BURN_ERF_PROFILE_JSON {}",
            serde_json::json!({
                "execution_os":"linux", "backend":backend, "cases":profile.input_bits.len(),
                "strict_mismatches":profile.strict_mismatches, "special_mismatches":profile.special_mismatches,
                "max_absolute_error":profile.max_absolute_error, "max_ulp_error":profile.max_ulp_error,
                "input_bits":profile.input_bits, "output_bits":profile.output_bits,
                "strict_erf_conformance":profile.strict_mismatches == 0, "gelu_conformance":true,
            })
        );
        println!("BURN_ERF_PROFILE_OK os=linux backend={backend} flags_only=false strict_failures_reported=true gelu=forward,backward");
        return;
    }
    if let Err(error) = d3os_ai::validation::erf::run(&device, gradient) {
        eprintln!("BURN_ERF_CONFORMANCE_FAIL {error}");
        std::process::exit(1);
    }
    device.sync().expect("erf synchronization");
    println!("BURN_ERF_CONFORMANCE_OK os=linux backend={backend} erf_ulp=8 erf=forward,backward gelu=forward,backward shared_views=verified");
}
