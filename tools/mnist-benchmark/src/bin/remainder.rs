//! Linux CUDA execution of the public FP32 remainder corpus shared with D3OS.
use d3os_ai::{
    burn::tensor::Device,
    validation::remainder::{self, Domain},
};
fn main() {
    let finite = std::env::args().any(|arg| arg == "--finite");
    let wide = std::env::args().any(|arg| arg == "--finite-wide");
    let domain = if wide {
        Domain::FiniteWide
    } else if finite {
        Domain::Finite
    } else {
        Domain::Ieee
    };
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let report = match remainder::run(&device, domain) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("BURN_REMAINDER_FAIL os=linux backend={backend} finite={finite} wide={wide} {error}");
            std::process::exit(1);
        }
    };
    device.sync().expect("remainder synchronization");
    println!(
        "BURN_REMAINDER_OK os=linux backend={backend} domain={} pairs={} scalar_cases={} gradients={} reference=libm_fmod_floor finite_bits=exact zero_sign=unspecified signed_views=true broadcast=true empty=true",
        if wide {
            "finite_wide"
        } else if finite {
            "finite"
        } else {
            "ieee"
        },
        report.pairs,
        report.scalar_cases,
        report.gradients
    );
}
