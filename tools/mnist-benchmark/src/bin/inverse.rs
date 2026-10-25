//! Linux CUDA execution of shared inverse math and autodiff conformance.
use d3os_ai::{
    burn::tensor::Device,
    validation::inverse::{self, Domain},
};
fn main() {
    let finite = std::env::args().any(|arg| arg == "--finite");
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    if std::env::args().any(|arg| arg == "--diagnostic") {
        let d = inverse::diagnose(&device, if finite { Domain::Finite } else { Domain::Ieee });
        device.sync().expect("inverse diagnostic synchronization");
        let strict = d.operations.iter().all(|(_, r)| r.is_ok()) && d.gradients.is_ok();
        println!(
            "BURN_INVERSE_DIAGNOSTIC_JSON {}",
            serde_json::json!({"os":"linux","backend":backend,"domain":if finite {"finite"}else{"ieee"},"strict_conformance":strict,"ulp_bound":8,"operations":d.operations.iter().map(|(op,r)|serde_json::json!({"operation":op,"cases":r.as_ref().ok(),"error":r.as_ref().err()})).collect::<Vec<_>>(),"gradient_cases":d.gradients.as_ref().ok(),"gradient_error":d.gradients.as_ref().err()})
        );
        println!("BURN_INVERSE_DIAGNOSTIC_OK os=linux backend={backend} strict_failures_reported=true performance=false");
        return;
    }
    let report = match inverse::run(&device, if finite { Domain::Finite } else { Domain::Ieee }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("BURN_INVERSE_FAIL os=linux backend={backend} finite={finite} {e}");
            std::process::exit(1);
        }
    };
    device.sync().expect("inverse synchronization");
    println!(
        "BURN_INVERSE_OK os=linux backend={backend} domain={} unary_ops=6 unary_inputs={} atan2_pairs={} ulp=8 gradients={} signed_views=true broadcast=true empty=true",
        if finite { "finite" } else { "ieee" },
        report.unary_inputs,
        report.atan2_pairs,
        report.gradients
    );
}
