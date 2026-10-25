//! Linux CUDA spatial convolution and analytic gradient conformance.
use d3os_ai::{burn::tensor::Device, validation::spatial_convolution};
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let args: Vec<_> = std::env::args().skip(1).collect();
    let empty = args.iter().any(|s| s == "--empty");
    let selected = args
        .iter()
        .position(|s| s == "--case")
        .map(|i| args.get(i + 1).expect("case index").parse::<usize>().expect("numeric case"));
    let exact_inputs = args.iter().any(|s| s == "--tf32-exact");
    let empty_backward = args.iter().any(|s| s == "--empty-backward");
    match spatial_convolution::check(&device, empty, selected, exact_inputs, empty_backward) {
        Ok((forward, gradients)) => println!(
            "BURN_SPATIAL_CONVOLUTION_OK os=linux backend={backend} forward={forward} gradients={gradients} empty={empty} selected={selected:?} tf32_exact_inputs={exact_inputs} empty_backward={empty_backward} reference=f64 signed_views=true shared=true"
        ),
        Err(error) => {
            eprintln!("BURN_SPATIAL_CONVOLUTION_FAIL os=linux backend={backend} {error}");
            std::process::exit(1);
        }
    }
}
