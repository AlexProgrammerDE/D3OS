//! Matched Linux CUDA deformable convolution and derivative comparisons.
use d3os_ai::{burn::tensor::Device, validation::deform};
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let args: Vec<_> = std::env::args().skip(1).collect();
    let exact = args.iter().any(|a| a == "--tf32-exact");
    let empty = args.iter().any(|a| a == "--empty");
    let selected = args
        .iter()
        .position(|a| a == "--case")
        .map(|i| args.get(i + 1).expect("case").parse::<usize>().expect("numeric case"));
    let backward = !args.iter().any(|a| a == "--forward-only");
    match deform::check(&device, empty, exact, selected, backward) {
        Ok((forward, gradients)) => println!(
            "BURN_DEFORM_OK os=linux backend={backend} forward={forward} gradients={gradients} empty={empty} tf32_exact_inputs={exact} selected={selected:?} backward={backward} reference=f64 signed_views=true shared=true"
        ),
        Err(error) => {
            eprintln!("BURN_DEFORM_FAIL os=linux backend={backend} {error}");
            std::process::exit(1);
        }
    }
}
