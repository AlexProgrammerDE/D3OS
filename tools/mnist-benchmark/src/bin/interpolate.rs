//! Linux CUDA interpolation forward and gradient comparisons.
use d3os_ai::{
    burn::tensor::Device,
    validation::interpolate::{self as check, Mode},
};
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let args: Vec<_> = std::env::args().skip(1).collect();
    let wide = !args.iter().any(|s| s == "--compact");
    let result = if args.iter().any(|s| s == "--empty-only") {
        check::check_empty(&device).map(|cases| format!("BURN_INTERPOLATE_EMPTY_OK os=linux backend={backend} cases={cases}"))
    } else if args.iter().any(|s| s == "--ieee-only") {
        check::check_ieee(&device).map(|cases| format!("BURN_INTERPOLATE_IEEE_OK os=linux backend={backend} cases={cases} bits=exact"))
    } else if let Some(at) = args.iter().position(|s| s == "--gradient-mode") {
        let mode = match args.get(at + 1).map(String::as_str) {
            Some("linear") => Mode::Linear,
            Some("cubic") => Mode::Cubic,
            Some("lanczos3") => Mode::Lanczos3,
            _ => panic!("expected linear, cubic or lanczos3"),
        };
        check::check_gradients(&device, &[mode]).map(|cases| format!("BURN_INTERPOLATE_GRADIENT_OK os=linux backend={backend} mode={mode:?} cases={cases}"))
    } else {
        check::check_forward(&device,wide).and_then(|forward|check::check_gradients(&device,&[Mode::Nearest,Mode::Exact]).map(|gradients|format!("BURN_INTERPOLATE_OK os=linux backend={backend} modes=5 align_corners=both forward={forward} gradient_modes=2 gradients={gradients} reference=f64 signed_views=true wide_coordinates={wide}")))
    };
    match result {
        Ok(marker) => {
            device.sync().expect("interpolation sync");
            println!("{marker}");
        }
        Err(error) => {
            eprintln!("BURN_INTERPOLATE_FAIL os=linux backend={backend} {error}");
            std::process::exit(1);
        }
    }
}
