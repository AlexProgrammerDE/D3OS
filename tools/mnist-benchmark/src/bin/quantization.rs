//! Linux execution of the public Burn conformance shared with D3OS.
use d3os_ai::burn::tensor::Device;

fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    println!("BURN_QUANTIZATION_REFERENCE_START os=linux backend={backend} burn=0.22.1");
    let result = d3os_ai::validation::quantization::integer(&device).and_then(|()| device.sync().map_err(|e| format!("quantization sync: {e:?}")));
    if let Err(error) = result {
        eprintln!("BURN_QUANTIZATION_REFERENCE_FAIL {error}");
        std::process::exit(1);
    }
    println!(
        "BURN_QUANTIZATION_REFERENCE_OK os=linux backend={backend} formats=6 rounding=ties_even packed_bytes=exact views=verified serialization=verified dynamic=verified"
    );
}
