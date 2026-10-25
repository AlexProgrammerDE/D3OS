//! Public ordinary dtype storage on an explicitly selected backend.
use d3os_ai::burn::tensor::Device;

fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    if std::env::args().any(|a| a == "--flex32-direct") {
        d3os_ai::validation::storage::flex32_direct(&device).expect("direct Flex32 upload diagnostic");
        device.sync().expect("direct Flex32 synchronization");
        println!("BURN_STORAGE_FLEX32_DIRECT_OK os=linux backend={backend}");
        return;
    }
    let include_flex32 = !std::env::args().any(|a| a == "--no-flex32");
    d3os_ai::validation::storage::run(&device, include_flex32).expect("ordinary storage conformance");
    device.sync().expect("storage synchronization");
    let (dtypes, flex32) = if include_flex32 { (15, "direct") } else { (14, "excluded_upstream_failure") };
    println!(
        "BURN_STORAGE_CONFORMANCE_OK os=linux backend={backend} dtypes={dtypes} flex32={flex32} bytes=1,2,4,8 views=signed,broadcast,overlap fill=true bool_storage=explicit_backend reload=true half_bits=exhaustive empty=true scope=storage"
    );
}
