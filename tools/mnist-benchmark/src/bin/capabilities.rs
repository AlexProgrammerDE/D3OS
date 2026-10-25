//! Query actual backend declarations on the explicitly selected Linux device.
use d3os_ai::{burn::tensor::Device, validation::capabilities};

fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let rows = capabilities::declared(&device);
    for row in &rows {
        println!(
            "BURN_DTYPE_CAPABILITY os=linux backend={backend} name={} storage={} arithmetic={} accelerated={}",
            row.name, row.storage, row.arithmetic, row.accelerated
        );
    }
    device.sync().expect("capability query synchronization");
    println!("BURN_CAPABILITIES_OK os=linux backend={backend} rows={} flags_only=true", rows.len());
}
