//! Diagnostic: an execution error is recorded as evidence, never called a pass.
use d3os_ai::{
    burn::tensor::{Device, try_read_sync},
    validation::quantized_empty::{self, Case},
};
fn main() {
    #[cfg(feature = "cuda")]
    let (device, backend) = (Device::cuda(0), "cuda");
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    let (device, backend) = (Device::flex(), "flex_cpu");
    let name = std::env::args().nth(1).unwrap_or_else(|| String::from("mask"));
    let case = match name.as_str() {
        "direct" => Case::Direct,
        "select" => Case::Select,
        "mask" => Case::Mask,
        _ => panic!("case must be direct, select, or mask"),
    };
    let result = try_read_sync(quantized_empty::run(&device, case)).expect("synchronous quantized-empty probe");
    let row = match result {
        Ok(bytes) => serde_json::json!({"os":"linux","backend":backend,"case":name,"status":"ok","record_bytes":bytes}),
        Err(error) => serde_json::json!({"os":"linux","backend":backend,"case":name,"status":"error","error":error}),
    };
    println!("BURN_QUANTIZED_EMPTY_RESULT {row}");
    let _ = device.sync();
}
