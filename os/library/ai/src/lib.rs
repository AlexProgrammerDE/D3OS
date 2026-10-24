//! Burn-based model APIs shared by D3OS applications and host validation.
#![no_std]

extern crate alloc;

pub use burn;
#[cfg(feature = "native")]
pub use d3os_burn_native as native;

pub mod alexnet;
pub mod mnist;
#[cfg(target_os = "none")]
mod platform;
pub mod qwen;
#[cfg(any(test, target_os = "none"))]
mod timer_queue;
#[cfg(feature = "training")]
pub mod validation;

/// Open the native D3OS GPU for Burn tensors and modules.
#[cfg(feature = "native")]
pub fn native_device() -> Result<burn::tensor::Device, native::NativeError> {
    Ok(burn::tensor::Device::new(native::NativeDevice::open()?))
}
