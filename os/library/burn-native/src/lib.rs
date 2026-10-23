//! Native tensor storage and execution for the D3OS Burn backend.
#![no_std]

extern crate alloc;

mod device;
pub mod layout;
mod tensor;

pub use device::NativeDevice;
pub use tensor::NativeTensor;
