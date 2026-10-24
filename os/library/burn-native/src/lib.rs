//! Native tensor storage and execution for the D3OS Burn backend.
#![no_std]

extern crate alloc;

pub mod convolution;
mod device;
pub mod layout;
pub mod pooling;
mod tensor;

pub use device::NativeDevice;
pub use tensor::NativeTensor;
