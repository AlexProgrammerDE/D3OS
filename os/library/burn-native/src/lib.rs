//! Native tensor storage and execution for the D3OS Burn backend.
#![no_std]

extern crate alloc;

pub mod backend;
pub mod convolution;
mod device;
pub mod layout;
pub mod pooling;
pub mod random;
mod tensor;
mod updates;

pub use device::NativeDevice;
pub use tensor::NativeTensor;

pub use backend::Native;

pub use d3os_compute::Error as NativeError;
