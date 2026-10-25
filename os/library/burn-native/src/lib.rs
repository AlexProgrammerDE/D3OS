//! Native tensor storage and execution for the D3OS Burn backend.
#![no_std]

extern crate alloc;

pub mod backend;
mod compaction;
pub mod convolution;
pub mod deform;
mod device;
mod indexing;
pub mod interpolate;
pub mod layout;
mod linear;
pub mod pooling;
mod predicate;
pub mod quantization;
pub mod random;
pub mod range;
mod sort;
pub mod spatial_convolution;
mod storage;
mod tensor;
mod updates;
mod vector;

pub use device::NativeDevice;
pub use tensor::NativeTensor;

pub use backend::Native;

pub use d3os_compute::Error as NativeError;
