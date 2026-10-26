//! Native tensor storage and execution for the D3OS Burn backend.
#![no_std]

extern crate alloc;

pub mod backend;
mod compaction;
pub mod convolution;
pub mod deform;
mod device;
mod float_arithmetic;
mod float_extrema;
mod float_reduction;
mod indexing;
mod indices;
mod integer_arithmetic;
mod integer_reduction;
pub mod interpolate;
pub mod layout;
mod linear;
mod packed_gather;
mod packed_scatter;
mod packed_updates;
pub mod pooling;
mod predicate;
pub mod quantization;
pub mod random;
pub mod range;
mod reduction;
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
