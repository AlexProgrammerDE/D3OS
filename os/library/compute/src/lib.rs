//! Native compute primitives shared by D3OS applications and host verification.
#![no_std]

extern crate alloc;

pub mod cpu;
#[cfg(feature = "nvidia")]
pub mod nvidia;
pub mod raster;
pub mod transformer;

use alloc::vec::Vec;
use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidShape,
    ShapeMismatch,
    InvalidParameter,
    SizeOverflow,
    AllocationFailed,
    NonFinite,
    DeviceUnavailable,
    DeviceFault,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidShape => "tensor dimensions must be nonzero, with a rank from 1 to 8",
            Self::ShapeMismatch => "tensor dimensions do not match the operation",
            Self::InvalidParameter => "invalid operation parameter",
            Self::SizeOverflow => "tensor size exceeds the address space",
            Self::AllocationFailed => "cannot allocate tensor storage",
            Self::NonFinite => "operation requires finite values",
            Self::DeviceUnavailable => "native NVIDIA compute is unavailable",
            Self::DeviceFault => "native NVIDIA compute failed",
        })
    }
}

impl core::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

pub fn element_count(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() || shape.len() > 8 || shape.contains(&0) {
        return Err(Error::InvalidShape);
    }
    let count = shape.iter().try_fold(1usize, |count, &dim| count.checked_mul(dim).ok_or(Error::SizeOverflow))?;
    if count > isize::MAX as usize / size_of::<f32>() {
        return Err(Error::SizeOverflow);
    }
    Ok(count)
}

pub trait Buffer {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Contiguous, row-major tensor. The backend owns its storage.
#[derive(Debug)]
pub struct Tensor<B: Backend> {
    shape: Vec<usize>,
    buffer: B::Buffer,
}

impl<B: Backend> Tensor<B> {
    pub fn from_buffer(shape: &[usize], buffer: B::Buffer) -> Result<Self> {
        if element_count(shape)? != buffer.len() {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { shape: shape.into(), buffer })
    }

    pub fn from_vec(backend: &B, shape: &[usize], values: Vec<f32>) -> Result<Self> {
        if element_count(shape)? != values.len() {
            return Err(Error::ShapeMismatch);
        }
        Self::from_buffer(shape, backend.upload(values)?)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn buffer(&self) -> &B::Buffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn reshape(mut self, shape: &[usize]) -> Result<Self> {
        if element_count(shape)? != self.len() {
            return Err(Error::ShapeMismatch);
        }
        self.shape = shape.into();
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Conv2d {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

impl Default for Conv2d {
    fn default() -> Self {
        Self {
            stride: [1, 1],
            padding: [0, 0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pool2d {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
}

pub fn output_extent(input: usize, kernel: usize, stride: usize, padding: usize) -> Result<usize> {
    if input == 0 || kernel == 0 || stride == 0 {
        return Err(Error::InvalidParameter);
    }
    let padded = padding.checked_mul(2).and_then(|p| input.checked_add(p)).ok_or(Error::SizeOverflow)?;
    let remaining = padded.checked_sub(kernel).ok_or(Error::ShapeMismatch)?;
    (remaining / stride).checked_add(1).ok_or(Error::SizeOverflow)
}

/// Synchronous tensor operations. Successful return means that the output is ready.
/// Explicit command scopes can defer work until readback or scope completion.
pub trait Backend: Sized {
    type Buffer: Buffer;

    /// Group operations while retaining their order. Outputs are ready when
    /// this call returns, including nested scopes. CPU reads synchronize inside
    /// the scope; temporary buffers remain owned until queued work completes.
    fn with_commands<T, E: From<Error>>(&self, work: impl FnOnce() -> core::result::Result<T, E>) -> core::result::Result<T, E> {
        work()
    }

    fn name(&self) -> &'static str;
    fn upload(&self, values: Vec<f32>) -> Result<Self::Buffer>;
    /// Upload finite, exact BF16 weights. Backends can keep packed storage for supported
    /// weight operations or expand it without changing the represented values.
    fn upload_bf16(&self, values: impl ExactSizeIterator<Item = u16>) -> Result<Self::Buffer> {
        let mut expanded = Vec::new();
        expanded.try_reserve_exact(values.len()).map_err(|_| Error::AllocationFailed)?;
        for value in values {
            if value & 0x7f80 == 0x7f80 {
                return Err(Error::NonFinite);
            }
            expanded.push(f32::from_bits(u32::from(value) << 16));
        }
        self.upload(expanded)
    }
    /// Upload borrowed little-endian BF16 bytes, including unaligned sources.
    /// The backend copies the values before returning and rejects non-finite data.
    fn upload_bf16_le(&self, bytes: &[u8]) -> Result<Self::Buffer> {
        if !bytes.len().is_multiple_of(2) {
            return Err(Error::InvalidParameter);
        }
        element_count(&[bytes.len() / 2])?;
        self.upload_bf16(bytes.chunks_exact(2).map(|pair| u16::from_le_bytes([pair[0], pair[1]])))
    }
    fn download(&self, tensor: &Tensor<Self>) -> Result<Vec<f32>>;
    /// Index of the largest value, choosing the first index for equal values.
    /// Rejects any non-finite input, including values that cannot win.
    fn argmax(&self, input: &Tensor<Self>) -> Result<usize> {
        let values = self.download(input)?;
        let mut best = 0;
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::NonFinite);
            }
            if value > values[best] {
                best = index;
            }
        }
        Ok(best)
    }
    fn matmul(&self, a: &Tensor<Self>, b: &Tensor<Self>) -> Result<Tensor<Self>>;
    /// Weights have shape [output_features, input_features]. Bias has shape [output_features].
    fn linear(&self, input: &Tensor<Self>, weights: &Tensor<Self>, bias: &Tensor<Self>) -> Result<Tensor<Self>>;
    /// Input and output use NCHW. Weights use [output_channels, input_channels, height, width].
    fn conv2d(&self, input: &Tensor<Self>, weights: &Tensor<Self>, bias: &Tensor<Self>, config: Conv2d) -> Result<Tensor<Self>>;
    fn relu(&self, input: Tensor<Self>) -> Result<Tensor<Self>>;
    fn max_pool2d(&self, input: &Tensor<Self>, config: Pool2d) -> Result<Tensor<Self>>;
    /// Normalize each row of a matrix with the numerically stable softmax algorithm.
    fn softmax(&self, input: &Tensor<Self>) -> Result<Tensor<Self>>;
}
