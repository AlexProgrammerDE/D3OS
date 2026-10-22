//! Versioned D3OS compute syscall protocol. Applications pass owned buffer handles.

use crate::{Error, Result};

pub const VERSION: u32 = 1;
pub const INFO: u32 = 0;
pub const ALLOCATE: u32 = 1;
pub const FREE: u32 = 2;
pub const UPLOAD: u32 = 3;
pub const DOWNLOAD: u32 = 4;
pub const LAUNCH: u32 = 5;

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Request {
    pub version: u32,
    pub operation: u32,
    pub handle: u64,
    pub pointer: u64,
    pub elements: u64,
    pub offset: u64,
    pub kernel: u32,
    pub buffer_count: u32,
    pub buffers: [u64; 16],
    pub offsets: [u64; 16],
    pub parameter_count: u32,
    pub parameters: [u32; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Kernel {
    Matmul = 0,
    Project = 1,
    Relu = 2,
    BiasRow = 3,
    BiasChannel = 4,
    Im2col = 5,
    MaxPool = 6,
    Softmax = 7,
}

#[derive(Clone, Copy, Debug)]
pub struct Work {
    pub threads: u32,
    pub blocks: u32,
    /// Element extents accessed by each pointer, in kernel argument order.
    pub extents: [u64; 3],
    pub pointers: usize,
    pub output: usize,
    /// Exact input/output aliasing is valid only for these elementwise kernels.
    pub in_place: bool,
}

fn product(values: &[u32]) -> Result<u32> {
    if values.contains(&0) {
        return Err(Error::InvalidField);
    }
    values.iter().try_fold(1u32, |v, &n| v.checked_mul(n).ok_or(Error::InvalidField))
}

impl Kernel {
    pub fn from_raw(raw: u32) -> Result<Self> {
        Ok(match raw {
            0 => Self::Matmul,
            1 => Self::Project,
            2 => Self::Relu,
            3 => Self::BiasRow,
            4 => Self::BiasChannel,
            5 => Self::Im2col,
            6 => Self::MaxPool,
            7 => Self::Softmax,
            _ => return Err(Error::Unsupported),
        })
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Matmul => "tensor_matmul_tf32x3",
            Self::Project => "tensor_project_tf32x3",
            Self::Relu => "inference_relu",
            Self::BiasRow => "bias_row",
            Self::BiasChannel => "bias_channel",
            Self::Im2col => "im2col",
            Self::MaxPool => "max_pool",
            Self::Softmax => "inference_softmax",
        }
    }

    pub fn work(self, p: &[u32]) -> Result<Work> {
        let mut extents = [0; 3];
        let (count, pointers, output, in_place) = match (self, p) {
            (Self::Matmul | Self::Project, &[m, k, n]) => {
                extents = [product(&[m, k])? as u64, product(&[k, n])? as u64, product(&[m, n])? as u64];
                (product(&[m.div_ceil(16), n.div_ceil(8), 32])?, 3, 2, false)
            }
            (Self::Relu, &[n]) => {
                product(&[n])?;
                extents = [n as u64, n as u64, 0];
                (n, 2, 1, true)
            }
            (Self::BiasRow, &[count, cols]) => {
                if count == 0 || cols == 0 || count % cols != 0 {
                    return Err(Error::InvalidField);
                }
                extents = [count as u64, cols as u64, count as u64];
                (count, 3, 2, true)
            }
            (Self::BiasChannel, &[count, spatial, channels]) => {
                let image = product(&[spatial, channels])?;
                if count == 0 || count % image != 0 {
                    return Err(Error::InvalidField);
                }
                extents = [count as u64, channels as u64, count as u64];
                (count, 3, 2, true)
            }
            (Self::Im2col, &[channels, height, width, kh, kw, oh, ow, sh, sw, ph, pw]) => {
                let output = product(&[channels, kh, kw, oh, ow])?;
                validate_extent(height, kh, sh, ph, oh)?;
                validate_extent(width, kw, sw, pw, ow)?;
                extents[0] = product(&[channels, height, width])? as u64;
                extents[1] = output as u64;
                (output, 2, 1, false)
            }
            (Self::MaxPool, &[planes, height, width, kh, kw, sh, sw, oh, ow]) => {
                validate_extent(height, kh, sh, 0, oh)?;
                validate_extent(width, kw, sw, 0, ow)?;
                let output = product(&[planes, oh, ow])?;
                extents[0] = product(&[planes, height, width])? as u64;
                extents[1] = output as u64;
                (output, 2, 1, false)
            }
            (Self::Softmax, &[rows, cols]) => {
                let n = product(&[rows, cols])?;
                extents = [n as u64, n as u64, 0];
                (rows, 2, 1, true)
            }
            _ => return Err(Error::InvalidField),
        };
        let blocks = count.div_ceil(128);
        blocks.checked_mul(128).ok_or(Error::InvalidField)?;
        Ok(Work {
            threads: 128,
            blocks,
            extents,
            pointers,
            output,
            in_place,
        })
    }
}

fn validate_extent(input: u32, kernel: u32, stride: u32, padding: u32, output: u32) -> Result<()> {
    if [input, kernel, stride, output].contains(&0) {
        return Err(Error::InvalidField);
    }
    let padded = padding.checked_mul(2).and_then(|p| input.checked_add(p)).ok_or(Error::InvalidField)?;
    if padded.checked_sub(kernel).map(|v| v / stride + 1) != Some(output) {
        return Err(Error::InvalidField);
    }
    Ok(())
}
