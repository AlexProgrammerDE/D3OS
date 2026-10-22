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
    Copy = 8,
    Add = 9,
    SiluMul = 10,
    RmsNorm = 11,
    AttentionPrepare = 12,
    AttentionScores = 13,
    AttentionOutput = 14,
    DeltaConvolution = 15,
    DeltaPrepare = 16,
    DeltaUpdate = 17,
    DeltaOutput = 18,
    Zero = 19,
    Raster = 20,
    Palette = 21,
    Fuzz = 22,
    RasterBinned = 23,
}

#[derive(Clone, Copy, Debug)]
pub struct Work {
    pub threads: u32,
    pub blocks: u32,
    /// Element extents accessed by each pointer, in kernel argument order.
    pub extents: [u64; 16],
    pub pointers: usize,
    pub output: usize,
    /// Exact input/output aliasing is valid only for these elementwise kernels.
    pub in_place: bool,
    /// Additional buffers modified by stateful kernels.
    pub mutated: u16,
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
            8 => Self::Copy,
            9 => Self::Add,
            10 => Self::SiluMul,
            11 => Self::RmsNorm,
            12 => Self::AttentionPrepare,
            13 => Self::AttentionScores,
            14 => Self::AttentionOutput,
            15 => Self::DeltaConvolution,
            16 => Self::DeltaPrepare,
            17 => Self::DeltaUpdate,
            18 => Self::DeltaOutput,
            19 => Self::Zero,
            20 => Self::Raster,
            21 => Self::Palette,
            22 => Self::Fuzz,
            23 => Self::RasterBinned,
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
            Self::Copy => "tensor_copy",
            Self::Add => "tensor_add",
            Self::SiluMul => "silu_mul",
            Self::RmsNorm => "rms_norm",
            Self::AttentionPrepare => "attention_prepare",
            Self::AttentionScores => "attention_scores",
            Self::AttentionOutput => "attention_output",
            Self::DeltaConvolution => "delta_convolution",
            Self::DeltaPrepare => "delta_prepare",
            Self::DeltaUpdate => "delta_update",
            Self::DeltaOutput => "delta_output",
            Self::Zero => "buffer_zero",
            Self::Raster => "raster_commands",
            Self::Palette => "palette_scale",
            Self::Fuzz => "raster_fuzz",
            Self::RasterBinned => "raster_binned",
        }
    }

    pub fn work(self, p: &[u32]) -> Result<Work> {
        let mut extents = [0; 16];
        let mut mutated = 0;
        let (count, pointers, output, in_place) = match (self, p) {
            (Self::Fuzz, &[width, height, commands, words]) => {
                extents[..3].copy_from_slice(&[product(&[width, height])? as u64, product(&[commands, 12])? as u64, product(&[words])? as u64]);
                (1, 3, 0, false)
            }
            (Self::Raster, &[width, height, commands, words]) => {
                let n = product(&[width, height])?;
                extents[..4].copy_from_slice(&[n as u64, product(&[commands, 12])? as u64, product(&[words])? as u64, n as u64]);
                (n, 4, 3, false)
            }
            (Self::RasterBinned, &[width, height, commands, words, bins]) => {
                let n = product(&[width, height])?;
                let header = width.checked_add(height).and_then(|v| v.checked_add(1)).ok_or(Error::InvalidField)?;
                if bins < header {
                    return Err(Error::InvalidField);
                }
                extents[..5].copy_from_slice(&[n as u64, product(&[commands, 12])? as u64, product(&[words])? as u64, n as u64, bins as u64]);
                (n, 5, 3, false)
            }
            (Self::Palette, &[width, height, scale]) => {
                if !matches!(scale, 1..=4) {
                    return Err(Error::InvalidField);
                }
                let n = product(&[width, height])?;
                let output = product(&[n, scale, scale])?;
                extents[..3].copy_from_slice(&[n as u64, 256, output as u64]);
                (output, 3, 2, false)
            }
            (Self::Zero, &[n]) => {
                product(&[n])?;
                extents[0] = n as u64;
                (n, 1, 0, false)
            }
            (Self::Matmul | Self::Project, &[m, k, n]) => {
                extents[..3].copy_from_slice(&[product(&[m, k])? as u64, product(&[k, n])? as u64, product(&[m, n])? as u64]);
                (product(&[m.div_ceil(16), n.div_ceil(8), 32])?, 3, 2, false)
            }
            (Self::Relu | Self::Copy, &[n]) => {
                product(&[n])?;
                extents[..2].fill(n as u64);
                (n, 2, 1, true)
            }
            (Self::BiasRow, &[count, cols]) => {
                if count == 0 || cols == 0 || count % cols != 0 {
                    return Err(Error::InvalidField);
                }
                extents[..3].copy_from_slice(&[count as u64, cols as u64, count as u64]);
                (count, 3, 2, true)
            }
            (Self::BiasChannel, &[count, spatial, channels]) => {
                let image = product(&[spatial, channels])?;
                if count == 0 || count % image != 0 {
                    return Err(Error::InvalidField);
                }
                extents[..3].copy_from_slice(&[count as u64, channels as u64, count as u64]);
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
                extents[..2].fill(n as u64);
                (rows, 2, 1, true)
            }
            (Self::Add | Self::SiluMul, &[n]) => {
                product(&[n])?;
                extents[..3].fill(n as u64);
                (n, 3, 2, true)
            }
            (Self::RmsNorm, &[rows, width, epsilon, offset]) => {
                positive_float(epsilon)?;
                if !matches!(f32::from_bits(offset), 0. | 1.) {
                    return Err(Error::InvalidField);
                }
                let n = product(&[rows, width])? as u64;
                extents[..3].copy_from_slice(&[n, width as u64, n]);
                (rows, 3, 2, false)
            }
            (Self::AttentionPrepare, &[heads, dim, rotary, position, theta, epsilon, stride]) => {
                positive_float(theta)?;
                positive_float(epsilon)?;
                if rotary == 0 || rotary % 2 != 0 || rotary > dim || !matches!(stride, 1 | 2) || position == u32::MAX {
                    return Err(Error::InvalidField);
                }
                let n = product(&[heads, dim])?;
                extents[..3].copy_from_slice(&[product(&[n, stride])? as u64, dim as u64, n as u64]);
                (heads, 3, 2, false)
            }
            (Self::AttentionScores, &[heads, kv_heads, dim, positions]) => {
                let n = product(&[heads, positions])?;
                if kv_heads == 0 || heads % kv_heads != 0 {
                    return Err(Error::InvalidField);
                }
                extents[..3].copy_from_slice(&[product(&[heads, dim])? as u64, product(&[positions, kv_heads, dim])? as u64, n as u64]);
                (n, 3, 2, false)
            }
            (Self::AttentionOutput, &[heads, kv_heads, dim, positions]) => {
                if kv_heads == 0 || heads % kv_heads != 0 {
                    return Err(Error::InvalidField);
                }
                let n = product(&[heads, dim])?;
                extents[..4].copy_from_slice(&[
                    product(&[heads, positions])? as u64,
                    product(&[positions, kv_heads, dim])? as u64,
                    product(&[n, 2])? as u64,
                    n as u64,
                ]);
                (n, 4, 3, false)
            }
            (Self::DeltaConvolution, &[channels, kernel]) => {
                let history = product(&[channels, kernel])? as u64;
                extents[..4].copy_from_slice(&[channels as u64, history, history, channels as u64]);
                mutated = 1 << 2;
                (channels, 4, 3, false)
            }
            (Self::DeltaPrepare, &[heads, key, value]) => {
                let channels = delta_channels(heads, key, value)?;
                extents[..6].copy_from_slice(&[
                    channels as u64,
                    heads as u64,
                    heads as u64,
                    heads as u64,
                    heads as u64,
                    product(&[heads, 4])? as u64,
                ]);
                (heads, 6, 5, false)
            }
            (Self::DeltaUpdate, &[heads, key, value]) => {
                let channels = delta_channels(heads, key, value)?;
                let n = product(&[heads, value])?;
                extents[..4].copy_from_slice(&[channels as u64, product(&[heads, 4])? as u64, product(&[heads, key, value])? as u64, n as u64]);
                mutated = 1 << 2;
                (n, 4, 3, false)
            }
            (Self::DeltaOutput, &[heads, value, epsilon]) => {
                positive_float(epsilon)?;
                let n = product(&[heads, value])? as u64;
                extents[..4].copy_from_slice(&[n, value as u64, n, n]);
                (heads, 4, 3, false)
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
            mutated,
        })
    }
}

impl Work {
    pub fn validate_aliases(&self, addresses: &[u64]) -> Result<()> {
        if addresses.len() != self.pointers {
            return Err(Error::InvalidField);
        }
        let writes = self.mutated | (1 << self.output);
        for output in 0..self.pointers {
            if writes & (1 << output) == 0 {
                continue;
            }
            let output_end = addresses[output].checked_add(self.extents[output] * 4).ok_or(Error::InvalidField)?;
            for i in 0..self.pointers {
                if i == output {
                    continue;
                }
                let end = addresses[i].checked_add(self.extents[i] * 4).ok_or(Error::InvalidField)?;
                let overlap = addresses[output] < end && addresses[i] < output_end;
                if overlap
                    && !(self.in_place && output == self.output && i == 0 && addresses[i] == addresses[output] && self.extents[i] == self.extents[output])
                {
                    return Err(Error::InvalidField);
                }
            }
        }
        Ok(())
    }
}

fn positive_float(bits: u32) -> Result<()> {
    let value = f32::from_bits(bits);
    if !value.is_finite() || value <= 0. {
        return Err(Error::InvalidField);
    }
    Ok(())
}

fn delta_channels(heads: u32, key: u32, value: u32) -> Result<u32> {
    product(&[heads, key, value])?;
    let channels = key.checked_mul(2).and_then(|n| n.checked_add(value)).ok_or(Error::InvalidField)?;
    product(&[heads, channels])
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
