//! Versioned D3OS compute syscall protocol. Applications pass owned buffer handles.

use crate::{Error, Result};

pub const VERSION: u32 = 3;
pub const INFO: u32 = 0;
pub const ALLOCATE: u32 = 1;
pub const FREE: u32 = 2;
pub const UPLOAD: u32 = 3;
pub const DOWNLOAD: u32 = 4;
pub const LAUNCH: u32 = 5;
pub const PROFILE: u32 = 6;
/// Allocate an output and enqueue a full writer. Readback and scope completion
/// wait for that writer; outside command scopes this operation is synchronous.
pub const LAUNCH_ALLOCATE: u32 = 7;
/// Enable GPU timestamp instrumentation for subsequent launches (elements 0 or 1).
pub const DEVICE_TIMING: u32 = 8;
/// Open a process-scoped command batch. Nested scopes are supported.
pub const BEGIN_COMMANDS: u32 = 9;
/// Complete queued work before closing one command scope.
pub const END_COMMANDS: u32 = 10;
pub const QUEUE_PROFILE: u32 = 11;
pub const KERNEL_COUNT: usize = 38;

/// Device-wide submission counters, including waits outside kernel dispatch.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct QueueProfile {
    pub batches: u64,
    pub launches: u64,
    pub cycles: u64,
    pub wait_cycles: u64,
}

/// Device-wide CPU cycle counters. Read snapshots around an exclusive workload
/// and subtract them. Wait cycles include GPU execution and completion polling.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ProfileEntry {
    pub calls: u64,
    /// CPU recording and metadata preparation, excluding queue submission.
    pub cycles: u64,
    /// Immediate-launch waits only. QueueProfile includes all batch waits.
    pub wait_cycles: u64,
    /// GPU start-to-completion nanoseconds for instrumented launches only.
    pub device_ns: u64,
}

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
    ProjectBf16 = 24,
    DequantizeBf16 = 25,
    ArgMax = 26,
    RasterBinnedBytes = 27,
    FuzzBytes = 28,
    PaletteBytes = 29,
    DeltaFused = 30,
    GatedProjectBf16 = 31,
    Embedding = 32,
    AttentionPrepareBatch = 33,
    ProjectBf16Split = 34,
    SumProjectionPartials = 35,
    DeltaFusedBatch = 36,
    AttentionFusedBatch = 37,
}

const _: () = assert!(Kernel::AttentionFusedBatch as usize + 1 == KERNEL_COUNT);

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
    /// These kernels write every output element for every valid work shape.
    /// Partial writers must never expose uncleared allocations through this API.
    pub fn allocates_output(self) -> bool {
        matches!(
            self,
            Self::Matmul
                | Self::Project
                | Self::Copy
                | Self::RmsNorm
                | Self::ProjectBf16
                | Self::ArgMax
                | Self::DeltaFused
                | Self::DeltaFusedBatch
                | Self::GatedProjectBf16
                | Self::Embedding
                | Self::AttentionPrepareBatch
                | Self::AttentionFusedBatch
                | Self::ProjectBf16Split
                | Self::SumProjectionPartials
        )
    }

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
            24 => Self::ProjectBf16,
            25 => Self::DequantizeBf16,
            26 => Self::ArgMax,
            27 => Self::RasterBinnedBytes,
            28 => Self::FuzzBytes,
            29 => Self::PaletteBytes,
            30 => Self::DeltaFused,
            31 => Self::GatedProjectBf16,
            32 => Self::Embedding,
            33 => Self::AttentionPrepareBatch,
            34 => Self::ProjectBf16Split,
            35 => Self::SumProjectionPartials,
            36 => Self::DeltaFusedBatch,
            37 => Self::AttentionFusedBatch,
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
            Self::ProjectBf16 => "tensor_project_bf16_weights",
            Self::DequantizeBf16 => "dequantize_bf16",
            Self::ArgMax => "argmax_partials",
            Self::RasterBinnedBytes => "raster_binned_bytes",
            Self::FuzzBytes => "raster_fuzz_bytes",
            Self::PaletteBytes => "palette_scale_bytes",
            Self::DeltaFused => "delta_fused",
            Self::DeltaFusedBatch => "delta_fused_batch",
            Self::GatedProjectBf16 => "gated_project_bf16_weights",
            Self::Embedding => "embedding_gather",
            Self::AttentionPrepareBatch => "attention_prepare_batch",
            Self::AttentionFusedBatch => "attention_fused_batch",
            Self::ProjectBf16Split => "project_bf16_split",
            Self::SumProjectionPartials => "sum_projection_partials",
        }
    }

    /// Select the cuTile specialization without changing the public request ABI.
    pub fn entry(self, parameters: &[u32]) -> &'static str {
        match (self, parameters) {
            (Self::GatedProjectBf16, [k, _]) if k.is_multiple_of(2) => "gated_project_paired_bf16_weights",
            (Self::ProjectBf16, [1, k, n]) if !k.is_multiple_of(2) && *n > 8192 => "vector_project_unpaired_wide_bf16_weights",
            (Self::Project, [1, _, n]) if *n <= 8192 => "vector_project_tf32x3",
            (Self::Project, [1, _, _]) => "vector_project_wide_tf32x3",
            (Self::ProjectBf16, [1, k, n]) if k.is_multiple_of(2) && *n <= 8192 => "vector_project_paired_bf16_weights",
            (Self::ProjectBf16, [1, _, n]) if *n <= 8192 => "vector_project_bf16_weights",
            (Self::ProjectBf16, [1, _, _]) => "vector_project_wide_bf16_weights",
            _ => self.symbol(),
        }
    }

    pub fn work(self, p: &[u32]) -> Result<Work> {
        let mut extents = [0; 16];
        let mut mutated = 0;
        let mut threads = 128;
        let (count, pointers, output, in_place) = match (self, p) {
            (Self::AttentionFusedBatch, &[rows, heads, kv_heads, dim, position]) => {
                let positions = position.checked_add(rows).ok_or(Error::InvalidField)?;
                if kv_heads == 0 || heads % kv_heads != 0 || !(1..=256).contains(&dim) {
                    return Err(Error::InvalidField);
                }
                let n = product(&[rows, heads, dim])?;
                let cache = product(&[positions, kv_heads, dim])?;
                extents[..5].copy_from_slice(&[u64::from(n), u64::from(cache), u64::from(cache), u64::from(product(&[n, 2])?), u64::from(n)]);
                (product(&[rows, heads, 128])?, 5, 4, false)
            }
            (Self::ProjectBf16Split, &[m, k, n, splits]) => {
                extents[..3].copy_from_slice(&[
                    u64::from(product(&[m, k])?),
                    u64::from(product(&[k, n])?).div_ceil(2),
                    u64::from(product(&[splits, m, n])?),
                ]);
                (product(&[splits, m.div_ceil(16), n.div_ceil(32), 128])?, 3, 2, false)
            }
            (Self::SumProjectionPartials, &[m, n, splits]) => {
                let output = product(&[m, n])?;
                extents[..2].copy_from_slice(&[u64::from(product(&[splits, output])?), u64::from(output)]);
                (output, 2, 1, false)
            }
            (Self::AttentionPrepareBatch, &[heads, dim, rotary, position, theta, epsilon, stride, rows]) => {
                positive_float(theta)?;
                positive_float(epsilon)?;
                if rotary == 0 || rotary % 2 != 0 || rotary > dim || !matches!(stride, 1 | 2) {
                    return Err(Error::InvalidField);
                }
                position.checked_add(rows).ok_or(Error::InvalidField)?;
                let n = product(&[rows, heads, dim])?;
                extents[..3].copy_from_slice(&[u64::from(product(&[n, stride])?), u64::from(dim), u64::from(n)]);
                (product(&[rows, heads, 128])?, 3, 2, false)
            }
            (Self::Embedding, &[vocab, width, rows, packed]) => {
                if packed > 1 {
                    return Err(Error::InvalidField);
                }
                let table = u64::from(product(&[vocab, width])?);
                let output = product(&[rows, width])?;
                extents[..3].copy_from_slice(&[if packed == 1 { table.div_ceil(2) } else { table }, u64::from(rows), u64::from(output)]);
                (output, 3, 2, false)
            }
            (Self::DeltaFused | Self::DeltaFusedBatch, [heads, key, value, kernel, epsilon, tail @ ..]) => {
                let (heads, key, value, kernel, epsilon) = (*heads, *key, *value, *kernel, *epsilon);
                let rows = match (self, tail) {
                    (Self::DeltaFused, []) => 1,
                    (Self::DeltaFusedBatch, [rows]) => *rows,
                    _ => return Err(Error::InvalidField),
                };
                let channels = delta_channels(heads, key, value)?;
                if key > 128 || value > 128 || !(1..=8).contains(&kernel) {
                    return Err(Error::InvalidField);
                }
                positive_float(epsilon)?;
                let history = product(&[channels, kernel])? as u64;
                extents[..11].copy_from_slice(&[
                    u64::from(product(&[rows, channels])?),
                    u64::from(product(&[rows, heads, value])?),
                    u64::from(product(&[rows, heads])?),
                    u64::from(product(&[rows, heads])?),
                    history,
                    heads as u64,
                    heads as u64,
                    value as u64,
                    history,
                    product(&[heads, key, value])? as u64,
                    u64::from(product(&[rows, heads, value])?),
                ]);
                mutated = (1 << 8) | (1 << 9);
                threads = 256;
                (product(&[heads, threads])?, 11, 10, false)
            }
            (Self::ArgMax, &[n]) => {
                product(&[n])?;
                extents[..2].copy_from_slice(&[u64::from(n), 64]);
                (4096, 2, 1, false)
            }
            (Self::GatedProjectBf16, &[k, n]) => {
                product(&[k])?;
                let weights = u64::from(product(&[k, n])?).div_ceil(2);
                extents[..4].copy_from_slice(&[u64::from(k), weights, weights, u64::from(n)]);
                (product(&[n.div_ceil(4), 128])?, 4, 3, false)
            }
            (Self::ProjectBf16, &[m, k, n]) => {
                extents[..3].copy_from_slice(&[product(&[m, k])? as u64, u64::from(product(&[k, n])?).div_ceil(2), product(&[m, n])? as u64]);
                let columns = if m == 1 && n <= 8192 { 4 } else { 32 };
                (product(&[m.div_ceil(16), n.div_ceil(columns), 128])?, 3, 2, false)
            }
            (Self::DequantizeBf16, &[n, half]) => {
                product(&[n])?;
                if half > 1 {
                    return Err(Error::InvalidField);
                }
                extents[..2].copy_from_slice(&[(u64::from(n) + u64::from(half)).div_ceil(2), u64::from(n)]);
                (n, 2, 1, false)
            }
            (Self::Fuzz | Self::FuzzBytes, &[width, height, commands, words]) => {
                let n = product(&[width, height])? as u64;
                let resources = product(&[words])? as u64;
                let packed = self == Self::FuzzBytes;
                threads = 32;
                extents[..3].copy_from_slice(&[
                    if packed { n.div_ceil(4) } else { n },
                    product(&[commands, 12])? as u64,
                    if packed { resources.div_ceil(4) } else { resources },
                ]);
                (if packed { product(&[width, 32])? } else { 32 }, 3, 0, false)
            }
            (Self::Raster, &[width, height, commands, words]) => {
                let n = product(&[width, height])?;
                extents[..4].copy_from_slice(&[n as u64, product(&[commands, 12])? as u64, product(&[words])? as u64, n as u64]);
                (product(&[n.div_ceil(32), 128])?, 4, 3, false)
            }
            (Self::RasterBinned | Self::RasterBinnedBytes, &[width, height, commands, words, bins]) => {
                let n = product(&[width, height])?;
                let header = width.checked_add(height).and_then(|v| v.checked_add(1)).ok_or(Error::InvalidField)?;
                if bins < header {
                    return Err(Error::InvalidField);
                }
                let pixels = if self == Self::RasterBinnedBytes {
                    u64::from(n).div_ceil(4)
                } else {
                    u64::from(n)
                };
                let textures = u64::from(product(&[words])?);
                let textures = if self == Self::RasterBinnedBytes { textures.div_ceil(4) } else { textures };
                extents[..5].copy_from_slice(&[pixels, product(&[commands, 12])? as u64, textures, pixels, bins as u64]);
                (product(&[n.div_ceil(32), 128])?, 5, 3, false)
            }
            (Self::Palette | Self::PaletteBytes, &[width, height, scale]) => {
                if !matches!(scale, 1..=4) {
                    return Err(Error::InvalidField);
                }
                let n = product(&[width, height])?;
                let output = product(&[n, scale, scale])?;
                extents[..3].copy_from_slice(&[
                    if self == Self::PaletteBytes { u64::from(n).div_ceil(4) } else { u64::from(n) },
                    256,
                    output as u64,
                ]);
                (output, 3, 2, false)
            }
            (Self::Zero, &[n]) => {
                product(&[n])?;
                extents[0] = n as u64;
                (n, 1, 0, false)
            }
            (Self::Matmul | Self::Project, &[m, k, n]) => {
                extents[..3].copy_from_slice(&[product(&[m, k])? as u64, product(&[k, n])? as u64, product(&[m, n])? as u64]);
                let columns = if self == Self::Project && m == 1 && n <= 8192 { 4 } else { 32 };
                (product(&[m.div_ceil(16), n.div_ceil(columns), 128])?, 3, 2, false)
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
                (product(&[rows, 128])?, 2, 1, true)
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
                (product(&[rows, 128])?, 3, 2, false)
            }
            (Self::AttentionPrepare, &[heads, dim, rotary, position, theta, epsilon, stride]) => {
                positive_float(theta)?;
                positive_float(epsilon)?;
                if rotary == 0 || rotary % 2 != 0 || rotary > dim || !matches!(stride, 1 | 2) || position == u32::MAX {
                    return Err(Error::InvalidField);
                }
                let n = product(&[heads, dim])?;
                extents[..3].copy_from_slice(&[product(&[n, stride])? as u64, dim as u64, n as u64]);
                (product(&[heads, 128])?, 3, 2, false)
            }
            (Self::AttentionScores, &[heads, kv_heads, dim, positions]) => {
                let n = product(&[heads, positions])?;
                if kv_heads == 0 || heads % kv_heads != 0 {
                    return Err(Error::InvalidField);
                }
                extents[..3].copy_from_slice(&[product(&[heads, dim])? as u64, product(&[positions, kv_heads, dim])? as u64, n as u64]);
                (product(&[heads, positions.div_ceil(32), 128])?, 3, 2, false)
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
                (product(&[heads, dim.div_ceil(128), 128])?, 4, 3, false)
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
                (product(&[heads, 128])?, 6, 5, false)
            }
            (Self::DeltaUpdate, &[heads, key, value]) => {
                let channels = delta_channels(heads, key, value)?;
                let n = product(&[heads, value])?;
                extents[..4].copy_from_slice(&[channels as u64, product(&[heads, 4])? as u64, product(&[heads, key, value])? as u64, n as u64]);
                mutated = 1 << 2;
                (product(&[heads, value.div_ceil(128), 128])?, 4, 3, false)
            }
            (Self::DeltaOutput, &[heads, value, epsilon]) => {
                positive_float(epsilon)?;
                let n = product(&[heads, value])? as u64;
                extents[..4].copy_from_slice(&[n, value as u64, n, n]);
                (product(&[heads, 128])?, 4, 3, false)
            }
            _ => return Err(Error::InvalidField),
        };
        let blocks = count.div_ceil(threads);
        blocks.checked_mul(threads).ok_or(Error::InvalidField)?;
        Ok(Work {
            threads,
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
