//! Versioned D3OS compute syscall protocol. Applications pass owned buffer handles.

use crate::{Error, Result};

pub mod argwhere;
pub mod cast;
pub mod convolution;
pub mod deform;
pub mod elementwise;
pub mod float_arithmetic;
pub mod indexing;
pub mod integer_arithmetic;
pub mod interpolate;
pub mod optimizer;
pub mod packed_gather;
pub mod packed_scatter;
pub mod packed_updates;
pub mod pooling;
pub mod predicate;
pub mod quantization;
pub mod random;
pub mod range;
pub mod reduction;
pub mod scan;
pub mod sort;
pub mod spatial_convolution;
pub mod storage;
pub mod updates;
pub mod words;

pub const VERSION: u32 = 44;
/// END_COMMANDS flag: an inner recording scope joins its parent's completion.
/// The outermost scope always waits. Zero retains synchronous nested scopes.
pub const DEFER_NESTED_COMPLETION: u64 = 1;
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
pub const SERVICE_PROFILE: u32 = 12;
pub const MEMORY_PROFILE: u32 = 13;
pub const SERVICE_OPERATIONS: usize = 14;

/// Compute-buffer bytes at a synchronized boundary, excluding driver firmware
/// and channel allocations. Cached and retired bytes belong to the device pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct MemoryProfile {
    pub process_live_bytes: u64,
    pub device_cached_bytes: u64,
    pub device_retired_bytes: u64,
}
pub const KERNEL_COUNT: usize = 82;

/// CPU service costs by request operation, including queue work within it.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ServiceEntry {
    pub calls: u64,
    pub cycles: u64,
    pub queue_cycles: u64,
}

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
    /// Row-major FP32 product. An optional fourth parameter partitions the
    /// reduction and writes [splits, rows, columns] for SumProjectionPartials.
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
    ResidualNorm = 38,
    ProjectResidualBf16 = 39,
    GroupedProjectBf16 = 40,
    ProjectQuantized = 41,
    Dequantize = 42,
    EmbeddingQuantized = 43,
    TensorElementwise = 44,
    TensorReduction = 45,
    Fill = 46,
    TensorWords = 47,
    Gather = 48,
    Scatter = 49,
    Conv2d = 50,
    Conv2dWindows = 51,
    Conv2dFold = 52,
    Pool = 53,
    PoolBackward = 54,
    MaskWhere = 55,
    ViewAssign = 56,
    Random = 57,
    Range = 58,
    Adam = 59,
    TensorQuantized = 60,
    TensorScale = 61,
    /// Row-major I32 matrix product with wrapping accumulation.
    IntMatmul = 62,
    TensorScan = 63,
    TensorSort = 64,
    ArgWhere = 65,
    TensorStorage = 66,
    TensorCast = 67,
    Interpolate = 68,
    SpatialWindows = 69,
    SpatialFold = 70,
    DeformWindows = 71,
    DeformCoordinates = 72,
    DeformInput = 73,
    TensorPredicate = 74,
    PackedGather = 75,
    PackedAssign = 76,
    PackedMaskWhere = 77,
    PackedBoolScatter = 78,
    PackedScatterAssign = 79,
    IntegerArithmetic = 80,
    FloatArithmetic = 81,
}

const _: () = assert!(Kernel::FloatArithmetic as usize + 1 == KERNEL_COUNT);

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

fn quantized_words(elements: u32, format: u32) -> Result<u64> {
    let (block, bytes) = match format {
        8 => (32, 34),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        _ => return Err(Error::InvalidField),
    };
    if elements == 0 || !elements.is_multiple_of(block) {
        return Err(Error::InvalidField);
    }
    Ok((u64::from(elements / block) * bytes).div_ceil(4))
}

impl Kernel {
    /// These kernels write every output element for every valid work shape.
    /// Partial writers must never expose uncleared allocations through this API.
    pub fn allocates_output(self) -> bool {
        matches!(
            self,
            Self::TensorElementwise
                | Self::TensorQuantized
                | Self::TensorScale
                | Self::Adam
                | Self::MaskWhere
                | Self::Random
                | Self::Range
                | Self::Conv2d
                | Self::Conv2dWindows
                | Self::Conv2dFold
                | Self::Pool
                | Self::PoolBackward
                | Self::TensorReduction
                | Self::Fill
                | Self::TensorWords
                | Self::Gather
                | Self::Matmul
                | Self::IntMatmul
                | Self::TensorScan
                | Self::TensorSort
                | Self::ArgWhere
                | Self::TensorStorage
                | Self::IntegerArithmetic
                | Self::FloatArithmetic
                | Self::PackedGather
                | Self::PackedMaskWhere
                | Self::TensorPredicate
                | Self::TensorCast
                | Self::Interpolate
                | Self::DeformWindows
                | Self::DeformCoordinates
                | Self::SpatialWindows
                | Self::SpatialFold
                | Self::Im2col
                | Self::MaxPool
                | Self::Project
                | Self::Relu
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
                | Self::AttentionOutput
                | Self::ProjectBf16Split
                | Self::SumProjectionPartials
                | Self::ResidualNorm
                | Self::ProjectResidualBf16
                | Self::DeltaConvolution
                | Self::DeltaUpdate
                | Self::DeltaOutput
                | Self::GroupedProjectBf16
                | Self::ProjectQuantized
                | Self::Dequantize
                | Self::EmbeddingQuantized
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
            38 => Self::ResidualNorm,
            39 => Self::ProjectResidualBf16,
            40 => Self::GroupedProjectBf16,
            41 => Self::ProjectQuantized,
            42 => Self::Dequantize,
            43 => Self::EmbeddingQuantized,
            44 => Self::TensorElementwise,
            45 => Self::TensorReduction,
            46 => Self::Fill,
            47 => Self::TensorWords,
            48 => Self::Gather,
            49 => Self::Scatter,
            50 => Self::Conv2d,
            51 => Self::Conv2dWindows,
            52 => Self::Conv2dFold,
            53 => Self::Pool,
            54 => Self::PoolBackward,
            55 => Self::MaskWhere,
            56 => Self::ViewAssign,
            57 => Self::Random,
            58 => Self::Range,
            59 => Self::Adam,
            60 => Self::TensorQuantized,
            61 => Self::TensorScale,
            62 => Self::IntMatmul,
            63 => Self::TensorScan,
            64 => Self::TensorSort,
            65 => Self::ArgWhere,
            66 => Self::TensorStorage,
            67 => Self::TensorCast,
            68 => Self::Interpolate,
            69 => Self::SpatialWindows,
            70 => Self::SpatialFold,
            71 => Self::DeformWindows,
            72 => Self::DeformCoordinates,
            73 => Self::DeformInput,
            74 => Self::TensorPredicate,
            75 => Self::PackedGather,
            76 => Self::PackedAssign,
            77 => Self::PackedMaskWhere,
            78 => Self::PackedBoolScatter,
            79 => Self::PackedScatterAssign,
            80 => Self::IntegerArithmetic,
            81 => Self::FloatArithmetic,
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
            Self::ResidualNorm => "residual_norm",
            Self::ProjectResidualBf16 => "project_residual_bf16_weights",
            Self::GroupedProjectBf16 => "group_project_bf16_weights",
            Self::ProjectQuantized => "project_quantized",
            Self::Dequantize => "dequantize_weights",
            Self::EmbeddingQuantized => "embedding_quantized",
            Self::TensorElementwise => "burn_strided_copy",
            Self::TensorReduction => "burn_sum",
            Self::Fill => "burn_fill",
            Self::TensorWords => "burn_i32_add",
            Self::Gather => "burn_gather",
            Self::Scatter => "burn_scatter_add_f32",
            Self::Conv2d => "burn_conv2d",
            Self::Conv2dWindows => "burn_conv2d_windows",
            Self::Conv2dFold => "burn_conv2d_fold",
            Self::Pool => "burn_max_pool",
            Self::PoolBackward => "burn_avg_pool_backward",
            Self::MaskWhere => "burn_mask_where",
            Self::ViewAssign => "burn_view_assign",
            Self::Random => "burn_random_words",
            Self::Range => "burn_i32_range",
            Self::Adam => "burn_adam",
            Self::TensorQuantized => "burn_quantize",
            Self::TensorScale => "burn_scale_encode",
            Self::IntMatmul => "burn_i32_matmul",
            Self::TensorScan => "burn_cumsum",
            Self::TensorSort => "burn_sort_f32_ascending",
            Self::ArgWhere => "burn_argwhere",
            Self::TensorStorage => "burn_storage_copy",
            Self::TensorCast => "burn_cast",
            Self::Interpolate => "burn_interpolate_nearest",
            Self::SpatialWindows => "burn_spatial_windows",
            Self::SpatialFold => "burn_spatial_fold",
            Self::DeformWindows => "burn_deform_windows",
            Self::DeformCoordinates => "burn_deform_coordinates",
            Self::DeformInput => "burn_deform_input",
            Self::TensorPredicate => "burn_predicate",
            Self::PackedGather => "burn_packed_gather",
            Self::PackedAssign => "burn_packed_assign",
            Self::PackedMaskWhere => "burn_packed_mask",
            Self::PackedBoolScatter => "burn_packed_bool_scatter",
            Self::PackedScatterAssign => "burn_packed_scatter_assign",
            Self::IntegerArithmetic => "burn_integer_arithmetic",
            Self::FloatArithmetic => "burn_float_arithmetic",
        }
    }

    /// Select the cuTile specialization without changing the public request ABI.
    pub fn entry(self, parameters: &[u32]) -> &'static str {
        match (self, parameters) {
            (Self::Interpolate, p) if p.len() == interpolate::PARAMETERS => {
                interpolate::Mode::from_raw(p[6]).map_or(self.symbol(), |mode| mode.symbol(p[14] != 0))
            }
            (Self::TensorStorage, p) if p.len() == storage::PARAMETERS => storage::Mode::from_raw(p[2] >> 8).map_or(self.symbol(), |mode| mode.symbol()),
            (Self::TensorSort, p) if p.len() == sort::PARAMETERS => sort::Mode::from_raw(p[2]).map_or(self.symbol(), |mode| mode.symbol()),
            (Self::TensorScan, p) if p.len() == scan::PARAMETERS => scan::Scan::from_raw(p[2]).map_or(self.symbol(), |op| op.symbol()),
            (Self::TensorScale, [_, _, op]) => quantization::ScaleOperation::from_raw(*op).map_or(self.symbol(), |op| op.symbol()),
            (Self::TensorQuantized, p) if p.len() == quantization::PARAMETERS => {
                quantization::Operation::from_raw(p[4]).map_or(self.symbol(), |op| op.symbol())
            }
            (Self::Random, p) if p.len() == random::PARAMETERS => random::Mode::from_raw(p[5]).map_or(self.symbol(), |op| op.symbol()),
            (Self::Pool | Self::PoolBackward, p) if p.len() == pooling::PARAMETERS => {
                pooling::Mode::from_raw(p[31]).map_or(self.symbol(), |op| op.symbol(self == Self::PoolBackward))
            }
            (Self::Scatter, p) if p.len() == indexing::PARAMETERS => indexing::Scatter::from_raw(p[0] >> 16).map_or(self.symbol(), |op| op.symbol()),
            (Self::TensorWords, p) if p.len() == elementwise::PARAMETERS => words::WordOp::from_raw(p[2] & 0xff).map_or(self.symbol(), |op| op.symbol()),
            (Self::TensorReduction, p) if p.len() == reduction::PARAMETERS => reduction::Reduction::from_raw(p[2]).map_or(self.symbol(), |op| op.symbol()),
            (Self::TensorElementwise, p) if p.len() == elementwise::PARAMETERS => {
                elementwise::Elementwise::from_raw(p[2] & 0xff).map_or(self.symbol(), |op| op.symbol())
            }
            (Self::ProjectQuantized, [m, _, _, 8]) if *m > 16 => "project_m32_q8_0",
            (Self::ProjectQuantized, [m, _, _, 11]) if *m > 16 => "project_m32_q3k",
            (Self::ProjectQuantized, [m, _, _, 12]) if *m > 16 => "project_m32_q4k",
            (Self::ProjectQuantized, [m, _, _, 13]) if *m > 16 => "project_m32_q5k",
            (Self::ProjectQuantized, [m, _, _, 14]) if *m > 16 => "project_m32_q6k",
            (Self::ProjectQuantized, [1, _, _, 8]) => "vector_project_q8_0",
            (Self::ProjectQuantized, [1, _, _, 11]) => "vector_project_q3k",
            (Self::ProjectQuantized, [_, _, _, 11]) => "project_q3k",
            (Self::Dequantize, [_, 11]) => "dequantize_q3k",
            (Self::EmbeddingQuantized, [_, _, _, 11]) => "embedding_q3k",
            (Self::ProjectQuantized, [1, _, _, 12]) => "vector_project_q4k",
            (Self::ProjectQuantized, [_, _, _, 12]) => "project_q4k",
            (Self::Dequantize, [_, 12]) => "dequantize_q4k",
            (Self::EmbeddingQuantized, [_, _, _, 12]) => "embedding_q4k",
            (Self::ProjectQuantized, [1, _, _, 13]) => "vector_project_q5k",
            (Self::ProjectQuantized, [_, _, _, 13]) => "project_q5k",
            (Self::Dequantize, [_, 13]) => "dequantize_q5k",
            (Self::EmbeddingQuantized, [_, _, _, 13]) => "embedding_q5k",
            (Self::ProjectQuantized, [1, _, _, 14]) => "vector_project_q6k",
            (Self::ProjectQuantized, [_, _, _, 14]) => "project_q6k",
            (Self::Dequantize, [_, 14]) => "dequantize_q6k",
            (Self::EmbeddingQuantized, [_, _, _, 14]) => "embedding_q6k",
            (Self::Matmul, [_, _, _, _]) => "tensor_matmul_tf32x3_split",
            (Self::Matmul, [m, k, n]) if *m >= 64 && *k >= 256 && *n >= 64 => "tensor_matmul_tf32x3_k16",
            (Self::AttentionFusedBatch, [_, _, _, _, _, _]) => "attention_split_batch",
            (Self::AttentionOutput, [_, _, _]) => "attention_merge_splits",
            (Self::DeltaFused, [_, _, _, kernel, _]) if *kernel <= 4 => "delta_fused_four_taps",
            (Self::DeltaConvolution, [_, _, _]) => "delta_convolution_batch",
            (Self::DeltaUpdate, [_, _, _, _, _]) => "delta_grouped_recurrent_batch",
            (Self::DeltaUpdate, [_, _, _, _]) => "delta_recurrent_batch",
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
            (Self::ArgWhere, p) => {
                extents[..2].copy_from_slice(&argwhere::validate(p)?);
                (p[2], 2, 1, false)
            }
            (Self::TensorSort, p) => {
                extents[..3].copy_from_slice(&sort::validate(p)?);
                (p[0], 3, 2, false)
            }
            (Self::TensorScan, p) => {
                extents[..2].copy_from_slice(&scan::validate(p)?);
                (product(&[p[0], 128])?, 2, 1, false)
            }
            (Self::TensorQuantized, p) => {
                let (ranges, count) = quantization::validate(p)?;
                extents[..5].copy_from_slice(&ranges);
                (count, 5, 4, false)
            }
            (Self::TensorScale, p) => {
                let (ranges, count) = quantization::validate_scales(p)?;
                extents[..2].copy_from_slice(&ranges);
                (count, 2, 1, false)
            }
            (Self::Adam, p) => {
                extents[..6].copy_from_slice(&optimizer::validate(p)?);
                (p[0], 6, 5, false)
            }
            (Self::Random, p) => {
                extents[..1].copy_from_slice(&random::validate(p)?);
                (p[0], 1, 0, false)
            }
            (Self::MaskWhere, p) => {
                extents[..4].copy_from_slice(&updates::validate_mask(p)?);
                (p[0], 4, 3, false)
            }
            (Self::ViewAssign, p) => {
                extents[..2].copy_from_slice(&updates::validate_assign(p)?);
                (p[0], 2, 0, false)
            }
            (Self::Pool | Self::PoolBackward, p) => {
                let backward = self == Self::PoolBackward;
                let (geometry, ranges) = pooling::validate(p, backward)?;
                let count = if backward { geometry.images } else { geometry.outputs };
                if count == 0 {
                    return Err(Error::InvalidField);
                }
                extents[..2].copy_from_slice(&ranges);
                (count, 2, 1, false)
            }
            (Self::DeformWindows | Self::DeformCoordinates | Self::DeformInput, p) => {
                let g = deform::geometry(p)?;
                if g.columns == 0 {
                    return Err(Error::InvalidField);
                }
                let mask = if p[17] != 0 { u64::from(g.masks) } else { 1 };
                match self {
                    Self::DeformWindows => {
                        extents[..4].copy_from_slice(&[u64::from(g.images), u64::from(g.offsets), mask, u64::from(g.columns)]);
                        (g.columns, 4, 3, false)
                    }
                    Self::DeformCoordinates => {
                        extents[..5].copy_from_slice(&[u64::from(g.images), u64::from(g.offsets), mask, u64::from(g.columns), u64::from(g.coordinates)]);
                        (g.masks, 5, 4, false)
                    }
                    _ => {
                        extents[..4].copy_from_slice(&[u64::from(g.offsets), mask, u64::from(g.columns), u64::from(g.images)]);
                        (g.columns, 4, 3, false)
                    }
                }
            }
            (Self::SpatialWindows, p) => {
                let (geometry, ranges) = spatial_convolution::validate_windows(p)?;
                if geometry.columns == 0 {
                    return Err(Error::InvalidField);
                }
                extents[..2].copy_from_slice(&ranges);
                (geometry.columns, 2, 1, false)
            }
            (Self::SpatialFold, p) => {
                let geometry = spatial_convolution::geometry(p)?;
                if geometry.images == 0 {
                    return Err(Error::InvalidField);
                }
                extents[..2].copy_from_slice(&[u64::from(geometry.columns), u64::from(geometry.images)]);
                (geometry.images, 2, 1, false)
            }
            (Self::Conv2d, p) => {
                let (geometry, ranges) = convolution::validate(p)?;
                if geometry.outputs == 0 {
                    return Err(Error::InvalidField);
                }
                extents[..4].copy_from_slice(&ranges);
                (geometry.blocks.checked_mul(128).ok_or(Error::InvalidField)?, 4, 3, false)
            }
            (Self::Conv2dWindows, p) => {
                let (geometry, ranges) = convolution::validate_windows(p)?;
                if geometry.columns == 0 {
                    return Err(Error::InvalidField);
                }
                extents[..2].copy_from_slice(&ranges);
                (geometry.columns, 2, 1, false)
            }
            (Self::Conv2dFold, p) => {
                let geometry = convolution::geometry(p)?;
                if geometry.images == 0 {
                    return Err(Error::InvalidField);
                }
                extents[..2].copy_from_slice(&[u64::from(geometry.columns), u64::from(geometry.images)]);
                (geometry.images, 2, 1, false)
            }
            (Self::Scatter, p) => {
                extents[..4].copy_from_slice(&indexing::validate_scatter(p)?);
                mutated = 1 << 2;
                let count = if indexing::Scatter::from_raw(p[0] >> 16)?.ordered() {
                    p[2] / p[8 + p[1] as usize]
                } else {
                    p[2]
                };
                (count, 4, 3, false)
            }
            (Self::Gather, p) => {
                extents[..4].copy_from_slice(&indexing::validate(p)?);
                mutated = 1 << 2;
                (p[2], 4, 3, false)
            }
            (Self::TensorWords, p) => {
                extents[..3].copy_from_slice(&words::validate(p)?);
                (p[0], 3, 2, false)
            }
            (Self::Range, p) => {
                extents[0] = range::validate(p)?;
                (p[0], 1, 0, false)
            }
            (Self::Fill, &[count, _bits]) => {
                if count > i32::MAX as u32 {
                    return Err(Error::InvalidField);
                }
                extents[0] = product(&[count])? as u64;
                (count, 1, 0, false)
            }
            (Self::TensorReduction, p) => {
                extents[..2].copy_from_slice(&reduction::validate(p)?);
                (product(&[p[0], 128])?, 2, 1, false)
            }
            (Self::Interpolate, p) => {
                let (sizes, count) = interpolate::validate(p)?;
                if count == 0 {
                    return Err(Error::InvalidField);
                }
                extents[..2].copy_from_slice(&sizes);
                (count, 2, 1, false)
            }
            (Self::PackedGather, p) => {
                extents[..4].copy_from_slice(&packed_gather::validate(p)?);
                mutated = 1 << 2;
                (packed_gather::work_items(p)?, 4, 3, false)
            }
            (Self::PackedAssign, p) => {
                extents[..2].copy_from_slice(&packed_updates::validate_assign(p)?);
                (p[0], 2, 0, false)
            }
            (Self::PackedMaskWhere, p) => {
                extents[..4].copy_from_slice(&packed_updates::validate_mask(p)?);
                (packed_updates::mask_work_items(p)?, 4, 3, false)
            }
            (Self::FloatArithmetic, p) => {
                extents[..3].copy_from_slice(&float_arithmetic::validate(p)?);
                (float_arithmetic::work_items(p)?, 3, 2, false)
            }
            (Self::IntegerArithmetic, p) => {
                extents[..3].copy_from_slice(&integer_arithmetic::validate(p)?);
                (integer_arithmetic::work_items(p)?, 3, 2, false)
            }
            (Self::PackedScatterAssign, p) => {
                extents[..4].copy_from_slice(&packed_scatter::validate(p)?);
                mutated = 1 << 2;
                (packed_scatter::work_items(p)?, 4, 3, false)
            }
            (Self::PackedBoolScatter, p) => {
                extents[..4].copy_from_slice(&packed_updates::validate_bool_scatter(p)?);
                mutated = 1 << 2;
                (p[2], 4, 3, false)
            }
            (Self::TensorPredicate, p) => {
                extents[..3].copy_from_slice(&predicate::validate(p)?);
                (u32::try_from(extents[2]).map_err(|_| Error::InvalidField)?, 3, 2, false)
            }
            (Self::TensorCast, p) => {
                let (source, output) = cast::validate(p)?;
                extents[..2].copy_from_slice(&[source, output]);
                (u32::try_from(output).map_err(|_| Error::InvalidField)?, 2, 1, false)
            }
            (Self::TensorStorage, p) => {
                let (mode, _, source, output) = storage::validate(p)?;
                if mode == storage::Mode::Copy {
                    extents[..2].copy_from_slice(&[source, output]);
                    (u32::try_from(output).map_err(|_| Error::InvalidField)?, 2, 1, false)
                } else {
                    extents[0] = output;
                    (u32::try_from(output).map_err(|_| Error::InvalidField)?, 1, 0, false)
                }
            }
            (Self::TensorElementwise, p) => {
                extents[..3].copy_from_slice(&elementwise::validate(p)?);
                (p[0], 3, 2, false)
            }
            (Self::ProjectQuantized, &[m, k, n, format]) => {
                quantized_words(k, format)?;
                extents[..3].copy_from_slice(&[product(&[m, k])? as u64, quantized_words(product(&[k, n])?, format)?, product(&[m, n])? as u64]);
                (product(&[m.div_ceil(32), n.div_ceil(if m == 1 { 4 } else { 32 }), 128])?, 3, 2, false)
            }
            (Self::Dequantize, &[n, format]) => {
                extents[..2].copy_from_slice(&[quantized_words(n, format)?, product(&[n])? as u64]);
                (n, 2, 1, false)
            }
            (Self::EmbeddingQuantized, &[vocab, width, rows, format]) => {
                quantized_words(width, format)?;
                extents[..3].copy_from_slice(&[
                    quantized_words(product(&[vocab, width])?, format)?,
                    product(&[rows])? as u64,
                    product(&[rows, width])? as u64,
                ]);
                (product(&[rows, width])?, 3, 2, false)
            }
            (Self::GroupedProjectBf16, &[k, q, z, h]) => {
                if !k.is_multiple_of(2) || !q.is_multiple_of(4) || !z.is_multiple_of(4) || !h.is_multiple_of(4) {
                    return Err(Error::InvalidField);
                }
                let total = q
                    .checked_add(z)
                    .and_then(|n| h.checked_mul(2).and_then(|h| n.checked_add(h)))
                    .ok_or(Error::InvalidField)?;
                extents[..6].copy_from_slice(&[
                    product(&[k])? as u64,
                    (product(&[k, q])? / 2) as u64,
                    (product(&[k, z])? / 2) as u64,
                    (product(&[k, h])? / 2) as u64,
                    (product(&[k, h])? / 2) as u64,
                    product(&[total])? as u64,
                ]);
                (product(&[total.div_ceil(4), 128])?, 6, 5, false)
            }
            (Self::AttentionFusedBatch, &[rows, heads, kv_heads, dim, position, splits]) => {
                let positions = position.checked_add(rows).ok_or(Error::InvalidField)?;
                if kv_heads == 0 || heads % kv_heads != 0 || !(1..=256).contains(&dim) || !(1..=32).contains(&splits) {
                    return Err(Error::InvalidField);
                }
                positions.checked_add(splits * 32 - 1).ok_or(Error::InvalidField)?;
                let n = product(&[rows, heads, dim])?;
                let cache = product(&[positions, kv_heads, dim])?;
                extents[..5].copy_from_slice(&[
                    u64::from(n),
                    u64::from(cache),
                    u64::from(cache),
                    u64::from(product(&[n, 2])?),
                    u64::from(product(&[rows, heads, splits, dim + 2])?),
                ]);
                (product(&[rows, heads, splits, 128])?, 5, 4, false)
            }
            (Self::AttentionOutput, &[heads, dim, splits]) => {
                if !(1..=256).contains(&dim) || !(1..=32).contains(&splits) {
                    return Err(Error::InvalidField);
                }
                let n = product(&[heads, dim])?;
                extents[..3].copy_from_slice(&[u64::from(product(&[heads, splits, dim + 2])?), u64::from(product(&[n, 2])?), u64::from(n)]);
                (product(&[heads, 128])?, 3, 2, false)
            }
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
            (Self::Matmul | Self::ProjectBf16Split, &[m, k, n, splits]) => {
                let weights = u64::from(product(&[k, n])?);
                extents[..3].copy_from_slice(&[
                    u64::from(product(&[m, k])?),
                    if self == Self::ProjectBf16Split { weights.div_ceil(2) } else { weights },
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
            (Self::ProjectResidualBf16, &[k, n]) => {
                if !k.is_multiple_of(2) || !(1..=8192).contains(&n) {
                    return Err(Error::InvalidField);
                }
                let weights = u64::from(product(&[k, n])?).div_ceil(2);
                extents[..4].copy_from_slice(&[u64::from(n), u64::from(k), weights, u64::from(n)]);
                (product(&[n.div_ceil(4), 128])?, 4, 3, true)
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
            (Self::IntMatmul, &[m, k, n]) => {
                extents[..3].copy_from_slice(&[u64::from(product(&[m, k])?), u64::from(product(&[k, n])?), u64::from(product(&[m, n])?)]);
                (product(&[m, n.div_ceil(128), 128])?, 3, 2, false)
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
            (Self::ResidualNorm, &[rows, width, epsilon]) => {
                positive_float(epsilon)?;
                if !(1..=1024).contains(&width) {
                    return Err(Error::InvalidField);
                }
                let n = product(&[rows, width])?;
                extents[..4].copy_from_slice(&[u64::from(n), u64::from(n), u64::from(width), u64::from(n)]);
                mutated = 1;
                (product(&[rows, 128])?, 4, 3, false)
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
            (Self::DeltaConvolution, &[channels, kernel, rows]) => {
                if !(1..=8).contains(&kernel) {
                    return Err(Error::InvalidField);
                }
                let history = product(&[channels, kernel])? as u64;
                let n = product(&[rows, channels])? as u64;
                extents[..4].copy_from_slice(&[n, history, history, n]);
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
            (Self::DeltaUpdate, &[heads, key, value, rows, key_heads]) => {
                if !(1..=128).contains(&key) || !(1..=128).contains(&value) {
                    return Err(Error::InvalidField);
                }
                product(&[key_heads])?;
                if !heads.is_multiple_of(key_heads) {
                    return Err(Error::InvalidField);
                }
                let channels = product(&[2, key_heads, key])?
                    .checked_add(product(&[heads, value])?)
                    .ok_or(Error::InvalidField)?;
                let scalars = product(&[rows, heads])? as u64;
                extents[..7].copy_from_slice(&[
                    product(&[rows, channels])? as u64,
                    scalars,
                    scalars,
                    heads as u64,
                    heads as u64,
                    product(&[heads, key, value])? as u64,
                    product(&[rows, heads, value])? as u64,
                ]);
                mutated = 1 << 5;
                (product(&[heads, value.div_ceil(32), 128])?, 7, 6, false)
            }
            (Self::DeltaUpdate, &[heads, key, value, rows]) => {
                if !(1..=128).contains(&key) || !(1..=128).contains(&value) {
                    return Err(Error::InvalidField);
                }
                let channels = delta_channels(heads, key, value)?;
                let scalars = product(&[rows, heads])? as u64;
                extents[..7].copy_from_slice(&[
                    product(&[rows, channels])? as u64,
                    scalars,
                    scalars,
                    heads as u64,
                    heads as u64,
                    product(&[heads, key, value])? as u64,
                    product(&[rows, heads, value])? as u64,
                ]);
                mutated = 1 << 5;
                (product(&[heads, value.div_ceil(32), 128])?, 7, 6, false)
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
