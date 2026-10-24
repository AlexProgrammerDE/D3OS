use crate::{NativeDevice, layout::Layout};
use alloc::{sync::Arc, vec::Vec};
use burn_backend::{BoolStore, DType, Shape, Slice, TensorData, TensorMetadata};
use d3os_compute::{
    Error, Result,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::words::{ValueType, WordOp};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise, reduction::Reduction};

/// A Burn tensor primitive backed by shared, process-owned GPU storage.
#[derive(Clone, Debug)]
pub struct NativeTensor {
    buffer: Arc<GpuBuffer<u32>>,
    layout: Layout,
    dtype: DType,
}

fn supported(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::I32 | DType::Bool(BoolStore::U32))
}

fn word_dtype(value: ValueType) -> DType {
    match value {
        ValueType::F32 => DType::F32,
        ValueType::I32 => DType::I32,
        ValueType::Bool => DType::Bool(BoolStore::U32),
    }
}

impl NativeTensor {
    /// Fill dense GPU storage with exact raw words without a host upload or a
    /// separate clear. Boolean words must have their canonical zero/one values.
    pub fn full(shape: Shape, dtype: DType, word: u32) -> Result<Self> {
        if !supported(dtype) || (matches!(dtype, DType::Bool(_)) && word > 1) {
            return Err(Error::InvalidParameter);
        }
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Self::empty(shape, dtype);
        }
        let buffer = Nvidia.produce_words(layout.len(), Kernel::Fill, &[], &[layout.len(), word as usize])?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    pub fn empty(shape: Shape, dtype: DType) -> Result<Self> {
        if !supported(dtype) {
            return Err(Error::InvalidParameter);
        }
        let layout = Layout::contiguous(shape)?;
        // The driver requires a nonempty physical allocation. An empty logical
        // tensor owns one unused word and never dispatches element work.
        let buffer = Nvidia.allocate_words(layout.len().max(1))?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    /// Upload host data without changing floating-point bit representations.
    pub fn from_data(data: TensorData) -> Result<Self> {
        let data = match data.dtype() {
            DType::Bool(_) => data.try_cast(DType::Bool(BoolStore::U32)).map_err(|_| Error::InvalidParameter)?,
            DType::F32 | DType::I32 => data,
            _ => return Err(Error::InvalidParameter),
        };
        let layout = Layout::contiguous(data.shape().clone())?;
        let dtype = data.dtype();
        if layout.is_empty() {
            return Self::empty(layout.shape().clone(), dtype);
        }
        let buffer = Nvidia.upload_bytes(data.as_bytes())?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    pub fn with_layout(&self, layout: Layout) -> Result<Self> {
        if layout.storage_elements() != self.layout.storage_elements() {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            layout,
            dtype: self.dtype,
        })
    }

    pub fn reshape(mut self, shape: Shape) -> Result<Self> {
        self.layout = self.layout.reshape(shape)?;
        Ok(self)
    }

    pub fn permute(mut self, axes: &[usize]) -> Result<Self> {
        self.layout = self.layout.permute(axes)?;
        Ok(self)
    }

    pub fn flip(mut self, axes: &[usize]) -> Result<Self> {
        self.layout = self.layout.flip(axes)?;
        Ok(self)
    }

    pub fn expand(mut self, shape: Shape) -> Result<Self> {
        self.layout = self.layout.expand(shape)?;
        Ok(self)
    }

    pub fn slice(mut self, slices: &[Slice]) -> Result<Self> {
        self.layout = self.layout.slice(slices)?;
        Ok(self)
    }

    pub fn unfold(mut self, dimension: usize, size: usize, step: usize) -> Result<Self> {
        self.layout = self.layout.unfold(dimension, size, step)?;
        Ok(self)
    }

    fn elementwise(&self, rhs: &Self, op: Elementwise, scalar_bits: u32, scalar_rhs: bool, dtype: DType) -> Result<Self> {
        let shape = crate::layout::broadcast(self.layout.shape(), rhs.layout.shape())?;
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype,
            });
        }
        let left = self.layout.expand(shape.clone())?;
        let right = rhs.layout.expand(shape)?;
        let parameters = left.elementwise_parameters(&right, op, scalar_bits, scalar_rhs)?;
        let buffer = Nvidia.produce_words(layout.len(), Kernel::TensorElementwise, &[(&self.buffer, 0), (&rhs.buffer, 0)], &parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    /// Materialize a view on the GPU, preserving the stored bits of every dtype.
    pub fn copy(&self) -> Result<Self> {
        self.elementwise(self, Elementwise::Copy, 0, false, self.dtype)
    }

    pub fn binary(&self, rhs: &Self, operation: Elementwise) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 || !operation.binary() {
            return Err(Error::InvalidParameter);
        }
        let dtype = if operation.comparison() { DType::Bool(BoolStore::U32) } else { DType::F32 };
        self.elementwise(rhs, operation, 0, false, dtype)
    }

    pub fn scalar(&self, operation: Elementwise, value: f32) -> Result<Self> {
        if self.dtype != DType::F32 || !operation.binary() {
            return Err(Error::InvalidParameter);
        }
        let dtype = if operation.comparison() { DType::Bool(BoolStore::U32) } else { DType::F32 };
        self.elementwise(self, operation, value.to_bits(), true, dtype)
    }

    pub fn mask_fill(&self, mask: &Self, value: f32) -> Result<Self> {
        if self.dtype != DType::F32 || mask.dtype != DType::Bool(BoolStore::U32) {
            return Err(Error::InvalidParameter);
        }
        self.elementwise(mask, Elementwise::MaskFill, value.to_bits(), false, DType::F32)
    }

    pub fn unary(&self, operation: Elementwise) -> Result<Self> {
        if self.dtype != DType::F32 || !operation.unary() {
            return Err(Error::InvalidParameter);
        }
        let dtype = if operation.boolean() { DType::Bool(BoolStore::U32) } else { DType::F32 };
        self.elementwise(self, operation, 0, false, dtype)
    }

    fn words(&self, rhs: &Self, op: WordOp, scalar: u32, scalar_rhs: bool) -> Result<Self> {
        if self.dtype != word_dtype(op.input()) || rhs.dtype != if op.mask_fill() { DType::Bool(BoolStore::U32) } else { self.dtype } {
            return Err(Error::InvalidParameter);
        }
        let shape = crate::layout::broadcast(self.layout.shape(), rhs.layout.shape())?;
        let layout = Layout::contiguous(shape.clone())?;
        let dtype = word_dtype(op.output());
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype,
            });
        }
        let left = self.layout.expand(shape.clone())?;
        let right = rhs.layout.expand(shape)?;
        let parameters = left.word_parameters(&right, op, scalar, scalar_rhs)?;
        let buffer = Nvidia.produce_words(layout.len(), Kernel::TensorWords, &[(&self.buffer, 0), (&rhs.buffer, 0)], &parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    pub fn word_binary(&self, rhs: &Self, operation: WordOp) -> Result<Self> {
        if !operation.binary() {
            return Err(Error::InvalidParameter);
        }
        self.words(rhs, operation, 0, false)
    }

    /// Scalar words retain their exact I32 representation. Boolean scalars must
    /// use canonical zero/one words, including for empty logical results.
    pub fn word_scalar(&self, operation: WordOp, value: u32) -> Result<Self> {
        if !operation.binary() || (operation.input() == ValueType::Bool && value > 1) {
            return Err(Error::InvalidParameter);
        }
        self.words(self, operation, value, true)
    }

    pub fn word_unary(&self, operation: WordOp) -> Result<Self> {
        if !operation.unary() {
            return Err(Error::InvalidParameter);
        }
        self.words(self, operation, 0, false)
    }

    pub fn mask_fill_i32(&self, mask: &Self, value: i32) -> Result<Self> {
        self.words(mask, WordOp::IntMaskFill, value as u32, false)
    }

    pub fn mask_fill_bool(&self, mask: &Self, value: bool) -> Result<Self> {
        self.words(mask, WordOp::BoolMaskFill, u32::from(value), false)
    }

    /// Convert supported native dtypes. Float-to-int truncates toward zero,
    /// clamps overflow to I32 bounds, and maps NaN to zero, as Rust casts do.
    pub fn cast(&self, dtype: DType) -> Result<Self> {
        if !supported(dtype) {
            return Err(Error::InvalidParameter);
        }
        if self.dtype == dtype || (self.dtype == DType::Bool(BoolStore::U32) && dtype == DType::I32) {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout: self.layout.clone(),
                dtype,
            });
        }
        let op = match (self.dtype, dtype) {
            (DType::F32, DType::I32) => WordOp::FloatToInt,
            (DType::F32, DType::Bool(_)) => WordOp::FloatToBool,
            (DType::I32, DType::F32) => WordOp::IntToFloat,
            (DType::I32, DType::Bool(_)) => WordOp::IntToBool,
            (DType::Bool(_), DType::F32) => WordOp::BoolToFloat,
            _ => return Err(Error::InvalidParameter),
        };
        self.word_unary(op)
    }

    /// FP32 matrix multiplication with Tensor Core accumulation and broadcast batches.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        let shape = crate::layout::matmul_shape(self.layout.shape(), rhs.layout.shape())?;
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: DType::F32,
            });
        }
        let rank = shape.num_dims();
        let rows = shape[rank - 2];
        let cols = shape[rank - 1];
        let inner = self.layout.shape()[self.layout.shape().num_dims() - 1];
        if inner == 0 {
            return Self::empty(shape, DType::F32);
        }
        let matrix_contiguous = |tensor: &Self| {
            let rank = tensor.layout.shape().num_dims();
            let rows = tensor.layout.shape()[rank - 2];
            let cols = tensor.layout.shape()[rank - 1];
            (rows == 1 || tensor.layout.strides()[rank - 2] == cols as i64) && (cols == 1 || tensor.layout.strides()[rank - 1] == 1)
        };
        // Materialize only an input with a noncontiguous matrix, before batch
        // expansion. Broadcast batches then reuse its original resident storage.
        let left = if matrix_contiguous(self) { self.clone() } else { self.copy()? };
        let right = if matrix_contiguous(rhs) { rhs.clone() } else { rhs.copy()? };
        let mut left_shape = shape.to_vec();
        left_shape[rank - 1] = inner;
        let mut right_shape = shape.to_vec();
        right_shape[rank - 2] = inner;
        let left_layout = left.layout.expand(Shape::from(left_shape))?;
        let right_layout = right.layout.expand(Shape::from(right_shape))?;
        let batches = layout.len() / (rows * cols);
        let parameters = [rows, inner, cols];
        let buffer = if batches == 1 {
            Nvidia.produce_words(
                rows * cols,
                Kernel::Matmul,
                &[(&left.buffer, left_layout.offset()), (&right.buffer, right_layout.offset())],
                &parameters,
            )?
        } else {
            let buffer = Nvidia.allocate_words(layout.len())?;
            for batch in 0..batches {
                Nvidia.dispatch_words(
                    Kernel::Matmul,
                    &[
                        (&left.buffer, left_layout.element_offset(batch * rows * inner)?),
                        (&right.buffer, right_layout.element_offset(batch * inner * cols)?),
                        (&buffer, batch * rows * cols),
                    ],
                    &parameters,
                )?;
            }
            buffer
        };
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype: DType::F32,
        })
    }

    /// Reduce one dimension, preserving its rank. Extrema propagate the first
    /// NaN and use the first logical index for equal values, including zeros.
    pub fn reduce(&self, axis: usize, operation: Reduction) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        let layout = Layout::contiguous(self.layout.reduction_shape(axis)?)?;
        let dtype = if operation.index() { DType::I32 } else { DType::F32 };
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype,
            });
        }
        let parameters = self.layout.reduction_parameters(axis, operation)?;
        let buffer = Nvidia.produce_words(layout.len(), Kernel::TensorReduction, &[(&self.buffer, 0)], &parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    /// Explicit host readback gathers the requested view in logical order.
    /// GPU arithmetic must use the resident storage and its layout instead.
    pub fn read(&self) -> Result<TensorData> {
        let mut words = Vec::new();
        words.try_reserve_exact(self.layout.storage_elements()).map_err(|_| Error::AllocationFailed)?;
        words.resize(self.layout.storage_elements(), 0u32);
        Nvidia.read_words_into(&self.buffer, 0, &mut words)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.layout.len().checked_mul(4).ok_or(Error::SizeOverflow)?)
            .map_err(|_| Error::AllocationFailed)?;
        for element in 0..self.layout.len() {
            let offset = self.layout.element_offset(element)?;
            let word = words.get(offset).ok_or(Error::DeviceFault)?;
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        Ok(TensorData::from_bytes_vec(bytes, self.layout.shape().clone(), self.dtype))
    }
}

impl TensorMetadata for NativeTensor {
    type Device = NativeDevice;
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn shape(&self) -> Shape {
        self.layout.shape().clone()
    }
    fn device(&self) -> NativeDevice {
        NativeDevice
    }
    fn can_mut(&self) -> bool {
        Arc::strong_count(&self.buffer) == 1 && self.layout.is_contiguous()
    }
}
