use crate::{NativeDevice, layout::Layout};
use alloc::{sync::Arc, vec::Vec};
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{BoolStore, DType, Shape, Slice, TensorData, TensorMetadata};
use d3os_compute::{
    Error, Result,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::scan::Scan;
use d3os_nvidia::compute::words::{ValueType, WordOp};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise, indexing::Scatter};

/// A Burn tensor primitive backed by shared, process-owned GPU storage.
#[derive(Clone, Debug)]
pub struct NativeTensor {
    pub(crate) buffer: Arc<GpuBuffer<u32>>,
    pub(crate) layout: Layout,
    pub(crate) dtype: DType,
}

fn supported(dtype: DType) -> bool {
    crate::storage::width(dtype).is_ok()
}

fn word_dtype(value: ValueType) -> DType {
    match value {
        ValueType::F32 => DType::F32,
        ValueType::I32 => DType::I32,
        ValueType::Bool => DType::Bool(BoolStore::U32),
    }
}

impl NativeTensor {
    /// Fill dense storage from the low raw bits without a host upload.
    /// Use `full_scalar` for numeric values in their destination dtype.
    /// Boolean fills must have their canonical zero/one values.
    pub fn full(shape: Shape, dtype: DType, word: u32) -> Result<Self> {
        Self::fill_bits(shape, dtype, u64::from(word))
    }

    pub fn empty(shape: Shape, dtype: DType) -> Result<Self> {
        if !supported(dtype) {
            return Err(Error::InvalidParameter);
        }
        let layout = Layout::contiguous(shape)?;
        // The driver requires a nonempty physical allocation. An empty logical
        // tensor owns one unused word and never dispatches element work.
        let buffer = Nvidia.allocate_words((crate::storage::width(dtype)?.words(layout.len() as u32) as usize).max(1))?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    /// Upload host data without changing floating-point bit representations.
    pub fn from_data(data: TensorData) -> Result<Self> {
        let data = match data.dtype() {
            DType::Bool(BoolStore::Native) => data.try_cast(DType::Bool(BoolStore::U32)).map_err(|_| Error::InvalidParameter)?,
            dtype if supported(dtype) => data,
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
        if !crate::storage::word_sized(self.dtype) || !crate::storage::word_sized(rhs.dtype) || !crate::storage::word_sized(dtype) {
            return Err(Error::InvalidParameter);
        }
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
        if crate::storage::word_sized(self.dtype) {
            self.elementwise(self, Elementwise::Copy, 0, false, self.dtype)
        } else {
            self.copy_storage()
        }
    }

    pub fn binary(&self, rhs: &Self, operation: Elementwise) -> Result<Self> {
        if self.dtype != DType::F32 && d3os_nvidia::compute::float_arithmetic::Operation::from_elementwise(operation).is_ok() {
            return self.float_binary(rhs, operation);
        }
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
        if self.dtype != DType::F32 {
            return Err(Error::InvalidParameter);
        }
        if mask.dtype != DType::Bool(BoolStore::U32) {
            return self.mask_fill_scalar(mask, value.into());
        }
        self.elementwise(mask, Elementwise::MaskFill, value.to_bits(), false, DType::F32)
    }

    pub fn unary(&self, operation: Elementwise) -> Result<Self> {
        if self.dtype != DType::F32 && d3os_nvidia::compute::float_arithmetic::Operation::from_elementwise(operation).is_ok() {
            return self.float_unary(operation);
        }
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
        if ((self.dtype.is_int() || self.dtype.is_uint()) && self.dtype != DType::I32)
            || (self.dtype == DType::I32 && operation == WordOp::Power && rhs.dtype != DType::I32)
        {
            return self.integer_binary(rhs, operation);
        }
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
        if (self.dtype.is_int() || self.dtype.is_uint()) && self.dtype != DType::I32 {
            return self.integer_unary(operation);
        }
        if !operation.unary() {
            return Err(Error::InvalidParameter);
        }
        self.words(self, operation, 0, false)
    }

    pub fn mask_fill_i32(&self, mask: &Self, value: i32) -> Result<Self> {
        if self.dtype != DType::I32 {
            return Err(Error::InvalidParameter);
        }
        if mask.dtype != DType::Bool(BoolStore::U32) {
            return self.mask_fill_scalar(mask, value.into());
        }
        self.words(mask, WordOp::IntMaskFill, value as u32, false)
    }

    pub fn mask_fill_bool(&self, mask: &Self, value: bool) -> Result<Self> {
        if !self.dtype.is_bool() {
            return Err(Error::InvalidParameter);
        }
        if self.dtype != DType::Bool(BoolStore::U32) || mask.dtype != DType::Bool(BoolStore::U32) {
            return self.mask_fill_scalar(mask, value.into());
        }
        self.words(mask, WordOp::BoolMaskFill, u32::from(value), false)
    }

    /// Convert supported native dtypes. Float-to-int truncates toward zero,
    /// clamps overflow to destination bounds and maps NaN to zero. Integer
    /// narrowing retains the low bits; float rounding uses ties to even.
    pub fn cast(&self, dtype: DType) -> Result<Self> {
        if !supported(dtype) {
            return Err(Error::InvalidParameter);
        }
        if self.dtype == dtype || (matches!(self.dtype, DType::F32 | DType::Flex32) && matches!(dtype, DType::F32 | DType::Flex32)) {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout: self.layout.clone(),
                dtype,
            });
        }
        let op = match (self.dtype, dtype) {
            (DType::F32, DType::I32) => WordOp::FloatToInt,
            (DType::F32, DType::Bool(BoolStore::U32)) => WordOp::FloatToBool,
            (DType::I32, DType::F32) => WordOp::IntToFloat,
            (DType::I32, DType::Bool(BoolStore::U32)) => WordOp::IntToBool,
            (DType::Bool(BoolStore::U32), DType::F32) => WordOp::BoolToFloat,
            _ => return self.cast_storage(dtype),
        };
        self.word_unary(op)
    }

    /// Read arbitrary elements on the GPU. Only the four-byte bounds result
    /// crosses to the host; invalid device indices leave inputs unchanged.
    pub fn gather(&self, axis: usize, indices: &Self) -> Result<Self> {
        let indices = indices.checked_indices()?;
        let indices = &indices;
        if !crate::storage::word_sized(self.dtype) {
            return self.packed_gather(axis, indices);
        }
        if indices.dtype != DType::I32 {
            return Err(Error::InvalidParameter);
        }
        let layout = Layout::contiguous(self.layout.gather_shape(&indices.layout, axis)?)?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        let parameters = self.layout.gather_parameters(&indices.layout, axis)?;
        let status = Nvidia.produce_words(1, Kernel::Fill, &[], &[1, 0])?;
        let buffer = Nvidia.produce_words(
            layout.len(),
            Kernel::Gather,
            &[(&self.buffer, 0), (&indices.buffer, 0), (&status, 0)],
            &parameters,
        )?;
        let mut invalid = [0];
        Nvidia.read_words_into(&status, 0, &mut invalid)?;
        if invalid[0] != 0 {
            return Err(Error::InvalidParameter);
        }
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype: self.dtype,
        })
    }

    pub fn select(&self, axis: usize, indices: &Self) -> Result<Self> {
        let indices = indices.checked_indices()?;
        self.gather(axis, &self.selection_indices(axis, &indices)?)
    }

    fn selection_indices(&self, axis: usize, indices: &Self) -> Result<Self> {
        if indices.dtype != DType::I32 || indices.layout.shape().num_dims() != 1 || axis >= self.layout.shape().num_dims() {
            return Err(Error::InvalidParameter);
        }
        let mut shape = self.layout.shape().to_vec();
        shape[axis] = indices.layout.len();
        let mut strides = alloc::vec![0;shape.len()];
        strides[axis] = indices.layout.strides()[0];
        let layout = Layout::from_parts(Shape::from(shape), strides, indices.layout.offset() as i64, indices.layout.storage_elements())?;
        indices.with_layout(layout)
    }

    /// Accumulate repeated indices without modifying shared inputs. Floating
    /// additions preserve subnormals; their ordering is unspecified for collisions.
    pub fn scatter_add(&self, axis: usize, indices: &Self, values: &Self) -> Result<Self> {
        if crate::indexed_accumulation::uses_typed_accumulation(self.dtype, IndexingUpdateOp::Add) {
            return self.indexed_accumulation(axis, indices, values, IndexingUpdateOp::Add);
        }
        let operation = match self.dtype {
            DType::F32 => Scatter::FloatAdd,
            DType::I32 => Scatter::IntAdd,
            _ => return Err(Error::InvalidParameter),
        };
        self.scatter(axis, indices, values, operation)
    }

    pub fn scatter_update(&self, axis: usize, indices: &Self, values: &Self, update: IndexingUpdateOp) -> Result<Self> {
        if crate::indexed_accumulation::uses_typed_accumulation(self.dtype, update) {
            return self.indexed_accumulation(axis, indices, values, update);
        }
        use IndexingUpdateOp::*;
        let indices = indices.checked_indices()?;
        if update == Assign && !matches!(self.dtype, DType::F32 | DType::I32) {
            return self.packed_scatter_assign(axis, &indices, values);
        }
        let operation = match (self.dtype, update) {
            (DType::F32, Add) => Scatter::FloatAdd,
            (DType::I32, Add) => Scatter::IntAdd,
            (DType::F32, Assign) => Scatter::FloatAssign,
            (DType::I32, Assign) => Scatter::IntAssign,
            (DType::F32, Mul) => Scatter::FloatMultiply,
            (DType::I32, Mul) => Scatter::IntMultiply,
            (DType::F32, Min) => Scatter::FloatMin,
            (DType::I32, Min) => Scatter::IntMin,
            (DType::F32, Max) => Scatter::FloatMax,
            (DType::I32, Max) => Scatter::IntMax,
            _ => return Err(Error::InvalidParameter),
        };
        self.scatter(axis, &indices, values, operation)
    }

    pub fn scatter_or(&self, axis: usize, indices: &Self, values: &Self) -> Result<Self> {
        let indices = indices.checked_indices()?;
        let indices = &indices;
        if self.dtype == DType::Bool(BoolStore::U8) {
            return self.packed_scatter_or(axis, indices, values);
        }
        if self.dtype != DType::Bool(BoolStore::U32) {
            return Err(Error::InvalidParameter);
        }
        self.scatter(axis, indices, values, Scatter::BoolOr)
    }

    pub fn select_update(&self, axis: usize, indices: &Self, values: &Self, update: IndexingUpdateOp) -> Result<Self> {
        if crate::indexed_accumulation::uses_typed_accumulation(self.dtype, update) {
            crate::indexed_accumulation::validate_axis_update(self, axis, indices, values, update, crate::indexed_accumulation::AxisRoute::Select)?;
            if values.layout.is_empty() {
                return Ok(self.clone());
            }
            let indices = indices.checked_indices()?;
            return self.indexed_accumulation(axis, &self.selection_indices(axis, &indices)?, values, update);
        }
        let indices = indices.checked_indices()?;
        self.scatter_update(axis, &self.selection_indices(axis, &indices)?, values, update)
    }

    pub fn select_or(&self, axis: usize, indices: &Self, values: &Self) -> Result<Self> {
        let indices = indices.checked_indices()?;
        self.scatter_or(axis, &self.selection_indices(axis, &indices)?, values)
    }

    fn scatter(&self, axis: usize, indices: &Self, values: &Self, operation: Scatter) -> Result<Self> {
        let indices = indices.checked_indices()?;
        let indices = &indices;
        if indices.dtype != DType::I32 || values.dtype != self.dtype {
            return Err(Error::InvalidParameter);
        }
        let shape = self.layout.gather_shape(&indices.layout, axis)?;
        if values.layout.shape() != &shape {
            return Err(Error::ShapeMismatch);
        }
        if indices.layout.is_empty() {
            return Ok(self.clone());
        }
        if self.layout.shape()[axis] == 0 {
            return Err(Error::InvalidParameter);
        }
        let layout = Layout::contiguous(self.layout.shape().clone())?;
        let parameters = layout.scatter_parameters(&indices.layout, axis, operation)?;
        let values = if values.layout.is_contiguous() { values.clone() } else { values.copy()? };
        let output = self.copy()?;
        let status = Nvidia.produce_words(1, Kernel::Fill, &[], &[1, 0])?;
        Nvidia.dispatch_words(
            Kernel::Scatter,
            &[
                (&values.buffer, values.layout.offset()),
                (&indices.buffer, 0),
                (&status, 0),
                (&output.buffer, 0),
            ],
            &parameters,
        )?;
        let mut invalid = [0];
        Nvidia.read_words_into(&status, 0, &mut invalid)?;
        if invalid[0] != 0 {
            return Err(Error::InvalidParameter);
        }
        Ok(output)
    }

    /// Inclusive resident prefixes along a checked axis, with a dense output.
    pub fn scan(&self, axis: usize, operation: Scan) -> Result<Self> {
        let operation = match self.dtype {
            DType::F32 if !operation.is_integer() => operation,
            DType::I32 => operation.integer(),
            _ => return Err(Error::InvalidParameter),
        };
        if axis >= self.layout.shape().num_dims() {
            return Err(Error::InvalidParameter);
        }
        let layout = Layout::contiguous(self.layout.shape().clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        let parameters = self.layout.scan_parameters(axis, operation)?;
        let buffer = Nvidia.produce_words(layout.len(), Kernel::TensorScan, &[(&self.buffer, 0)], &parameters)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype: self.dtype,
        })
    }

    /// Explicit host readback gathers the requested view in logical order.
    /// GPU arithmetic must use the resident storage and its layout instead.
    pub fn read(&self) -> Result<TensorData> {
        let width = crate::storage::width(self.dtype)?;
        let word_count = width.words(self.layout.storage_elements() as u32) as usize;
        let mut words = Vec::new();
        words.try_reserve_exact(word_count).map_err(|_| Error::AllocationFailed)?;
        words.resize(word_count, 0u32);
        Nvidia.read_words_into(&self.buffer, 0, &mut words)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.layout.len().checked_mul(width.bytes() as usize).ok_or(Error::SizeOverflow)?)
            .map_err(|_| Error::AllocationFailed)?;
        for element in 0..self.layout.len() {
            let offset = self.layout.element_offset(element)?;
            if width == d3os_nvidia::compute::storage::Width::Four {
                bytes.extend_from_slice(&words.get(offset).ok_or(Error::DeviceFault)?.to_le_bytes());
            } else {
                let address = offset.checked_mul(width.bytes() as usize).ok_or(Error::SizeOverflow)?;
                for byte in address..address + width.bytes() as usize {
                    bytes.push((words.get(byte / 4).ok_or(Error::DeviceFault)? >> ((byte % 4) * 8)) as u8);
                }
            }
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
