use crate::{NativeDevice, layout::Layout};
use alloc::{sync::Arc, vec::Vec};
use burn_backend::{BoolStore, DType, Shape, Slice, TensorData, TensorMetadata};
use d3os_compute::{
    Error, Result,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise};

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

impl NativeTensor {
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
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 || matches!(operation, Elementwise::Copy | Elementwise::MaskFill) {
            return Err(Error::InvalidParameter);
        }
        let dtype = if operation.comparison() { DType::Bool(BoolStore::U32) } else { DType::F32 };
        self.elementwise(rhs, operation, 0, false, dtype)
    }

    pub fn scalar(&self, operation: Elementwise, value: f32) -> Result<Self> {
        if self.dtype != DType::F32 || matches!(operation, Elementwise::Copy | Elementwise::MaskFill) {
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
