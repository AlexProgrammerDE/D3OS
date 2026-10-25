//! Ordinary dtype storage retains packed bytes without a compute conversion.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{BoolStore, DType, Scalar, Shape, bf16, f16};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{
    Kernel,
    storage::{Mode, Width},
};

pub(crate) fn format(dtype: DType) -> Result<d3os_nvidia::compute::cast::Format> {
    use d3os_nvidia::compute::cast::Format;
    match dtype {
        DType::F64 => Ok(Format::F64),
        DType::F32 => Ok(Format::F32),
        DType::Flex32 => Ok(Format::Flex32),
        DType::F16 => Ok(Format::F16),
        DType::BF16 => Ok(Format::BF16),
        DType::I64 => Ok(Format::I64),
        DType::I32 => Ok(Format::I32),
        DType::I16 => Ok(Format::I16),
        DType::I8 => Ok(Format::I8),
        DType::U64 => Ok(Format::U64),
        DType::U32 => Ok(Format::U32),
        DType::U16 => Ok(Format::U16),
        DType::U8 => Ok(Format::U8),
        DType::Bool(BoolStore::U8) => Ok(Format::Bool8),
        DType::Bool(BoolStore::U32) => Ok(Format::Bool32),
        _ => Err(Error::InvalidParameter),
    }
}

pub(crate) fn width(dtype: DType) -> Result<Width> {
    Ok(format(dtype)?.width())
}

pub(crate) fn word_sized(dtype: DType) -> bool {
    width(dtype) == Ok(Width::Four)
}

impl NativeTensor {
    pub(crate) fn fill_bits(shape: Shape, dtype: DType, bits: u64) -> Result<Self> {
        let width = width(dtype)?;
        if dtype.is_bool() && bits > 1 {
            return Err(Error::InvalidParameter);
        }
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Self::empty(shape, dtype);
        }
        let bits = if width == Width::Eight {
            bits
        } else {
            bits & ((1u64 << (width.bytes() * 8)) - 1)
        };
        let buffer = if width == Width::Four {
            Nvidia.produce_words(layout.len(), Kernel::Fill, &[], &[layout.len(), bits as u32 as usize])?
        } else {
            let parameters = layout.storage_parameters(width, Mode::Fill, bits)?;
            Nvidia.produce_words(width.words(layout.len() as u32) as usize, Kernel::TensorStorage, &[], &parameters)?
        };
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    /// Numeric scalar metadata uses the destination dtype before GPU fill.
    pub fn full_scalar(shape: Shape, dtype: DType, value: Scalar) -> Result<Self> {
        let bits = match dtype {
            DType::F64 => value.elem::<f64>().to_bits(),
            DType::F32 | DType::Flex32 => u64::from(value.elem::<f32>().to_bits()),
            DType::F16 => u64::from(value.elem::<f16>().to_bits()),
            DType::BF16 => u64::from(value.elem::<bf16>().to_bits()),
            DType::I64 => value.elem::<i64>() as u64,
            DType::I32 => u64::from(value.elem::<i32>() as u32),
            DType::I16 => u64::from(value.elem::<i16>() as u16),
            DType::I8 => u64::from(value.elem::<i8>() as u8),
            DType::U64 => value.elem::<u64>(),
            DType::U32 => u64::from(value.elem::<u32>()),
            DType::U16 => u64::from(value.elem::<u16>()),
            DType::U8 => u64::from(value.elem::<u8>()),
            DType::Bool(BoolStore::U8 | BoolStore::U32) => u64::from(value.elem::<bool>()),
            _ => return Err(Error::InvalidParameter),
        };
        Self::fill_bits(shape, dtype, bits)
    }

    pub(crate) fn cast_storage(&self, dtype: DType) -> Result<Self> {
        let source = format(self.dtype)?;
        let destination = format(dtype)?;
        let layout = Layout::contiguous(self.layout.shape().clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype,
            });
        }
        let mut parameters = self.layout.storage_parameters(source.width(), Mode::Copy, 0)?;
        parameters[2] = source as usize | ((destination as usize) << 8);
        let buffer = Nvidia.produce_words(
            destination.width().words(layout.len() as u32) as usize,
            Kernel::TensorCast,
            &[(&self.buffer, 0)],
            &parameters,
        )?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype,
        })
    }

    pub(crate) fn copy_storage(&self) -> Result<Self> {
        let width = width(self.dtype)?;
        let layout = Layout::contiguous(self.layout.shape().clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        let parameters = self.layout.storage_parameters(width, Mode::Copy, 0)?;
        let buffer = Nvidia.produce_words(
            width.words(layout.len() as u32) as usize,
            Kernel::TensorStorage,
            &[(&self.buffer, 0)],
            &parameters,
        )?;
        Ok(Self {
            buffer: Arc::new(buffer),
            layout,
            dtype: self.dtype,
        })
    }
}
