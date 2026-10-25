//! Packed masks and partial updates preserve raw values and word ownership.
use crate::{
    NativeTensor,
    layout::Layout,
    storage::{scalar_bits, width},
};
use alloc::sync::Arc;
use burn_backend::{BoolStore, DType, Scalar};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, packed_updates as abi, storage::Width};

pub(crate) fn assignment(target: &Layout, source: &Layout, width: Width) -> Result<(Kernel, [usize; 32])> {
    let mut p = target.assignment_parameters(source)?;
    if width == Width::Four {
        return Ok((Kernel::ViewAssign, p));
    }
    p[2] = width as usize;
    let raw = p.map(|v| v as u32);
    abi::validate_assign(&raw).map_err(|_| Error::InvalidParameter)?;
    Ok((Kernel::PackedAssign, p))
}
impl NativeTensor {
    pub fn mask_fill_scalar(&self, mask: &Self, scalar: Scalar) -> Result<Self> {
        if mask.dtype == DType::Bool(BoolStore::U32) {
            match self.dtype {
                DType::F32 => return self.mask_fill(mask, scalar.elem::<f32>()),
                DType::I32 => return self.mask_fill_i32(mask, scalar.elem::<i32>()),
                DType::Bool(BoolStore::U32) => return self.mask_fill_bool(mask, scalar.elem::<bool>()),
                _ => {}
            }
        }
        self.packed_mask(mask, self, Some(scalar_bits(self.dtype, scalar)?))
    }
    pub(crate) fn packed_mask(&self, mask: &Self, value: &Self, scalar: Option<u64>) -> Result<Self> {
        let width = width(self.dtype)?;
        if value.dtype != self.dtype || !matches!(mask.dtype, DType::Bool(BoolStore::U8 | BoolStore::U32)) {
            return Err(Error::InvalidParameter);
        }
        let shape = crate::layout::broadcast(&crate::layout::broadcast(self.layout.shape(), value.layout.shape())?, mask.layout.shape())?;
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        let left = self.layout.expand(shape.clone())?;
        let right = value.layout.expand(shape.clone())?;
        let mask = mask.clone().expand(shape)?;
        Nvidia.record_commands(|| {
            let mask = if mask.layout.is_contiguous() { mask } else { mask.copy()? };
            let narrow = mask.dtype == DType::Bool(BoolStore::U8);
            let remainder = if narrow { mask.layout.offset() % 4 } else { 0 };
            let pointer_offset = if narrow { mask.layout.offset() / 4 } else { mask.layout.offset() };
            let mut p = left.elementwise_parameters(&right, d3os_nvidia::compute::elementwise::Elementwise::Copy, 0, false)?;
            p[2] = width as usize | if narrow { abi::MASK_U8 as usize } else { 0 } | remainder << abi::REMAINDER_SHIFT;
            if let Some(bits) = scalar {
                p[2] |= abi::SCALAR as usize;
                p[3] = bits as u32 as usize;
                p[5] = (bits >> 32) as usize;
                p[7] = 0;
                p[24..32].fill(0);
            }
            let buffer = Nvidia.produce_words(
                width.words(layout.len() as u32) as usize,
                Kernel::PackedMaskWhere,
                &[(&self.buffer, 0), (&value.buffer, 0), (&mask.buffer, pointer_offset)],
                &p,
            )?;
            Ok(Self {
                buffer: Arc::new(buffer),
                layout,
                dtype: self.dtype,
            })
        })
    }
    pub(crate) fn packed_scatter_or(&self, axis: usize, indices: &Self, values: &Self) -> Result<Self> {
        if self.dtype != DType::Bool(BoolStore::U8) || values.dtype != self.dtype || indices.dtype != DType::I32 {
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
        let mut p = layout.gather_parameters(&indices.layout, axis)?;
        let values = if values.layout.is_contiguous() { values.clone() } else { values.copy()? };
        p[0] |= (values.layout.offset() % 4) << abi::REMAINDER_SHIFT;
        let output = self.copy()?;
        let status = Nvidia.produce_words(1, Kernel::Fill, &[], &[1, 0])?;
        Nvidia.dispatch_words(
            Kernel::PackedBoolScatter,
            &[
                (&values.buffer, values.layout.offset() / 4),
                (&indices.buffer, 0),
                (&status, 0),
                (&output.buffer, 0),
            ],
            &p,
        )?;
        let mut invalid = [0];
        Nvidia.read_words_into(&status, 0, &mut invalid)?;
        if invalid[0] != 0 {
            return Err(Error::InvalidParameter);
        }
        Ok(output)
    }
}
