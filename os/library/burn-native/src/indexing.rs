//! ND indexing composes exact device indices with resident row selection.
use crate::{
    NativeTensor,
    layout::{Layout, nd_shape},
};
use burn_backend::{DType, Shape, Slice, tensor::IndexingUpdateOp};
use d3os_compute::{Backend as _, Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::words::WordOp;

fn dense_reshape(tensor: &NativeTensor, shape: Shape) -> Result<NativeTensor> {
    if tensor.layout.is_contiguous() {
        tensor.clone().reshape(shape)
    } else {
        tensor.copy()?.reshape(shape)
    }
}

impl NativeTensor {
    // Coordinate validation must precede the final linear bounds check. An
    // invalid tuple such as [1, -1] can otherwise alias a valid preceding row.
    fn nd_linear_indices(&self, indices: &Self, width: usize) -> Result<Self> {
        let tuples = indices.layout.len() / width;
        let indices = dense_reshape(indices, Shape::new([tuples, width]))?;
        let mut linear: Option<Self> = None;
        let mut invalid: Option<Self> = None;
        for axis in 0..width {
            let column = indices
                .clone()
                .slice(&[Slice::full(), Slice::new(axis as isize, Some((axis + 1) as isize), 1)])?;
            let column = dense_reshape(&column, Shape::new([tuples]))?;
            let negative = column.word_scalar(WordOp::Less, 0)?;
            let high = column.word_scalar(WordOp::GreaterEqual, self.layout.shape()[axis] as u32)?;
            let bad = negative.word_binary(&high, WordOp::BoolOr)?;
            invalid = Some(match invalid {
                Some(mask) => mask.word_binary(&bad, WordOp::BoolOr)?,
                None => bad,
            });
            linear = Some(match linear {
                Some(prefix) => prefix
                    .word_scalar(WordOp::Multiply, self.layout.shape()[axis] as u32)?
                    .word_binary(&column, WordOp::Add)?,
                None => column,
            });
        }
        linear.ok_or(Error::InvalidShape)?.mask_fill_i32(&invalid.ok_or(Error::InvalidShape)?, -1)
    }

    /// Gather slices addressed by leading coordinate tuples. Shape metadata
    /// stays on the CPU; coordinates and tensor payloads stay on the GPU.
    pub fn gather_nd(&self, indices: &Self) -> Result<Self> {
        if !matches!(self.dtype, DType::F32 | DType::I32) || indices.dtype != DType::I32 {
            return Err(Error::InvalidParameter);
        }
        let shape = nd_shape(self.layout.shape(), indices.layout.shape())?;
        let layout = Layout::contiguous(shape.clone())?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        if self.layout.is_empty() {
            return Err(Error::InvalidParameter);
        }
        let width = *indices.layout.shape().last().ok_or(Error::InvalidShape)?;
        let rows = self.layout.shape()[..width].iter().product();
        let slices = self.layout.shape()[width..].iter().product();
        Nvidia.with_commands(|| {
            let data = dense_reshape(self, Shape::new([rows, slices]))?;
            let linear = self.nd_linear_indices(indices, width)?;
            data.select(0, &linear)?.reshape(shape)
        })
    }

    /// Update ND slices through a private dense destination and the checked
    /// scatter operation. Non-additive duplicates have no Burn API guarantee.
    pub fn scatter_nd(&self, indices: &Self, values: &Self, update: IndexingUpdateOp) -> Result<Self> {
        if !matches!(self.dtype, DType::F32 | DType::I32) || indices.dtype != DType::I32 || values.dtype != self.dtype {
            return Err(Error::InvalidParameter);
        }
        let shape = nd_shape(self.layout.shape(), indices.layout.shape())?;
        if values.layout.shape() != &shape {
            return Err(Error::ShapeMismatch);
        }
        if values.layout.is_empty() {
            return Ok(self.clone());
        }
        if self.layout.is_empty() {
            return Err(Error::InvalidParameter);
        }
        let width = *indices.layout.shape().last().ok_or(Error::InvalidShape)?;
        let rows = self.layout.shape()[..width].iter().product();
        let slices = self.layout.shape()[width..].iter().product();
        let tuples = indices.layout.len() / width;
        Nvidia.with_commands(|| {
            let data = dense_reshape(self, Shape::new([rows, slices]))?;
            let values = dense_reshape(values, Shape::new([tuples, slices]))?;
            let linear = self.nd_linear_indices(indices, width)?;
            data.select_update(0, &linear, &values, update)?.reshape(self.layout.shape().clone())
        })
    }
}
