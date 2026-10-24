use crate::{NativeTensor, layout::Layout};
use alloc::{sync::Arc, vec, vec::Vec};
use burn_backend::{BoolStore, DType, Shape, Slice};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise};

impl NativeTensor {
    /// Select exact stored words from `value` where `mask` is true. All three
    /// inputs broadcast, and unselected NaNs cannot affect the selected value.
    pub fn mask_where(&self, mask: &Self, value: &Self) -> Result<Self> {
        if self.dtype != value.dtype || mask.dtype != DType::Bool(BoolStore::U32) {
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
        let parameters = left.elementwise_parameters(&right, Elementwise::Copy, 0, false)?;
        let mask = mask.clone().expand(shape)?;
        Nvidia.record_commands(|| {
            let mask = if mask.layout.is_contiguous() { mask } else { mask.copy()? };
            let buffer = Nvidia.produce_words(
                layout.len(),
                Kernel::MaskWhere,
                &[(&self.buffer, 0), (&value.buffer, 0), (&mask.buffer, mask.layout.offset())],
                &parameters,
            )?;
            Ok(Self {
                buffer: Arc::new(buffer),
                layout,
                dtype: self.dtype,
            })
        })
    }

    /// Copy the logical destination on the GPU, then replace a checked slice.
    /// The original tensor and every shared or overlapping source remain intact.
    pub fn slice_assign(&self, slices: &[Slice], value: &Self) -> Result<Self> {
        if self.dtype != value.dtype {
            return Err(Error::InvalidParameter);
        }
        let target = Layout::contiguous(self.layout.shape().clone())?.slice(slices)?;
        let source = value.layout.expand(target.shape().clone())?;
        if target.is_empty() {
            return Ok(self.clone());
        }
        let parameters = target.assignment_parameters(&source)?;
        Nvidia.record_commands(|| {
            let output = self.copy()?;
            Nvidia.dispatch_words(Kernel::ViewAssign, &[(&output.buffer, 0), (&value.buffer, 0)], &parameters)?;
            Ok(output)
        })
    }

    /// Concatenate tensors through resident strided copies. Empty operands do
    /// not dispatch writes. Every source is validated before output allocation.
    pub fn cat(tensors: &[Self], axis: usize) -> Result<Self> {
        let first = tensors.first().ok_or(Error::InvalidParameter)?;
        let mut shape = first.layout.shape().to_vec();
        let length = shape.get_mut(axis).ok_or(Error::InvalidParameter)?;
        *length = 0;
        for tensor in tensors {
            if tensor.dtype != first.dtype {
                return Err(Error::InvalidParameter);
            }
            if tensor.layout.shape().num_dims() != shape.len() || tensor.layout.shape().iter().zip(&shape).enumerate().any(|(a, (&s, &d))| a != axis && s != d)
            {
                return Err(Error::ShapeMismatch);
            }
            shape[axis] = shape[axis].checked_add(tensor.layout.shape()[axis]).ok_or(Error::SizeOverflow)?;
        }
        let layout = Layout::contiguous(Shape::from(shape))?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: first.buffer.clone(),
                layout,
                dtype: first.dtype,
            });
        }
        let mut writes = Vec::with_capacity(tensors.len());
        let mut start = 0;
        for tensor in tensors {
            let end = start + tensor.layout.shape()[axis];
            if !tensor.layout.is_empty() {
                let mut slices = vec![Slice::full(); axis + 1];
                slices[axis] = Slice::new(start as isize, Some(end as isize), 1);
                writes.push((tensor, layout.slice(&slices)?.assignment_parameters(&tensor.layout)?));
            }
            start = end;
        }
        Nvidia.record_commands(|| {
            // This partial-writer path starts from initialized storage. No
            // uninitialized allocation is exposed if any later dispatch fails.
            let buffer = Nvidia.allocate_words(layout.len())?;
            for (source, parameters) in writes {
                Nvidia.dispatch_words(Kernel::ViewAssign, &[(&buffer, 0), (&source.buffer, 0)], &parameters)?;
            }
            Ok(Self {
                buffer: Arc::new(buffer),
                layout,
                dtype: first.dtype,
            })
        })
    }

    /// Add explicit padding to every dimension with an exact raw fill word.
    pub fn pad(&self, padding: &[(usize, usize)], word: u32) -> Result<Self> {
        if padding.len() != self.layout.shape().num_dims() || (matches!(self.dtype, DType::Bool(_)) && word > 1) {
            return Err(Error::InvalidParameter);
        }
        let dimensions: Vec<_> = self
            .layout
            .shape()
            .iter()
            .zip(padding)
            .map(|(&d, &(before, after))| d.checked_add(before).and_then(|n| n.checked_add(after)).ok_or(Error::SizeOverflow))
            .collect::<Result<_>>()?;
        let layout = Layout::contiguous(Shape::from(dimensions))?;
        if layout.is_empty() {
            return Ok(Self {
                buffer: self.buffer.clone(),
                layout,
                dtype: self.dtype,
            });
        }
        let parameters = if self.layout.is_empty() {
            None
        } else {
            let slices: Vec<_> = padding
                .iter()
                .zip(self.layout.shape().iter())
                .map(|(&(before, _), &d)| Slice::new(before as isize, Some((before + d) as isize), 1))
                .collect();
            Some(layout.slice(&slices)?.assignment_parameters(&self.layout)?)
        };
        Nvidia.record_commands(|| {
            let output = Self::full(layout.shape().clone(), self.dtype, word)?;
            if let Some(parameters) = parameters {
                Nvidia.dispatch_words(Kernel::ViewAssign, &[(&output.buffer, 0), (&self.buffer, 0)], &parameters)?;
            }
            Ok(output)
        })
    }
}
