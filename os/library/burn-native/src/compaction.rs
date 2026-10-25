//! GPU compaction with a four-byte dynamic-shape readback.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{BoolStore, DType, Shape};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, argwhere, scan::Scan};

impl NativeTensor {
    pub fn argwhere(&self) -> Result<Self> {
        if !matches!(self.dtype, DType::Bool(BoolStore::U8 | BoolStore::U32)) {
            return Err(Error::InvalidParameter);
        }
        let rank = self.layout.shape().num_dims();
        let empty = || -> Result<Self> {
            Ok(Self {
                buffer: self.buffer.clone(),
                layout: Layout::contiguous(Shape::new([0, rank]))?,
                dtype: DType::I32,
            })
        };
        let count = self.layout.len();
        if count == 0 {
            return empty();
        }
        Nvidia.record_commands(|| {
            let mask = self.cast(DType::I32)?;
            let mask = if mask.layout.is_contiguous() { mask } else { mask.copy()? };
            let prefix = mask.reshape(Shape::new([count]))?.scan(0, Scan::IntSum)?;
            let mut selected = [0];
            Nvidia.read_words_into(&prefix.buffer, count - 1, &mut selected)?;
            if selected[0] as usize > count {
                return Err(Error::InvalidParameter);
            }
            if selected[0] == 0 {
                return empty();
            }
            if rank == 0 {
                return Ok(Self {
                    buffer: self.buffer.clone(),
                    layout: Layout::contiguous(Shape::new([selected[0] as usize, 0]))?,
                    dtype: DType::I32,
                });
            }
            let layout = Layout::contiguous(Shape::new([selected[0] as usize, rank]))?;
            let mut parameters = [0u32; argwhere::PARAMETERS];
            parameters[..4].copy_from_slice(&[count as u32, rank as u32, selected[0], (count as u32).ilog2() + 1]);
            parameters[4..12].fill(1);
            for (axis, &dim) in self.layout.shape().iter().enumerate() {
                parameters[4 + axis] = dim as u32;
            }
            argwhere::validate(&parameters).map_err(|_| Error::InvalidParameter)?;
            let buffer = Nvidia.produce_words(layout.len(), Kernel::ArgWhere, &[(&prefix.buffer, 0)], &parameters.map(|v| v as usize))?;
            Ok(Self {
                buffer: Arc::new(buffer),
                layout,
                dtype: DType::I32,
            })
        })
    }
}
