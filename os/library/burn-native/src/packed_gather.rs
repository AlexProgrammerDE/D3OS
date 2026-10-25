//! Gather packed values directly without changing their stored representation.
use crate::{NativeTensor, layout::Layout, storage::width};
use alloc::sync::Arc;
use burn_backend::DType;
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::Kernel;

impl NativeTensor {
    pub(crate) fn packed_gather(&self, axis: usize, indices: &Self) -> Result<Self> {
        let width = width(self.dtype)?;
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
        if self.layout.shape()[axis] == 0 {
            return Err(Error::InvalidParameter);
        }
        let mut parameters = self.layout.gather_parameters(&indices.layout, axis)?;
        parameters[0] |= (width as usize) << 8;
        let status = Nvidia.produce_words(1, Kernel::Fill, &[], &[1, 0])?;
        let buffer = Nvidia.produce_words(
            width.words(layout.len() as u32) as usize,
            Kernel::PackedGather,
            &[(&self.buffer, 0), (&indices.buffer, 0), (&status, 0)],
            &parameters,
        )?;
        let mut invalid = [0u32];
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
}
