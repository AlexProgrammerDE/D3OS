//! Packed indexed assignment uses a private destination and ordered row owners.
use crate::{NativeTensor, layout::Layout, storage::width};
use burn_backend::{BoolStore, DType};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::Kernel;

impl NativeTensor {
    pub(crate) fn packed_scatter_assign(&self, axis: usize, indices: &Self, values: &Self) -> Result<Self> {
        if values.dtype != self.dtype || indices.dtype != DType::I32 || matches!(self.dtype, DType::Bool(BoolStore::U8 | BoolStore::U32)) {
            return Err(Error::InvalidParameter);
        }
        let width = width(self.dtype)?;
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
        Nvidia.record_commands(|| {
            // A nonzero logical offset need not start at a physical word.
            // Snapshot into offset-zero storage before binding the dense ABI.
            let values = if values.layout.is_contiguous() && values.layout.offset() == 0 {
                values.clone()
            } else {
                values.copy()?
            };
            let output = self.copy()?;
            let layout = Layout::contiguous(self.layout.shape().clone())?;
            let mut parameters = layout.gather_parameters(&indices.layout, axis)?;
            parameters[0] |= (width as usize) << 8;
            let status = Nvidia.produce_words(1, Kernel::Fill, &[], &[1, 0])?;
            Nvidia.dispatch_words(
                Kernel::PackedScatterAssign,
                &[(&values.buffer, 0), (&indices.buffer, 0), (&status, 0), (&output.buffer, 0)],
                &parameters,
            )?;
            let mut invalid = [0u32];
            Nvidia.read_words_into(&status, 0, &mut invalid)?;
            if invalid[0] != 0 {
                return Err(Error::InvalidParameter);
            }
            Ok(output)
        })
    }
}
