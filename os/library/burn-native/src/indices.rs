//! Validate resident integer coordinates before narrowing their representation.
use crate::NativeTensor;
use burn_backend::{BoolStore, DType};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{predicate::Predicate, words::WordOp};

impl NativeTensor {
    pub(crate) fn checked_indices(&self) -> Result<Self> {
        if !self.dtype.is_int() && !self.dtype.is_uint() {
            return Err(Error::InvalidParameter);
        }
        if self.dtype == DType::I32 {
            return Ok(self.clone());
        }
        if self.layout.is_empty() {
            return self.cast(DType::I32);
        }
        Nvidia.record_commands(|| {
            let boolean = DType::Bool(BoolStore::U32);
            let mut invalid = if self.dtype.is_int() {
                Some(self.compare_scalar(0i32.into(), Predicate::Less, boolean)?)
            } else {
                None
            };
            // Narrow dtypes cannot represent this bound. Converting the scalar
            // to them would wrap and reject otherwise valid coordinates.
            if matches!(self.dtype, DType::I64 | DType::U64 | DType::U32) {
                let high = self.compare_scalar(i32::MAX.into(), Predicate::Greater, boolean)?;
                invalid = Some(match invalid {
                    Some(negative) => negative.word_binary(&high, WordOp::BoolOr)?,
                    None => high,
                });
            }
            let indices = self.cast(DType::I32)?;
            match invalid {
                // Keep invalid values resident until the indexing kernel
                // reports its isolated bounds status, without narrowing aliases.
                Some(mask) => indices.mask_fill_i32(&mask, -1),
                None => Ok(indices),
            }
        })
    }
}
