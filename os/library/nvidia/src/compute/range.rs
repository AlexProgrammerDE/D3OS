//! Dense I32 indices generated on the GPU without host tensor data.
use crate::{Error, Result};

pub const PARAMETERS: usize = 3;

/// Parameters are element count, signed first value, and unsigned positive step.
/// Prove the last value fits I32 before exposing a full-writer allocation.
pub fn validate(parameters: &[u32]) -> Result<u64> {
    let &[count, first, step] = parameters else { return Err(Error::InvalidField) };
    if count == 0 || count > i32::MAX as u32 || step == 0 {
        return Err(Error::InvalidField);
    }
    let last = i64::from(first as i32) + i64::from(count - 1) * i64::from(step);
    if last > i64::from(i32::MAX) {
        return Err(Error::InvalidField);
    }
    Ok(u64::from(count))
}
