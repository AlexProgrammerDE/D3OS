//! Full-writer coordinate extraction from an inclusive boolean prefix count.
use crate::{Error, Result};

pub const PARAMETERS: usize = 16;

/// Input length, rank, selected count, search steps, eight dimensions, four zeros.
/// Prefix contents affect values, but never output addresses or initialized extent.
pub fn validate(parameters: &[u32]) -> Result<[u64; 2]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let rank = p[1] as usize;
    if !(1..=8).contains(&rank)
        || p[0] == 0
        || p[0] > i32::MAX as u32
        || p[2] == 0
        || p[2] > p[0]
        || p[3] != p[0].ilog2() + 1
        || p[4..12].iter().any(|&dim| dim == 0 || dim > i32::MAX as u32)
        || p[4 + rank..12].iter().any(|&dim| dim != 1)
        || p[12..].iter().any(|&v| v != 0)
    {
        return Err(Error::InvalidField);
    }
    let count = p[4..12].iter().try_fold(1u32, |n, &d| n.checked_mul(d)).ok_or(Error::InvalidField)?;
    let output = p[2].checked_mul(p[1]).filter(|&n| n <= i32::MAX as u32).ok_or(Error::InvalidField)?;
    if count != p[0] {
        return Err(Error::InvalidField);
    }
    Ok([u64::from(count), u64::from(output)])
}
