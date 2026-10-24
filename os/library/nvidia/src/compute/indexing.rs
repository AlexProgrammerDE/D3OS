//! Checked strided gather metadata. Device index values need a runtime bounds check.
use crate::{Error, Result};

pub const PARAMETERS: usize = 32;

/// rank, axis, count, source extent/offset, index extent/offset, axis length,
/// eight index dimensions, eight source strides, and eight index strides.
pub fn validate(p: &[u32]) -> Result<[u64; 4]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let rank = p[0] as usize;
    let axis = p[1] as usize;
    if !(1..=8).contains(&rank)
        || axis >= rank
        || p[2] == 0
        || p[2] > i32::MAX as u32
        || p[7] > i32::MAX as u32
        || p[8..16].iter().any(|&d| d == 0 || d > i32::MAX as u32)
        || p[8 + rank..16].iter().any(|&d| d != 1)
        || p[16 + rank..24].iter().chain(&p[24 + rank..32]).any(|&s| s != 0)
    {
        return Err(Error::InvalidField);
    }
    let count = p[8..16].iter().try_fold(1u32, |n, &d| n.checked_mul(d)).ok_or(Error::InvalidField)?;
    if count != p[2] {
        return Err(Error::InvalidField);
    }
    for (input, extent, offset, strides) in [(0, p[3], p[4], &p[16..24]), (1, p[5], p[6], &p[24..32])] {
        if extent == 0 || extent > i32::MAX as u32 || offset > extent {
            return Err(Error::InvalidField);
        }
        if input == 0 && p[7] == 0 {
            continue;
        }
        let mut low = i64::from(offset);
        let mut high = low;
        for (a, (&d, &s)) in p[8..16].iter().zip(strides).enumerate() {
            let dim = if input == 0 && a == axis { p[7] } else { d };
            let span = i64::from(dim - 1) * i64::from(s as i32);
            low = low.checked_add(span.min(0)).ok_or(Error::InvalidField)?;
            high = high.checked_add(span.max(0)).ok_or(Error::InvalidField)?;
        }
        if low < 0 || high >= i64::from(extent) {
            return Err(Error::InvalidField);
        }
    }
    Ok([u64::from(p[3]), u64::from(p[5]), 1, u64::from(p[2])])
}
