//! Checked masked selection and nonoverlapping strided writes.
use super::elementwise::{self, PARAMETERS};
use crate::{Error, Result};

fn views(parameters: &[u32]) -> Result<[u64; 3]> {
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    if p[2] != 0 || p[3] != 0 {
        return Err(Error::InvalidField);
    }
    elementwise::validate_layout(p)
}

/// Two signed source views, a dense mask, and a dense full-writer output.
pub fn validate_mask(parameters: &[u32]) -> Result<[u64; 4]> {
    let [a, b, count] = views(parameters)?;
    Ok([a, b, count, count])
}

/// The first view is writable and must map each logical element to a distinct
/// address. Sorted absolute strides prove this for dense, permuted, reversed,
/// stepped, and padded layouts. Overlapping windows and broadcast writes fail.
/// The proof deliberately rejects layouts whose injectivity it cannot establish.
pub fn validate_assign(parameters: &[u32]) -> Result<[u64; 2]> {
    let [destination, source, _] = views(parameters)?;
    let p: &[u32; PARAMETERS] = parameters.try_into().map_err(|_| Error::InvalidField)?;
    let mut axes = [(0u64, 0u32); 8];
    let mut count = 0;
    for axis in 0..p[1] as usize {
        if p[8 + axis] > 1 {
            axes[count] = (i64::from(p[16 + axis] as i32).unsigned_abs(), p[8 + axis]);
            count += 1;
        }
    }
    axes[..count].sort_unstable_by_key(|&(stride, _)| stride);
    let mut span = 0u64;
    for &(stride, dimension) in &axes[..count] {
        if stride <= span {
            return Err(Error::InvalidField);
        }
        span = span.checked_add(stride * u64::from(dimension - 1)).ok_or(Error::InvalidField)?;
    }
    Ok([destination, source])
}
