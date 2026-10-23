//! QMD dependencies use GPU virtual addresses, in 256-byte units.
use crate::{Error, Record, Result, abi::QmdFields as Q};

/// Schedule `next` after this grid completes, without an intermediate CPU wait.
/// The final grid retains its release semaphore; intermediate grids do not.
pub fn chain(qmd: &mut Record<256>, next: u64) -> Result<()> {
    if next == 0 || next & 255 != 0 || next >> 40 != 0 {
        return Err(Error::InvalidField);
    }
    qmd.set_bits(Q::DEPENDENT_QMD0_POINTER, next >> 8)?;
    qmd.set_bits(Q::DEPENDENT_QMD0_ACTION, 1)?; // QMD_INVALIDATE_COPY_SCHEDULE
    qmd.set_bits(Q::DEPENDENT_QMD0_PREFETCH, 1)?;
    qmd.set_bits(Q::DEPENDENT_QMD0_ENABLE, 1)?;
    qmd.set_bits(Q::RELEASE0_ENABLE, 0)
}
