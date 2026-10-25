//! Adam's resident parameter and moment update.
use crate::{Error, Result};

pub const PARAMETERS: usize = 9;
pub const STATE: u32 = 1;
pub const AMSGRAD: u32 = 2;
pub const MAXIMUM: u32 = 4;

/// Count, state flags, and seven FP32 coefficients. The output contains the
/// parameter, first moment, second moment, and optional AMSGrad maximum.
pub fn validate(p: &[u32]) -> Result<[u64; 6]> {
    let p: &[u32; PARAMETERS] = p.try_into().map_err(|_| Error::InvalidField)?;
    let [count, flags, ..] = *p;
    if count == 0 || flags & !(STATE | AMSGRAD | MAXIMUM) != 0 || (flags & MAXIMUM != 0 && flags & (STATE | AMSGRAD) != (STATE | AMSGRAD)) {
        return Err(Error::InvalidField);
    }
    let count = u64::from(count);
    let outputs = count * if flags & AMSGRAD == 0 { 3 } else { 4 };
    if outputs > u64::from(u32::MAX) {
        return Err(Error::InvalidField);
    }
    let moment = if flags & STATE == 0 { 0 } else { count };
    Ok([count, count, moment, moment, if flags & MAXIMUM == 0 { 0 } else { count }, outputs])
}
