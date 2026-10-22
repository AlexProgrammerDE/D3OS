//! CUDA ELF metadata records needed by the native launch path.
use crate::{Error, Result};

/// Maximum EIATTR_MIN_STACK_SIZE across the module. A common allocation can
/// serve every entry point; symbol-specific selection is not required.
pub fn stack_bytes(info: &[u8]) -> Result<u32> {
    let mut remaining = info;
    let mut maximum = 0;
    while !remaining.is_empty() {
        let header = remaining.get(..4).ok_or(Error::InvalidField)?;
        let length = u16::from_le_bytes([header[2], header[3]]) as usize;
        let payload = if header[0] == 4 { length } else { 0 };
        let record = remaining.get(4..4 + payload).ok_or(Error::InvalidField)?;
        if header[1] == 0x12 {
            if header[0] != 4 || record.is_empty() || !record.len().is_multiple_of(8) {
                return Err(Error::InvalidField);
            }
            for entry in record.chunks_exact(8) {
                maximum = maximum.max(u32::from_le_bytes(entry[4..8].try_into().map_err(|_| Error::InvalidField)?));
            }
        }
        remaining = &remaining[4 + payload..];
    }
    Ok(maximum)
}
