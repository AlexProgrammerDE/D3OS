//! CUDA ELF metadata records needed by the native launch path.
use crate::{Error, Result};

/// Fixed CTA dimensions emitted by cuTile and other CUDA compilers.
/// A caller must honor all three dimensions when this record is present.
pub fn required_threads(info: &[u8]) -> Result<Option<[u32; 3]>> {
    let mut remaining = info;
    let mut dimensions = None;
    while !remaining.is_empty() {
        let header = remaining.get(..4).ok_or(Error::InvalidField)?;
        let payload = if header[0] == 4 {
            u16::from_le_bytes([header[2], header[3]]) as usize
        } else {
            0
        };
        let record = remaining.get(4..4 + payload).ok_or(Error::InvalidField)?;
        if header[1] == 0x10 {
            if header[0] != 4 || record.len() != 12 || dimensions.is_some() {
                return Err(Error::InvalidField);
            }
            let mut value = [0u32; 3];
            let mut total = 1u32;
            for (axis, bytes) in value.iter_mut().zip(record.chunks_exact(4)) {
                *axis = u32::from_le_bytes(bytes.try_into().map_err(|_| Error::InvalidField)?);
                total = total.checked_mul(*axis).filter(|&n| (1..=1024).contains(&n)).ok_or(Error::InvalidField)?;
            }
            dimensions = Some(value);
        }
        remaining = &remaining[4 + payload..];
    }
    Ok(dimensions)
}

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

/// CUDA's reserved shared prefix plus static storage, rounded to 128 bytes.
/// Ada provides up to 100 KiB, including the reserved prefix.
pub fn shared_bytes(static_bytes: u64) -> Result<u32> {
    let bytes = static_bytes.checked_add(0x400 + 127).ok_or(Error::InvalidField)? & !127;
    if bytes > 100 * 1024 {
        return Err(Error::InvalidField);
    }
    Ok(bytes as u32)
}

/// Smallest supported Ada shared-memory partition. QMD encodes its size in
/// 4 KiB units plus one. Keep the established 32 KiB minimum for small kernels.
pub fn shared_partition(bytes: u32) -> Result<u32> {
    let kib = match bytes {
        0..=32768 => 32,
        32769..=65536 => 64,
        65537..=102400 => 100,
        _ => return Err(Error::InvalidField),
    };
    Ok(kib / 4 + 1)
}

/// Reject blocks that exceed Ada's 64K-register budget. Registers are allocated
/// in groups of eight per thread, and partial warps still occupy 32 lanes.
/// This validates a launch limit, rather than predicting kernel occupancy.
pub fn validate_ada_registers(registers: u64, threads: u32) -> Result<()> {
    if registers > 255 || !(1..=1024).contains(&threads) {
        return Err(Error::InvalidField);
    }
    let required = registers.div_ceil(8) * 8 * 32 * u64::from(threads.div_ceil(32));
    if required > 65536 {
        return Err(Error::InvalidField);
    }
    Ok(())
}
