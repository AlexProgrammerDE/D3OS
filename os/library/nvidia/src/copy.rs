//! Linear virtual-address copies for NVIDIA's Ampere DMA COPY B engine.
//!
//! Method fields follow RM 570's published class/clc7b5.h and clc56f.h.

use crate::{Error, Result, abi::AMPERE_DMA_COPY_B};

const fn method(subchannel: u32, offset: u32, words: u32) -> u32 {
    (2 << 28) | (words << 16) | (subchannel << 13) | (offset >> 2)
}

/// Copy disjoint virtual ranges, flush system memory, and report completion.
/// `report` contains a copy-engine word and a separate FIFO tail at `report + 8`.
/// The caller must own the command and report storage until both become one.
pub fn commands(destination: u64, source: u64, bytes: u32, report: u64) -> Result<[u32; 21]> {
    let end = |address: u64, bytes: u64| address.checked_add(bytes).filter(|&end| end <= 1 << 49);
    let source_end = end(source, u64::from(bytes)).ok_or(Error::InvalidField)?;
    let destination_end = end(destination, u64::from(bytes)).ok_or(Error::InvalidField)?;
    if bytes == 0
        || source == 0
        || destination == 0
        || report == 0
        || bytes > 1 << 31
        || report & 7 != 0
        || end(report, 16).is_none()
        || (source < destination_end && destination < source_end)
        || (report < source_end && source < report + 16)
        || (report < destination_end && destination < report + 16)
    {
        return Err(Error::InvalidField);
    }
    Ok([
        method(4, 0, 1),
        AMPERE_DMA_COPY_B,
        method(4, 0x400, 4), // OFFSET_IN/OUT_UPPER/LOWER
        (source >> 32) as u32,
        source as u32,
        (destination >> 32) as u32,
        destination as u32,
        method(4, 0x418, 1), // LINE_LENGTH_IN
        bytes,
        method(4, 0x240, 3), // SET_SEMAPHORE_A/B/PAYLOAD
        (report >> 32) as u32,
        report as u32,
        1,
        method(4, 0x300, 1), // LAUNCH_DMA
        // Non-pipelined, system flush, one-word release, pitched layouts.
        2 | (1 << 2) | (1 << 3) | (1 << 7) | (1 << 8),
        method(0, 0x5c, 5), // SEM_ADDR_LO/HI/PAYLOAD_LO/HI/EXECUTE
        (report + 8) as u32,
        ((report + 8) >> 32) as u32,
        1,
        0,
        1 | (1 << 24) | (1 << 20), // 64-bit release, wait for idle
    ])
}
