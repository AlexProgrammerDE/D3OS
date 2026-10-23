//! Immutable texture snapshots with aligned offsets for incremental GPU upload.
use crate::{Error, Result};
use alloc::vec::Vec;
use hashbrown::HashMap;

pub struct TextureAtlas {
    data: Vec<u8>,
    entries: HashMap<usize, (u32, usize)>,
    capacity: usize,
}

impl TextureAtlas {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || !capacity.is_multiple_of(4) || capacity > u32::MAX as usize {
            return Err(Error::InvalidParameter);
        }
        Ok(Self {
            data: Vec::new(),
            entries: HashMap::new(),
            capacity,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn remaining(&self) -> usize {
        self.capacity - self.data.len()
    }

    /// Reuse matching contents at the same identity, including a prefix of a
    /// longer snapshot. Changed bytes append a snapshot so pending draws retain
    /// their previous contents. The caller supplies only valid readable bytes.
    pub fn snapshot(&mut self, identity: usize, bytes: &[u8]) -> Result<u32> {
        if bytes.is_empty() {
            return Err(Error::InvalidParameter);
        }
        if let Some(&(offset, length)) = self.entries.get(&identity)
            && bytes.len() <= length
            && &self.data[offset as usize..offset as usize + bytes.len()] == bytes
        {
            return Ok(offset);
        }
        let padded = bytes.len().checked_add(3).ok_or(Error::SizeOverflow)? & !3;
        if padded > self.remaining() {
            return Err(Error::AllocationFailed);
        }
        self.entries.try_reserve(1).map_err(|_| Error::AllocationFailed)?;
        self.data.try_reserve(padded).map_err(|_| Error::AllocationFailed)?;
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        self.data.resize(offset + padded, 0);
        self.entries.insert(identity, (offset as u32, bytes.len()));
        Ok(offset as u32)
    }

    /// The caller must complete pending draws before discarding their snapshots.
    /// CPU storage capacity remains available for the next generation.
    pub fn clear(&mut self) {
        self.data.clear();
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reused_identity_preserves_previous_contents_and_zero_padding() {
        let mut atlas = TextureAtlas::new(32).unwrap();
        assert_eq!(atlas.snapshot(9, &[1, 2, 3]).unwrap(), 0);
        assert_eq!(atlas.snapshot(9, &[1, 2, 3]).unwrap(), 0);
        assert_eq!(atlas.snapshot(9, &[8, 2, 3]).unwrap(), 4);
        assert_eq!(atlas.snapshot(9, &[1, 2]).unwrap(), 8);
        assert_eq!(atlas.snapshot(10, &[1, 2, 3]).unwrap(), 12);
        assert_eq!(atlas.bytes(), &[1, 2, 3, 0, 8, 2, 3, 0, 1, 2, 0, 0, 1, 2, 3, 0]);
        assert_eq!(atlas.remaining(), 16);
    }

    #[test]
    fn full_atlas_reuses_existing_entries_without_mutating_failed_inserts() {
        let mut atlas = TextureAtlas::new(8).unwrap();
        atlas.snapshot(1, &[1; 5]).unwrap();
        assert_eq!(atlas.remaining(), 0);
        assert_eq!(atlas.snapshot(1, &[1; 5]).unwrap(), 0);
        assert_eq!(atlas.snapshot(1, &[2; 5]), Err(Error::AllocationFailed));
        assert_eq!(atlas.snapshot(1, &[]), Err(Error::InvalidParameter));
        assert_eq!(atlas.bytes(), &[1, 1, 1, 1, 1, 0, 0, 0]);
        atlas.clear();
        assert_eq!(atlas.remaining(), 8);
        assert_eq!(atlas.snapshot(1, &[2; 3]).unwrap(), 0);
        assert_eq!(atlas.bytes(), &[2, 2, 2, 0]);
        for capacity in [0, 3, usize::MAX] {
            assert!(TextureAtlas::new(capacity).is_err());
        }
    }

    #[test]
    fn shorter_requests_reuse_a_valid_prefix_without_reading_beyond_the_slice() {
        let mut atlas = TextureAtlas::new(32).unwrap();
        assert_eq!(atlas.snapshot(1, &[1, 2, 3, 4, 5]).unwrap(), 0);
        assert_eq!(atlas.snapshot(1, &[1, 2]).unwrap(), 0);
        assert_eq!(atlas.snapshot(1, &[1]).unwrap(), 0);
        assert_eq!(atlas.remaining(), 24);
        assert_eq!(atlas.snapshot(1, &[1, 9]).unwrap(), 8);
        assert_eq!(atlas.snapshot(1, &[1, 9, 3]).unwrap(), 12);
        assert_eq!(&atlas.bytes()[..8], &[1, 2, 3, 4, 5, 0, 0, 0]);
        assert_eq!(atlas.snapshot(1, &[1, 9]).unwrap(), 12);
    }
}
