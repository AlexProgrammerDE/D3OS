//! Immutable texture snapshots with aligned offsets for incremental GPU upload.
use crate::{Error, Result};
use alloc::vec::Vec;
use hashbrown::HashMap;

struct Entry {
    offset: u32,
    length: usize,
    validated: Option<(u64, usize)>,
}

pub struct TextureAtlas {
    data: Vec<u8>,
    entries: HashMap<usize, Entry>,
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
        self.snapshot_inner(identity, None, bytes)
    }

    /// Avoid comparing bytes already validated in this generation. The caller
    /// must change the generation before changing or reusing source storage.
    /// Generations must not repeat while the atlas retains their entries.
    pub fn snapshot_versioned(&mut self, identity: usize, generation: u64, bytes: &[u8]) -> Result<u32> {
        self.snapshot_inner(identity, Some(generation), bytes)
    }

    fn snapshot_inner(&mut self, identity: usize, generation: Option<u64>, bytes: &[u8]) -> Result<u32> {
        if bytes.is_empty() {
            return Err(Error::InvalidParameter);
        }
        if let Some(entry) = self.entries.get_mut(&identity)
            && bytes.len() <= entry.length
        {
            if let Some((version, length)) = entry.validated
                && generation == Some(version)
                && bytes.len() <= length
            {
                return Ok(entry.offset);
            }
            if &self.data[entry.offset as usize..entry.offset as usize + bytes.len()] == bytes {
                // A matching prefix cannot validate an unseen suffix after
                // source storage changes generation.
                entry.validated = generation.map(|version| (version, bytes.len()));
                return Ok(entry.offset);
            }
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
        self.entries.insert(
            identity,
            Entry {
                offset: offset as u32,
                length: bytes.len(),
                validated: generation.map(|version| (version, bytes.len())),
            },
        );
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
    fn new_generations_validate_only_the_requested_prefix() {
        let mut atlas = TextureAtlas::new(32).unwrap();
        assert_eq!(atlas.snapshot_versioned(1, 0, &[1, 2, 3, 4, 5]).unwrap(), 0);
        assert_eq!(atlas.snapshot_versioned(1, 1, &[1, 2]).unwrap(), 0);
        assert_eq!(atlas.snapshot_versioned(1, 1, &[1, 2, 9, 4, 5]).unwrap(), 8);
        assert_eq!(atlas.snapshot_versioned(1, 1, &[1, 2]).unwrap(), 8);
        assert_eq!(atlas.snapshot_versioned(1, 2, &[1, 2, 9, 4, 5]).unwrap(), 8);
        assert_eq!(&atlas.bytes()[..8], &[1, 2, 3, 4, 5, 0, 0, 0]);
        assert_eq!(atlas.snapshot(1, &[1, 2, 8]).unwrap(), 16);
        assert_eq!(atlas.snapshot_versioned(1, 3, &[1, 2, 8]).unwrap(), 16);
        assert_eq!(atlas.snapshot_versioned(1, 4, &[1]).unwrap(), 16);
        assert_eq!(atlas.snapshot_versioned(1, 4, &[1, 7, 8]).unwrap(), 20);
        assert_eq!(atlas.bytes().len(), 24);
    }

    #[test]
    fn full_versioned_atlas_preserves_failed_inserts_and_revalidates_after_clear() {
        let mut atlas = TextureAtlas::new(8).unwrap();
        assert_eq!(atlas.snapshot_versioned(1, 7, &[1, 2, 3, 4, 5]).unwrap(), 0);
        assert_eq!(atlas.snapshot_versioned(1, 7, &[1, 2, 3]).unwrap(), 0);
        assert_eq!(atlas.snapshot_versioned(1, 8, &[1, 2, 3]).unwrap(), 0);
        assert_eq!(atlas.snapshot_versioned(1, 8, &[1, 2, 3, 9, 5]), Err(Error::AllocationFailed));
        assert_eq!(atlas.bytes(), &[1, 2, 3, 4, 5, 0, 0, 0]);
        atlas.clear();
        assert_eq!(atlas.snapshot_versioned(1, 8, &[1, 2, 3, 9, 5]).unwrap(), 0);
        assert_eq!(atlas.bytes(), &[1, 2, 3, 9, 5, 0, 0, 0]);
        assert_eq!(atlas.snapshot_versioned(1, 8, &[]), Err(Error::InvalidParameter));
    }

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
