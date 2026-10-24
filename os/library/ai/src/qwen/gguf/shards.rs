//! Checked, independently seekable firmware parts of one model file.
use super::{Error, ReadAt, Result};
use alloc::vec::Vec;

#[derive(Clone, Debug)]
pub struct Index {
    length: u64,
    part_bytes: u64,
    parts: usize,
    sha256: [u8; 32],
}

impl Index {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 60 || &bytes[..8] != b"D3MODEL\0" {
            return Err(Error::InvalidArchive);
        }
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let part_bytes = u64::from(word(12));
        let length = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let parts = word(24) as usize;
        if word(8) != 1
            || !(1 << 20..=1 << 30).contains(&part_bytes)
            || !part_bytes.is_power_of_two()
            || length == 0
            || !(1..=64).contains(&parts)
            || length.div_ceil(part_bytes) != parts as u64
        {
            return Err(Error::InvalidArchive);
        }
        Ok(Self {
            length,
            part_bytes,
            parts,
            sha256: bytes[28..60].try_into().unwrap(),
        })
    }

    pub fn parts(&self) -> usize {
        self.parts
    }

    /// Identity recorded by the host preparer; the reader does not hash data.
    pub fn source_sha256(&self) -> &[u8; 32] {
        &self.sha256
    }

    pub fn part_len(&self, part: usize) -> Option<u64> {
        (part < self.parts).then(|| self.part_bytes.min(self.length - part as u64 * self.part_bytes))
    }
}

pub struct Sharded<R> {
    index: Index,
    sources: Vec<R>,
}

impl<R: ReadAt> Sharded<R> {
    pub fn new(index: Index, sources: Vec<R>) -> Result<Self> {
        if sources.len() != index.parts || sources.iter().enumerate().any(|(part, source)| Some(source.len()) != index.part_len(part)) {
            return Err(Error::InvalidArchive);
        }
        Ok(Self { index, sources })
    }

    pub fn index(&self) -> &Index {
        &self.index
    }
}

impl<R: ReadAt> ReadAt for Sharded<R> {
    fn len(&self) -> u64 {
        self.index.length
    }

    fn read_at(&mut self, mut offset: u64, mut output: &mut [u8]) -> Result<()> {
        if offset.checked_add(output.len() as u64).is_none_or(|end| end > self.len()) {
            return Err(Error::SourceRead);
        }
        while !output.is_empty() {
            let part = (offset / self.index.part_bytes) as usize;
            let position = offset % self.index.part_bytes;
            let count = output.len().min((self.index.part_bytes - position) as usize);
            self.sources[part].read_at(position, &mut output[..count])?;
            output = &mut output[count..];
            offset += count as u64;
        }
        Ok(())
    }
}
