//! Version 1 tensor archives contain named, little-endian, contiguous f32 tensors.
//! Parsing checks the complete file before a backend receives any allocation.
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use d3os_compute::{Backend, Tensor, element_count};

use crate::{Error, Result};

pub const MAGIC: &[u8; 8] = b"D3TENSR\0";
pub const VERSION: u32 = 1;

pub struct Record<'a> {
    shape: Vec<usize>,
    bytes: &'a [u8],
}

impl Record<'_> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn load<B: Backend>(&self, backend: &B) -> Result<Tensor<B>> {
        Ok(Tensor::from_vec(backend, &self.shape, self.to_vec()?)?)
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.bytes.len() / 4)
            .map_err(|_| d3os_compute::Error::AllocationFailed)?;
        for bytes in self.bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(bytes.try_into().map_err(|_| Error::InvalidArchive)?);
            if !value.is_finite() {
                return Err(d3os_compute::Error::NonFinite.into());
            }
            values.push(value);
        }
        Ok(values)
    }
}

pub struct Archive<'a> {
    records: BTreeMap<&'a str, Record<'a>>,
}

impl<'a> Archive<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let mut cursor = Cursor(bytes);
        if cursor.take(8)? != MAGIC || cursor.u32()? != VERSION {
            return Err(Error::InvalidArchive);
        }
        let count = cursor.u32()?;
        if count == 0 || count > 64 {
            return Err(Error::InvalidArchive);
        }
        let mut records = BTreeMap::new();
        for _ in 0..count {
            let name_len = cursor.u16()? as usize;
            let rank = cursor.u16()? as usize;
            let elements = usize::try_from(cursor.u64()?).map_err(|_| Error::InvalidArchive)?;
            if !(1..=256).contains(&name_len) || !(1..=8).contains(&rank) {
                return Err(Error::InvalidArchive);
            }
            let name = core::str::from_utf8(cursor.take(name_len)?).map_err(|_| Error::InvalidArchive)?;
            if name.as_bytes().contains(&0) {
                return Err(Error::InvalidArchive);
            }
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(usize::try_from(cursor.u64()?).map_err(|_| Error::InvalidArchive)?);
            }
            if element_count(&shape).map_err(|_| Error::InvalidArchive)? != elements {
                return Err(Error::InvalidArchive);
            }
            let bytes = cursor.take(elements.checked_mul(4).ok_or(Error::InvalidArchive)?)?;
            if records.insert(name, Record { shape, bytes }).is_some() {
                return Err(Error::InvalidArchive);
            }
        }
        if !cursor.0.is_empty() {
            return Err(Error::InvalidArchive);
        }
        Ok(Self { records })
    }

    pub fn take(&mut self, name: &str) -> Result<Record<'a>> {
        self.records.remove(name).ok_or_else(|| Error::MissingTensor(name.into()))
    }

    pub fn finish(self) -> Result<()> {
        if !self.records.is_empty() {
            return Err(Error::UnexpectedTensor);
        }
        Ok(())
    }
}

struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(count).ok_or(Error::InvalidArchive)?;
        self.0 = tail;
        Ok(head)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().map_err(|_| Error::InvalidArchive)?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| Error::InvalidArchive)?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| Error::InvalidArchive)?))
    }
}
