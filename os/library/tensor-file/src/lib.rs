//! Named, little-endian tensors: version 1 stores f32, version 2 adds per-record dtypes.
//! Parsing validates the complete file structure without an execution backend.
#![no_std]

extern crate alloc;

pub mod matrix;
pub mod quantized;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    InvalidArchive,
    InvalidShape,
    ShapeMismatch,
    SizeOverflow,
    MissingTensor(String),
    UnexpectedTensor,
    NonFinite,
    AllocationFailed,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => f.write_str("invalid tensor shape"),
            Self::ShapeMismatch => f.write_str("tensor storage does not match its shape"),
            Self::SizeOverflow => f.write_str("tensor size exceeds the address space"),
            Self::InvalidArchive => f.write_str("invalid or unsupported tensor file"),
            Self::MissingTensor(name) => write!(f, "missing tensor {name}"),
            Self::UnexpectedTensor => f.write_str("tensor file contains unexpected records"),
            Self::NonFinite => f.write_str("tensor file contains a nonfinite value"),
            Self::AllocationFailed => f.write_str("cannot allocate tensor data"),
        }
    }
}
impl core::error::Error for Error {}
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
}

pub fn element_count(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() || shape.len() > 8 || shape.contains(&0) {
        return Err(Error::InvalidShape);
    }
    shape.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).filter(|&n| n <= isize::MAX as usize / 4).ok_or(Error::SizeOverflow)
    })
}

pub const MAGIC: &[u8; 8] = b"D3TENSR\0";
pub const VERSION: u32 = 1;
pub const TYPED_VERSION: u32 = 2;

pub struct Record<'a> {
    shape: Vec<usize>,
    bytes: &'a [u8],
    width: usize,
}

impl Record<'_> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        if self.width == 2 { DType::Bf16 } else { DType::F32 }
    }
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut values = Vec::new();
        values.try_reserve_exact(self.bytes.len() / self.width).map_err(|_| Error::AllocationFailed)?;
        for bytes in self.bytes.chunks_exact(self.width) {
            let value = if self.width == 2 {
                f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().map_err(|_| Error::InvalidArchive)?)) << 16)
            } else {
                f32::from_le_bytes(bytes.try_into().map_err(|_| Error::InvalidArchive)?)
            };
            if !value.is_finite() {
                return Err(Error::NonFinite);
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
        if cursor.take(8)? != MAGIC {
            return Err(Error::InvalidArchive);
        }
        let version = cursor.u32()?;
        if version != VERSION && version != TYPED_VERSION {
            return Err(Error::InvalidArchive);
        }
        let count = cursor.u32()?;
        if count == 0 || count > 4096 {
            return Err(Error::InvalidArchive);
        }
        let mut records = BTreeMap::new();
        for _ in 0..count {
            let name_len = cursor.u16()? as usize;
            let rank = cursor.u16()? as usize;
            let elements = usize::try_from(cursor.u64()?).map_err(|_| Error::InvalidArchive)?;
            let width = if version == TYPED_VERSION {
                match cursor.u16()? {
                    0 => 4,
                    1 => 2,
                    _ => return Err(Error::InvalidArchive),
                }
            } else {
                4
            };
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
            let bytes = cursor.take(elements.checked_mul(width).ok_or(Error::InvalidArchive)?)?;
            if records.insert(name, Record { shape, bytes, width }).is_some() {
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
