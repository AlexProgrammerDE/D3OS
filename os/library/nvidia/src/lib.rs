//! Checked wire formats for the native D3OS NVIDIA firmware protocol.
#![no_std]

extern crate alloc;

pub mod abi;
pub mod compute;
pub mod context;
pub mod cubin;
pub mod firmware;
pub mod rpc;

use core::marker::PhantomData;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    InvalidFirmware,
    InvalidQueue,
    QueueFull,
    InvalidChecksum,
    Unsupported,
    InvalidField,
}

pub type Result<T> = core::result::Result<T, Error>;

/// A typed field in an unaligned, little-endian firmware record.
pub struct Field<T> {
    offset: usize,
    value: PhantomData<T>,
}

impl<T> Field<T> {
    pub const fn new(offset: usize) -> Self {
        Self { offset, value: PhantomData }
    }
}

pub trait WireValue: Sized {
    const SIZE: usize;
    fn decode(bytes: &[u8]) -> Self;
    fn encode(self, bytes: &mut [u8]);
}

macro_rules! integer {
    ($($ty:ty),*) => { $(
        impl WireValue for $ty {
            const SIZE: usize = size_of::<Self>();
            fn decode(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().expect("checked wire field size"))
            }
            fn encode(self, bytes: &mut [u8]) { bytes.copy_from_slice(&self.to_le_bytes()); }
        }
    )* };
}
integer!(u8, u16, u32, u64, i8, i16, i32, i64);

impl<const N: usize> WireValue for [u8; N] {
    const SIZE: usize = N;
    fn decode(bytes: &[u8]) -> Self {
        bytes.try_into().expect("checked wire field size")
    }
    fn encode(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self);
    }
}

/// Fixed-size records support overlapping ABI fields without packed references.
#[derive(Clone)]
pub struct Record<const N: usize> {
    bytes: [u8; N],
}

impl<const N: usize> Default for Record<N> {
    fn default() -> Self {
        Self { bytes: [0; N] }
    }
}

impl<const N: usize> Record<N> {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let bytes = bytes.get(..N).ok_or(Error::Truncated)?;
        Ok(Self {
            bytes: bytes.try_into().map_err(|_| Error::Truncated)?,
        })
    }

    pub fn get<T: WireValue>(&self, field: Field<T>) -> Result<T> {
        let end = field.offset.checked_add(T::SIZE).ok_or(Error::Truncated)?;
        Ok(T::decode(self.bytes.get(field.offset..end).ok_or(Error::Truncated)?))
    }

    pub fn set<T: WireValue>(&mut self, field: Field<T>, value: T) -> Result<()> {
        let end = field.offset.checked_add(T::SIZE).ok_or(Error::Truncated)?;
        value.encode(self.bytes.get_mut(field.offset..end).ok_or(Error::Truncated)?);
        Ok(())
    }

    pub fn as_bytes(&self) -> &[u8; N] {
        &self.bytes
    }
    pub fn into_bytes(self) -> [u8; N] {
        self.bytes
    }
}

/// Inclusive hardware bit range, numbered from the least significant bit.
#[derive(Clone, Copy)]
pub struct Bits {
    high: usize,
    low: usize,
}

impl Bits {
    pub const fn new(high: usize, low: usize) -> Self {
        Self { high, low }
    }
    fn width(self, bytes: usize) -> Result<usize> {
        let width = self.high.checked_sub(self.low).and_then(|n| n.checked_add(1)).ok_or(Error::InvalidField)?;
        if width > 64 || self.high / 8 >= bytes {
            return Err(Error::InvalidField);
        }
        Ok(width)
    }
}

impl<const N: usize> Record<N> {
    pub fn set_bits(&mut self, field: Bits, value: u64) -> Result<()> {
        let width = field.width(N)?;
        if width < 64 && value >> width != 0 {
            return Err(Error::InvalidField);
        }
        for i in 0..width {
            let bit = field.low + i;
            let mask = 1u8 << (bit % 8);
            self.bytes[bit / 8] = (self.bytes[bit / 8] & !mask) | (((value >> i) as u8 & 1) << (bit % 8));
        }
        Ok(())
    }

    pub fn get_bits(&self, field: Bits) -> Result<u64> {
        let width = field.width(N)?;
        Ok((0..width).fold(0, |value, i| {
            value | (u64::from((self.bytes[(field.low + i) / 8] >> ((field.low + i) % 8)) & 1) << i)
        }))
    }
}
