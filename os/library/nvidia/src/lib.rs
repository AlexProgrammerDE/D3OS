//! Checked wire formats for the native D3OS NVIDIA firmware protocol.
#![no_std]

extern crate alloc;

pub mod abi;
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
