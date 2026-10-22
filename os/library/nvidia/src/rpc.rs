//! Framing and bounds checks for GSP's page-based command and status queues.

use crate::{
    Error, Record, Result,
    abi::{self, QueueElement, RpcHeader},
};
use alloc::{vec, vec::Vec};

pub const PAGE: usize = 4096;
pub const MAX_PAGES: usize = 16;
pub const PREFIX: usize = QueueElement::SIZE + RpcHeader::SIZE;
pub const MAX_PAYLOAD: usize = PAGE * MAX_PAGES - PREFIX;

pub fn checksum(bytes: &[u8]) -> u32 {
    let mut value = 0u64;
    for chunk in bytes.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        value ^= u64::from_le_bytes(word);
    }
    value as u32 ^ (value >> 32) as u32
}

pub fn encode(function: u32, sequence: u32, payload: &[u8]) -> Result<Vec<u8>> {
    if payload.len() > MAX_PAYLOAD {
        return Err(Error::InvalidQueue);
    }
    let length = PREFIX + payload.len();
    let pages = length.div_ceil(PAGE);
    let mut output = vec![0u8; pages * PAGE];
    let mut element = Record::<{ QueueElement::SIZE }>::default();
    element.set(QueueElement::seqNum, sequence)?;
    element.set(QueueElement::elemCount, pages as u32)?;
    let mut rpc = Record::<{ RpcHeader::SIZE }>::default();
    rpc.set(RpcHeader::header_version, 3 << 24)?;
    rpc.set(RpcHeader::signature, abi::NV_VGPU_MSG_SIGNATURE_VALID)?;
    rpc.set(RpcHeader::length, (RpcHeader::SIZE + payload.len()) as u32)?;
    rpc.set(RpcHeader::function, function)?;
    rpc.set(RpcHeader::rpc_result, abi::NV_VGPU_MSG_RESULT_RPC_PENDING)?;
    rpc.set(RpcHeader::rpc_result_private, abi::NV_VGPU_MSG_RESULT_RPC_PENDING)?;
    output[..QueueElement::SIZE].copy_from_slice(element.as_bytes());
    output[QueueElement::SIZE..PREFIX].copy_from_slice(rpc.as_bytes());
    output[PREFIX..length].copy_from_slice(payload);
    element.set(QueueElement::checkSum, checksum(&output[..length]))?;
    output[..QueueElement::SIZE].copy_from_slice(element.as_bytes());
    Ok(output)
}

pub struct Response<'a> {
    pub function: u32,
    pub result: u32,
    pub sequence: u32,
    pub pages: usize,
    pub payload: &'a [u8],
}

pub fn decode(bytes: &[u8]) -> Result<Response<'_>> {
    let element = Record::<{ QueueElement::SIZE }>::from_bytes(bytes)?;
    let rpc = Record::<{ RpcHeader::SIZE }>::from_bytes(bytes.get(QueueElement::SIZE..).ok_or(Error::Truncated)?)?;
    let pages = element.get(QueueElement::elemCount)? as usize;
    let length = rpc.get(RpcHeader::length)? as usize;
    let total = QueueElement::SIZE.checked_add(length).ok_or(Error::InvalidQueue)?;
    if pages == 0
        || pages > MAX_PAGES
        || length < RpcHeader::SIZE
        || total.div_ceil(PAGE) != pages
        || rpc.get(RpcHeader::signature)? != abi::NV_VGPU_MSG_SIGNATURE_VALID
    {
        return Err(Error::InvalidQueue);
    }
    let message = bytes.get(..total).ok_or(Error::Truncated)?;
    if checksum(message) != 0 {
        return Err(Error::InvalidChecksum);
    }
    Ok(Response {
        function: rpc.get(RpcHeader::function)?,
        result: rpc.get(RpcHeader::rpc_result)?,
        sequence: element.get(QueueElement::seqNum)?,
        pages,
        payload: &message[PREFIX..],
    })
}

pub fn available(write: usize, read: usize, count: usize) -> Result<usize> {
    if count < 2 || write >= count || read >= count {
        return Err(Error::InvalidQueue);
    }
    Ok(if read > write { read - write - 1 } else { count - (write - read) - 1 })
}

/// First contiguous part and wrapped part of a page-aligned queue transfer.
pub fn spans(start: usize, pages: usize, count: usize) -> Result<(usize, usize)> {
    if count < 2 || start >= count || pages == 0 || pages >= count {
        return Err(Error::InvalidQueue);
    }
    let first = pages.min(count - start);
    Ok((first, pages - first))
}
