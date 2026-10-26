//! Checked NVIDIA firmware containers and VBIOS FWSEC preparation.

use crate::{Error, Record, Result, abi::*};
use alloc::vec::Vec;

fn region(data: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    data.get(offset..offset.checked_add(size).ok_or(Error::Truncated)?).ok_or(Error::Truncated)
}
fn patch(data: &mut [u8], offset: usize, bytes: &[u8]) -> Result<()> {
    data.get_mut(offset..offset.checked_add(bytes.len()).ok_or(Error::Truncated)?)
        .ok_or(Error::Truncated)?
        .copy_from_slice(bytes);
    Ok(())
}
fn u16_at(data: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(region(data, offset, 2)?.try_into().map_err(|_| Error::Truncated)?))
}
fn u32_at(data: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(region(data, offset, 4)?.try_into().map_err(|_| Error::Truncated)?))
}

pub fn binary(data: &[u8]) -> Result<(Record<{ BinaryHeader::SIZE }>, &[u8])> {
    let header = Record::<{ BinaryHeader::SIZE }>::from_bytes(data)?;
    if header.get(BinaryHeader::bin_magic)? != 0x10de || header.get(BinaryHeader::bin_ver)? != 1 {
        return Err(Error::InvalidFirmware);
    }
    let image = region(
        data,
        header.get(BinaryHeader::data_offset)? as usize,
        header.get(BinaryHeader::data_size)? as usize,
    )?;
    Ok((header, image))
}

pub struct Booter {
    pub image: Vec<u8>,
    pub code_offset: u32,
    pub code_size: u32,
    pub data_offset: u32,
    pub data_size: u32,
}

pub fn signed_booter(data: &[u8]) -> Result<Booter> {
    let (header, image) = binary(data)?;
    let signed = Record::<{ SignedHeader::SIZE }>::from_bytes(data.get(header.get(BinaryHeader::header_offset)? as usize..).ok_or(Error::Truncated)?)?;
    let load_offset = signed.get(SignedHeader::header_offset)? as usize;
    let load = Record::<{ LoadHeader::SIZE }>::from_bytes(data.get(load_offset..).ok_or(Error::Truncated)?)?;
    if load.get(LoadHeader::num_apps)? != 1 {
        return Err(Error::Unsupported);
    }
    let app = Record::<{ LoadApp::SIZE }>::from_bytes(data.get(load_offset + LoadHeader::SIZE..).ok_or(Error::Truncated)?)?;
    let patch_offset = u32_at(data, signed.get(SignedHeader::patch_loc)? as usize)? as usize;
    let patch_signature = u32_at(data, signed.get(SignedHeader::patch_sig)? as usize)? as usize;
    let signatures = u32_at(data, signed.get(SignedHeader::num_sig)? as usize)? as usize;
    let signatures_size = signed.get(SignedHeader::sig_prod_size)? as usize;
    if signatures == 0 || !signatures_size.is_multiple_of(signatures) {
        return Err(Error::InvalidFirmware);
    }
    let signature_offset = (signed.get(SignedHeader::sig_prod_offset)? as usize)
        .checked_add(patch_signature)
        .ok_or(Error::InvalidFirmware)?;
    let signature = region(data, signature_offset, signatures_size / signatures)?;
    let mut image = image.to_vec();
    patch(&mut image, patch_offset, signature)?;
    let code_offset = app.get(LoadApp::offset)?;
    let code_size = app.get(LoadApp::size)?;
    let data_offset = load.get(LoadHeader::os_data_offset)?;
    let data_size = load.get(LoadHeader::os_data_size)?;
    region(&image, code_offset as usize, code_size as usize)?;
    region(&image, data_offset as usize, data_size as usize)?;
    Ok(Booter {
        image,
        code_offset,
        code_size,
        data_offset,
        data_size,
    })
}

pub struct Fwsec {
    pub descriptor: Record<{ FalconDescriptor::SIZE }>,
    pub image: Vec<u8>,
}

pub fn fwsec(vbios: &[u8], frts_offset: u64) -> Result<Fwsec> {
    // NVIDIA VBIOS pointers in the extension image are relative to the preceding image.
    let (mut offset, mut base_size, mut expansion) = (0usize, None, None);
    for _ in 0..32 {
        let rom_signature = u16_at(vbios, offset)?;
        let pcir = offset + usize::from(u16_at(vbios, offset + 24)?);
        let signature = region(vbios, pcir, 4)?;
        let code = region(vbios, pcir + 20, 1)?[0];
        if !((rom_signature == 0xaa55 && signature == b"PCIR") || (rom_signature == 0x4e56 && signature == b"NPDS" && code == 0xe0)) {
            return Err(Error::InvalidFirmware);
        }
        let size = usize::from(u16_at(vbios, pcir + 16)?) * 512;
        if size == 0 {
            return Err(Error::InvalidFirmware);
        }
        region(vbios, offset, size)?;
        match code {
            0 => base_size = Some(size),
            0xe0 => {
                expansion = Some(offset.checked_sub(base_size.ok_or(Error::InvalidFirmware)?).ok_or(Error::InvalidFirmware)?);
                break;
            }
            _ => {}
        }
        offset = offset.checked_add(size).ok_or(Error::InvalidFirmware)?;
    }
    let expansion = expansion.ok_or(Error::Unsupported)?;
    let bit_offset = vbios.windows(6).position(|bytes| bytes == b"\xff\xb8BIT\0").ok_or(Error::InvalidFirmware)?;
    let bit = Record::<{ BitHeader::SIZE }>::from_bytes(&vbios[bit_offset..])?;
    let header_size = usize::from(bit.get(BitHeader::HeaderSize)?);
    let token_size = usize::from(bit.get(BitHeader::TokenSize)?);
    if header_size < BitHeader::SIZE || token_size < 6 {
        return Err(Error::InvalidFirmware);
    }
    for i in 0..usize::from(bit.get(BitHeader::TokenEntries)?) {
        // BIT v1 tokens contain a 16-bit data pointer and can be six bytes.
        let token = region(vbios, bit_offset + header_size + i * token_size, 6)?;
        if u32::from(token[0]) != BIT_TOKEN_FALCON_DATA || token[1] != 2 || u16_at(token, 2)? < 4 {
            continue;
        }
        let pointer = usize::from(u16_at(token, 4)?);
        let falcon = Record::<{ FalconData::SIZE }>::from_bytes(region(vbios, pointer, FalconData::SIZE)?)?;
        let table_offset = expansion
            .checked_add(falcon.get(FalconData::FalconUcodeTablePtr)? as usize)
            .ok_or(Error::InvalidFirmware)?;
        let table = Record::<{ FalconTableHeader::SIZE }>::from_bytes(region(vbios, table_offset, FalconTableHeader::SIZE)?)?;
        let head = usize::from(table.get(FalconTableHeader::HeaderSize)?);
        let size = usize::from(table.get(FalconTableHeader::EntrySize)?);
        if head < FalconTableHeader::SIZE || size < FalconTableEntry::SIZE {
            return Err(Error::InvalidFirmware);
        }
        for j in 0..usize::from(table.get(FalconTableHeader::EntryCount)?) {
            let entry = Record::<{ FalconTableEntry::SIZE }>::from_bytes(region(vbios, table_offset + head + j * size, size)?)?;
            if u32::from(entry.get(FalconTableEntry::ApplicationID)?) != FALCON_UCODE_ENTRY_APPID_FWSEC_PROD {
                continue;
            }
            let desc_offset = expansion
                .checked_add(entry.get(FalconTableEntry::DescPtr)? as usize)
                .ok_or(Error::InvalidFirmware)?;
            let descriptor = Record::<{ FalconDescriptor::SIZE }>::from_bytes(region(vbios, desc_offset, FalconDescriptor::SIZE)?)?;
            let descriptor_size = (u32_at(vbios, desc_offset)? >> 16) as usize;
            if descriptor_size < FalconDescriptor::SIZE + 384 {
                return Err(Error::InvalidFirmware);
            }
            let image_size = (descriptor.get(FalconDescriptor::StoredSize)? as usize)
                .checked_add(255)
                .ok_or(Error::InvalidFirmware)?
                & !255;
            let mut image = region(vbios, desc_offset + descriptor_size, image_size)?.to_vec();
            let signature = region(vbios, desc_offset + descriptor_size - 384, 384)?;
            let imem = descriptor.get(FalconDescriptor::IMEMLoadSize)? as usize;
            let interface_offset = imem
                .checked_add(descriptor.get(FalconDescriptor::InterfaceOffset)? as usize)
                .ok_or(Error::InvalidFirmware)?;
            let interface = Record::<{ InterfaceHeader::SIZE }>::from_bytes(region(&image, interface_offset, InterfaceHeader::SIZE)?)?;
            let head = usize::from(interface.get(InterfaceHeader::headerSize)?);
            let stride = usize::from(interface.get(InterfaceHeader::entrySize)?);
            if head < InterfaceHeader::SIZE || stride < InterfaceEntry::SIZE {
                return Err(Error::InvalidFirmware);
            }
            for k in 0..usize::from(interface.get(InterfaceHeader::entryCount)?) {
                let entry = Record::<{ InterfaceEntry::SIZE }>::from_bytes(region(&image, interface_offset + head + k * stride, stride)?)?;
                if entry.get(InterfaceEntry::id)? != FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER {
                    continue;
                }
                let mapper_offset = imem + entry.get(InterfaceEntry::dmemOffset)? as usize;
                let mut mapper = Record::<{ DmemMapper::SIZE }>::from_bytes(region(&image, mapper_offset, DmemMapper::SIZE)?)?;
                if mapper.get(DmemMapper::cmd_in_buffer_size)? < FrtsCommand::SIZE as u32 {
                    return Err(Error::InvalidFirmware);
                }
                mapper.set(DmemMapper::init_cmd, 0x15)?;
                let command_offset = imem + mapper.get(DmemMapper::cmd_in_buffer_offset)? as usize;
                let mut read = Record::<{ ReadVbios::SIZE }>::default();
                read.set(ReadVbios::version, 1)?;
                read.set(ReadVbios::size, ReadVbios::SIZE as u32)?;
                read.set(ReadVbios::flags, 2)?;
                let mut frts = Record::<{ FrtsRegion::SIZE }>::default();
                frts.set(FrtsRegion::version, 1)?;
                frts.set(FrtsRegion::size, FrtsRegion::SIZE as u32)?;
                frts.set(
                    FrtsRegion::frtsRegionOffset4K,
                    (frts_offset >> 12).try_into().map_err(|_| Error::InvalidFirmware)?,
                )?;
                frts.set(FrtsRegion::frtsRegionSize, 0x100)?;
                frts.set(FrtsRegion::frtsRegionMediaType, 2)?;
                let mut command = Record::<{ FrtsCommand::SIZE }>::default();
                command.set(FrtsCommand::readVbiosDesc, read.into_bytes())?;
                command.set(FrtsCommand::frtsRegionDesc, frts.into_bytes())?;
                patch(&mut image, mapper_offset, mapper.as_bytes())?;
                patch(&mut image, command_offset, command.as_bytes())?;
                patch(&mut image, imem + descriptor.get(FalconDescriptor::PKCDataOffset)? as usize, signature)?;
                return Ok(Fwsec { descriptor, image });
            }
            return Err(Error::Unsupported);
        }
    }
    Err(Error::Unsupported)
}
