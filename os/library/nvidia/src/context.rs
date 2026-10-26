//! Checked graphics-image relocation and virtual RAMIN binding.
//!
//! Field definitions follow NVIDIA's published Volta/Turing context format:
//! https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/volta/gv100/dev_ctxsw.ref.txt
//! https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/turing/tu104/dev_ram.ref.txt

use crate::{Bits, Error, Field, Record, Result};

pub const HEADER_BYTES: usize = 256;
pub const INSTANCE_BYTES: usize = 4096;

/// Relocate a firmware-generated image without changing its saved engine state.
/// The caller must copy the remaining image bytes from the same template.
pub fn relocate_header(template: &[u8], allocation_bytes: u64, virtual_address: u64) -> Result<[u8; HEADER_BYTES]> {
    let mut header = Record::<HEADER_BYTES>::from_bytes(template)?;
    let image_bytes = u64::from(header.get(Field::<u32>::new(4))?);
    if header.get(Field::<u32>::new(0xfc))? != 0x600dc0de
        || header.get_bits(Bits::new(0x0c * 8 + 31, 0x0c * 8 + 16))? != 0xcafe
        || image_bytes < HEADER_BYTES as u64
        || image_bytes > allocation_bytes
        || virtual_address == 0
        || virtual_address & 4095 != 0
        || virtual_address.checked_add(allocation_bytes).is_none_or(|end| end > 1 << 49)
    {
        return Err(Error::InvalidField);
    }
    header.set_bits(Bits::new(0x78 * 8 + 16, 0x78 * 8), virtual_address >> 32)?;
    header.set(Field::<u32>::new(0x7c), virtual_address as u32)?;
    Ok(header.into_bytes())
}

/// Bind an initialized image while the channel is disabled and idle.
/// RAMIN's engine pointer has a 40-bit address and a separate virtual-mode bit.
pub fn bind_virtual_image(instance: &mut Record<INSTANCE_BYTES>, virtual_address: u64) -> Result<()> {
    if virtual_address == 0 || virtual_address & 4095 != 0 || virtual_address >= 1 << 40 {
        return Err(Error::InvalidField);
    }
    instance.set(Field::<u32>::new(0x210), virtual_address as u32 | 4)?;
    instance.set(Field::<u32>::new(0x214), (virtual_address >> 32) as u32)?;
    instance.set_bits(Bits::new(0x218 * 8 + 5, 0x218 * 8), 0)?; // VEID zero.
    instance.set_bits(Bits::new(0xac * 8 + 16, 0xac * 8 + 16), 1)?;
    Ok(())
}
