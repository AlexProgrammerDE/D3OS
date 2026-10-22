//! GPU-owned VRAM and Ada's five-level virtual address space.

use super::{Device, Error, Result};
use alloc::vec::Vec;

#[derive(Clone, Copy)]
pub(super) struct Allocation {
    pub physical: u64,
    pub virtual_address: u64,
    pub bytes: u64,
}

pub(super) struct Memory {
    pub root: u64,
    physical: u64,
    virtual_address: u64,
    limit: u64,
    tables: Vec<(usize, u64, u64)>,
}

impl Memory {
    pub(super) fn new(device: &Device) -> Result<Self> {
        let mut memory = Self {
            root: 0,
            physical: 16 << 20,
            virtual_address: 0x1000000000,
            limit: device.vram_size.checked_sub(256 << 20).ok_or(Error::Alignment)?,
            tables: Vec::new(),
        };
        memory.root = memory.allocate_physical(device, 4096, 4096)?;
        Ok(memory)
    }

    pub(super) fn allocate_physical(&mut self, device: &Device, bytes: u64, alignment: u64) -> Result<u64> {
        let start = self.reserve_physical(bytes, alignment)?;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        for offset in (start..self.physical).step_by(4) {
            vram.write(offset as usize, 0)?;
        }
        Ok(start)
    }

    fn reserve_physical(&mut self, bytes: u64, alignment: u64) -> Result<u64> {
        if bytes == 0 || !alignment.is_power_of_two() {
            return Err(Error::Alignment);
        }
        let start = self.physical.checked_add(alignment - 1).ok_or(Error::Alignment)? & !(alignment - 1);
        let end = start.checked_add(bytes).and_then(|n| n.checked_add(4095)).ok_or(Error::Alignment)? & !4095;
        if end > self.limit {
            return Err(Error::Alignment);
        }
        self.physical = end;
        Ok(start)
    }

    pub(super) fn reserve_virtual(&mut self, bytes: u64, alignment: u64) -> Result<u64> {
        if bytes == 0 || !alignment.is_power_of_two() {
            return Err(Error::Alignment);
        }
        let start = self.virtual_address.checked_add(alignment - 1).ok_or(Error::Alignment)? & !(alignment - 1);
        self.virtual_address = start.checked_add(bytes).ok_or(Error::Alignment)?;
        if self.virtual_address >= 1 << 48 {
            return Err(Error::Alignment);
        }
        Ok(start)
    }

    fn write64(device: &Device, address: u64, value: u64) -> Result<()> {
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        vram.write(address as usize, value as u32)?;
        vram.write(address as usize + 4, (value >> 32) as u32)
    }

    pub(super) fn page_tables(&mut self, device: &Device, address: u64, depth: usize) -> Result<Vec<u64>> {
        if address >= 1 << 48 || depth == 0 || depth > 5 {
            return Err(Error::Alignment);
        }
        let shifts = [47, 38, 29, 21, 12];
        let mut table = self.root;
        let mut result = alloc::vec![table];
        for level in 1..depth {
            let prefix = address >> shifts[level - 1];
            let next = if let Some((_, _, physical)) = self.tables.iter().find(|(l, p, _)| *l == level && *p == prefix) {
                *physical
            } else {
                let physical = self.allocate_physical(device, 4096, 4096)?;
                let index = (address >> shifts[level - 1])
                    & if level == 1 {
                        3
                    } else if level == 4 {
                        255
                    } else {
                        511
                    };
                if level == 4 {
                    // PDE0 has separate 64 KiB and 4 KiB entries. Select
                    // the small-page aperture in the upper 64-bit word.
                    Self::write64(device, table + index * 16, 1 << 5)?;
                    Self::write64(device, table + index * 16 + 8, 2 | ((physical >> 12) << 8))?;
                } else {
                    Self::write64(device, table + index * 8, 0x22 | ((physical >> 12) << 8))?;
                }
                self.tables.push((level, prefix, physical));
                physical
            };
            result.push(next);
            table = next;
        }
        Ok(result)
    }

    pub(super) fn allocate(&mut self, device: &Device, bytes: u64, alignment: u64) -> Result<Allocation> {
        let bytes = bytes.checked_add(4095).ok_or(Error::Alignment)? & !4095;
        let physical = self.allocate_physical(device, bytes, alignment.max(4096))?;
        let virtual_address = self.map_range(device, physical, bytes, alignment.max(4096), false)?;
        Ok(Allocation {
            physical,
            virtual_address,
            bytes,
        })
    }

    /// Driver-owned buffer storage. Callers must clear this allocation before
    /// returning a handle or data to an application.
    pub(super) fn allocate_buffer(&mut self, device: &Device, bytes: u64) -> Result<Allocation> {
        let bytes = bytes.checked_add(4095).ok_or(Error::Alignment)? & !4095;
        let alignment = if bytes >= 2 << 20 { 2 << 20 } else { 4096 };
        let physical = self.reserve_physical(bytes, alignment)?;
        let virtual_address = self.map_range(device, physical, bytes, alignment, false)?;
        Ok(Allocation {
            physical,
            virtual_address,
            bytes,
        })
    }

    pub(super) fn map_system(&mut self, device: &Device, physical: u64, bytes: u64) -> Result<u64> {
        self.map_range(device, physical, bytes, 4096, true)
    }

    fn map_range(&mut self, device: &Device, physical: u64, bytes: u64, alignment: u64, system: bool) -> Result<u64> {
        if physical == 0 || physical & 4095 != 0 || bytes == 0 || bytes & 4095 != 0 || physical.checked_add(bytes).is_none_or(|end| end > 1 << 52) {
            return Err(Error::Alignment);
        }
        let virtual_address = self.reserve_virtual(bytes, alignment)?;
        let mut offset = 0;
        while offset < bytes {
            let va = virtual_address + offset;
            let large = !system && (va | (physical + offset)) & ((2 << 20) - 1) == 0 && bytes - offset >= 2 << 20;
            let tables = self.page_tables(device, va, if large { 4 } else { 5 })?;
            // GPU-owned VRAM must use the L2 cache. VOL bypasses caching and
            // is needed only for CPU/GPU coherent shared system-memory pages.
            // MMU-v2 aperture 2 denotes system-coherent RAM; 0 denotes VRAM.
            let pte = 1 | (u64::from(system) << 3) | (u64::from(system) << 2) | (((physical + offset) >> 12) << 8) | (6u64 << 56);
            if large {
                // A valid PTE in PDE0 maps 2 MiB directly instead of pointing
                // to 512 small-page entries. Keep the unused upper word zero.
                let entry = tables[3] + ((va >> 21) & 255) * 16;
                Self::write64(device, entry, pte)?;
                Self::write64(device, entry + 8, 0)?;
                offset += 2 << 20;
            } else {
                Self::write64(device, tables[4] + ((va >> 12) & 511) * 8, pte)?;
                offset += 4096;
            }
        }
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        device.registers.write(0xb830b0, (1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))?;
        Ok(virtual_address)
    }
}
