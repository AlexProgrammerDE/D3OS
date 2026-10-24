use crate::device::qemu_cfg::Selector::Signature;
use crate::{
    memory::vma::{VirtualMemoryArea, VmaType},
    process_manager,
};
use alloc::{
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::sync::atomic::{Ordering, fence};
use spin::Mutex;
use x86_64::instructions::port::{PortReadOnly, PortWriteOnly};
use x86_64::structures::paging::PageTableFlags;

const SELECTOR_PORT: u16 = 0x510;
const DATA_PORT: u16 = 0x511;
static PORT_ACCESS: Mutex<()> = Mutex::new(());

#[allow(dead_code)]
#[repr(u16)]
enum Selector {
    Signature = 0x0000,
    Id = 0x0001,
    RootDirectory = 0x0019,
}

pub fn is_available() -> bool {
    let _ports = PORT_ACCESS.lock();
    let mut selector_port = PortWriteOnly::<u16>::new(SELECTOR_PORT);
    let mut data_port = PortReadOnly::<u8>::new(DATA_PORT);
    let id: [u8; 4];

    unsafe {
        selector_port.write(Signature as u16);
        id = [data_port.read(), data_port.read(), data_port.read(), data_port.read()]
    }

    id[0] == b'Q' && id[1] == b'E' && id[2] == b'M' && id[3] == b'U'
}

#[derive(Clone, Debug)]
pub struct FirmwareFile {
    pub name: String,
    pub size: usize,
    selector: u16,
}

#[derive(Clone, Copy, Debug)]
pub enum Error {
    Protocol,
    Memory,
    Transfer,
}

const STAGING_BYTES: usize = 1 << 20;

/// Read-only DMA access to user-provided firmware files. The mount owns this
/// allocation for its lifetime, including after a failed or timed-out DMA.
pub struct Files {
    pub entries: Vec<FirmwareFile>,
    address: usize,
    allocation: Arc<VirtualMemoryArea>,
    failed: bool,
}

impl Files {
    pub fn probe() -> Result<Option<Self>, Error> {
        if !is_available() {
            return Ok(None);
        }
        let _ports = PORT_ACCESS.lock();
        let mut selector = PortWriteOnly::<u16>::new(SELECTOR_PORT);
        let mut data = PortReadOnly::<u8>::new(DATA_PORT);
        let mut read = |key, buffer: &mut [u8]| unsafe {
            selector.write(key);
            for byte in buffer {
                *byte = data.read();
            }
        };
        let mut id = [0u8; 4];
        read(Selector::Id as u16, &mut id);
        if u32::from_le_bytes(id) & 2 == 0 {
            return Ok(None);
        }
        let mut count = [0u8; 4];
        read(Selector::RootDirectory as u16, &mut count);
        let count = u32::from_be_bytes(count);
        if count > 4096 {
            return Err(Error::Protocol);
        }
        let mut entries = Vec::new();
        for _ in 0..count {
            let mut entry = [0u8; 64];
            for byte in &mut entry {
                *byte = unsafe { data.read() };
            }
            let end = entry[8..].iter().position(|&b| b == 0).ok_or(Error::Protocol)? + 8;
            let name = core::str::from_utf8(&entry[8..end]).map_err(|_| Error::Protocol)?;
            let Some(name) = name.strip_prefix("opt/d3os/") else {
                continue;
            };
            if name.is_empty() || name.contains('/') || name == "." || name == ".." || entries.iter().any(|file: &FirmwareFile| file.name == name) {
                return Err(Error::Protocol);
            }
            let key = u16::from_be_bytes([entry[4], entry[5]]);
            if !(0x20..0x4000).contains(&key) {
                return Err(Error::Protocol);
            }
            entries.push(FirmwareFile {
                name: name.to_string(),
                size: u32::from_be_bytes(entry[..4].try_into().map_err(|_| Error::Protocol)?) as usize,
                selector: key,
            });
        }
        if entries.is_empty() {
            return Ok(None);
        }
        let kernel = process_manager().read().kernel_process().ok_or(Error::Memory)?;
        // The explicit identity mapping makes this CPU virtual address also
        // the guest physical address required by the firmware DMA interface.
        let range = kernel.virtual_address_space.kernel_alloc_map_identity(
            (STAGING_BYTES / 4096 + 1) as u64,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE,
            VmaType::KernelBuffer,
            "fwcfg-dma",
        );
        let address = range.start.start_address().as_u64() as usize;
        let allocation = kernel
            .virtual_address_space
            .is_address_within_vma(address as u64, VmaType::KernelBuffer)
            .ok_or(Error::Memory)?;
        unsafe {
            (address as *mut u8).write_bytes(0, STAGING_BYTES + 4096);
        }
        Ok(Some(Self {
            entries,
            address,
            allocation,
            failed: false,
        }))
    }

    fn transfer(&mut self, control: u32, length: u32, target: u64) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Transfer);
        }
        let descriptor = self.address as *mut u32;
        unsafe {
            descriptor.write_volatile(control.to_be());
            descriptor.add(1).write_volatile(length.to_be());
            (descriptor.add(2) as *mut u64).write_volatile(target.to_be());
        }
        fence(Ordering::Release);
        unsafe {
            PortWriteOnly::<u32>::new(0x514).write(((self.address as u64 >> 32) as u32).to_be());
            PortWriteOnly::<u32>::new(0x518).write((self.address as u32).to_be());
        }
        for _ in 0..10_000_000 {
            let status = u32::from_be(unsafe { descriptor.read_volatile() });
            if status == 0 {
                fence(Ordering::Acquire);
                return Ok(());
            }
            if status & 1 != 0 {
                break;
            }
            core::hint::spin_loop();
        }
        self.failed = true;
        Err(Error::Transfer)
    }

    pub fn read(&mut self, file: &FirmwareFile, offset: usize, buffer: &mut [u8]) -> Result<usize, Error> {
        if offset >= file.size || buffer.is_empty() {
            return Ok(0);
        }
        let _ports = PORT_ACCESS.lock();
        let count = buffer.len().min(file.size - offset).min(STAGING_BYTES);
        let select = (u32::from(file.selector) << 16) | (1 << 3);
        if offset != 0 {
            self.transfer(select | (1 << 2), offset as u32, 0)?;
        }
        let flags = (1 << 1) | if offset == 0 { select } else { 0 };
        self.transfer(flags, count as u32, (self.address + 4096) as u64)?;
        unsafe {
            core::ptr::copy_nonoverlapping((self.address + 4096) as *const u8, buffer.as_mut_ptr(), count);
        }
        Ok(count)
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        if !self.failed {
            if let Some(kernel) = process_manager().read().kernel_process() {
                kernel.virtual_address_space.unmap_vma(self.allocation.clone(), true);
            }
        }
    }
}
