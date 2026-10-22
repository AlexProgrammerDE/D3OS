//! GSP firmware boot allocations and the management RPC queues.

use super::{Device, Error, Result};
use crate::memory::vma::{VirtualMemoryArea, VmaType};
use crate::{pci_bus, process_manager};
use alloc::{sync::Arc, vec::Vec};
use core::sync::atomic::{Ordering, fence};
use d3os_nvidia::{Record, abi::*, firmware, rpc};
use goblin::elf::Elf;
use log::info;
use pci_types::{CommandRegister, ConfigRegionAccess};
use x86_64::structures::paging::PageTableFlags;

const QUEUE_SIZE: usize = 0x40000;
const COMMAND: usize = 0x1000;
const STATUS: usize = COMMAND + QUEUE_SIZE;

pub(super) struct DmaRegion {
    pub(super) address: usize,
    pub(super) size: usize,
    vma: Arc<VirtualMemoryArea>,
}

impl DmaRegion {
    pub(super) fn new(bytes: usize) -> Result<Self> {
        let pages = bytes.checked_add(4095).ok_or(Error::Alignment)? / 4096;
        if pages == 0 {
            return Err(Error::Alignment);
        }
        let kernel = process_manager().read().kernel_process().ok_or(Error::Alignment)?;
        let range = kernel.virtual_address_space.kernel_alloc_map_identity(
            pages as u64,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE,
            VmaType::KernelBuffer,
            "nvidia-dma",
        );
        let address = range.start.start_address().as_u64() as usize;
        let vma = kernel
            .virtual_address_space
            .is_address_within_vma(address as u64, VmaType::KernelBuffer)
            .ok_or(Error::Alignment)?;
        unsafe {
            (address as *mut u8).write_bytes(0, pages * 4096);
        }
        Ok(Self {
            address,
            size: pages * 4096,
            vma,
        })
    }

    fn check(&self, offset: usize, size: usize) -> Result<()> {
        if offset.checked_add(size).is_none_or(|end| end > self.size) {
            return Err(Error::Alignment);
        }
        Ok(())
    }

    fn copy(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.check(offset, bytes.len())?;
        // Queue memory is shared with an external DMA agent.
        for (i, byte) in bytes.iter().copied().enumerate() {
            unsafe {
                ((self.address + offset + i) as *mut u8).write_volatile(byte);
            }
        }
        Ok(())
    }

    fn read(&self, offset: usize, size: usize) -> Result<Vec<u8>> {
        self.check(offset, size)?;
        Ok((0..size)
            .map(|i| unsafe { ((self.address + offset + i) as *const u8).read_volatile() })
            .collect())
    }

    fn read32(&self, offset: usize) -> Result<u32> {
        self.check(offset, 4)?;
        if offset & 3 != 0 {
            return Err(Error::Alignment);
        }
        Ok(unsafe { ((self.address + offset) as *const u32).read_volatile() })
    }

    fn write32(&self, offset: usize, value: u32) -> Result<()> {
        self.check(offset, 4)?;
        if offset & 3 != 0 {
            return Err(Error::Alignment);
        }
        unsafe {
            ((self.address + offset) as *mut u32).write_volatile(value);
        }
        Ok(())
    }
}

impl Drop for DmaRegion {
    fn drop(&mut self) {
        // Live regions remain owned by the static Device for the firmware lifetime.
        // Failed preparation can release its buffers before bus mastering starts.
        if let Some(kernel) = process_manager().read().kernel_process() {
            kernel.virtual_address_space.unmap_vma(self.vma.clone(), true);
        }
    }
}

pub(super) struct Gsp {
    regions: Vec<DmaRegion>,
    queue: usize,
    sequence: u32,
    pub(super) libos: u64,
    pub(super) metadata: u64,
}

impl Gsp {
    fn allocate(&mut self, bytes: usize, data: Option<&[u8]>) -> Result<usize> {
        let region = DmaRegion::new(bytes)?;
        if let Some(data) = data {
            region.copy(0, data)?;
        }
        self.regions.push(region);
        Ok(self.regions.len() - 1)
    }

    pub(super) fn prepare(device: &Device, image_file: &[u8], loader_file: &[u8]) -> Result<Self> {
        let mut gsp = Self {
            regions: Vec::new(),
            queue: 0,
            sequence: 0,
            libos: 0,
            metadata: 0,
        };
        gsp.queue = gsp.allocate(STATUS + QUEUE_SIZE, None)?;
        let queue = &gsp.regions[gsp.queue];
        for i in 0..queue.size / 4096 {
            queue.copy(i * 8, &((queue.address + i * 4096) as u64).to_le_bytes())?;
        }
        let mut header = Record::<{ QueueHeader::SIZE }>::default();
        header.set(QueueHeader::size, QUEUE_SIZE as u32)?;
        header.set(QueueHeader::entryOff, 4096)?;
        header.set(QueueHeader::msgSize, 4096)?;
        header.set(QueueHeader::msgCount, (QUEUE_SIZE / 4096 - 1) as u32)?;
        header.set(QueueHeader::flags, 1)?;
        header.set(QueueHeader::rxHdrOff, QueueHeader::SIZE as u32)?;
        queue.copy(COMMAND, header.as_bytes())?;
        let mut queue_args = Record::<{ QueueArguments::SIZE }>::default();
        queue_args.set(QueueArguments::sharedMemPhysAddr, queue.address as u64)?;
        queue_args.set(QueueArguments::pageTableEntryCount, (queue.size / 4096) as u32)?;
        queue_args.set(QueueArguments::cmdQueueOffset, COMMAND as u64)?;
        queue_args.set(QueueArguments::statQueueOffset, STATUS as u64)?;
        let mut args = Record::<{ GspArguments::SIZE }>::default();
        args.set(GspArguments::messageQueueInitArguments, queue_args.into_bytes())?;
        args.set(GspArguments::bDmemStack, 1)?;
        let args_index = gsp.allocate(4096, Some(args.as_bytes()))?;
        let args_address = gsp.regions[args_index].address as u64;
        let log_index = gsp.allocate(2 << 20, None)?;
        let logs_address = gsp.regions[log_index].address as u64;
        let mut libos_data = Vec::new();
        for (i, name) in [b"LOGINIT".as_slice(), b"LOGINTR", b"LOGRM", b"LOGMNOC", b"LOGKRNL", b"RMARGS"]
            .iter()
            .enumerate()
        {
            let mut identifier = [0u8; 8];
            identifier[8 - name.len()..].copy_from_slice(name);
            let mut region = Record::<{ LibosRegion::SIZE }>::default();
            region.set(LibosRegion::id8, u64::from_be_bytes(identifier))?;
            region.set(LibosRegion::pa, if i == 5 { args_address } else { logs_address + (i * 0x10000) as u64 })?;
            region.set(LibosRegion::size, if i == 5 { 4096 } else { 0x10000 })?;
            region.set(LibosRegion::kind, 1)?;
            region.set(LibosRegion::loc, 1)?;
            libos_data.extend_from_slice(region.as_bytes());
        }
        let index = gsp.allocate(4096, Some(&libos_data))?;
        gsp.libos = gsp.regions[index].address as u64;

        let elf = Elf::parse(image_file).map_err(|_| Error::Firmware)?;
        let section = |name: &str| -> Result<&[u8]> {
            let section = elf
                .section_headers
                .iter()
                .find(|section| elf.shdr_strtab.get_at(section.sh_name) == Some(name))
                .ok_or(Error::Firmware)?;
            image_file
                .get(section.sh_offset as usize..section.sh_offset.checked_add(section.sh_size).ok_or(Error::Firmware)? as usize)
                .ok_or(Error::Firmware)
        };
        let image = section(".fwimage")?;
        let signature = section(".fwsignature_ad10x")?;
        if signature.len() != 4096 {
            return Err(Error::Firmware);
        }
        let pages3 = image.len().div_ceil(4096);
        let pages2 = pages3.div_ceil(512);
        let pages1 = pages2.div_ceil(512);
        let pages0 = pages1.div_ceil(512);
        let counts = [pages0, pages1, pages2, pages3];
        if pages0 != 1 {
            return Err(Error::Firmware);
        }
        let offsets = [0, pages0 * 4096, (pages0 + pages1) * 4096, (pages0 + pages1 + pages2) * 4096];
        // Firmware uses a radix tree specifically so its image pages need not
        // be physically contiguous. Stay within the frame allocator's 2 MiB
        // allocation limit instead of requesting one large firmware extent.
        let radix = gsp.allocate(offsets[3], None)?;
        let mut image_pages = Vec::with_capacity(pages3);
        for chunk in image.chunks(2 << 20) {
            let index = gsp.allocate(chunk.len(), Some(chunk))?;
            let region = &gsp.regions[index];
            image_pages.extend((0..region.size / 4096).map(|page| (region.address + page * 4096) as u64));
        }
        let root = &gsp.regions[radix];
        for level in 0..3 {
            for i in 0..counts[level + 1] {
                let address = if level == 2 {
                    image_pages[i]
                } else {
                    (root.address + offsets[level + 1] + i * 4096) as u64
                };
                root.copy(offsets[level] + i * 8, &address.to_le_bytes())?;
            }
        }
        let radix_address = root.address as u64;
        let index = gsp.allocate(signature.len(), Some(signature))?;
        let signature_address = gsp.regions[index].address as u64;
        let (loader_header, loader) = firmware::binary(loader_file)?;
        let desc_offset = loader_header.get(BinaryHeader::header_offset)? as usize;
        let descriptor = Record::<{ RiscvDescriptor::SIZE }>::from_bytes(loader_file.get(desc_offset..).ok_or(Error::Firmware)?)?;
        let index = gsp.allocate(loader.len(), Some(loader))?;
        let loader_address = gsp.regions[index].address as u64;
        let vga_size = 1u64 << 20;
        let vga = device.vram_size.checked_sub(vga_size).ok_or(Error::Firmware)?;
        let frts = vga.checked_sub(1 << 20).ok_or(Error::Firmware)?;
        let boot = frts.checked_sub(loader.len() as u64).ok_or(Error::Firmware)?;
        let fw = boot.checked_sub(image.len() as u64).ok_or(Error::Firmware)? & !0xffff;
        let heap_size = 0x8100000u64;
        let heap = fw.checked_sub(heap_size).ok_or(Error::Firmware)? & !0xfffff;
        let wpr_start = heap.checked_sub(4096).ok_or(Error::Firmware)? & !0xfffff;
        let non_wpr = wpr_start.checked_sub(1 << 20).ok_or(Error::Firmware)? & !0xfffff;
        let mut metadata = Record::<{ WprMeta::SIZE }>::default();
        metadata.set(WprMeta::magic, GSP_FW_WPR_META_MAGIC)?;
        metadata.set(WprMeta::revision, u64::from(GSP_FW_WPR_META_REVISION))?;
        metadata.set(WprMeta::sysmemAddrOfRadix3Elf, radix_address)?;
        metadata.set(WprMeta::sizeOfRadix3Elf, image.len() as u64)?;
        metadata.set(WprMeta::sysmemAddrOfBootloader, loader_address)?;
        metadata.set(WprMeta::sizeOfBootloader, loader.len() as u64)?;
        metadata.set(WprMeta::bootloaderCodeOffset, u64::from(descriptor.get(RiscvDescriptor::monitorCodeOffset)?))?;
        metadata.set(WprMeta::bootloaderDataOffset, u64::from(descriptor.get(RiscvDescriptor::monitorDataOffset)?))?;
        metadata.set(WprMeta::bootloaderManifestOffset, u64::from(descriptor.get(RiscvDescriptor::manifestOffset)?))?;
        metadata.set(WprMeta::sysmemAddrOfSignature, signature_address)?;
        metadata.set(WprMeta::sizeOfSignature, 4096)?;
        metadata.set(WprMeta::vgaWorkspaceOffset, vga)?;
        metadata.set(WprMeta::vgaWorkspaceSize, vga_size)?;
        metadata.set(WprMeta::gspFwWprEnd, vga)?;
        metadata.set(WprMeta::frtsSize, 1 << 20)?;
        metadata.set(WprMeta::frtsOffset, frts)?;
        metadata.set(WprMeta::bootBinOffset, boot)?;
        metadata.set(WprMeta::gspFwOffset, fw)?;
        metadata.set(WprMeta::gspFwHeapSize, heap_size)?;
        metadata.set(WprMeta::fbSize, device.vram_size)?;
        metadata.set(WprMeta::gspFwHeapOffset, heap)?;
        metadata.set(WprMeta::gspFwWprStart, wpr_start)?;
        metadata.set(WprMeta::nonWprHeapSize, 1 << 20)?;
        metadata.set(WprMeta::nonWprHeapOffset, non_wpr)?;
        metadata.set(WprMeta::gspFwRsvdStart, non_wpr)?;
        let index = gsp.allocate(4096, Some(metadata.as_bytes()))?;
        gsp.metadata = gsp.regions[index].address as u64;
        info!(
            "NVIDIA_GSP_PREPARED image_bytes={} radix={radix_address:#x} libos={:#x} metadata={:#x}",
            image.len(),
            gsp.libos,
            gsp.metadata
        );
        gsp.prefill(device)?;
        Ok(gsp)
    }

    fn prefill(&mut self, device: &Device) -> Result<()> {
        let address = device.address;
        let bus = pci_bus();
        let config = bus.config_space();
        let mut info = Record::<{ SystemInfo::SIZE }>::default();
        info.set(SystemInfo::gpuPhysAddr, device.registers.base.as_ptr() as u64)?;
        info.set(SystemInfo::gpuPhysFbAddr, device.vram_address)?;
        info.set(SystemInfo::gpuPhysInstAddr, device.bar3)?;
        info.set(
            SystemInfo::nvDomainBusDeviceFunc,
            (u64::from(address.bus()) << 8) | (u64::from(address.device()) << 3) | u64::from(address.function()),
        )?;
        info.set(SystemInfo::pciConfigMirrorBase, 0x88000)?;
        info.set(SystemInfo::pciConfigMirrorSize, 4096)?;
        info.set(SystemInfo::PCIDeviceID, unsafe { config.read(address, 0) })?;
        info.set(SystemInfo::PCISubDeviceID, unsafe { config.read(address, 0x2c) })?;
        info.set(SystemInfo::PCIRevisionID, unsafe { config.read(address, 8) } & 0xff)?;
        info.set(SystemInfo::bIsPassthru, 1)?;
        info.set(SystemInfo::maxUserVa, 0x7ffffffff000)?;
        self.send(device, NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, info.as_bytes())?;
        let keys = [b"RMForcePcieConfigSave".as_slice(), b"RMSecBusResetEnable"];
        let mut entries = Vec::new();
        let mut names = Vec::new();
        for key in keys {
            let mut entry = Record::<{ RegistryEntry::SIZE }>::default();
            entry.set(
                RegistryEntry::nameOffset,
                (RegistryTable::SIZE + keys.len() * RegistryEntry::SIZE + names.len()) as u32,
            )?;
            entry.set(RegistryEntry::r#type, REGISTRY_TABLE_ENTRY_TYPE_DWORD as u8)?;
            entry.set(RegistryEntry::data, 1)?;
            entry.set(RegistryEntry::length, 4)?;
            entries.extend_from_slice(entry.as_bytes());
            names.extend_from_slice(key);
            names.push(0);
        }
        let mut table = Record::<{ RegistryTable::SIZE }>::default();
        table.set(RegistryTable::size, (RegistryTable::SIZE + entries.len() + names.len()) as u32)?;
        table.set(RegistryTable::numEntries, keys.len() as u32)?;
        let mut data = table.as_bytes().to_vec();
        data.extend_from_slice(&entries);
        data.extend_from_slice(&names);
        self.send(device, NV_VGPU_MSG_FUNCTION_SET_REGISTRY, &data)
    }

    pub(super) fn send(&mut self, device: &Device, function: u32, payload: &[u8]) -> Result<()> {
        let message = rpc::encode(function, self.sequence, payload)?;
        let memory = &self.regions[self.queue];
        let write = memory.read32(COMMAND + 16)? as usize;
        let read = if memory.read32(STATUS + 28)? == 4096 {
            memory.read32(STATUS + memory.read32(STATUS + 24)? as usize)? as usize
        } else {
            0
        };
        let count = memory.read32(COMMAND + 12)? as usize;
        let pages = message.len() / 4096;
        if rpc::available(write, read, count)? < pages {
            return Err(d3os_nvidia::Error::QueueFull.into());
        }
        let (first, second) = rpc::spans(write, pages, count)?;
        memory.copy(COMMAND + 4096 + write * 4096, &message[..first * 4096])?;
        if second > 0 {
            memory.copy(COMMAND + 4096, &message[first * 4096..])?;
        }
        fence(Ordering::SeqCst);
        memory.write32(COMMAND + 16, ((write + pages) % count) as u32)?;
        fence(Ordering::SeqCst);
        self.sequence = self.sequence.wrapping_add(1);
        device.registers.write(0x110c00, 0)
    }

    fn receive(&mut self) -> Result<Option<(u32, u32, Vec<u8>)>> {
        let memory = &self.regions[self.queue];
        if memory.read32(STATUS + 28)? != 4096 {
            return Ok(None);
        }
        let count = memory.read32(STATUS + 12)? as usize;
        if memory.read32(STATUS + 8)? != 4096 || count != QUEUE_SIZE / 4096 - 1 {
            return Err(d3os_nvidia::Error::InvalidQueue.into());
        }
        fence(Ordering::SeqCst);
        let read = memory.read32(COMMAND + QueueHeader::SIZE)? as usize;
        let write = memory.read32(STATUS + 16)? as usize;
        if read >= count || write >= count {
            return Err(d3os_nvidia::Error::InvalidQueue.into());
        }
        if read == write {
            return Ok(None);
        }
        let pages = memory.read32(STATUS + 4096 + read * 4096 + 40)? as usize;
        let (first, second) = rpc::spans(read, pages, count)?;
        let mut bytes = memory.read(STATUS + 4096 + read * 4096, first * 4096)?;
        if second > 0 {
            bytes.extend_from_slice(&memory.read(STATUS + 4096, second * 4096)?);
        }
        let response = rpc::decode(&bytes)?;
        let result = (response.function, response.result, response.payload.to_vec());
        memory.write32(COMMAND + QueueHeader::SIZE, ((read + pages) % count) as u32)?;
        fence(Ordering::SeqCst);
        Ok(Some(result))
    }

    pub(super) fn drain_diagnostics(&mut self) {
        for _ in 0..63 {
            match self.receive() {
                Ok(Some((function, result, payload))) => {
                    log::warn!("NVIDIA queued event function={function:#x} result={result:#x} bytes={}", payload.len());
                    if function == 0x1004 {
                        if let Ok(header) = Record::<{ RcError::SIZE }>::from_bytes(&payload) {
                            log::warn!(
                                "NVIDIA engine fault engine={:?} channel={:?} kind={:?}",
                                header.get(RcError::nv2080EngineType),
                                header.get(RcError::chid),
                                header.get(RcError::exceptType)
                            );
                        }
                    }
                    if function != 0x1020 {
                        log::warn!("NVIDIA event payload {:02x?}", payload);
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    log::warn!("NVIDIA diagnostic queue error: {error}");
                    break;
                }
            }
        }
    }

    pub(super) fn call(&mut self, device: &Device, function: u32, payload: &[u8]) -> Result<Vec<u8>> {
        self.send(device, function, payload)?;
        self.wait_for(device, function)
    }

    pub(super) fn wait_ready(&mut self, device: &Device) -> Result<()> {
        self.wait_for(device, NV_VGPU_MSG_EVENT_GSP_INIT_DONE).map(|_| ())
    }

    fn wait_for(&mut self, device: &Device, expected: u32) -> Result<Vec<u8>> {
        let start = crate::timer().systime_ms();
        loop {
            if let Some((function, result, payload)) = self.receive()? {
                info!("NVIDIA GSP event function={function:#x} result={result:#x} bytes={}", payload.len());
                if function == 0x1020 {
                    log::debug!("NVIDIA NOCAT {:02x?}", payload);
                }
                if result != 0 {
                    return Err(Error::Rpc { function, result });
                }
                if function == expected {
                    return Ok(payload);
                }
                if function == NV_VGPU_MSG_EVENT_OS_ERROR_LOG {
                    let text = payload.get(12..).unwrap_or(&payload);
                    log::warn!("NVIDIA GSP firmware log: {}", core::str::from_utf8(text).unwrap_or("invalid UTF-8"));
                    return Err(Error::Firmware);
                }
                if function == NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER {
                    device.cpu_sequence(&payload, self.libos)?;
                }
                if function == NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED {
                    return Err(Error::Firmware);
                }
            }
            if crate::timer().systime_ms().wrapping_sub(start) >= 15000 {
                log::warn!("NVIDIA GSP timed out waiting for function={expected:#x}");
                return Err(Error::Firmware);
            }
            core::hint::spin_loop();
        }
    }
}

impl Device {
    fn cpu_sequence(&self, payload: &[u8], libos: u64) -> Result<()> {
        let read32 = |offset| -> Result<u32> {
            Ok(u32::from_le_bytes(
                payload
                    .get(offset..offset + 4)
                    .ok_or(Error::Firmware)?
                    .try_into()
                    .map_err(|_| Error::Firmware)?,
            ))
        };
        let count = read32(4)? as usize;
        if count > 4096 || 40 + count * 4 > payload.len() {
            return Err(Error::Firmware);
        }
        let mut cursor = 0usize;
        let mut next = || -> Result<u32> {
            if cursor >= count {
                return Err(Error::Firmware);
            }
            let value = read32(40 + cursor * 4)?;
            cursor += 1;
            Ok(value)
        };
        let mut saved = [0u32; 8];
        loop {
            let op = match next() {
                Ok(op) => op,
                Err(_) => break,
            };
            match op {
                0 => {
                    let offset = next()? as usize;
                    self.registers.write(offset, next()?)?;
                }
                1 => {
                    let offset = next()? as usize;
                    let value = next()?;
                    let mask = next()?;
                    let old = self.registers.read(offset).ok_or(Error::Register(offset))?;
                    self.registers.write(offset, (old & !mask) | (value & mask))?;
                }
                2 => {
                    let offset = next()? as usize;
                    let mask = next()?;
                    let value = next()?;
                    next()?;
                    next()?;
                    self.wait_register(offset, mask, value)?;
                }
                3 => {
                    crate::timer().wait((next()? as usize).div_ceil(1000));
                }
                4 => {
                    let offset = next()? as usize;
                    let index = next()? as usize;
                    *saved.get_mut(index).ok_or(Error::Firmware)? = self.registers.read(offset).ok_or(Error::Register(offset))?;
                }
                5 => {
                    self.reset_falcon(0x110000, false)?;
                    self.disable_context(0x110000)?;
                }
                6 => self.start_falcon(0x110000)?,
                7 => self.wait_register(0x110100, 1 << 4, 1 << 4)?,
                8 => {
                    self.reset_falcon(0x110000, true)?;
                    self.registers.write(0x110040, libos as u32)?;
                    self.registers.write(0x110044, (libos >> 32) as u32)?;
                    self.start_falcon(0x840000)?;
                    self.wait_register(0x1180f8, 1 << 26, 1 << 26)?;
                    let value = self.registers.read(0x840040).ok_or(Error::Firmware)?;
                    if value != 0 {
                        return Err(Error::Rpc { function: op, result: value });
                    }
                }
                _ => return Err(Error::Firmware),
            }
        }
        Ok(())
    }

    pub(super) fn enable_dma(&self) -> Result<()> {
        let bus = pci_bus();
        let devices = bus.search_by_ids(super::VENDOR, super::TARGET);
        let header = devices.first().ok_or(Error::Firmware)?;
        header
            .write()
            .update_command(bus.config_space(), |command| command | CommandRegister::BUS_MASTER_ENABLE);
        let _ = unsafe { bus.config_space().read(self.address, 4) };
        Ok(())
    }
}

pub(super) fn asset(name: &str) -> Result<&'static [u8]> {
    crate::initrd()
        .entries()
        .find(|entry| entry.filename().as_str() == Ok(name))
        .map(|entry| entry.data())
        .ok_or(Error::Firmware)
}
