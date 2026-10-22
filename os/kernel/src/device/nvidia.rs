//! Native NVIDIA device discovery and register access.
//!
//! Register names follow NVIDIA's published Ada/Ampere hardware headers.
//! Device presence is distinct from firmware readiness and compute availability.

use core::ptr::NonNull;
use log::{info, warn};
use pci_types::{Bar, CommandRegister};
use spin::{Mutex, Once};
use x86_64::structures::paging::PageTableFlags;

use crate::memory::vma::VmaType;
use crate::device::pci::EcamFunction;
use crate::{pci_bus, process_manager};

const VENDOR: u16 = 0x10de;
const TARGET: u16 = 0x2757;
const PMC_BOOT_0: usize = 0;
const PMC_BOOT_42: usize = 0xa00;
const SECURE_BOOT_PROGRESS: usize = 0x118234;
const VRAM_SIZE_MIB: usize = 0x1183a4;
const WPR2_ADDR_HI: usize = 0x1fa828;

static DEVICE: Once<Mutex<Device>> = Once::new();

struct Mmio {
    base: NonNull<u32>,
    bytes: usize,
}

// Access is serialized by DEVICE; the mapped register aperture outlives it.
unsafe impl Send for Mmio {}

impl Mmio {
    fn map(address: u64, bytes: u64, tag: &str) -> Option<Self> {
        if address == 0 || address & 0xfff != 0 || bytes == 0 {
            return None;
        }
        let end = address.checked_add(bytes)?;
        let kernel = process_manager().read().kernel_process()?;
        kernel.virtual_address_space.kernel_map_devm_identity(
            address,
            end,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_CACHE | PageTableFlags::NO_EXECUTE,
            VmaType::DeviceMemory,
            tag,
        );
        Some(Self {
            base: NonNull::new(address as *mut u32)?,
            bytes: bytes.try_into().ok()?,
        })
    }

    fn read(&self, offset: usize) -> Option<u32> {
        if offset & 3 != 0 || offset.checked_add(4)? > self.bytes {
            return None;
        }
        // The BAR is mapped as uncached device memory and accesses are aligned.
        Some(unsafe { self.base.as_ptr().add(offset / 4).read_volatile() })
    }
}

struct Device {
    registers: Mmio,
}

fn memory_bar(bar: Bar) -> Option<(u64, u64)> {
    match bar {
        Bar::Memory32 { address, size, .. } => Some((u64::from(address), u64::from(size))),
        Bar::Memory64 { address, size, .. } => Some((address, size)),
        Bar::Io { .. } => None,
    }
}

pub fn init() {
    let bus = pci_bus();
    let matches = bus.search_by_ids(VENDOR, TARGET);
    let Some(header) = matches.first() else { return };
    let mut header = header.write();
    let address = header.header().address();
    // Keep bus mastering off until the firmware DMA buffers and queues exist.
    header.update_command(bus.config_space(), |command| {
        (command | CommandRegister::MEMORY_ENABLE | CommandRegister::INTERRUPT_DISABLE) & !CommandRegister::BUS_MASTER_ENABLE
    });
    let Some((bar0, size0)) = header.bar(0, bus.config_space()).and_then(memory_bar) else {
        warn!("NVIDIA BAR0 is unavailable");
        return;
    };
    let Some((bar1, size1)) = header.bar(1, bus.config_space()).and_then(memory_bar) else {
        warn!("NVIDIA BAR1 is unavailable");
        return;
    };
    let Some(registers) = Mmio::map(bar0, size0, "nvidia-registers") else {
        warn!("NVIDIA register aperture is invalid");
        return;
    };
    let device = Device { registers };
    let Some(boot0) = device.registers.read(PMC_BOOT_0).filter(|v| *v != u32::MAX && *v != 0) else {
        warn!("NVIDIA BAR0 does not respond");
        return;
    };
    let details = device.registers.read(PMC_BOOT_42).unwrap();
    let architecture = (details >> 24) & 0x3f;
    let implementation = (details >> 20) & 0xf;
    let vram = device.registers.read(VRAM_SIZE_MIB).unwrap();
    let progress = device.registers.read(SECURE_BOOT_PROGRESS).unwrap();
    let wpr2 = device.registers.read(WPR2_ADDR_HI).unwrap();
    if let Some(ecam) = EcamFunction::map(address) {
        let id = ecam.read(0).unwrap();
        let extended = ecam.read(0x100).unwrap();
        info!("NVIDIA PCIe configuration: id={id:#010x} extended_capability={extended:#010x}");
    } else {
        warn!("NVIDIA PCIe extended configuration is unavailable");
    }
    info!(
        "NVIDIA_PROBE_OK pci={:02x}:{:02x}.{} boot0={boot0:#010x} architecture={architecture:#x} implementation={implementation} vram_mib={vram}",
        address.bus(),
        address.device(),
        address.function()
    );
    info!("NVIDIA apertures: BAR0={bar0:#x}+{size0:#x} BAR1={bar1:#x}+{size1:#x}");
    info!("NVIDIA firmware state: boot_progress={progress:#010x} wpr2={wpr2:#010x}; compute is not initialized");
    DEVICE.call_once(|| Mutex::new(device));
}
