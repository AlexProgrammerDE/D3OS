//! Native NVIDIA device discovery and register access.
//!
//! Register names follow NVIDIA's published Ada/Ampere hardware headers.
//! Device presence is distinct from firmware readiness and compute availability.

use alloc::vec::Vec;
use core::ptr::NonNull;
use d3os_nvidia::firmware;

mod compute;
mod flcn;
mod gsp;
mod mmu;
mod rm;
mod service;
pub(crate) use service::{request as compute_request, release_process};

#[derive(Debug)]
enum Error {
    Protocol(d3os_nvidia::Error),
    Register(usize),
    Alignment,
    Firmware,
    Rpc { function: u32, result: u32 },
    Timeout { offset: usize, value: u32, mask: u32, expected: u32 },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Protocol(error) => write!(f, "firmware protocol error: {error:?}"),
            Self::Register(offset) => write!(f, "invalid register or state at {offset:#x}"),
            Self::Firmware => f.write_str("invalid firmware state or boot data"),
            Self::Rpc { function, result } => write!(f, "GSP RPC {function:#x} failed: {result:#x}"),
            Self::Alignment => f.write_str("firmware buffer address or size is invalid"),
            Self::Timeout { offset, value, mask, expected } => {
                write!(f, "register timeout at {offset:#x}: value={value:#x} mask={mask:#x} expected={expected:#x}")
            }
        }
    }
}

impl From<d3os_nvidia::Error> for Error {
    fn from(value: d3os_nvidia::Error) -> Self {
        Self::Protocol(value)
    }
}
type Result<T> = core::result::Result<T, Error>;
use log::{info, warn};
use pci_types::{Bar, CommandRegister};
use spin::{Mutex, Once};
use x86_64::structures::paging::PageTableFlags;

use crate::device::pci::EcamFunction;
use crate::memory::vma::VmaType;
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

    fn write(&self, offset: usize, value: u32) -> Result<()> {
        if offset & 3 != 0 || offset.checked_add(4).is_none_or(|end| end > self.bytes) {
            return Err(Error::Register(offset));
        }
        unsafe {
            self.base.as_ptr().add(offset / 4).write_volatile(value);
        }
        Ok(())
    }

    fn copy_from(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        if offset & 3 != 0 || bytes.len() & 3 != 0 || offset.checked_add(bytes.len()).is_none_or(|end| end > self.bytes) {
            return Err(Error::Alignment);
        }
        for (i, word) in bytes.chunks_exact(4).enumerate() {
            self.write(offset + i * 4, u32::from_le_bytes(word.try_into().map_err(|_| Error::Alignment)?))?;
        }
        Ok(())
    }

    fn read_bytes(&self, offset: usize, size: usize) -> Result<Vec<u8>> {
        if offset & 3 != 0 || size & 3 != 0 || offset.checked_add(size).is_none_or(|end| end > self.bytes) {
            return Err(Error::Alignment);
        }
        let mut bytes = Vec::with_capacity(size);
        for i in (offset..offset + size).step_by(4) {
            bytes.extend_from_slice(&self.read(i).ok_or(Error::Register(i))?.to_le_bytes());
        }
        Ok(bytes)
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
    address: pci_types::PciAddress,
    bar3: u64,
    gsp: Option<gsp::Gsp>,
    compute: Option<rm::Compute>,
    boot0: u32,
    vram_address: u64,
    vram_aperture: u64,
    vram_size: u64,
    vram: Option<Mmio>,
    fwsec_started: bool,
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
    let bar3 = header.bar(3, bus.config_space()).and_then(memory_bar).map(|(address, _)| address).unwrap_or(0);
    let Some(registers) = Mmio::map(bar0, size0, "nvidia-registers") else {
        warn!("NVIDIA register aperture is invalid");
        return;
    };
    let Some(boot0) = registers.read(PMC_BOOT_0).filter(|v| *v != u32::MAX && *v != 0) else {
        warn!("NVIDIA BAR0 does not respond");
        return;
    };
    let details = registers.read(PMC_BOOT_42).unwrap();
    let architecture = (details >> 24) & 0x3f;
    let implementation = (details >> 20) & 0xf;
    let vram = registers.read(VRAM_SIZE_MIB).unwrap();
    let progress = registers.read(SECURE_BOOT_PROGRESS).unwrap();
    let wpr2 = registers.read(WPR2_ADDR_HI).unwrap();
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
    DEVICE.call_once(|| {
        Mutex::new(Device {
            registers,
            address,
            bar3,
            gsp: None,
            compute: None,
            boot0,
            vram_address: bar1,
            vram_aperture: size1,
            vram_size: u64::from(vram) << 20,
            vram: None,
            fwsec_started: false,
        })
    });
}

pub fn bootstrap() {
    let Some(device) = DEVICE.get() else { return };
    let mut device = device.lock();
    if device.fwsec_started {
        return;
    }
    device.fwsec_started = true;
    let result = (|| -> Result<()> {
        let bios = device.registers.read_bytes(0x300000, 0x100000)?;
        let frts = device.vram_size.checked_sub(2 << 20).ok_or(Error::Alignment)?;
        let fwsec = firmware::fwsec(&bios, frts)?;
        info!("NVIDIA_FWSEC_PREPARED bytes={} frts={frts:#x}", fwsec.image.len());
        let vram = Mmio::map(device.vram_address, device.vram_aperture, "nvidia-vram").ok_or(Error::Alignment)?;
        vram.copy_from(2 << 20, &fwsec.image)?;
        device.vram = Some(vram);
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        device.run_fwsec(&fwsec.descriptor, 2 << 20)?;
        let wpr = device.registers.read(WPR2_ADDR_HI).ok_or(Error::Register(WPR2_ADDR_HI))?;
        if wpr == 0 {
            return Err(Error::Register(WPR2_ADDR_HI));
        }
        info!("NVIDIA_FWSEC_BOOT_OK wpr2={wpr:#x}; GSP and compute queues are not initialized");
        let Ok(image) = gsp::asset("usr/share/nvidia/gsp-570.144.bin") else {
            info!("NVIDIA GSP assets are absent from the initrd");
            return Ok(());
        };
        let loader = gsp::asset("usr/share/nvidia/bootloader-570.144.bin")?;
        let booter = firmware::signed_booter(gsp::asset("usr/share/nvidia/booter_load-570.144.bin")?)?;
        let gsp = gsp::Gsp::prepare(&device, image, loader)?;
        let (libos, metadata) = (gsp.libos, gsp.metadata);
        device.vram.as_ref().ok_or(Error::Firmware)?.copy_from(4 << 20, &booter.image)?;
        // Keep all DMA buffers alive even if firmware boot subsequently fails.
        device.gsp = Some(gsp);
        device.enable_dma()?;
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        device.run_gsp_booter(&booter, 4 << 20, libos, metadata)?;
        let mut gsp = device.gsp.take().ok_or(Error::Firmware)?;
        let result = gsp.wait_ready(&device);
        device.gsp = Some(gsp);
        result?;
        info!("NVIDIA_GSP_BOOT_OK; management firmware is running, compute is not initialized");
        info!(
            "NVIDIA BAR1 mapping before compute={:#x}",
            device.registers.read(0x1704).ok_or(Error::Firmware)?
        );
        device.registers.write(0x1704, 0)?;
        let mut compute = rm::Compute::new(&device)?;
        let result = compute.initialize(&mut device).and_then(|()| {
            compute.queue_test(&device)?;
            if let Ok(cubin) = gsp::asset("usr/share/nvidia/vector-add.cubin") {
                compute.vector_add_test(&device, cubin)?;
            }
            if let Ok(cubin) = gsp::asset("usr/share/nvidia/tensor-matmul.cubin") {
                compute.tensor_test(&device, cubin)?;
            }
            Ok(())
        });
        compute.healthy = result.is_ok();
        device.compute = Some(compute);
        if result.is_err() {
            if let Some(gsp) = device.gsp.as_mut() {
                gsp.drain_diagnostics();
            }
        }
        result?;
        Ok(())
    })();
    if let Err(error) = result {
        warn!("NVIDIA initialization failed: {error}");
    }
}
