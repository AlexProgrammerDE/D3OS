//! Ada's authenticated Falcon firmware loader.
//!
//! Based on NVIDIA's published Falcon registers and the tinygrad direct-PCI
//! boot sequence. See the reference license in os/library/nvidia/LICENSE.tinygrad.

use super::{Device, Error, Result};
use d3os_nvidia::{Record, abi::FalconDescriptor, firmware::Booter};

const GSP: usize = 0x110000;
const CPUCTL: usize = 0x100;
const HWCFG2: usize = 0xf4;
const BCR_CTRL: usize = 0x1668;

impl Device {
    pub(super) fn wait_register(&self, offset: usize, mask: u32, expected: u32) -> Result<()> {
        let start = crate::timer().systime_ms();
        loop {
            let value = self.registers.read(offset).ok_or(Error::Register(offset))?;
            if value & mask == expected {
                return Ok(());
            }
            if crate::timer().systime_ms().wrapping_sub(start) >= 5000 {
                return Err(Error::Timeout { offset, value, mask, expected });
            }
            core::hint::spin_loop();
        }
    }

    fn write_register(&self, offset: usize, value: u32) -> Result<()> {
        self.registers.write(offset, value)
    }

    fn update_register(&self, offset: usize, mask: u32, value: u32) -> Result<()> {
        let old = self.registers.read(offset).ok_or(Error::Register(offset))?;
        self.write_register(offset, (old & !mask) | (value & mask))
    }

    pub(super) fn reset_falcon(&self, base: usize, riscv: bool) -> Result<()> {
        self.write_register(base + 0x3c0, 1)?;
        crate::timer().wait(100);
        self.write_register(base + 0x3c0, 0)?;
        self.wait_register(base + HWCFG2, 1 << 12, 0)?;
        if riscv {
            self.write_register(base + BCR_CTRL, (1 << 4) | (1 << 8))?;
        } else if self.registers.read(base + HWCFG2).ok_or(Error::Register(base + HWCFG2))? & (1 << 10) != 0 {
            self.write_register(base + BCR_CTRL, 0)?;
            self.wait_register(base + BCR_CTRL, 1, 1)?;
            self.write_register(base + 0x84, self.boot0)?;
        }
        Ok(())
    }

    fn transfer_falcon(&self, base: usize, command: u32, destination: u32, source_offset: u32, source: u64, bytes: u32) -> Result<()> {
        if source & 255 != 0 {
            return Err(Error::Alignment);
        }
        self.wait_register(base + 0x118, 1, 0)?;
        self.write_register(base + 0x110, (source >> 8) as u32)?;
        self.write_register(base + 0x128, ((source >> 40) & 0x1ff) as u32)?;
        for transferred in (0..bytes).step_by(256) {
            self.wait_register(base + 0x118, 1, 0)?;
            self.write_register(base + 0x114, destination.checked_add(transferred).ok_or(Error::Alignment)?)?;
            self.write_register(base + 0x11c, source_offset.checked_add(transferred).ok_or(Error::Alignment)?)?;
            self.write_register(base + 0x118, command)?;
        }
        self.wait_register(base + 0x118, 2, 2)
    }

    pub(super) fn disable_context(&self, base: usize) -> Result<()> {
        self.update_register(base + 0x624, 1 << 7, 1 << 7)?;
        self.write_register(base + 0x10c, 0)
    }

    pub(super) fn start_falcon(&self, base: usize) -> Result<()> {
        let control = self.registers.read(base + CPUCTL).ok_or(Error::Register(base + CPUCTL))?;
        self.write_register(base + if control & (1 << 6) != 0 { 0x130 } else { CPUCTL }, 2)
    }

    pub(super) fn run_gsp_booter(&self, booter: &Booter, image: u64, libos: u64, metadata: u64) -> Result<()> {
        const SEC2: usize = 0x840000;
        self.reset_falcon(GSP, true)?;
        self.write_register(GSP + 0x40, libos as u32)?;
        self.write_register(GSP + 0x44, (libos >> 32) as u32)?;
        self.reset_falcon(SEC2, false)?;
        self.disable_context(SEC2)?;
        self.update_register(SEC2 + 0x600, 7, 4)?;
        self.transfer_falcon(SEC2, 0x614, 0, booter.code_offset, image, booter.code_size)?;
        self.transfer_falcon(SEC2, 0x600, 0, 0, image + u64::from(booter.data_offset), booter.data_size)?;
        self.write_register(SEC2 + 0x1210, 0x10)?;
        self.write_register(SEC2 + 0x119c, 1)?;
        self.write_register(SEC2 + 0x1198, 3)?;
        self.write_register(SEC2 + 0x1180, 1)?;
        self.write_register(SEC2 + 0x104, booter.code_offset)?;
        self.write_register(SEC2 + 0x40, metadata as u32)?;
        self.write_register(SEC2 + 0x44, (metadata >> 32) as u32)?;
        self.start_falcon(SEC2)?;
        self.wait_register(SEC2 + CPUCTL, 1 << 4, 1 << 4)?;
        let status = self.registers.read(SEC2 + 0x40).ok_or(Error::Firmware)?;
        if status != 0 {
            return Err(Error::Rpc { function: 3, result: status });
        }
        self.write_register(GSP + 0x80, 0)?;
        self.wait_register(GSP + 0x1388, 1 << 7, 1 << 7)
    }

    pub(super) fn run_fwsec(&self, descriptor: &Record<{ FalconDescriptor::SIZE }>, image: u64) -> Result<()> {
        self.reset_falcon(GSP, false)?;
        self.disable_context(GSP)?;
        // FB target, physical addressing, context DMA slot zero.
        self.update_register(GSP + 0x600, 7, 4)?;
        let imem_virtual = descriptor.get(FalconDescriptor::IMEMVirtBase)?;
        let imem_size = descriptor.get(FalconDescriptor::IMEMLoadSize)?;
        self.transfer_falcon(
            GSP,
            0x614,
            descriptor.get(FalconDescriptor::IMEMPhysBase)?,
            imem_virtual,
            image.checked_sub(u64::from(imem_virtual)).ok_or(Error::Alignment)?,
            imem_size,
        )?;
        self.transfer_falcon(
            GSP,
            0x600,
            descriptor.get(FalconDescriptor::DMEMPhysBase)?,
            0,
            image + u64::from(imem_size),
            descriptor.get(FalconDescriptor::DMEMLoadSize)?,
        )?;
        self.write_register(GSP + 0x1210, descriptor.get(FalconDescriptor::PKCDataOffset)?)?;
        self.write_register(GSP + 0x119c, u32::from(descriptor.get(FalconDescriptor::EngineIdMask)?))?;
        self.write_register(GSP + 0x1198, u32::from(descriptor.get(FalconDescriptor::UcodeId)?))?;
        self.write_register(GSP + 0x1180, 1)?;
        self.write_register(GSP + 0x104, imem_virtual)?;
        self.start_falcon(GSP)?;
        self.wait_register(GSP + CPUCTL, 1 << 4, 1 << 4)
    }
}
