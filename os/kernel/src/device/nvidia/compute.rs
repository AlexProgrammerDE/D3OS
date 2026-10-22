//! SASS loading, QMD construction, and submission to the native compute FIFO.

use super::{Device, Error, Result, rm::Compute};
use alloc::vec::Vec;
use core::sync::atomic::{Ordering, fence};
use d3os_nvidia::{Record, abi::*};
use goblin::elf::{
    Elf,
    section_header::{SHF_ALLOC, SHT_PROGBITS},
};
use log::info;

impl Compute {
    fn method(commands: &mut Vec<u32>, method: u32, values: &[u32]) {
        commands.push((2 << 28) | ((values.len() as u32) << 16) | (1 << 13) | (method >> 2));
        commands.extend_from_slice(values);
    }

    fn submit(&mut self, device: &Device, commands: &[u32]) -> Result<()> {
        if commands.is_empty() || commands.len() > 1024 {
            return Err(Error::Alignment);
        }
        let allocation = if let Some(allocation) = self.command_buffer {
            allocation
        } else {
            let allocation = self.memory.allocate(device, 4096, 4096)?;
            self.command_buffer = Some(allocation);
            allocation
        };
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        for (i, word) in commands.iter().copied().enumerate() {
            vram.write(allocation.physical as usize + i * 4, word)?;
        }
        let entry = allocation.virtual_address | (1 << 41) | ((commands.len() as u64) << 42);
        let slot = self.put % self.entries;
        vram.write(self.fifo.physical as usize + slot * 8, entry as u32)?;
        vram.write(self.fifo.physical as usize + slot * 8 + 4, (entry >> 32) as u32)?;
        fence(Ordering::SeqCst);
        self.put = (self.put + 1) % self.entries;
        vram.write(self.fifo.physical as usize + self.entries * 8 + 0x8c, self.put as u32)?;
        fence(Ordering::SeqCst);
        device.registers.write(0xbb0090, self.token)
    }

    pub(super) fn queue_test(&mut self, device: &Device) -> Result<()> {
        let signal = self.memory.allocate(device, 4096, 4096)?;
        for value in [0x12345678u32, 0x87654321, 0xd305c0de] {
            self.submit(
                device,
                &[
                    (2 << 28) | (5 << 16) | (0x5c >> 2),
                    signal.virtual_address as u32,
                    (signal.virtual_address >> 32) as u32,
                    value,
                    0,
                    1 | (1 << 24) | (1 << 20),
                ],
            )?;
            let start = crate::timer().systime_ms();
            while device.vram.as_ref().ok_or(Error::Firmware)?.read(signal.physical as usize) != Some(value) {
                if crate::timer().systime_ms().wrapping_sub(start) >= 5000 {
                    return Err(Error::Firmware);
                }
                core::hint::spin_loop();
            }
            fence(Ordering::SeqCst);
        }
        info!("NVIDIA_QUEUE_OK submissions=3 backend=nvidia");
        Ok(())
    }

    pub(super) fn vector_add_test(&mut self, device: &Device, cubin: &[u8]) -> Result<()> {
        let elf = Elf::parse(cubin).map_err(|_| Error::Firmware)?;
        let mut image = Vec::new();
        let mut entry = None;
        let mut registers = 0;
        for section in &elf.section_headers {
            if section.sh_type != SHT_PROGBITS || section.sh_flags & u64::from(SHF_ALLOC) == 0 {
                continue;
            }
            let alignment = section.sh_addralign.max(128) as usize;
            if !alignment.is_power_of_two() {
                return Err(Error::Firmware);
            }
            let offset = image.len().checked_add(alignment - 1).ok_or(Error::Firmware)? & !(alignment - 1);
            let bytes = cubin
                .get(section.sh_offset as usize..section.sh_offset.checked_add(section.sh_size).ok_or(Error::Firmware)? as usize)
                .ok_or(Error::Firmware)?;
            image.resize(offset, 0);
            image.extend_from_slice(bytes);
            if elf.shdr_strtab.get_at(section.sh_name) == Some(".text.vector_add") {
                entry = Some((offset as u64, section.sh_size));
                registers = (section.sh_info >> 24) as u64;
            }
        }
        let (entry, code_bytes) = entry.ok_or(Error::Firmware)?;
        let program = self.memory.allocate(device, image.len() as u64 + 4096, 4096)?;
        while image.len() & 3 != 0 {
            image.push(0);
        }
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        vram.copy_from(program.physical as usize, &image)?;
        let data = self.memory.allocate(device, 4 * 4096, 4096)?;
        let launch = self.memory.allocate(device, 4096, 4096)?;
        const COUNT: u32 = 257;
        for i in 0..COUNT {
            vram.write((data.physical + u64::from(i) * 4) as usize, (i as f32 * 0.25).to_bits())?;
            vram.write((data.physical + 4096 + u64::from(i) * 4) as usize, (i as f32 * -0.125 + 3.0).to_bits())?;
            vram.write((data.physical + 8192 + u64::from(i) * 4) as usize, f32::NAN.to_bits())?;
        }
        let mut constants = Record::<384>::default();
        use d3os_nvidia::Field;
        for (i, value) in [128, 1, 1, COUNT.div_ceil(128), 1, 1].into_iter().enumerate() {
            constants.set(Field::<u32>::new(i * 4), value)?;
        }
        constants.set(Field::<u64>::new(24), 0x729400000000)?;
        constants.set(Field::<u64>::new(32), 0x729300000000)?;
        constants.set(Field::<u64>::new(40), 0xfffdc0)?;
        constants.set(Field::<u64>::new(0x160), data.virtual_address)?;
        constants.set(Field::<u64>::new(0x168), data.virtual_address + 4096)?;
        constants.set(Field::<u64>::new(0x170), data.virtual_address + 8192)?;
        constants.set(Field::<u32>::new(0x178), COUNT)?;
        vram.copy_from(launch.physical as usize + 256, constants.as_bytes())?;
        let mut qmd = Record::<256>::default();
        use QmdFields as Q;
        for (field, value) in [
            (Q::QMD_MAJOR_VERSION, 3),
            (Q::QMD_GROUP_ID, 0x3f),
            (Q::SM_GLOBAL_CACHING_ENABLE, 1),
            (Q::SHARED_MEMORY_SIZE, 0x400),
            (Q::SHADER_LOCAL_MEMORY_HIGH_SIZE, 0x240),
            (Q::REGISTER_COUNT_V, registers),
            (Q::INVALIDATE_TEXTURE_HEADER_CACHE, 1),
            (Q::INVALIDATE_TEXTURE_SAMPLER_CACHE, 1),
            (Q::INVALIDATE_TEXTURE_DATA_CACHE, 1),
            (Q::INVALIDATE_SHADER_DATA_CACHE, 1),
            (Q::API_VISIBLE_CALL_LIMIT, 1),
            (Q::SAMPLER_INDEX, 1),
            (Q::BARRIER_COUNT, 1),
            (Q::CWD_MEMBAR_TYPE, u64::from(NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR)),
            (Q::MIN_SM_CONFIG_SHARED_MEM_SIZE, 9),
            (Q::TARGET_SM_CONFIG_SHARED_MEM_SIZE, 9),
            (Q::MAX_SM_CONFIG_SHARED_MEM_SIZE, 0x1a),
            (Q::PROGRAM_PREFETCH_SIZE, (code_bytes >> 8).min(0x1ff)),
            (Q::SASS_VERSION, self.sass_version),
            (Q::CTA_RASTER_WIDTH, u64::from(COUNT.div_ceil(128))),
            (Q::CTA_RASTER_HEIGHT, 1),
            (Q::CTA_RASTER_DEPTH, 1),
            (Q::CTA_THREAD_DIMENSION0, 128),
            (Q::CTA_THREAD_DIMENSION1, 1),
            (Q::CTA_THREAD_DIMENSION2, 1),
            (Q::CONSTANT_BUFFER_VALID_0, 1),
            (Q::CONSTANT_BUFFER_INVALIDATE_0, 1),
            (Q::CONSTANT_BUFFER_SIZE_SHIFTED4_0, 384),
            (Q::PROGRAM_ADDRESS_LOWER, (program.virtual_address + entry) & 0xffffffff),
            (Q::PROGRAM_ADDRESS_UPPER, (program.virtual_address + entry) >> 32),
            (Q::PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, ((program.virtual_address + entry) >> 8) & 0xffffffff),
            (Q::PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, (program.virtual_address + entry) >> 40),
            (Q::CONSTANT_BUFFER_ADDR_LOWER_0, (launch.virtual_address + 256) & 0xffffffff),
            (Q::CONSTANT_BUFFER_ADDR_UPPER_0, (launch.virtual_address + 256) >> 32),
            (Q::RELEASE0_ENABLE, 1),
            (Q::RELEASE0_STRUCTURE_SIZE, 2),
            (Q::RELEASE0_PAYLOAD64B, 1),
            (Q::RELEASE0_ADDRESS_LOWER, (data.virtual_address + 12288) & 0xffffffff),
            (Q::RELEASE0_ADDRESS_UPPER, (data.virtual_address + 12288) >> 32),
            (Q::RELEASE0_PAYLOAD_LOWER, 1),
            (Q::RELEASE0_PAYLOAD_UPPER, 0),
        ] {
            qmd.set_bits(field, value)?;
        }
        vram.copy_from(launch.physical as usize, qmd.as_bytes())?;
        let mut commands = Vec::new();
        let semaphore = |commands: &mut Vec<u32>, address: u64, value: u32, wait: bool| {
            commands.extend_from_slice(&[
                (2 << 28) | (5 << 16) | (0x5c >> 2),
                address as u32,
                (address >> 32) as u32,
                value,
                0,
                1 | (1 << 24) | (u32::from(wait) << 20),
            ]);
        };
        semaphore(&mut commands, data.virtual_address + 12296, 0x1111, false);

        Self::method(&mut commands, NVC6C0_SET_OBJECT, &[ADA_COMPUTE_A]);
        semaphore(&mut commands, data.virtual_address + 12304, 0x2000, true);
        Self::method(&mut commands, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, &[0x7293, 0]);
        Self::method(&mut commands, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, &[0x7294, 0]);
        semaphore(&mut commands, data.virtual_address + 12312, 0x2001, true);
        let local = self.local_memory.ok_or(Error::Firmware)?;
        Self::method(
            &mut commands,
            NVC6C0_SET_SHADER_LOCAL_MEMORY_A,
            &[(local.virtual_address >> 32) as u32, local.virtual_address as u32],
        );
        Self::method(
            &mut commands,
            NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A,
            &[(self.local_bytes_per_tpc >> 32) as u32, self.local_bytes_per_tpc as u32, 0xff],
        );
        semaphore(&mut commands, data.virtual_address + 12320, 0x2002, true);
        Self::method(&mut commands, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, &[0x1011]);
        semaphore(&mut commands, data.virtual_address + 12328, 0x2003, true);
        Self::method(&mut commands, NVC6C0_SEND_PCAS_A, &[(launch.virtual_address >> 8) as u32]);
        semaphore(&mut commands, data.virtual_address + 12336, 0x2004, true);
        Self::method(
            &mut commands,
            NVC6C0_SEND_SIGNALING_PCAS2_B,
            &[NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE],
        );
        semaphore(&mut commands, data.virtual_address + 12344, 0x2222, true);
        info!("NVIDIA VRAM mode before submit={:?}", device.registers.read(0x1704));
        self.submit(device, &commands)?;
        info!(
            "NVIDIA_KERNEL_SUBMITTED qmd={:#x} entry={:#x}",
            launch.virtual_address,
            program.virtual_address + entry
        );
        let start = crate::timer().systime_ms();
        while vram.read(data.physical as usize + 12288) != Some(1) {
            if crate::timer().systime_ms().wrapping_sub(start) >= 5000 {
                log::warn!(
                    "NVIDIA kernel timeout GPGet={:?} GPPut={:?}",
                    vram.read(self.fifo.physical as usize + self.entries * 8 + 0x88),
                    vram.read(self.fifo.physical as usize + self.entries * 8 + 0x8c)
                );
                for stage in 0..7 {
                    info!("NVIDIA FIFO stage={stage} value={:?}", vram.read(data.physical as usize + 12296 + stage * 8));
                }
                info!("NVIDIA notifier {:02x?}", vram.read_bytes(self.notifier as usize, 0x100)?);
                info!(
                    "NVIDIA output first={:?} last={:?}",
                    vram.read(data.physical as usize + 8192),
                    vram.read(data.physical as usize + 8192 + (COUNT as usize - 1) * 4)
                );
                for offset in [
                    0x400100, 0x400700, 0x409800, 0x409804, 0x409808, 0x40980c, 0x409810, 0x409814, 0x409818, 0x40981c, 0x1fa100,
                ] {
                    info!("NVIDIA diagnostic reg={offset:#x} value={:?}", device.registers.read(offset));
                }
                return Err(Error::Firmware);
            }
            core::hint::spin_loop();
        }
        fence(Ordering::SeqCst);
        for i in 0..COUNT {
            let actual = vram.read((data.physical + 8192 + u64::from(i) * 4) as usize).ok_or(Error::Firmware)?;
            let expected = i as f32 * 0.125 + 3.0;
            if actual != expected.to_bits() {
                log::warn!("NVIDIA vector-add mismatch index={i} actual={:#x} expected={:#x}", actual, expected.to_bits());
                return Err(Error::Firmware);
            }
        }
        info!("NVIDIA_KERNEL_OK vector-add count={COUNT} backend=nvidia");
        Ok(())
    }
}
