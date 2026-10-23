//! SASS loading, QMD construction, and submission to the native compute FIFO.

use super::{Device, Error, Result, gsp::DmaRegion, rm::Compute};
use alloc::vec::Vec;
use core::sync::atomic::{Ordering, fence};
use d3os_nvidia::{Field, Record, abi::*};
use goblin::elf::{
    Elf,
    section_header::{SHF_ALLOC, SHT_PROGBITS},
};
use log::{info, warn};

#[derive(Clone)]
pub(super) struct Program {
    registers: u64,
    local_bytes: u64,
    required_threads: Option<[u32; 3]>,
    source: (usize, usize),
    // Launch addresses and dependency links belong only to submission copies.
    qmd: Record<256>,
}

// Each slot owns its QMD, constants, and completion cache line. A dependency
// chain completes before any slot or retired buffer can be reused.
const SLOTS: usize = 32;
const SLOT_BYTES: usize = 2048;
const CONSTANT_BYTES: usize = 1024;
const SIGNAL: usize = 1536;
const COMMANDS: usize = SLOTS * SLOT_BYTES;

struct Pending {
    qmd: Record<256>,
    symbol: &'static str,
    dimensions: [u32; 2],
}

pub(super) struct Submission {
    region: DmaRegion,
    address: u64,
    pending: Vec<Pending>,
    invalidate_code: bool,
}

impl Compute {
    pub(super) fn ensure_local_memory(&mut self, device: &Device, required: u64) -> Result<()> {
        if self.local_bytes_per_thread >= required {
            return Ok(());
        }
        let per_thread = required.checked_add(31).ok_or(Error::Firmware)? & !31;
        if per_thread >= 1 << 24 || self.local_warps_per_tpc == 0 || self.local_tpc_count == 0 {
            return Err(Error::Firmware);
        }
        let per_tpc = per_thread
            .checked_mul(32)
            .and_then(|v| v.checked_mul(self.local_warps_per_tpc))
            .and_then(|v| v.checked_add(0x7fff))
            .ok_or(Error::Firmware)?
            & !0x7fff;
        let bytes = per_tpc
            .checked_mul(self.local_tpc_count)
            .and_then(|v| v.checked_add(0x1ffff))
            .ok_or(Error::Firmware)?
            & !0x1ffff;
        // Driver-private scratch is never exposed through a userspace handle.
        // Valid shaders initialize their local values before reading them.
        let allocation = self.memory.allocate_buffer(device, bytes)?;
        self.local_memory = Some(allocation);
        self.local_bytes_per_thread = per_thread;
        self.local_bytes_per_tpc = per_tpc;
        self.engine_configured = false;
        info!("NVIDIA_LOCAL_MEMORY_READY per_thread={per_thread} per_tpc={per_tpc} bytes={bytes}");
        Ok(())
    }

    pub(super) fn prepare_submission(&mut self, device: &Device) -> Result<()> {
        let bytes = COMMANDS + 4096;
        let region = DmaRegion::new(bytes)?;
        let physical = region.address as u64;
        // Retain DMA ownership even if installing the mapping later fails.
        self.submission = Some(Submission {
            region,
            address: 0,
            pending: Vec::with_capacity(SLOTS),
            invalidate_code: false,
        });
        let address = self.memory.map_system(device, physical, bytes as u64)?;
        self.submission.as_mut().ok_or(Error::Firmware)?.address = address;
        info!("NVIDIA_SUBMISSION_READY physical={physical:#x} virtual={address:#x} bytes={bytes} slots={SLOTS}");
        Ok(())
    }

    fn method(commands: &mut Vec<u32>, method: u32, values: &[u32]) {
        commands.push((2 << 28) | ((values.len() as u32) << 16) | (1 << 13) | (method >> 2));
        commands.extend_from_slice(values);
    }

    fn submit(&mut self, device: &Device, commands: &[u32]) -> Result<()> {
        if commands.is_empty() || commands.len() > 1024 {
            return Err(Error::Alignment);
        }
        let submission = self.submission.as_ref().ok_or(Error::Firmware)?;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        for (i, word) in commands.iter().copied().enumerate() {
            submission.region.write32(COMMANDS + i * 4, word)?;
        }
        fence(Ordering::SeqCst);
        let entry = (submission.address + COMMANDS as u64) | (1 << 41) | ((commands.len() as u64) << 42);
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

    fn load_program(&mut self, device: &Device, cubin: &'static [u8], symbol: &'static str) -> Result<Program> {
        let source = (cubin.as_ptr() as usize, cubin.len());
        if let Some(program) = self.programs.get(symbol) {
            if program.source != source {
                return Err(Error::Firmware);
            }
            return Ok(program.clone());
        }
        let elf = Elf::parse(cubin).map_err(|_| Error::Firmware)?;
        let mut image = Vec::new();
        let mut entry = None;
        let mut registers = 0;
        let mut stack_bytes = 0;
        let mut shared_bytes = 0x400u64;
        let mut required_threads = None;
        let mut constants = [None; 8];
        for section in &elf.section_headers {
            if elf.shdr_strtab.get_at(section.sh_name).and_then(|name| name.strip_prefix(".nv.info.")) == Some(symbol) {
                let metadata = cubin
                    .get(section.sh_offset as usize..section.sh_offset.checked_add(section.sh_size).ok_or(Error::Firmware)? as usize)
                    .ok_or(Error::Firmware)?;
                required_threads = d3os_nvidia::cubin::required_threads(metadata).map_err(|_| Error::Firmware)?;
            }
            if elf.shdr_strtab.get_at(section.sh_name).and_then(|name| name.strip_prefix(".nv.shared.")) == Some(symbol) {
                // CUDA reserves a 1 KiB shared-memory prefix. The static section
                // is NOBITS and needs allocation, rather than a VRAM image copy.
                shared_bytes = u64::from(d3os_nvidia::cubin::shared_bytes(section.sh_size).map_err(|_| Error::Firmware)?);
            }
            if elf.shdr_strtab.get_at(section.sh_name) == Some(".nv.info") {
                let metadata = cubin
                    .get(section.sh_offset as usize..section.sh_offset.checked_add(section.sh_size).ok_or(Error::Firmware)? as usize)
                    .ok_or(Error::Firmware)?;
                stack_bytes = stack_bytes.max(d3os_nvidia::cubin::stack_bytes(metadata).map_err(|_| Error::Firmware)?);
            }
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
            if let Some(name) = elf.shdr_strtab.get_at(section.sh_name).and_then(|name| name.strip_prefix(".nv.constant")) {
                let (bank, scope) = name.split_once('.').map_or((name, None), |(bank, scope)| (bank, Some(scope)));
                if let Ok(bank) = bank.parse::<usize>() {
                    if (1..8).contains(&bank) && scope.is_none_or(|scope| scope == symbol) {
                        let size = u32::try_from(section.sh_size).map_err(|_| Error::Firmware)?;
                        if size > 65536 {
                            return Err(Error::Firmware);
                        }
                        constants[bank] = Some((offset as u64, size));
                    }
                }
            }
            if elf.shdr_strtab.get_at(section.sh_name).and_then(|name| name.strip_prefix(".text.")) == Some(symbol) {
                entry = Some((offset as u64, section.sh_size));
                registers = (section.sh_info >> 24) as u64;
            }
        }
        let (entry, code_bytes) = entry.ok_or(Error::Firmware)?;
        // Every entry uses the same section layout. Upload the immutable image
        // once, while retaining entry-specific register and constant metadata.
        let program = if let Some(&allocation) = self.program_images.get(&source) {
            allocation
        } else {
            let allocation = self.memory.allocate(device, image.len() as u64 + 4096, 4096)?;
            while image.len() & 3 != 0 {
                image.push(0);
            }
            device.vram.as_ref().ok_or(Error::Firmware)?.copy_from(allocation.physical as usize, &image)?;
            self.program_images.insert(source, allocation);
            info!("NVIDIA_PROGRAM_IMAGE_READY bytes={} images={}", image.len(), self.program_images.len());
            allocation
        };
        let address = program.virtual_address + entry;
        let constants = constants.map(|binding| binding.map(|(offset, size)| (program.virtual_address + offset, size)));
        let qmd = self.program_qmd(address, code_bytes, registers, shared_bytes, &constants)?;
        let program = Program {
            registers,
            local_bytes: u64::from(stack_bytes).checked_add(0x240).ok_or(Error::Firmware)?,
            required_threads,
            source,
            qmd,
        };
        self.programs.insert(symbol, program.clone());
        Ok(program)
    }

    fn program_qmd(&self, address: u64, bytes: u64, registers: u64, shared_bytes: u64, constants: &[Option<(u64, u32)>; 8]) -> Result<Record<256>> {
        let shared_partition = u64::from(d3os_nvidia::cubin::shared_partition(shared_bytes as u32)?);
        let mut qmd = Record::<256>::default();
        use QmdFields as Q;
        let banks = [
            (
                Q::CONSTANT_BUFFER_ADDR_LOWER_1,
                Q::CONSTANT_BUFFER_ADDR_UPPER_1,
                Q::CONSTANT_BUFFER_SIZE_SHIFTED4_1,
                Q::CONSTANT_BUFFER_VALID_1,
                Q::CONSTANT_BUFFER_INVALIDATE_1,
            ),
            (
                Q::CONSTANT_BUFFER_ADDR_LOWER_2,
                Q::CONSTANT_BUFFER_ADDR_UPPER_2,
                Q::CONSTANT_BUFFER_SIZE_SHIFTED4_2,
                Q::CONSTANT_BUFFER_VALID_2,
                Q::CONSTANT_BUFFER_INVALIDATE_2,
            ),
            (
                Q::CONSTANT_BUFFER_ADDR_LOWER_3,
                Q::CONSTANT_BUFFER_ADDR_UPPER_3,
                Q::CONSTANT_BUFFER_SIZE_SHIFTED4_3,
                Q::CONSTANT_BUFFER_VALID_3,
                Q::CONSTANT_BUFFER_INVALIDATE_3,
            ),
            (
                Q::CONSTANT_BUFFER_ADDR_LOWER_4,
                Q::CONSTANT_BUFFER_ADDR_UPPER_4,
                Q::CONSTANT_BUFFER_SIZE_SHIFTED4_4,
                Q::CONSTANT_BUFFER_VALID_4,
                Q::CONSTANT_BUFFER_INVALIDATE_4,
            ),
            (
                Q::CONSTANT_BUFFER_ADDR_LOWER_5,
                Q::CONSTANT_BUFFER_ADDR_UPPER_5,
                Q::CONSTANT_BUFFER_SIZE_SHIFTED4_5,
                Q::CONSTANT_BUFFER_VALID_5,
                Q::CONSTANT_BUFFER_INVALIDATE_5,
            ),
            (
                Q::CONSTANT_BUFFER_ADDR_LOWER_6,
                Q::CONSTANT_BUFFER_ADDR_UPPER_6,
                Q::CONSTANT_BUFFER_SIZE_SHIFTED4_6,
                Q::CONSTANT_BUFFER_VALID_6,
                Q::CONSTANT_BUFFER_INVALIDATE_6,
            ),
            (
                Q::CONSTANT_BUFFER_ADDR_LOWER_7,
                Q::CONSTANT_BUFFER_ADDR_UPPER_7,
                Q::CONSTANT_BUFFER_SIZE_SHIFTED4_7,
                Q::CONSTANT_BUFFER_VALID_7,
                Q::CONSTANT_BUFFER_INVALIDATE_7,
            ),
        ];
        for (bank, (lower, upper, size, valid, invalidate)) in banks.into_iter().enumerate() {
            if let Some((address, bytes)) = constants[bank + 1] {
                qmd.set_bits(lower, address & 0xffffffff)?;
                qmd.set_bits(upper, address >> 32)?;
                qmd.set_bits(size, u64::from(bytes.div_ceil(16)))?;
                qmd.set_bits(valid, 1)?;
                qmd.set_bits(invalidate, 1)?;
            }
        }
        for (field, value) in [
            (Q::QMD_MAJOR_VERSION, 3),
            (Q::QMD_GROUP_ID, 0x3f),
            (Q::SM_GLOBAL_CACHING_ENABLE, 1),
            (Q::SHARED_MEMORY_SIZE, shared_bytes),
            (Q::REGISTER_COUNT_V, registers),
            (Q::INVALIDATE_TEXTURE_HEADER_CACHE, 1),
            (Q::INVALIDATE_TEXTURE_SAMPLER_CACHE, 1),
            (Q::INVALIDATE_TEXTURE_DATA_CACHE, 1),
            (Q::INVALIDATE_SHADER_DATA_CACHE, 1),
            (Q::API_VISIBLE_CALL_LIMIT, 1),
            (Q::SAMPLER_INDEX, 1),
            (Q::BARRIER_COUNT, 1),
            (Q::CWD_MEMBAR_TYPE, u64::from(NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR)),
            (Q::MIN_SM_CONFIG_SHARED_MEM_SIZE, shared_partition),
            (Q::TARGET_SM_CONFIG_SHARED_MEM_SIZE, shared_partition),
            (Q::MAX_SM_CONFIG_SHARED_MEM_SIZE, 0x1a),
            (Q::PROGRAM_PREFETCH_SIZE, (bytes >> 8).min(0x1ff)),
            (Q::SASS_VERSION, self.sass_version),
            (Q::CTA_RASTER_HEIGHT, 1),
            (Q::CTA_RASTER_DEPTH, 1),
            (Q::CTA_THREAD_DIMENSION1, 1),
            (Q::CTA_THREAD_DIMENSION2, 1),
            (Q::CONSTANT_BUFFER_VALID_0, 1),
            (Q::CONSTANT_BUFFER_INVALIDATE_0, 1),
            // QMD v3 encodes this size in 16-byte units.
            (Q::CONSTANT_BUFFER_SIZE_SHIFTED4_0, (CONSTANT_BYTES / 16) as u64),
            (Q::PROGRAM_ADDRESS_LOWER, address & 0xffffffff),
            (Q::PROGRAM_ADDRESS_UPPER, address >> 32),
            (Q::PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, (address >> 8) & 0xffffffff),
            (Q::PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, address >> 40),
            (Q::RELEASE0_ENABLE, 1),
            (Q::RELEASE0_PAYLOAD64B, 1),
            (Q::RELEASE0_PAYLOAD_LOWER, 1),
            (Q::RELEASE0_PAYLOAD_UPPER, 0),
        ] {
            qmd.set_bits(field, value)?;
        }
        Ok(qmd)
    }

    pub(super) fn launch(&mut self, device: &Device, cubin: &'static [u8], symbol: &'static str, arguments: &[u8], dimensions: [u32; 2]) -> Result<()> {
        self.last_wait_cycles = 0;
        self.last_device_ns = 0;
        let [threads, blocks] = dimensions;
        if arguments.len() > 512 || threads == 0 || threads > 1024 || blocks == 0 {
            return Err(Error::Alignment);
        }
        if self.submission.as_ref().ok_or(Error::Firmware)?.pending.len() == SLOTS {
            self.flush_pending(device)?;
        }
        let cached = self.programs.contains_key(symbol);
        let program = self.load_program(device, cubin, symbol)?;
        if program.required_threads.is_some_and(|required| required != [threads, 1, 1]) {
            return Err(Error::Alignment);
        }
        if d3os_nvidia::cubin::validate_ada_registers(program.registers, threads).is_err() {
            warn!(
                "NVIDIA launch exceeds Ada register budget symbol={symbol} registers={} threads={threads}",
                program.registers
            );
            return Err(Error::Alignment);
        }
        // Previous launches have completed before any scratch allocation or
        // engine-state change. A larger module stack needs a larger QMD limit.
        if self.local_bytes_per_thread < program.local_bytes {
            self.flush_pending(device)?;
            self.ensure_local_memory(device, program.local_bytes)?;
        }
        let submission = self.submission.as_ref().ok_or(Error::Firmware)?;
        let slot_offset = submission.pending.len() * SLOT_BYTES;
        let launch = submission.address + slot_offset as u64;
        let signal = launch + SIGNAL as u64;
        for offset in (0..64).step_by(4) {
            submission.region.write32(slot_offset + SIGNAL + offset, 0)?;
        }
        let mut constants = Record::<CONSTANT_BYTES>::default();
        for (i, value) in [threads, 1, 1, blocks, 1, 1].into_iter().enumerate() {
            constants.set(Field::<u32>::new(i * 4), value)?;
        }
        constants.set(Field::<u64>::new(24), 0x729400000000)?;
        constants.set(Field::<u64>::new(32), 0x729300000000)?;
        constants.set(Field::<u64>::new(40), 0xfffdc0)?;
        let mut constants = constants.into_bytes();
        constants[0x160..0x160 + arguments.len()].copy_from_slice(arguments);
        submission.region.copy(slot_offset + 256, &constants)?;
        let mut qmd = program.qmd.clone();
        use QmdFields as Q;
        for (field, value) in [
            (Q::SHADER_LOCAL_MEMORY_HIGH_SIZE, self.local_bytes_per_thread),
            (Q::CTA_RASTER_WIDTH, u64::from(blocks)),
            (Q::CTA_THREAD_DIMENSION0, u64::from(threads)),
            (Q::CONSTANT_BUFFER_ADDR_LOWER_0, (launch + 256) & 0xffffffff),
            (Q::CONSTANT_BUFFER_ADDR_UPPER_0, (launch + 256) >> 32),
            (Q::RELEASE0_STRUCTURE_SIZE, if self.device_timing { 0 } else { 2 }),
            (Q::RELEASE0_ADDRESS_LOWER, signal & 0xffffffff),
            (Q::RELEASE0_ADDRESS_UPPER, signal >> 32),
        ] {
            qmd.set_bits(field, value)?;
        }
        let submission = self.submission.as_mut().ok_or(Error::Firmware)?;
        if let Some(previous) = submission.pending.last_mut() {
            d3os_nvidia::qmd::chain(&mut previous.qmd, launch)?;
        }
        submission.invalidate_code |= !cached;
        submission.pending.push(Pending { qmd, symbol, dimensions });
        if !self.defer_launch || self.device_timing {
            self.flush_pending(device)?;
        }
        Ok(())
    }

    pub(super) fn pending_commands(&self) -> bool {
        self.submission.as_ref().is_some_and(|s| !s.pending.is_empty())
    }

    pub(super) fn flush_pending(&mut self, device: &Device) -> Result<()> {
        if !self.pending_commands() {
            return Ok(());
        }
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        let result = self.flush_pending_inner(device);
        let cycles = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start);
        self.last_flush_cycles = self.last_flush_cycles.wrapping_add(cycles);
        self.queue_profile.cycles = self.queue_profile.cycles.wrapping_add(cycles);
        if result.is_err() {
            self.healthy = false;
        }
        result
    }

    fn flush_pending_inner(&mut self, device: &Device) -> Result<()> {
        let submission = self.submission.as_ref().ok_or(Error::Firmware)?;
        let count = submission.pending.len();
        if self.device_timing && count != 1 {
            return Err(Error::Firmware);
        }
        for (index, pending) in submission.pending.iter().enumerate() {
            submission.region.copy(index * SLOT_BYTES, pending.qmd.as_bytes())?;
        }
        let last = submission.pending.last().ok_or(Error::Firmware)?;
        let symbol = last.symbol;
        let [threads, blocks] = last.dimensions;
        let launch = submission.address;
        let signal_offset = (count - 1) * SLOT_BYTES + SIGNAL;
        let signal = submission.address + signal_offset as u64;
        let invalidate_code = submission.invalidate_code;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;

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
        if !self.engine_configured {
            Self::method(&mut commands, NVC6C0_SET_OBJECT, &[ADA_COMPUTE_A]);
            Self::method(&mut commands, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, &[0x7293, 0]);
            Self::method(&mut commands, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, &[0x7294, 0]);
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
        }
        if invalidate_code {
            Self::method(&mut commands, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, &[0x1011]);
        }
        if self.device_timing {
            // FIFO four-word release writes payload followed by a GPU timestamp.
            commands.extend_from_slice(&[
                (2 << 28) | (5 << 16) | (0x5c >> 2),
                (signal + 16) as u32,
                ((signal + 16) >> 32) as u32,
                1,
                0,
                1 | (1 << 24) | (1 << 20) | (1 << 25),
            ]);
        }
        Self::method(&mut commands, NVC6C0_SEND_PCAS_A, &[(launch >> 8) as u32]);
        Self::method(
            &mut commands,
            NVC6C0_SEND_SIGNALING_PCAS2_B,
            &[NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE],
        );
        semaphore(&mut commands, signal + 56, 0x2222, true);
        self.submit(device, &commands)?;
        let submission = self.submission.as_ref().ok_or(Error::Firmware)?;
        log::debug!("NVIDIA_BATCH_SUBMITTED last={symbol} kernels={count} qmd={launch:#x}");
        let start = crate::timer().systime_ms();
        // Also wait for the FIFO tail before reusing the command buffer.
        let wait_start = unsafe { core::arch::x86_64::_rdtsc() };
        while submission.region.read32(signal_offset)? != 1 || submission.region.read32(signal_offset + 56)? != 0x2222 {
            if crate::timer().systime_ms().wrapping_sub(start) >= 5000 {
                self.dump_instance(device)?;
                log::warn!(
                    "NVIDIA batch timeout last={symbol} kernels={count} threads={threads} blocks={blocks} GPGet={:?} GPPut={:?}",
                    vram.read(self.fifo.physical as usize + self.entries * 8 + 0x88),
                    vram.read(self.fifo.physical as usize + self.entries * 8 + 0x8c)
                );
                info!(
                    "NVIDIA completion compute={:?} fifo={:?}",
                    submission.region.read32(signal_offset),
                    submission.region.read32(signal_offset + 56)
                );
                info!("NVIDIA notifier {:02x?}", vram.read_bytes(self.notifier as usize, 0x100)?);
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
        let wait_cycles = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(wait_start);
        self.last_wait_cycles = self.last_wait_cycles.wrapping_add(wait_cycles);
        self.queue_profile.wait_cycles = self.queue_profile.wait_cycles.wrapping_add(wait_cycles);
        if self.device_timing {
            let start = submission.region.read64(signal_offset + 24)?;
            let end = submission.region.read64(signal_offset + 8)?;
            if start == 0 || end < start {
                return Err(Error::Firmware);
            }
            self.last_device_ns = end - start;
        }
        self.engine_configured = true;
        let submission = self.submission.as_mut().ok_or(Error::Firmware)?;
        submission.pending.clear();
        submission.invalidate_code = false;
        self.queue_profile.batches = self.queue_profile.batches.wrapping_add(1);
        self.queue_profile.launches = self.queue_profile.launches.wrapping_add(count as u64);
        self.buffers.reclaim_retired();
        Ok(())
    }

    pub(super) fn timestamp_test(&mut self, device: &Device, cubin: &'static [u8]) -> Result<()> {
        let output = self.memory.allocate(device, 4096, 4096)?;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        for index in 0..128 {
            vram.write(output.physical as usize + index * 4, 1f32.to_bits())?;
        }
        let timing = self.device_timing;
        self.device_timing = true;
        let mut iterations = 32768u32;
        loop {
            let mut arguments = Record::<12>::default();
            arguments.set(Field::<u64>::new(0), output.virtual_address)?;
            arguments.set(Field::<u32>::new(8), iterations)?;
            let start = crate::timer().systime_ms();
            self.launch(device, cubin, "timestamp_probe", arguments.as_bytes(), [128, 1])?;
            let elapsed_ms = crate::timer().systime_ms().wrapping_sub(start);
            if self.last_device_ns >= 10_000_000 {
                if self.last_device_ns / 1_000_000 > elapsed_ms as u64 + 2 {
                    return Err(Error::Firmware);
                }
                for index in 0..128 {
                    if vram.read(output.physical as usize + index * 4) != Some(1f32.to_bits()) {
                        return Err(Error::Firmware);
                    }
                }
                info!(
                    "NVIDIA_DEVICE_TIMER_OK elapsed_ns={} host_ms={elapsed_ms} iterations={iterations}",
                    self.last_device_ns
                );
                break;
            }
            iterations = iterations.checked_mul(4).filter(|n| *n <= 33_554_432).ok_or(Error::Firmware)?;
        }
        self.device_timing = timing;
        Ok(())
    }

    pub(super) fn local_stack_test(&mut self, device: &Device, cubin: &'static [u8]) -> Result<()> {
        const COUNT: u32 = 257;
        let output = self.memory.allocate_buffer(device, 4096)?;
        let mut arguments = Record::<12>::default();
        arguments.set(Field::<u64>::new(0), output.virtual_address)?;
        arguments.set(Field::<u32>::new(8), COUNT)?;
        self.launch(device, cubin, "local_stack_probe", arguments.as_bytes(), [128, COUNT])?;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        for index in 0..COUNT {
            let seed = index.wrapping_mul(0x9e3779b9);
            let expected = (0u32..65536).fold(0u32, |sum, i| {
                let mut value = i ^ seed;
                for _ in 0..COUNT % 3 + 1 {
                    value = (value ^ value.wrapping_shl(13)).wrapping_mul(0x51a6d305);
                }
                sum.wrapping_add(value)
            });
            if vram.read(output.physical as usize + index as usize * 4) != Some(expected) {
                return Err(Error::Firmware);
            }
        }
        info!("NVIDIA_LOCAL_STACK_OK count={COUNT} backend=nvidia");
        Ok(())
    }

    pub(super) fn vector_add_test(&mut self, device: &Device, cubin: &'static [u8]) -> Result<()> {
        let data = self.memory.allocate(device, 3 * 4096, 4096)?;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        const COUNT: u32 = 257;
        for i in 0..COUNT {
            vram.write((data.physical + u64::from(i) * 4) as usize, (i as f32 * 0.25).to_bits())?;
            vram.write((data.physical + 4096 + u64::from(i) * 4) as usize, (i as f32 * -0.125 + 3.0).to_bits())?;
            vram.write((data.physical + 8192 + u64::from(i) * 4) as usize, f32::NAN.to_bits())?;
        }
        let mut arguments = Record::<28>::default();
        arguments.set(Field::<u64>::new(0), data.virtual_address)?;
        arguments.set(Field::<u64>::new(8), data.virtual_address + 4096)?;
        arguments.set(Field::<u64>::new(16), data.virtual_address + 8192)?;
        arguments.set(Field::<u32>::new(24), COUNT)?;
        self.launch(device, cubin, "vector_add", arguments.as_bytes(), [128, COUNT.div_ceil(128)])?;
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

    pub(super) fn tensor_test(&mut self, device: &Device, cubin: &'static [u8]) -> Result<()> {
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        let a_value = |i: u32| ((i * 7 % 23) as f32 - 11.0) / 8.0;
        let b_value = |i: u32| ((i * 13 % 19) as f32 - 9.0) / 8.0;
        // These inputs and their reference sums are exact in both formats.
        // CPU comparisons cannot pass unless the GPU overwrites the NaN output.
        for symbol in ["tensor_matmul_f16", "tensor_matmul_bf16"] {
            const SHAPES: [(u32, u32, u32); 5] = [(16, 16, 8), (17, 33, 9), (1, 65, 33), (37, 257, 65), (1, 1024, 1024)];
            for (rows, inner, cols) in SHAPES {
                let a = self.memory.allocate(device, u64::from(rows * inner) * 4, 4096)?;
                let b = self.memory.allocate(device, u64::from(inner * cols) * 4, 4096)?;
                let output = self.memory.allocate(device, u64::from(rows * cols + 2) * 4, 4096)?;
                for i in 0..rows * inner {
                    vram.write((a.physical + u64::from(i) * 4) as usize, a_value(i).to_bits())?;
                }
                for i in 0..inner * cols {
                    vram.write((b.physical + u64::from(i) * 4) as usize, b_value(i).to_bits())?;
                }
                const GUARD: u32 = 0x4b123456;
                vram.write(output.physical as usize, GUARD)?;
                for i in 0..rows * cols {
                    vram.write((output.physical + u64::from(i + 1) * 4) as usize, f32::NAN.to_bits())?;
                }
                vram.write((output.physical + u64::from(rows * cols + 1) * 4) as usize, GUARD)?;
                let mut arguments = Record::<36>::default();
                arguments.set(Field::<u64>::new(0), a.virtual_address)?;
                arguments.set(Field::<u64>::new(8), b.virtual_address)?;
                arguments.set(Field::<u64>::new(16), output.virtual_address + 4)?;
                arguments.set(Field::<u32>::new(24), rows)?;
                arguments.set(Field::<u32>::new(28), inner)?;
                arguments.set(Field::<u32>::new(32), cols)?;
                let tiles = rows.div_ceil(16) * cols.div_ceil(32);
                self.launch(device, cubin, symbol, arguments.as_bytes(), [128, tiles])?;
                if vram.read(output.physical as usize) != Some(GUARD) || vram.read((output.physical + u64::from(rows * cols + 1) * 4) as usize) != Some(GUARD) {
                    return Err(Error::Firmware);
                }
                for row in 0..rows {
                    for col in 0..cols {
                        let expected: f32 = (0..inner).map(|k| a_value(row * inner + k) * b_value(k * cols + col)).sum();
                        let actual = vram.read((output.physical + u64::from(row * cols + col + 1) * 4) as usize);
                        if actual != Some(expected.to_bits()) {
                            log::warn!("NVIDIA tensor mismatch symbol={symbol} row={row} col={col} actual={actual:?} expected={expected}");
                            return Err(Error::Firmware);
                        }
                    }
                }
            }
            info!("NVIDIA_TENSOR_OK kernel={symbol} cases={} backend=nvidia", SHAPES.len());
        }
        Ok(())
    }
}
