//! Coherent guest-RAM staging with synchronous GPU ownership transfers.
use super::{Device, Error, Result, gsp::DmaRegion, rm::Compute};
use alloc::vec;
use core::sync::atomic::{Ordering, fence};
use d3os_nvidia::compute::Kernel;

const STAGING_BYTES: usize = 1 << 20;

pub(super) struct Staging {
    region: DmaRegion,
    virtual_address: u64,
}

impl Compute {
    pub(super) fn prepare_staging(&mut self, device: &Device) -> Result<()> {
        // Bootstrap runs before userspace. Allocate here so runtime dispatch
        // never acquires the process-manager lock while holding DEVICE.
        // A separate final page retains copy commands and both completion
        // reports while the payload belongs to the copy engine.
        let region = DmaRegion::new(STAGING_BYTES + 4096)?;
        let physical = region.address as u64;
        self.staging = Some(Staging { region, virtual_address: 0 });
        let address = self.memory.map_system(device, physical, (STAGING_BYTES + 4096) as u64)?;
        self.staging.as_mut().ok_or(Error::Firmware)?.virtual_address = address;
        log::info!("NVIDIA_STAGING_READY physical={physical:#x} virtual={address:#x} bytes={STAGING_BYTES}");
        Ok(())
    }

    pub(super) fn dispatch(&mut self, device: &Device, kernel: Kernel, addresses: &[u64], parameters: &[u32]) -> Result<()> {
        let work = kernel.work(parameters)?;
        work.validate_aliases(addresses)?;
        let module = self.inference_module.ok_or(Error::Firmware)?;
        let mut arguments = alloc::vec::Vec::with_capacity(addresses.len() * 8 + parameters.len() * 4);
        for address in addresses {
            arguments.extend_from_slice(&address.to_le_bytes());
        }
        for parameter in parameters {
            arguments.extend_from_slice(&parameter.to_le_bytes());
        }
        let deferred = self.defer_launch && !self.device_timing;
        self.last_flush_cycles = 0;
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        let result = self.launch(device, module, kernel.entry(parameters), &arguments, [work.threads, work.blocks]);
        let elapsed = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start);
        let cycles = elapsed.saturating_sub(self.last_flush_cycles);
        let profile = &mut self.profile[kernel as usize];
        profile.calls = profile.calls.wrapping_add(1);
        profile.cycles = profile.cycles.wrapping_add(cycles);
        profile.wait_cycles = profile.wait_cycles.wrapping_add(if deferred { 0 } else { self.last_wait_cycles });
        profile.device_ns = profile.device_ns.wrapping_add(self.last_device_ns);
        if result.is_err() {
            self.healthy = false;
        }
        result
    }

    /// # Safety
    /// The CPU range contains `elements` readable (upload) or writable (download)
    /// words. The GPU range is owned, mapped, and contains that many words.
    pub(super) unsafe fn transfer(&mut self, device: &Device, gpu: u64, cpu: *mut u32, elements: usize, upload: bool) -> Result<()> {
        self.flush_pending(device)?;
        let deferred = self.defer_launch;
        self.defer_launch = false;
        let result = (|| {
            let staging = self.staging.as_ref().ok_or(Error::Firmware)?;
            let (pointer, address, capacity) = (staging.region.address as *mut u32, staging.virtual_address, STAGING_BYTES / 4);
            for offset in (0..elements).step_by(capacity) {
                let count = (elements - offset).min(capacity);
                let cpu = unsafe { cpu.add(offset) };
                let gpu = gpu.checked_add(offset as u64 * 4).ok_or(Error::Alignment)?;
                if upload {
                    // The previous DMA completed before the CPU regains this range.
                    // The bulk copy stays observable without clobbering SIMD state.
                    unsafe {
                        d3os_nvidia::dma::copy_owned(pointer.cast(), cpu.cast(), count * 4);
                    }
                }
                fence(Ordering::SeqCst);
                let addresses = if upload { [address, gpu] } else { [gpu, address] };
                self.copy_transfer(device, addresses[1], addresses[0], (count * 4) as u32)?;
                fence(Ordering::SeqCst);
                if !upload {
                    unsafe {
                        d3os_nvidia::dma::copy_owned(cpu.cast(), pointer.cast(), count * 4);
                    }
                }
            }
            Ok(())
        })();
        self.defer_launch = deferred;
        if result.is_err() {
            self.healthy = false;
        }
        result
    }

    fn copy_transfer(&mut self, device: &Device, destination: u64, source: u64, bytes: u32) -> Result<()> {
        const REPORT: usize = STAGING_BYTES + 4096 - 64;
        let record_start = unsafe { core::arch::x86_64::_rdtsc() };
        let staging = self.staging.as_ref().ok_or(Error::Firmware)?;
        let report = staging.virtual_address + REPORT as u64;
        let commands = d3os_nvidia::copy::commands(destination, source, bytes, report)?;
        for offset in [REPORT, REPORT + 8, REPORT + 12] {
            staging.region.write32(offset, 0)?;
        }
        for (index, word) in commands.iter().copied().enumerate() {
            staging.region.write32(STAGING_BYTES + index * 4, word)?;
        }
        let address = staging.virtual_address + STAGING_BYTES as u64;
        self.submit_fifo(device, address, commands.len())?;
        let staging = self.staging.as_ref().ok_or(Error::Firmware)?;
        let start = crate::timer().systime_ms();
        let wait_start = unsafe { core::arch::x86_64::_rdtsc() };
        while staging.region.read32(REPORT)? != 1 || staging.region.read64(REPORT + 8)? != 1 {
            if crate::timer().systime_ms().wrapping_sub(start) >= 5000 {
                self.dump_instance(device)?;
                log::warn!(
                    "NVIDIA copy timeout source={source:#x} destination={destination:#x} bytes={bytes} copy={:?} fifo={:?}",
                    staging.region.read32(REPORT),
                    staging.region.read64(REPORT + 8)
                );
                return Err(Error::Firmware);
            }
            core::hint::spin_loop();
        }
        fence(Ordering::SeqCst);
        let end = unsafe { core::arch::x86_64::_rdtsc() };
        self.queue_profile.batches = self.queue_profile.batches.wrapping_add(1);
        self.queue_profile.cycles = self.queue_profile.cycles.wrapping_add(end.wrapping_sub(record_start));
        self.queue_profile.wait_cycles = self.queue_profile.wait_cycles.wrapping_add(end.wrapping_sub(wait_start));
        Ok(())
    }

    pub(super) fn transfer_test(&mut self, device: &Device) -> Result<()> {
        const MAX: usize = 2 * STAGING_BYTES / 4 + 17;
        const GUARD: u32 = 0x51a6d305;
        let allocation = self.memory.allocate_buffer(device, (MAX as u64 + 2) * 4)?;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        let mut source = vec![0u32; MAX];
        let mut output = vec![0u32; MAX];
        for (i, word) in source.iter_mut().enumerate() {
            // Exercise arbitrary bit patterns, including signed zero and NaNs.
            *word = (i as u32).wrapping_mul(0x9e3779b9).rotate_left((i % 31) as u32);
        }
        source[..5].copy_from_slice(&[0, 0x80000000, 0x7fc01234, 0x7f800001, 0xff800000]);
        let shapes = [1, 257, STAGING_BYTES / 4 - 1, STAGING_BYTES / 4, STAGING_BYTES / 4 + 3, MAX];
        let start = crate::timer().systime_ms();
        for count in shapes {
            self.dispatch(device, Kernel::Zero, &[allocation.virtual_address], &[(allocation.bytes / 4) as u32])?;
            vram.write(allocation.physical as usize, GUARD)?;
            vram.write(allocation.physical as usize + (count + 1) * 4, GUARD)?;
            unsafe {
                self.transfer(device, allocation.virtual_address + 4, source.as_mut_ptr(), count, true)?;
                self.transfer(device, allocation.virtual_address + 4, output.as_mut_ptr(), count, false)?;
            }
            if output[..count] != source[..count]
                || vram.read(allocation.physical as usize) != Some(GUARD)
                || vram.read(allocation.physical as usize + (count + 1) * 4) != Some(GUARD)
            {
                return Err(Error::Firmware);
            }
            self.dispatch(device, Kernel::Zero, &[allocation.virtual_address], &[(allocation.bytes / 4) as u32])?;
            unsafe {
                self.transfer(device, allocation.virtual_address + 4, output.as_mut_ptr(), count, false)?;
            }
            if output[..count].iter().any(|&word| word != 0) {
                return Err(Error::Firmware);
            }
        }
        log::info!(
            "NVIDIA_TRANSFER_OK cases={} guards=ok bits=exact elapsed_ms={} engine=ampere_dma_copy_b",
            shapes.len(),
            crate::timer().systime_ms().wrapping_sub(start)
        );
        // A separate warm-buffer measurement includes CPU staging copies,
        // native kernel dispatch, synchronization, and both PCIe directions.
        let count = STAGING_BYTES / 4;
        let start = crate::timer().systime_ms();
        for _ in 0..64 {
            unsafe {
                self.transfer(device, allocation.virtual_address + 4, source.as_mut_ptr(), count, true)?;
                self.transfer(device, allocation.virtual_address + 4, output.as_mut_ptr(), count, false)?;
            }
        }
        let elapsed = crate::timer().systime_ms().wrapping_sub(start).max(1);
        if output[..count] != source[..count] {
            return Err(Error::Firmware);
        }
        log::info!(
            "NVIDIA_TRANSFER_BENCH bytes={} elapsed_ms={elapsed} mib_per_sec={}",
            STAGING_BYTES * 128,
            128_000 / elapsed
        );
        Ok(())
    }
}
