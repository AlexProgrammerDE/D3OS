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
        let region = DmaRegion::new(STAGING_BYTES)?;
        let physical = region.address as u64;
        self.staging = Some(Staging { region, virtual_address: 0 });
        let address = self.memory.map_system(device, physical, STAGING_BYTES as u64)?;
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
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        let result = self.launch(device, module, kernel.symbol(), &arguments, [work.threads, work.blocks]);
        let cycles = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start);
        let profile = &mut self.profile[kernel as usize];
        profile.calls = profile.calls.wrapping_add(1);
        profile.cycles = profile.cycles.wrapping_add(cycles);
        profile.wait_cycles = profile.wait_cycles.wrapping_add(self.last_wait_cycles);
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
        let staging = self.staging.as_ref().ok_or(Error::Firmware)?;
        let (pointer, address, capacity) = (staging.region.address as *mut u32, staging.virtual_address, staging.region.size / 4);
        for offset in (0..elements).step_by(capacity) {
            let count = (elements - offset).min(capacity);
            let cpu = unsafe { cpu.add(offset) };
            let gpu = gpu.checked_add(offset as u64 * 4).ok_or(Error::Alignment)?;
            if upload {
                // The GPU is an external DMA agent. Volatile staging accesses
                // preserve the CPU memory operations across ownership changes.
                for i in 0..count {
                    unsafe {
                        pointer.add(i).write_volatile(cpu.add(i).read());
                    }
                }
            }
            fence(Ordering::SeqCst);
            let addresses = if upload { [address, gpu] } else { [gpu, address] };
            self.dispatch(device, Kernel::Copy, &addresses, &[count as u32])?;
            fence(Ordering::SeqCst);
            if !upload {
                for i in 0..count {
                    unsafe {
                        cpu.add(i).write(pointer.add(i).read_volatile());
                    }
                }
            }
        }
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
            "NVIDIA_TRANSFER_OK cases={} guards=ok bits=exact elapsed_ms={}",
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
