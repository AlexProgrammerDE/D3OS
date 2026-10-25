use alloc::sync::{Arc, Weak};
#[cfg(feature = "linux_test")]
use alloc::vec::Vec;
use core::{arch::asm, sync::atomic::Ordering};
use linux_compat::{Host, Metadata, Result};
use naming::shared_types::{OpenOptions, SeekOrigin};
use syscall::return_vals::{Errno, SyscallResult};
use x86_64::{
    VirtAddr,
    structures::paging::{Page, PageTableFlags},
};

use crate::{memory::vma::VmaType, naming::api, process::process::Process};

pub struct KernelHost {
    process: Weak<Process>,
    #[cfg(feature = "linux_test")]
    pub output: spin::Mutex<[Vec<u8>; 2]>,
}

impl KernelHost {
    pub fn new(process: &Arc<Process>) -> Self {
        Self {
            process: Arc::downgrade(process),
            #[cfg(feature = "linux_test")]
            output: spin::Mutex::new([Vec::new(), Vec::new()]),
        }
    }

    fn process(&self) -> Result<Arc<Process>> {
        self.process.upgrade().ok_or(-3)
    }
}

// Each host is bound to one process. The syscall dispatcher locks its execution
// state and the kernel restores that process's page tables before entry.
unsafe impl Host for KernelHost {
    fn allocate_page(&self, address: u64) -> Result<()> {
        let process = self.process()?;
        let page = Page::from_start_address(VirtAddr::new(address)).map_err(|_| -22isize)?;
        process
            .virtual_address_space
            .user_alloc_map_full(Some(page), 1, VmaType::Heap, "linux")
            .ok_or(-12isize)?;
        let physical = process.virtual_address_space.get_phys(address).ok_or(-12isize)?;
        unsafe {
            core::ptr::write_bytes(physical.as_u64() as *mut u8, 0, 4096);
        }
        x86_64::instructions::tlb::flush(VirtAddr::new(address));
        Ok(())
    }

    fn unmap_page(&self, address: u64) -> Result<()> {
        let process = self.process()?;
        let vma = process.virtual_address_space.vma_for_address(address).ok_or(-12isize)?;
        process.virtual_address_space.unmap_vma(vma, true);
        x86_64::instructions::tlb::flush(VirtAddr::new(address));
        Ok(())
    }

    fn protect_page(&self, address: u64, protection: u32) -> Result<()> {
        let mut flags = PageTableFlags::USER_ACCESSIBLE;
        if protection != 0 {
            flags |= PageTableFlags::PRESENT;
        }
        if protection & 2 != 0 {
            flags |= PageTableFlags::WRITABLE;
        }
        if protection & 4 == 0 {
            flags |= PageTableFlags::NO_EXECUTE;
        }
        let page = Page::from_start_address(VirtAddr::new(address)).map_err(|_| -22isize)?;
        self.process()?.virtual_address_space.set_flags(Page::range(page, page + 1), flags);
        x86_64::instructions::tlb::flush(VirtAddr::new(address));
        Ok(())
    }

    fn copy_into(&self, address: u64, data: &[u8]) -> Result<()> {
        let process = self.process()?;
        let mut done = 0;
        while done < data.len() {
            let at = address.checked_add(done as u64).ok_or(-14isize)?;
            let physical = process.virtual_address_space.get_phys(at).ok_or(-14isize)?;
            let count = (4096 - at as usize % 4096).min(data.len() - done);
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr().add(done), physical.as_u64() as *mut u8, count);
            }
            done += count;
        }
        Ok(())
    }

    fn metadata(&self, path: &str) -> core::result::Result<Metadata, Errno> {
        let stat = api::stat(path)?;
        Ok(Metadata {
            size: stat.size,
            directory: stat.mode.is_directory(),
        })
    }

    fn open(&self, path: &str, options: OpenOptions) -> SyscallResult {
        api::open(path, options)
    }
    fn read(&self, handle: usize, buffer: &mut [u8]) -> SyscallResult {
        api::read(handle, buffer)
    }
    fn write(&self, handle: usize, buffer: &[u8]) -> SyscallResult {
        api::write(handle, buffer)
    }
    fn seek(&self, handle: usize, offset: isize, origin: SeekOrigin) -> SyscallResult {
        api::seek(handle, offset, origin)
    }
    fn close(&self, handle: usize) -> SyscallResult {
        api::close(handle)
    }
    fn duplicate(&self, handle: usize) -> SyscallResult {
        api::duplicate(handle)
    }
    fn unlink(&self, path: &str) -> core::result::Result<(), Errno> {
        api::unlink(path)
    }
    fn truncate(&self, path: &str) -> core::result::Result<(), Errno> {
        api::truncate(path, 0)
    }

    fn console_write(&self, descriptor: u64, buffer: &[u8]) -> Result<usize> {
        #[cfg(feature = "linux_test")]
        self.output.lock()[descriptor as usize - 1].extend_from_slice(buffer);
        #[cfg(not(feature = "linux_test"))]
        if descriptor == 2 {
            log::warn!("LINUX_STDERR {}", alloc::string::String::from_utf8_lossy(buffer));
        }
        Ok(crate::tty_output().write(buffer))
    }

    fn set_thread_pointer(&self, address: u64) {
        unsafe {
            asm!("wrfsbase {}", in(reg) address, options(nostack));
        }
    }

    fn thread_pointer(&self) -> u64 {
        let address: u64;
        unsafe {
            asm!("rdfsbase {}", out(reg) address, options(nostack));
        }
        address
    }

    fn random_word(&self) -> Result<u64> {
        for _ in 0..64 {
            let value: u64;
            let success: u8;
            unsafe {
                asm!("rdrand {value}", "setc {success}", value = out(reg) value, success = out(reg_byte) success, options(nostack));
            }
            if success != 0 {
                return Ok(value);
            }
        }
        Err(-11)
    }

    fn realtime_ms(&self) -> Result<u64> {
        crate::now().map(|n| n.timestamp_millis() as u64).ok_or(-22)
    }
    fn monotonic_ms(&self) -> u64 {
        crate::timer().systime_ms() as u64
    }
    fn cpu_time_ms(&self) -> u64 {
        self.process
            .upgrade()
            .map(|p| (p.utime.load(Ordering::Relaxed) + p.stime.load(Ordering::Relaxed)) * 10)
            .unwrap_or(0)
    }

    fn cpu_count(&self) -> usize {
        crate::process::scheduler::cpu_count() as usize
    }
    fn yield_now(&self) {
        crate::process::core_local_storage::scheduler().yield_now();
    }
}
