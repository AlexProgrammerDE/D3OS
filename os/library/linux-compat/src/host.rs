use naming::shared_types::{OpenOptions, SeekOrigin};
use syscall::return_vals::{Errno, SyscallResult};

use crate::Result;

pub struct Metadata {
    pub size: usize,
    pub directory: bool,
}

/// Kernel services used by the Linux execution personality.
///
/// # Safety
/// Implementations must give each process its own mappings. Allocated pages must
/// remain accessible at their guest virtual addresses while a syscall runs.
/// Permission changes and unmapping must invalidate cached translations before
/// returning to the process. The kernel must serialize operations on a process.
pub unsafe trait Host {
    fn allocate_page(&self, address: u64) -> Result<()>;
    fn unmap_page(&self, address: u64) -> Result<()>;
    fn protect_page(&self, address: u64, protection: u32) -> Result<()>;
    fn copy_into(&self, address: u64, data: &[u8]) -> Result<()>;

    fn metadata(&self, path: &str) -> core::result::Result<Metadata, Errno>;
    fn open(&self, path: &str, options: OpenOptions) -> SyscallResult;
    fn read(&self, handle: usize, buffer: &mut [u8]) -> SyscallResult;
    fn write(&self, handle: usize, buffer: &[u8]) -> SyscallResult;
    fn seek(&self, handle: usize, offset: isize, origin: SeekOrigin) -> SyscallResult;
    fn close(&self, handle: usize) -> SyscallResult;
    fn duplicate(&self, handle: usize) -> SyscallResult;
    fn unlink(&self, path: &str) -> core::result::Result<(), Errno>;
    fn truncate(&self, path: &str) -> core::result::Result<(), Errno>;
    fn console_write(&self, descriptor: u64, buffer: &[u8]) -> Result<usize>;

    fn set_thread_pointer(&self, address: u64);
    fn thread_pointer(&self) -> u64;
    fn random_word(&self) -> Result<u64>;
    fn realtime_ms(&self) -> Result<u64>;
    fn monotonic_ms(&self) -> u64;
    fn cpu_time_ms(&self) -> u64;
    fn cpu_count(&self) -> usize;
    fn yield_now(&self);
}
