#![no_std]

extern crate alloc;

pub mod elf;
pub mod host;
pub mod process;
pub mod stack;

pub use host::{Host, Metadata};
pub use process::{LinuxProcess, StartContext, SyscallFrame, SyscallOutcome};
pub type Result<T> = core::result::Result<T, isize>;

pub const PAGE_SIZE: u64 = 4096;
pub const LOW_EXECUTABLE_START: u64 = 4 * 1024 * 1024;
pub const LOW_EXECUTABLE_END: u64 = 128 * 1024 * 1024;

pub fn page_up(value: u64) -> Option<u64> {
    Some(value.checked_add(PAGE_SIZE - 1)? & !(PAGE_SIZE - 1))
}

pub fn linux_errno(d3os: isize) -> isize {
    match d3os {
        -2 => -2,        // ENOENT
        -3 => -24,       // EMFILE
        -4 | -9 => -9,   // EBADF
        -5 => -13,       // EACCES
        -15 => -30,      // EROFS
        -6 => -17,       // EEXIST
        -7 => -20,       // ENOTDIR
        -8 | -11 => -22, // EINVAL
        -10 => -39,      // ENOTEMPTY
        -12 => -16,      // EBUSY
        -13 => -95,      // EOPNOTSUPP
        -14 => -104,     // ECONNRESET
        -16 => -11,      // EAGAIN
        -17 => -3,       // ESRCH
        -19 => -32,      // EPIPE
        -20 => -12,      // ENOMEM
        -21 => -21,      // EISDIR
        _ => -5,         // EIO
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rounding_rejects_overflow() {
        assert_eq!(super::page_up(4097), Some(8192));
        assert_eq!(super::page_up(u64::MAX), None);
    }

    #[test]
    fn translates_incompatible_errno_numbers() {
        assert_eq!(super::linux_errno(-4), -9);
        assert_eq!(super::linux_errno(-8), -22);
        assert_eq!(super::linux_errno(-20), -12);
        assert_eq!(super::linux_errno(-18), -5);
        assert_eq!(super::linux_errno(-15), -30);
    }
}
