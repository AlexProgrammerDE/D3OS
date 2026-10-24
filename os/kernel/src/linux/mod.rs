//! Kernel integration for the Linux AMD64 execution personality.
mod host;
#[cfg(feature = "linux_test")]
pub mod test;

use alloc::{sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicU32, Ordering};
use linux_compat::{LinuxProcess, Result, SyscallFrame, SyscallOutcome, elf::Elf, process::normalize};
use naming::shared_types::OpenOptions;
use x86_64::VirtAddr;

use crate::{
    naming::api,
    process::{process::Process, thread::Thread},
};
pub use host::KernelHost;
pub type ExecutionState = LinuxProcess<KernelHost>;
static NEXT_PID: AtomicU32 = AtomicU32::new(1);

fn errno(error: syscall::return_vals::Errno) -> isize {
    linux_compat::linux_errno(error as isize)
}

pub fn read_file(path: &str) -> Result<Vec<u8>> {
    let size = api::stat(path).map_err(errno)?.size;
    let mut data = Vec::new();
    data.try_reserve_exact(size).map_err(|_| -12isize)?;
    data.resize(size, 0);
    let handle = api::open(path, OpenOptions::READONLY).map_err(errno)?;
    let result = (|| {
        let mut done = 0;
        while done < data.len() {
            let count = api::read(handle, &mut data[done..]).map_err(errno)?;
            if count == 0 {
                return Err(-8);
            }
            done += count;
        }
        Ok(data)
    })();
    let _ = api::close(handle);
    result
}

pub fn load(path: &str, name: &str, image: &[u8], args: &[&str]) -> Result<Arc<Thread>> {
    let elf = Elf::parse(image, 0).map_err(|_| -8isize)?;
    let interpreter = elf.interpreter.map(read_file).transpose()?;
    let mut directory = [0u8; 4096];
    let length = api::cwd(&mut directory).map_err(errno)?;
    let cwd = core::str::from_utf8(&directory[..length.saturating_sub(1)]).map_err(|_| -22isize)?;
    let executable = normalize(path, cwd)?;
    let process = crate::process_manager().write().create_process(name.into());
    let result = prepare(&process, &executable, image, interpreter.as_deref(), args, cwd);
    if result.is_err() {
        crate::process_manager().write().kill(process.id());
    }
    result
}

fn prepare(process: &Arc<Process>, path: &str, image: &[u8], interpreter: Option<&[u8]>, args: &[&str], cwd: &str) -> Result<Arc<Thread>> {
    let host = KernelHost::new(process);
    let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);
    let mut state = LinuxProcess::new(host, pid, cwd);
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push(path);
    argv.extend_from_slice(args);
    let start = state.start(image, interpreter, &argv)?;
    let entry: extern "sysv64" fn() = unsafe { core::mem::transmute(start.instruction as usize) };
    let thread = Thread::new_user_thread_with_stack(Arc::clone(process), VirtAddr::new(start.instruction), entry, Some(VirtAddr::new(start.stack)));
    *process.linux_state.lock() = Some(state);
    Ok(thread)
}

pub extern "sysv64" fn dispatch(frame: &SyscallFrame) -> isize {
    let process = crate::process_manager().read().current_process();
    let mut guard = process.linux_state.lock();
    if let Some(state) = guard.as_mut() {
        // The scheduler activated this process's page tables before syscall entry.
        match unsafe { state.dispatch(frame.number, frame.arguments()) } {
            SyscallOutcome::Return(value) => {
                #[cfg(feature = "linux_test")]
                log::info!("LINUX_SYSCALL {} => {value}", frame.number);
                value
            }
            SyscallOutcome::Exit(status) => {
                #[cfg(feature = "linux_test")]
                test::report(state, status);
                process.exit_status.store(status as usize, Ordering::Release);
                drop(guard);
                drop(process);
                crate::syscall::sys_concurrent::sys_process_exit();
            }
        }
    } else {
        drop(guard);
        crate::syscall::syscall_dispatcher::native_dispatch(frame.number, frame.arguments())
    }
}

pub fn user_fault(address: u64, instruction: u64) -> ! {
    log::error!("Linux process fault at {address:#x}, instruction {instruction:#x}");
    let process = crate::process_manager().read().current_process();
    process.exit_status.store(139, Ordering::Release);
    #[cfg(feature = "linux_test")]
    test::set_status(139);
    drop(process);
    crate::syscall::sys_concurrent::sys_process_exit();
}
