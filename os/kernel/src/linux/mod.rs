//! Kernel integration for the Linux AMD64 execution personality.
mod host;
#[cfg(feature = "linux_test")]
pub mod test;

use alloc::{borrow::Cow, boxed::Box, collections::BTreeMap, sync::Arc, vec::Vec};
use core::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU32, Ordering},
};
use linux_compat::{Host, LinuxProcess, Result, SyscallFrame, SyscallOutcome, elf::Elf, process::normalize};
use naming::shared_types::OpenOptions;
use x86_64::VirtAddr;

use crate::{
    naming::api,
    process::{process::Process, thread::Thread},
};
pub use host::KernelHost;
pub struct ExecutionState {
    program: LinuxProcess<KernelHost>,
    suspended: Option<Box<VforkParent>>,
    children: BTreeMap<u32, Child>,
    #[cfg(feature = "linux_test")]
    report_exit: bool,
}

struct VforkParent {
    program: LinuxProcess<KernelHost>,
    frame: SyscallFrame,
    thread_pointer: u64,
}

enum Child {
    Process { thread: usize, process: Arc<Process> },
    Exited(u8),
}

impl Deref for ExecutionState {
    type Target = LinuxProcess<KernelHost>;
    fn deref(&self) -> &Self::Target {
        &self.program
    }
}
impl DerefMut for ExecutionState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.program
    }
}

impl ExecutionState {
    fn resume_parent(&mut self, frame: &mut SyscallFrame) -> u32 {
        let pid = self.id();
        let parent = self.suspended.take().expect("Missing vfork parent");
        self.program = parent.program;
        self.host().set_thread_pointer(parent.thread_pointer);
        *frame = parent.frame;
        pid
    }
}

struct Launch<'a> {
    path: &'a str,
    image: &'a [u8],
    interpreter: Option<&'a [u8]>,
    arguments: &'a [&'a str],
    environment: Option<&'a [&'a str]>,
    cwd: &'a str,
    id: u32,
}
static NEXT_PID: AtomicU32 = AtomicU32::new(1);

fn errno(error: syscall::return_vals::Errno) -> isize {
    linux_compat::linux_errno(error as isize)
}

pub fn read_file(path: &str) -> Result<Cow<'static, [u8]>> {
    if let Some(data) = api::immutable_data(path).map_err(errno)? {
        return Ok(Cow::Borrowed(data));
    }
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
        Ok(Cow::Owned(data))
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
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push(executable.as_str());
    argv.extend_from_slice(args);
    let launch = Launch {
        path: &executable,
        image,
        interpreter: interpreter.as_deref(),
        arguments: &argv,
        environment: None,
        cwd,
        id: NEXT_PID.fetch_add(1, Ordering::Relaxed),
    };
    let result = prepare(&process, launch, None);
    if result.is_err() {
        crate::process_manager().write().kill(process.id());
    }
    result
}

fn prepare(process: &Arc<Process>, launch: Launch<'_>, inherited: Option<&ExecutionState>) -> Result<Arc<Thread>> {
    let host = KernelHost::new(process);
    let mut state = LinuxProcess::new(host, launch.id, launch.cwd);
    let start = if let Some(environment) = launch.environment {
        state.start_with_environment(launch.image, launch.interpreter, launch.arguments, environment)?
    } else {
        state.start(launch.image, launch.interpreter, launch.arguments)?
    };
    state.set_executable_path(launch.path);
    if let Some(inherited) = inherited {
        state.inherit_descriptors(inherited, true)?;
    }
    let entry: extern "sysv64" fn() = unsafe { core::mem::transmute(start.instruction as usize) };
    let thread = Thread::new_user_thread_with_stack(Arc::clone(process), VirtAddr::new(start.instruction), entry, Some(VirtAddr::new(start.stack)));
    *process.linux_state.lock() = Some(ExecutionState {
        program: state,
        suspended: None,
        children: BTreeMap::new(),
        #[cfg(feature = "linux_test")]
        report_exit: inherited.is_none(),
    });
    Ok(thread)
}

pub extern "sysv64" fn dispatch(frame: &mut SyscallFrame) -> isize {
    let process = crate::process_manager().read().current_process();
    let mut guard = process.linux_state.lock();
    if let Some(state) = guard.as_mut() {
        #[cfg(feature = "linux_test")]
        log::info!("LINUX_SYSCALL {} args={:x?}", frame.number, frame.arguments());
        // The scheduler activated this process's page tables before syscall entry.
        match unsafe { state.dispatch(frame.number, frame.arguments()) } {
            SyscallOutcome::Return(value) => {
                #[cfg(feature = "linux_test")]
                log::info!("LINUX_RESULT {} => {value}", frame.number);
                value
            }
            SyscallOutcome::Exit(status) => {
                if state.suspended.is_some() {
                    let pid = state.resume_parent(frame);
                    state.children.insert(pid, Child::Exited(status));
                    return pid as isize;
                }
                #[cfg(feature = "linux_test")]
                if state.report_exit {
                    test::report(state, status);
                }
                process.exit_status.store(status as usize, Ordering::Release);
                drop(guard);
                drop(process);
                crate::syscall::sys_concurrent::sys_process_exit();
            }
            SyscallOutcome::Vfork { stack } => {
                if state.suspended.is_some() {
                    return -38;
                }
                let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);
                let child = match state.vfork(KernelHost::new(&process), pid) {
                    Ok(child) => child,
                    Err(error) => return error,
                };
                let thread_pointer = state.host().thread_pointer();
                let parent = core::mem::replace(&mut state.program, child);
                state.suspended = Some(Box::new(VforkParent {
                    program: parent,
                    frame: *frame,
                    thread_pointer,
                }));
                frame.rsp = stack;
                0
            }
            SyscallOutcome::Exec { path, arguments, environment } => {
                if state.suspended.is_none() {
                    return -38;
                }
                let arguments: Vec<&str> = arguments.iter().map(|s| s.as_str()).collect();
                let environment: Vec<&str> = environment.iter().map(|s| s.as_str()).collect();
                let image = match read_file(&path) {
                    Ok(image) => image,
                    Err(error) => return error,
                };
                let elf = match Elf::parse(&image, 0) {
                    Ok(elf) => elf,
                    Err(_) => return -8,
                };
                let interpreter = match elf.interpreter.map(read_file).transpose() {
                    Ok(bytes) => bytes,
                    Err(error) => return error,
                };
                let child = crate::process_manager().write().create_process(path.clone());
                let launch = Launch {
                    path: &path,
                    image: &image,
                    interpreter: interpreter.as_deref(),
                    arguments: &arguments,
                    environment: Some(&environment),
                    cwd: state.cwd(),
                    id: state.id(),
                };
                let thread = match prepare(&child, launch, Some(state)) {
                    Ok(thread) => thread,
                    Err(error) => {
                        crate::process_manager().write().kill(child.id());
                        return error;
                    }
                };
                let pid = state.resume_parent(frame);
                state.children.insert(
                    pid,
                    Child::Process {
                        thread: thread.id(),
                        process: child,
                    },
                );
                crate::process::core_local_storage::scheduler().ready(thread);
                pid as isize
            }
            SyscallOutcome::Wait(request) => {
                let pid = request.pid;
                let child_id = if pid == -1 {
                    let manager = crate::process_manager().read();
                    state
                        .children
                        .iter()
                        .find(|(_, child)| match child {
                            Child::Exited(_) => true,
                            Child::Process { process, .. } => !manager.is_active_process(process.id()),
                        })
                        .map(|(&id, _)| id)
                        .or_else(|| state.children.keys().next().copied())
                } else {
                    u32::try_from(pid).ok().filter(|id| state.children.contains_key(id))
                };
                let Some(child_id) = child_id else {
                    return -10;
                };
                let child = &state.children[&child_id];
                let code = match child {
                    Child::Process { thread, process } => {
                        let active = crate::process_manager().read().is_active_process(process.id());
                        if active && request.nonblocking {
                            if request.information && request.output != 0 {
                                if let Err(error) = unsafe { state.write_memory(request.output, &[0; 128]) } {
                                    return error;
                                }
                            }
                            return 0;
                        }
                        // Child execution has its own state lock and address space.
                        let _ = crate::process::core_local_storage::scheduler().join(*thread);
                        process.exit_status.load(Ordering::Acquire) as u8
                    }
                    Child::Exited(status) => *status,
                };
                if request.output != 0 {
                    let result = if request.information {
                        let mut info = [0u8; 128];
                        info[..4].copy_from_slice(&17u32.to_le_bytes()); // SIGCHLD
                        info[8..12].copy_from_slice(&1u32.to_le_bytes()); // CLD_EXITED
                        info[16..20].copy_from_slice(&child_id.to_le_bytes());
                        info[24..28].copy_from_slice(&(code as u32).to_le_bytes());
                        unsafe { state.write_memory(request.output, &info) }
                    } else {
                        unsafe { state.write_memory(request.output, &((code as u32) << 8).to_le_bytes()) }
                    };
                    if let Err(error) = result {
                        return error;
                    }
                }
                if request.usage != 0 {
                    if let Err(error) = unsafe { state.write_memory(request.usage, &[0; 144]) } {
                        return error;
                    }
                }
                if !request.keep {
                    state.children.remove(&child_id);
                }
                if request.information { 0 } else { child_id as isize }
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
