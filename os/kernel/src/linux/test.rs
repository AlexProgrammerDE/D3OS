//! QEMU verification hooks. Compiler binaries and fixtures come from the ramdisk.
use crate::{
    naming::api,
    process::{core_local_storage::scheduler, thread::Thread},
};
use alloc::{string::String, vec, vec::Vec};
use core::{
    fmt::Write,
    sync::atomic::{AtomicUsize, Ordering},
};
use sha2::Digest;

static STATUS: AtomicUsize = AtomicUsize::new(usize::MAX);

pub fn set_status(status: u8) {
    STATUS.store(status as usize, Ordering::Release);
}

pub fn schedule() {
    scheduler().ready(Thread::new_kernel_thread(run, "linux-test"));
}

extern "sysv64" fn run() {
    let config = super::read_file("/linux.args").expect("Missing Linux test arguments");
    let config = core::str::from_utf8(&config).expect("Invalid Linux test arguments");
    let mut lines = config.lines();
    let program = lines.next().expect("Missing Linux test executable");
    let args: Vec<&str> = lines.collect();
    let repetitions = super::read_file("/linux.repeat")
        .ok()
        .and_then(|bytes| core::str::from_utf8(&bytes).ok()?.trim().parse::<usize>().ok())
        .unwrap_or(1);
    assert!((1..=10).contains(&repetitions));
    for iteration in 0..repetitions {
        STATUS.store(usize::MAX, Ordering::Release);
        let launcher = api::stat("/bin/native-launcher").is_ok() && repetitions == 1;
        let thread = if launcher {
            Thread::load_application("/bin/native-launcher", "native-launcher", &vec![]).expect("Native launcher load failed")
        } else {
            Thread::load_application(program, program, &args).expect("Linux process load failed")
        };
        let process = thread.process();
        let id = thread.id();
        scheduler().ready(thread);
        let _ = scheduler().join(id);
        let status = STATUS.load(Ordering::Acquire);
        if launcher {
            assert_eq!(super::read_file("/native-launch.tmp").expect("Native launch did not finish"), b"ok");
            log::info!("D3OS_NATIVE_LAUNCH_OK");
        } else {
            assert_eq!(status, process.exit_status.load(Ordering::Acquire));
        }
        assert!(!crate::process_manager().read().is_active_process(process.id()));
        drop(process);
        log::info!("D3OS_LINUX_REAPED iteration={} status={status}", iteration + 1);
        scheduler().sleep(150);
    }
    // A native program verifies syscall routing and reads the Linux output file.
    if api::stat("/bin/native-probe").is_ok() {
        let thread = Thread::load_application("/bin/native-probe", "native-probe", &vec![]).expect("Native probe load failed");
        let id = thread.id();
        scheduler().ready(thread);
        let _ = scheduler().join(id);
        assert_eq!(super::read_file("/native-check.tmp").expect("Native probe did not finish"), b"ok");
        log::info!("D3OS_NATIVE_PROBE_OK");
    }
    stop(STATUS.load(Ordering::Acquire) as u8);
}

pub fn report(state: &super::ExecutionState, status: u8) {
    let streams = state.host().output.lock();
    for (index, bytes) in streams.iter().enumerate() {
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3));
        log::info!("D3OS_LINUX_STREAM fd={} size={} fnv1a={hash:016x}", index + 1, bytes.len());
        for chunk in bytes.chunks(128) {
            let mut hex = String::new();
            for byte in chunk {
                let _ = write!(hex, "{byte:02x}");
            }
            log::info!("D3OS_LINUX_DATA fd={} hex={hex}", index + 1);
        }
    }
    if status == 0 {
        if let Ok(bytes) = super::read_file("/out.cubin") {
            let mut hash = String::new();
            for byte in sha2::Sha256::digest(&bytes) {
                let _ = write!(hash, "{byte:02x}");
            }
            log::info!("D3OS_LINUX_CUBIN size={} sha256={hash}", bytes.len());
        }
    }
    log::info!("D3OS_LINUX_EXIT status={status}");
    set_status(status);
}

fn stop(status: u8) -> ! {
    unsafe {
        x86_64::instructions::port::Port::<u32>::new(0xf4).write(if status == 0 { 0x10 } else { 0x11 });
    }
    loop {
        x86_64::instructions::hlt();
    }
}
