//! Compile portable kernels in userspace before publishing executable modules.
use super::{DEVICE, Error, Result};
use alloc::{format, string::String, vec, vec::Vec};
use core::{
    fmt::Write,
    sync::atomic::{AtomicU8, Ordering},
};
use sha2::{Digest, Sha256};
use spin::Once;
use syscall::return_vals::Errno;

use crate::{
    naming::api,
    process::{
        core_local_storage::{PreemptGuard, scheduler},
        thread::Thread,
    },
};

const FAMILIES: [&str; 3] = ["vector-add", "tensor-matmul", "inference"];
static COMPLETION: Once<crate::sync::wait_queue::WaitQueue> = Once::new();
// 0: not ready, 1: all modules compiled and validated, 2: failed.
static STATUS: AtomicU8 = AtomicU8::new(0);
static MODULES: Once<[Vec<u8>; 3]> = Once::new();

fn input(family: &str) -> Result<(String, bool)> {
    let tile = format!("/usr/share/nvidia/{family}.tilebc");
    let ptx = format!("/usr/share/nvidia/{family}.ptx");
    match (api::stat(&tile).is_ok(), api::stat(&ptx).is_ok()) {
        (true, false) => Ok((tile, true)),
        (false, true) => Ok((ptx, false)),
        _ => Err(Error::Firmware),
    }
}

/// Called once during boot, after the naming service and terminal exist.
pub(crate) fn schedule() {
    let Some(device) = DEVICE.get() else { return };
    if !device.lock().compute.as_ref().is_some_and(|compute| compute.healthy) || input("inference").is_err() {
        return;
    }
    let thread = Thread::new_kernel_thread(run, "nvidia-compiler");
    COMPLETION.call_once(crate::sync::wait_queue::WaitQueue::new);
    scheduler().ready(thread);
}

/// Wait without holding the GPU lock or disabling preemption. The compiler's
/// processes need the scheduler and normal process cleanup to make progress.
pub(super) fn wait() -> core::result::Result<(), Errno> {
    match STATUS.load(Ordering::Acquire) {
        1 => return Ok(()),
        2 => return Err(Errno::EUNKN),
        _ => (),
    }
    let completion = COMPLETION.get().ok_or(Errno::ENOTSUP)?;
    completion.wait(|| STATUS.load(Ordering::Acquire) != 0, "GPU compilation");
    if STATUS.load(Ordering::Acquire) == 1 { Ok(()) } else { Err(Errno::EUNKN) }
}

pub(super) fn inference() -> Option<&'static [u8]> {
    MODULES.get().map(|modules| modules[2].as_slice())
}

fn sha256(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(result, "{byte:02x}");
    }
    result
}

fn compile(family: &str, target: &str, sm: u32) -> Result<Vec<u8>> {
    let (input, tile) = input(family)?;
    let source = crate::linux::read_file(&input).map_err(|_| Error::Firmware)?;
    let input_hash = sha256(&source);
    let output = format!("/tmp/{family}-{target}.cubin");
    match api::unlink(&output) {
        Ok(()) | Err(Errno::ENOENT) => (),
        Err(_) => return Err(Error::Firmware),
    }
    let compiler = if tile { "/usr/cuda/bin/tileiras" } else { "/usr/cuda/bin/ptxas" };
    let target_option = format!("--gpu-name={target}");
    let options = vec![target_option.as_str(), "--opt-level=3", input.as_str(), "-o", output.as_str()];
    let start = crate::timer().systime_ms();
    let thread = Thread::load_application(compiler, compiler, &options).map_err(|_| Error::Firmware)?;
    let process = thread.process();
    let id = thread.id();
    scheduler().ready(thread);
    let _ = scheduler().join(id);
    let status = process.exit_status.load(Ordering::Acquire);
    if status != 0 {
        log::warn!("NVIDIA_COMPILE_FAIL family={family} target={target} exit={status}");
        return Err(Error::Firmware);
    }
    let bytes = crate::linux::read_file(&output).map_err(|_| Error::Firmware)?.into_owned();
    let elf = goblin::elf::Elf::parse(&bytes).map_err(|_| Error::Firmware)?;
    if elf.header.e_machine != 190 || (elf.header.e_flags >> 8) & 0xff != sm {
        log::warn!(
            "NVIDIA_COMPILE_FAIL family={family} target={target} machine={} flags={:#x}",
            elf.header.e_machine,
            elf.header.e_flags
        );
        return Err(Error::Firmware);
    }
    log::info!(
        "NVIDIA_COMPILE_MODULE os=d3os family={family} input={} input_sha256={input_hash} target={target} compile_ms={} bytes={} sha256={} exit=0",
        if tile { "tile_ir" } else { "ptx" },
        crate::timer().systime_ms() - start,
        bytes.len(),
        sha256(&bytes)
    );
    // The resident cache owns a private immutable copy. Temporary compiler
    // output cannot change an executable module after publication.
    api::unlink(&output).map_err(|_| Error::Firmware)?;
    Ok(bytes)
}

fn prepare() -> Result<()> {
    let device = DEVICE.get().ok_or(Error::Firmware)?;
    let boot0 = device.lock().boot0;
    // Ada's PMC architecture is 0x19. Other architectures need matching native
    // channel, MMU, firmware, and QMD support before the driver can use them.
    let (target, sm) = match boot0 >> 24 {
        0x19 => ("sm_89", 89),
        _ => return Err(d3os_nvidia::Error::Unsupported.into()),
    };
    let start = crate::timer().systime_ms();
    log::info!(
        "NVIDIA_COMPILE_START os=d3os target={target} boot0={boot0:#x} compiler_cpus={} phase=load",
        crate::process::scheduler::cpu_count()
    );
    let modules = [
        compile(FAMILIES[0], target, sm)?,
        compile(FAMILIES[1], target, sm)?,
        compile(FAMILIES[2], target, sm)?,
    ];
    let compile_ms = crate::timer().systime_ms() - start;
    let modules = MODULES.call_once(|| modules);
    let _preempt = PreemptGuard::new();
    let mut device = device.lock();
    let mut compute = device.compute.take().ok_or(Error::Firmware)?;
    let result = (|| {
        compute.vector_add_test(&device, &modules[0])?;
        compute.tensor_test(&device, &modules[1])?;
        compute.inference_module = Some(&modules[2]);
        compute.timestamp_test(&device, &modules[2])?;
        compute.log_clocks(&mut device)?;
        compute.local_stack_test(&device, &modules[2])?;
        compute.prepare_staging(&device)?;
        compute.transfer_test(&device)
    })();
    compute.healthy = result.is_ok();
    device.compute = Some(compute);
    result?;
    log::info!(
        "NVIDIA_COMPILE_READY os=d3os target={target} modules=3 compile_ms={compile_ms} validation_ms={} phase=load",
        crate::timer().systime_ms() - start - compile_ms
    );
    Ok(())
}

extern "sysv64" fn run() {
    let status = match prepare() {
        Ok(()) => 1,
        Err(error) => {
            log::warn!("NVIDIA_COMPILE_FAIL os=d3os error={error}");
            2
        }
    };
    STATUS.store(status, Ordering::Release);
    if let Some(completion) = COMPLETION.get() {
        completion.notify_all();
    }
}
