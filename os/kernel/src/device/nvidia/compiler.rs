//! Admit qualified cached modules or compile portable kernels in userspace.
use super::{
    DEVICE, Error, Result,
    cubin::{LegacyPolicy, LegacyQualification},
};
use alloc::{format, string::String, vec, vec::Vec};
use core::{
    fmt::Write,
    sync::atomic::{AtomicU8, Ordering},
};
use d3os_nvidia::cubin::qualification::{self, InputKind, ModuleQualification, RuntimeInventory};
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
static MODULES: Once<[CompiledModule; 3]> = Once::new();
static FAILURE: Once<Errno> = Once::new();
static QWEN_VECTOR: Once<Option<CompiledModule>> = Once::new();

struct CompiledModule {
    bytes: Vec<u8>,
    qualification: ModuleQualification,
    evidence: String,
}

/// Static initrd assets stay immutable while compiler processes use them.
struct CompilerAssets {
    bindings: Vec<(String, &'static [u8])>,
    sha256: [u8; 32],
}

fn immutable_file(path: &str) -> Result<&'static [u8]> {
    api::immutable_data(path)
        .map_err(|_| Error::Firmware)?
        .ok_or(d3os_nvidia::Error::Unsupported.into())
}

impl CompilerAssets {
    fn load() -> Result<Self> {
        let bytes = immutable_file("/usr/share/nvidia/runtime.qualification")?;
        let inventory = RuntimeInventory::parse(bytes)?;
        let mut bindings = Vec::new();
        for asset in inventory.records() {
            let path = format!("/{}", asset.path());
            let data = immutable_file(&path)?;
            if data.len() as u64 != asset.bytes() || <[u8; 32]>::from(Sha256::digest(data)) != *asset.sha256() {
                log::warn!("NVIDIA_COMPILE_REJECT asset={path} reason=runtime_identity");
                return Err(d3os_nvidia::Error::Unsupported.into());
            }
            bindings.push((path, data));
        }
        // Retain the inventory binding as well as its covered runtime assets.
        bindings.push((String::from("/usr/share/nvidia/runtime.qualification"), bytes));
        Ok(Self {
            bindings,
            sha256: Sha256::digest(inventory.canonical_bytes()).into(),
        })
    }

    fn binding(&self, path: &str) -> Result<&'static [u8]> {
        self.bindings
            .iter()
            .find(|(name, _)| name == path)
            .map(|(_, data)| *data)
            .ok_or(d3os_nvidia::Error::Unsupported.into())
    }

    fn verify_bindings(&self) -> Result<()> {
        for (path, bytes) in &self.bindings {
            if !core::ptr::eq(immutable_file(path)?, *bytes) {
                log::warn!("NVIDIA_COMPILE_REJECT asset={path} reason=runtime_binding_changed");
                return Err(d3os_nvidia::Error::Unsupported.into());
            }
        }
        Ok(())
    }
}

/// Admission belongs to the registered immutable module, not equal-content
/// bytes supplied by another caller. Other modules use the strict default.
pub(super) fn legacy_policy(bytes: &'static [u8]) -> Result<LegacyPolicy<'static>> {
    let Some(module) = MODULES
        .get()
        .into_iter()
        .flatten()
        .chain(QWEN_VECTOR.get().and_then(Option::as_ref))
        .find(|module| core::ptr::eq(module.bytes.as_slice(), bytes))
    else {
        return Ok(LegacyPolicy::Reject);
    };
    let qualified = LegacyQualification::from_trusted_admission(
        bytes,
        *module.qualification.cubin_sha256(),
        qualification::COMPILER_VERSION,
        module.evidence.as_str(),
    )
    .map_err(Error::Module)?;
    Ok(LegacyPolicy::QualifiedUnused(qualified))
}

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
        2 => return Err(*FAILURE.get().unwrap_or(&Errno::EUNKN)),
        _ => (),
    }
    let completion = COMPLETION.get().ok_or(Errno::ENOTSUP)?;
    completion.wait(|| STATUS.load(Ordering::Acquire) != 0, "GPU compilation");
    if STATUS.load(Ordering::Acquire) == 1 {
        Ok(())
    } else {
        Err(*FAILURE.get().unwrap_or(&Errno::EUNKN))
    }
}

pub(super) fn inference() -> Option<&'static [u8]> {
    MODULES.get().map(|modules| modules[2].bytes.as_slice())
}

/// Optional PTX entries retain the same immutable-module admission as Tile IR.
/// Missing optional code fails explicitly instead of falling back to another kernel.
pub(super) fn projection_module(symbol: &str) -> Result<Option<&'static [u8]>> {
    match symbol {
        "q3_components_dp4a" | "q4_components_dp4a" | "q3_warp_m16" => QWEN_VECTOR
            .get()
            .and_then(Option::as_ref)
            .map(|module| Some(module.bytes.as_slice()))
            .ok_or(d3os_nvidia::Error::Unsupported.into()),
        _ => Ok(None),
    }
}

fn sha256(bytes: &[u8]) -> String {
    sha256_hex(&Sha256::digest(bytes))
}

pub(super) fn sha256_hex(digest: &[u8]) -> String {
    let mut result = String::with_capacity(64);
    for byte in digest {
        let _ = write!(result, "{byte:02x}");
    }
    result
}

fn compile(family: &str, target: &str, sm: u32, assets: &CompilerAssets) -> Result<CompiledModule> {
    let (input, tile) = input(family)?;
    let source = assets.binding(&input)?;
    let receipt_path = format!("/usr/share/nvidia/{family}.qualification");
    let receipt_bytes = immutable_file(&receipt_path)?;
    let qualification = ModuleQualification::parse(receipt_bytes)?;
    let input_hash = sha256(&source);
    let compiler = if tile { "/usr/cuda/bin/tileiras" } else { "/usr/cuda/bin/ptxas" };
    let input_kind = if tile { InputKind::TileIr } else { InputKind::Ptx };
    if qualification.input_kind() != input_kind
        || *qualification.source_sha256() != <[u8; 32]>::from(Sha256::digest(source))
        || *qualification.compiler_sha256() != <[u8; 32]>::from(Sha256::digest(assets.binding(compiler)?))
        || *qualification.runtime_sha256() != assets.sha256
        || sm != 89
    {
        log::warn!("NVIDIA_COMPILE_REJECT family={family} reason=qualification_input_identity");
        return Err(d3os_nvidia::Error::Unsupported.into());
    }
    assets.verify_bindings()?;
    let start = crate::timer().systime_ms();
    let cache_path = format!("/usr/share/nvidia/{family}.cubin");
    let (bytes, loaded, completed, output, mode) = match api::stat(&cache_path) {
        Ok(_) => {
            // A file outside the sealed runtime inventory cannot become a cache
            // hit. Its qualified output hash is checked below as for compilation.
            let cached = assets.binding(&cache_path)?;
            (cached.to_vec(), start, start, None, "qualified_cache")
        }
        Err(Errno::ENOENT) => {
            let output = format!("/tmp/{family}-{target}.cubin");
            match api::unlink(&output) {
                Ok(()) | Err(Errno::ENOENT) => (),
                Err(_) => return Err(Error::Firmware),
            }
            let target_option = format!("--gpu-name={target}");
            let options = vec![target_option.as_str(), "--opt-level=3", input.as_str(), "-o", output.as_str()];
            let thread = Thread::load_application(compiler, compiler, &options).map_err(|_| Error::Firmware)?;
            let loaded = crate::timer().systime_ms();
            let process = thread.process();
            let id = thread.id();
            scheduler().ready(thread);
            let _ = scheduler().join(id);
            let completed = crate::timer().systime_ms();
            let status = process.exit_status.load(Ordering::Acquire);
            if status != 0 {
                log::warn!("NVIDIA_COMPILE_FAIL family={family} target={target} exit={status}");
                return Err(Error::Firmware);
            }
            let bytes = crate::linux::read_file(&output).map_err(|_| Error::Firmware)?.into_owned();
            (bytes, loaded, completed, Some(output), "native_compile")
        }
        Err(_) => return Err(Error::Firmware),
    };
    assets.verify_bindings()?;
    if !core::ptr::eq(immutable_file(&receipt_path)?, receipt_bytes) {
        return Err(d3os_nvidia::Error::Unsupported.into());
    }
    if <[u8; 32]>::from(Sha256::digest(&bytes)) != *qualification.cubin_sha256() {
        log::warn!(
            "NVIDIA_COMPILE_REJECT family={family} reason=qualified_output_identity sha256={}",
            sha256(&bytes)
        );
        return Err(d3os_nvidia::Error::Unsupported.into());
    }
    let elf = goblin::elf::Elf::parse(&bytes).map_err(|_| Error::Firmware)?;
    if elf.header.e_machine != 190 || (elf.header.e_flags >> 8) & 0xff != sm {
        log::warn!(
            "NVIDIA_COMPILE_FAIL family={family} target={target} machine={} flags={:#x}",
            elf.header.e_machine,
            elf.header.e_flags
        );
        return Err(Error::Firmware);
    }
    let validated = crate::timer().systime_ms();
    log::info!(
        "NVIDIA_COMPILE_MODULE os=d3os family={family} input={} input_sha256={input_hash} target={target} compile_ms={} image_load_ms={} execute_ms={} output_ms={} bytes={} sha256={} exit=0 mode={mode} compiler_executed={}",
        if tile { "tile_ir" } else { "ptx" },
        validated - start,
        loaded - start,
        completed - loaded,
        validated - completed,
        bytes.len(),
        sha256(&bytes),
        output.is_some()
    );
    // The resident cache owns a private immutable copy. Temporary compiler
    // output cannot change an executable module after publication.
    if let Some(output) = output {
        api::unlink(&output).map_err(|_| Error::Firmware)?;
    }
    let evidence = format!("{} sass_sha256={}", qualification::PROFILE_LABEL, sha256_hex(qualification.sass_sha256()));
    Ok(CompiledModule {
        bytes,
        qualification,
        evidence,
    })
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
    let assets = CompilerAssets::load()?;
    let modules = [
        compile(FAMILIES[0], target, sm, &assets)?,
        compile(FAMILIES[1], target, sm, &assets)?,
        compile(FAMILIES[2], target, sm, &assets)?,
    ];
    let qwen = match api::stat("/usr/share/nvidia/qwen-vector.ptx") {
        Ok(_) => Some(compile("qwen-vector", target, sm, &assets)?),
        Err(Errno::ENOENT) => None,
        Err(_) => return Err(Error::Firmware),
    };
    QWEN_VECTOR.call_once(|| qwen);
    let compile_ms = crate::timer().systime_ms() - start;
    let modules = MODULES.call_once(|| modules);
    let _preempt = PreemptGuard::new();
    let mut device = device.lock();
    let mut compute = device.compute.take().ok_or(Error::Firmware)?;
    let result = (|| {
        compute.vector_add_test(&device, &modules[0].bytes)?;
        compute.tensor_test(&device, &modules[1].bytes)?;
        compute.inference_module = Some(&modules[2].bytes);
        compute.timestamp_test(&device, &modules[2].bytes)?;
        compute.log_clocks(&mut device)?;
        compute.local_stack_test(&device, &modules[2].bytes)?;
        compute.relocation_test(&device, &modules[2].bytes)?;
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
            FAILURE.call_once(|| error.errno());
            log::warn!("NVIDIA_COMPILE_FAIL os=d3os error={error}");
            2
        }
    };
    STATUS.store(status, Ordering::Release);
    if let Some(completion) = COMPLETION.get() {
        completion.notify_all();
    }
}
