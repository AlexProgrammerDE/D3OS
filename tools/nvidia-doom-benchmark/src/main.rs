//! Linux reference for the same Rust DOOM draw collector and qualified kernels.
extern crate alloc;

#[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_feature = "avx2")))]
compile_error!("Build this matched reference for Linux x86-64 with RUSTFLAGS='-Ctarget-cpu=x86-64-v3 -Cno-redzone=yes'");

use std::{
    ffi::{CStr, CString, c_char, c_int},
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};

mod cuda;
#[path = "../../../os/application/doom/src/gpu.rs"]
mod gpu;
type GpuDevice = cuda::CudaDevice;

unsafe extern "C" {
    static DG_ScreenBuffer: *const u32;
    fn doomgeneric_Create(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn doomgeneric_Tick();
}

struct Logger;
static LOGGER: Logger = Logger;
impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            println!("{}", record.args());
        }
    }
    fn flush(&self) {}
}

static START: OnceLock<Instant> = OnceLock::new();
static FRAMEBUFFER: Mutex<Vec<u32>> = Mutex::new(Vec::new());
static CHECKSUM: AtomicU32 = AtomicU32::new(0);

fn milliseconds() -> i64 {
    START.get_or_init(Instant::now).elapsed().as_millis() as i64
}
fn exit_process() -> ! {
    std::process::exit(0)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn doom_engine_report(message: *const c_char) {
    println!("DOOM_ENGINE_REPORT {}", unsafe { CStr::from_ptr(message) }.to_string_lossy());
    gpu::report_profile();
    println!(
        "LINUX_DOOM_GPU_REPORT checksum={} presentation=headless_memory_copy width=800 height=600",
        CHECKSUM.load(Ordering::Relaxed)
    );
}

#[unsafe(no_mangle)]
extern "C" fn DG_Init() {
    START.get_or_init(Instant::now);
    FRAMEBUFFER.lock().unwrap().resize(800 * 600, 0);
}

#[unsafe(no_mangle)]
unsafe extern "C" fn DG_DrawFrame() {
    let start = gpu::cycles();
    let mut framebuffer = FRAMEBUFFER.lock().unwrap();
    for y in 0..400 {
        unsafe { std::ptr::copy_nonoverlapping(DG_ScreenBuffer.add(y * 640), framebuffer.as_mut_ptr().add((y + 100) * 800 + 80), 640) };
    }
    let sample = framebuffer[40000 + milliseconds() as usize % 300000];
    CHECKSUM.fetch_xor(sample, Ordering::Relaxed);
    drop(framebuffer);
    gpu::displayed(gpu::cycles().wrapping_sub(start));
}

#[unsafe(no_mangle)]
extern "C" fn DG_GetTicksMs() -> u32 {
    milliseconds() as u32
}
#[unsafe(no_mangle)]
extern "C" fn DG_SleepMs(milliseconds: u32) {
    std::thread::sleep(Duration::from_millis(milliseconds as u64));
}
#[unsafe(no_mangle)]
extern "C" fn DG_GetKey(_pressed: *mut c_int, _key: *mut c_char) -> c_int {
    0
}
#[unsafe(no_mangle)]
extern "C" fn DG_SetWindowTitle(_title: *const c_char) {}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    log::set_logger(&LOGGER).map_err(|_| "install benchmark logger")?;
    log::set_max_level(log::LevelFilter::Info);
    let mut args = std::env::args().skip(1);
    let cubin = args
        .next()
        .ok_or("usage: nvidia-doom-benchmark INFERENCE_CUBIN [--verify-gpu] -- ENGINE_ARGS...")?;
    let mut verify = false;
    let mut engine = Vec::new();
    for argument in args.by_ref() {
        if argument == "--" {
            break;
        }
        if argument != "--verify-gpu" {
            return Err(format!("unknown benchmark option: {argument}").into());
        }
        verify = true;
    }
    engine.push(CString::new("doom-linux-gpu")?);
    for argument in args {
        engine.push(CString::new(argument)?);
    }
    if engine.len() == 1 {
        return Err("explicit pinned engine arguments are required".into());
    }
    let device = GpuDevice::new(std::path::Path::new(&cubin))?;
    gpu::init(device, verify)?;
    assert!(gpu::enabled());
    println!(
        "LINUX_DOOM_GPU_READY verify={verify} scene_width=320 scene_height=200 output_width=640 output_height=400 presentation=headless_memory_copy native_window_cost_included=false"
    );
    let mut pointers: Vec<_> = engine.iter().map(|argument| argument.as_ptr().cast_mut()).collect();
    pointers.push(std::ptr::null_mut());
    unsafe {
        doomgeneric_Create(engine.len().try_into()?, pointers.as_mut_ptr());
        loop {
            doomgeneric_Tick();
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("LINUX_DOOM_GPU_FAIL {error}");
        std::process::exit(1);
    }
}
