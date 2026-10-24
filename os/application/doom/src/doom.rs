#![no_std]

extern crate alloc;

use alloc::ffi::CString;
use alloc::format;
use libc as _;

use ::time::systime;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use chrono::TimeDelta;
use concurrent::thread;
use core::ffi::{CStr, c_char, c_int};
use graphic::color;
use graphic::lfb::{FramebufferInfo, LFB, map_framebuffer};
use pc_keyboard::{DecodedKey, HandleControl, KeyCode, KeyEvent, KeyState, KeyboardLayout, Modifiers, layouts};
use spin::Mutex;
use spin::once::Once;

unsafe extern "C" {
    static DG_ScreenBuffer: *const u32;
    static mut key_strafeleft: c_int;
    static mut key_straferight: c_int;
    static mut key_use: c_int;
    static mut key_fire: c_int;

    fn doomgeneric_Create(argc: c_int, argv: *mut *mut c_char) -> c_int;
    fn doomgeneric_Tick();
}

const DOOMGENERIC_RESX: u32 = 640;
const DOOMGENERIC_RESY: u32 = 400;

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum DoomKey {
    RightArrow = 0xae,
    LeftArrow = 0xac,
    UpArrow = 0xad,
    DownArrow = 0xaf,
    Run = 0xb6,
    Escape = 27,
    Enter = 13,
    Tab = 9,
    F1 = 0x80 + 0x3b,
    F2 = 0x80 + 0x3c,
    F3 = 0x80 + 0x3d,
    F4 = 0x80 + 0x3e,
    F5 = 0x80 + 0x3f,
    F6 = 0x80 + 0x40,
    F7 = 0x80 + 0x41,
    F8 = 0x80 + 0x42,
    F9 = 0x80 + 0x43,
    F10 = 0x80 + 0x44,
    F11 = 0x80 + 0x57,
    F12 = 0x80 + 0x58,
    Backspace = 0x7f,
}

static START_TIME_MS: Once<TimeDelta> = Once::new();
static LOGGER: logger::Logger = logger::Logger::new();

mod gpu;

#[unsafe(no_mangle)]
unsafe extern "C" fn doom_engine_report(message: *const c_char) {
    log::info!("DOOM_ENGINE_REPORT {}", unsafe { CStr::from_ptr(message) }.to_string_lossy());
    gpu::report_profile();
}

static WINDOW_TITLE: Once<String> = Once::new();

static FRAMEBUFFER: Once<FramebufferInfo> = Once::new();
static SCREEN_OFFSET: Once<(u32, u32)> = Once::new();

static INPUT_BUFFER: Mutex<VecDeque<KeyEvent>> = Mutex::new(VecDeque::new());

struct FpsInfo {
    start_time: TimeDelta,
    counter: usize,
    last_fps: usize,
}

static FPS_INFO: Once<Mutex<FpsInfo>> = Once::new();

impl FpsInfo {
    fn add_frame(&mut self) {
        self.counter += 1;

        let time = systime();
        if (time - self.start_time).num_milliseconds() >= 1000 {
            self.last_fps = self.counter * 1000 / (time - self.start_time).num_milliseconds() as usize;
            if gpu::enabled() {
                log::info!("NVIDIA_DOOM_FPS fps={}", self.last_fps);
            }
            self.counter = 0;
            self.start_time = time;
        }
    }
}

impl TryFrom<KeyCode> for DoomKey {
    type Error = ();

    fn try_from(value: KeyCode) -> Result<Self, Self::Error> {
        match value {
            KeyCode::ArrowRight => Ok(Self::RightArrow),
            KeyCode::ArrowLeft => Ok(Self::LeftArrow),
            KeyCode::ArrowUp => Ok(Self::UpArrow),
            KeyCode::ArrowDown => Ok(Self::DownArrow),
            KeyCode::LShift | KeyCode::RShift => Ok(Self::Run),
            KeyCode::Escape => Ok(Self::Escape),
            KeyCode::Return => Ok(Self::Enter),
            KeyCode::Tab => Ok(Self::Tab),
            KeyCode::F1 => Ok(Self::F1),
            KeyCode::F2 => Ok(Self::F2),
            KeyCode::F3 => Ok(Self::F3),
            KeyCode::F4 => Ok(Self::F4),
            KeyCode::F5 => Ok(Self::F5),
            KeyCode::F6 => Ok(Self::F6),
            KeyCode::F7 => Ok(Self::F7),
            KeyCode::F8 => Ok(Self::F8),
            KeyCode::F9 => Ok(Self::F9),
            KeyCode::F10 => Ok(Self::F10),
            KeyCode::F11 => Ok(Self::F11),
            KeyCode::F12 => Ok(Self::F12),
            KeyCode::Backspace => Ok(Self::Backspace),
            _ => Err(()),
        }
    }
}

#[unsafe(no_mangle)]
pub fn main() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    let args: Vec<_> = runtime::env::args().skip(1).collect();
    let verify = args.iter().any(|arg| arg == "--verify-gpu");
    if verify || args.iter().any(|arg| arg == "--gpu") {
        gpu::init(verify).expect("native GPU renderer initialization failed");
    }
    let mut arguments = alloc::vec![
        CString::new("doom").unwrap(),
        // D3OS reports engine errors in its terminal, including timedemo results.
        CString::new("-nogui").unwrap(),
        CString::new("-iwad").unwrap(),
        CString::new("/usr/doom.wad").unwrap()
    ];
    for arg in args.iter().filter(|arg| arg.as_str() != "--gpu" && arg.as_str() != "--verify-gpu") {
        arguments.push(CString::new(arg.as_str()).unwrap());
    }
    let mut argv: Vec<_> = arguments.iter().map(|arg| arg.as_ptr().cast_mut()).collect();
    argv.push(core::ptr::null_mut());

    unsafe {
        // Keep letter events intact for menus and cheat codes as well as actions.
        key_strafeleft = c_int::from(b'a');
        key_straferight = c_int::from(b'd');
        key_use = c_int::from(b'e');
        key_fire = c_int::from(b' ');
        doomgeneric_Create(arguments.len() as c_int, argv.as_mut_ptr());

        // Enter the main loop, calling doomgeneric_Tick() repeatedly.
        // This function handles all game logic and rendering and calls our DG_* functions as needed.
        loop {
            doomgeneric_Tick();
        }
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn DG_Init() {
    START_TIME_MS.call_once(systime);

    FPS_INFO.call_once(|| {
        Mutex::new(FpsInfo {
            start_time: systime(),
            counter: 0,
            last_fps: 0,
        })
    });

    FRAMEBUFFER.call_once(|| map_framebuffer().unwrap());

    let fb_info = FRAMEBUFFER.get().unwrap();
    let x_offset = (fb_info.width - DOOMGENERIC_RESX) / 2;
    let y_offset = (fb_info.height - DOOMGENERIC_RESY) / 2;
    SCREEN_OFFSET.call_once(|| (x_offset, y_offset));

    // Start input reading thread
    thread::create(|| {
        loop {
            if let Some(event) = terminal::read::read_raw() {
                INPUT_BUFFER.lock().push_back(event);
            } else {
                thread::switch();
            }
        }
    });
}

#[unsafe(no_mangle)]
unsafe extern "C" fn DG_DrawFrame() {
    let start = gpu::cycles();
    if let Some(fb_info) = FRAMEBUFFER.get() {
        let (pos_x, pos_y) = *SCREEN_OFFSET.get().unwrap();
        let title = WINDOW_TITLE.get().unwrap().as_str();
        let title_x = pos_x + (DOOMGENERIC_RESX - title.len() as u32 * 8) / 2;
        let title_y = pos_y - 16;

        let fb_ptr = fb_info.addr as *mut u32;
        for y in 0..DOOMGENERIC_RESY {
            unsafe {
                let src = DG_ScreenBuffer.add(y as usize * DOOMGENERIC_RESX as usize);
                let dest = fb_ptr.add((pos_y + y) as usize * (fb_info.pitch / 4) as usize + pos_x as usize);

                dest.copy_from_nonoverlapping(src, DOOMGENERIC_RESX as usize);
            }
        }

        let mut fps_info = FPS_INFO.get().unwrap().lock();
        let fps_string = format!("FPS: {}", fps_info.last_fps);

        let mut lfb = LFB::new(fb_info.addr as *mut u8, fb_info.pitch, fb_info.width, fb_info.height, fb_info.bpp);
        lfb.draw_string(title_x, title_y, color::RED, color::BLACK, title);
        lfb.draw_string(pos_x, title_y, color::YELLOW, color::BLACK, fps_string.as_str());
        fps_info.add_frame();
        gpu::displayed(gpu::cycles().wrapping_sub(start));
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn DG_SleepMs(ms: u32) {
    thread::sleep(ms as usize);
}

#[unsafe(no_mangle)]
unsafe extern "C" fn DG_GetTicksMs() -> u32 {
    (systime() - *START_TIME_MS.get().unwrap()).num_milliseconds() as u32
}

#[unsafe(no_mangle)]
unsafe extern "C" fn DG_GetKey(pressed: *mut c_int, key: *mut c_char) -> c_int {
    let mut key_buffer = INPUT_BUFFER.lock();
    while let Some(key_event) = key_buffer.pop_front() {
        unsafe {
            pressed.write(match key_event.state {
                KeyState::Down => 1,
                _ => 0,
            });
        }

        let doom_key = DoomKey::try_from(key_event.code).map(|key| key as u8).ok().or_else(|| {
            match layouts::De105Key.map_keycode(key_event.code, &Modifiers::default(), HandleControl::Ignore) {
                DecodedKey::Unicode(c) if c.is_ascii() => Some(c as u8),
                _ => None,
            }
        });
        if let Some(doom_key) = doom_key {
            if gpu::enabled() {
                log::info!("NVIDIA_DOOM_INPUT key={} pressed={}", doom_key, key_event.state == KeyState::Down);
            }
            unsafe {
                key.write(doom_key as c_char);
            }
            return 1;
        }
    }

    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn DG_SetWindowTitle(title: *const c_char) {
    WINDOW_TITLE.call_once(|| unsafe { CStr::from_ptr(title as *mut c_char).to_str().unwrap_or("Doom").to_string() });
}
