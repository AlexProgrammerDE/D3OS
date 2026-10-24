use concurrent::thread;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use pc_keyboard::{KeyCode, KeyState};
use time::systime;

static STARTED: AtomicBool = AtomicBool::new(false);
static EVENTS: AtomicUsize = AtomicUsize::new(0);
const KEYS: [KeyCode; 6] = [
    KeyCode::ArrowUp,
    KeyCode::ArrowLeft,
    KeyCode::Spacebar,
    KeyCode::F5,
    KeyCode::Tab,
    KeyCode::Escape,
];
const COMPLETE: usize = (1 << (KEYS.len() * 2)) - 1;

pub fn run(panic_on_complete: bool) {
    thread::create(|| {
        STARTED.store(true, Ordering::Release);
        loop {
            let Some(event) = terminal::read::read_raw() else {
                continue;
            };
            let Some(index) = KEYS.iter().position(|key| *key == event.code) else {
                continue;
            };
            let state = match event.state {
                KeyState::Down => 0,
                KeyState::Up => 1,
                KeyState::SingleShot => continue,
            };
            let bit = index * 2 + state;
            EVENTS.fetch_or(1 << bit, Ordering::AcqRel);
        }
    })
    .expect("start raw input worker");
    while !STARTED.load(Ordering::Acquire) {
        thread::switch();
    }
    log::info!("RAW_INPUT_READY");
    let deadline = systime().num_milliseconds() + 15_000;
    while EVENTS.load(Ordering::Acquire) != COMPLETE && systime().num_milliseconds() < deadline {
        core::hint::spin_loop();
    }
    let events = EVENTS.load(Ordering::Acquire);
    if events == COMPLETE {
        log::info!("RAW_INPUT_OK mask={events:#x}");
        if panic_on_complete {
            panic!("intentional raw-input process cleanup");
        }
    } else {
        log::error!("RAW_INPUT_FAIL mask={events:#x}");
    }
}
