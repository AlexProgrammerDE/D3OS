/* ╔═════════════════════════════════════════════════════════════════════════╗
   ║ Module: tty                                                             ║
   ╟─────────────────────────────────────────────────────────────────────────╢
   ║ TTY-Input device (Workaround for missing pipes).                        ║
   ║ Buffers input from the terminal when an application is reading.         ║
   ║ Depending on the chosen TerminalMode, the application will block until  ║ 
   ║ terminal has written some input. Just a Workaround. Parallel reads      ║
   ║ where not considered and might cause problems.                          ║
   ╟─────────────────────────────────────────────────────────────────────────╢
   ║ Author: Sebastian Keller, Univ. Duesseldorf, 2025                       ║
   ╚═════════════════════════════════════════════════════════════════════════╝
*/

use alloc::collections::vec_deque::VecDeque;
use num_enum::{FromPrimitive, IntoPrimitive};
use spin::Mutex;
use terminal::TerminalMode;

use crate::process::core_local_storage::scheduler;

/// TTY-Input device (Workaround for missing pipes).
/// Buffers input from the terminal when an application is reading.
/// Depending on the chosen TerminalMode, the application will block until terminal has written some input.
///
/// NOTE: Just a Workaround. Parallel reads where not considered and might cause problems.
///
/// Author: Sebastian Keller
#[derive(Debug)]
pub struct TtyInput {
    input: Mutex<Input>,
}

#[derive(Debug)]
struct Input {
    buffer: VecDeque<u8>,
    state: TtyInputState,
    mode: TerminalMode,
}

/// TTY-Output device (Workaround for missing pipes).
/// Buffers output from from applications.
/// Applications don't block and can continue after writing.
/// The terminal collects the data from the buffer if present.
///
/// NOTE: Just a Workaround. Parallel writes where not considered and might cause problems.
/// NOTE: The application will write faster then the terminal can collect.
///       Writing a extreme amount of data will cause an out-of-memory error.
///
/// Author: Sebastian Keller
#[derive(Debug)]
pub struct TtyOutput {
    buffer: Mutex<VecDeque<u8>>,
}

#[derive(Debug, PartialEq, IntoPrimitive, FromPrimitive, Clone, Copy)]
#[repr(usize)]
pub enum TtyInputState {
    #[num_enum(default)]
    Idle = 0,
    Waiting = 1,
    Ready = 2,
}

impl TtyInput {
    pub const fn new() -> Self {
        TtyInput {
            input: Mutex::new(Input {
                buffer: VecDeque::new(),
                state: TtyInputState::Idle,
                mode: TerminalMode::Canonical,
            }),
        }
    }

    pub fn read(&self, buffer: &mut [u8], mode: TerminalMode) -> usize {
        {
            let mut input = self.input.lock();
            // A terminated reader can leave an undecoded key in the buffer.
            // Its bytes have no meaning in the next reader's terminal mode.
            if input.mode != mode {
                input.buffer.clear();
            }
            input.mode = mode;
            input.state = TtyInputState::Waiting;
        }

        loop {
            let mut input = self.input.lock();
            if input.state == TtyInputState::Ready {
                let count = buffer.len().min(input.buffer.len());
                for byte in &mut buffer[..count] {
                    *byte = input.buffer.pop_front().unwrap();
                }
                input.state = TtyInputState::Idle;
                return count;
            }
            drop(input);
            scheduler().switch_thread_no_interrupt();
        }
    }

    pub fn write(&self, bytes: &[u8], mode: TerminalMode) -> usize {
        let mut input = self.input.lock();
        if input.state != TtyInputState::Waiting {
            return 0; // Abort, no more readers
        }
        if input.mode != mode {
            return 0; //Abort, mismatched mode
        }

        input.buffer.extend(bytes);
        input.state = TtyInputState::Ready;
        bytes.len()
    }

    pub fn waiting_mode(&self) -> Option<TerminalMode> {
        let input = self.input.lock();
        (input.state == TtyInputState::Waiting).then_some(input.mode)
    }
}

impl TtyOutput {
    pub const fn new() -> Self {
        Self {
            buffer: Mutex::new(VecDeque::new()),
        }
    }

    pub fn write(&self, bytes: &[u8]) -> usize {
        let mut output_buffer = self.buffer.lock();
        let mut count = 0;
        for byte in bytes {
            output_buffer.push_back(*byte);
            count += 1;
        }
        if output_buffer.len() > 4096 {
//            info!("tty write, yield now, buffer size: {}", output_buffer.len());
            scheduler().sleep(100);
        }
        count
    }

    pub fn read(&self, buffer: &mut [u8]) -> usize {
        let mut output_buffer = self.buffer.lock();
        let mut count = 0;
        for byte in buffer {
            *byte = match output_buffer.pop_front() {
                Some(read_byte) => {
                    count += 1;
                    read_byte
                }
                None => break,
            };
        }

        count
    }
}
