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
use spin::Mutex;
use terminal::TerminalMode;

use crate::process::core_local_storage::scheduler;

/// Shared terminal input queue.
/// A read selects the encoding for subsequent input, including between calls.
/// Changing the encoding discards bytes from the previous mode.
/// Readers that use different modes must coordinate their access.
///
/// Author: Sebastian Keller
#[derive(Debug)]
pub struct TtyInput {
    input: Mutex<Input>,
}

#[derive(Debug)]
struct Input {
    buffer: VecDeque<u8>,
    mode: Option<TerminalMode>,
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

impl TtyInput {
    pub const fn new() -> Self {
        TtyInput {
            input: Mutex::new(Input {
                buffer: VecDeque::new(),
                mode: None,
            }),
        }
    }

    pub fn read(&self, buffer: &mut [u8], mode: TerminalMode) -> usize {
        if buffer.is_empty() {
            return 0;
        }
        {
            let mut input = self.input.lock();
            // A terminated reader can leave an undecoded key in the buffer.
            // Its bytes have no meaning in the next reader's terminal mode.
            if input.mode != Some(mode) {
                input.buffer.clear();
            }
            input.mode = Some(mode);
        }

        loop {
            let mut input = self.input.lock();
            if !input.buffer.is_empty() {
                let count = buffer.len().min(input.buffer.len());
                for byte in &mut buffer[..count] {
                    *byte = input.buffer.pop_front().unwrap();
                }
                return count;
            }
            drop(input);
            scheduler().switch_thread_no_interrupt();
        }
    }

    pub fn write(&self, bytes: &[u8], mode: TerminalMode) -> usize {
        let mut input = self.input.lock();
        if input.mode != Some(mode) {
            return 0; //Abort, mismatched mode
        }

        input.buffer.extend(bytes);
        bytes.len()
    }

    /// Keep the input encoding active while the application handles earlier input.
    pub fn input_mode(&self) -> Option<TerminalMode> {
        self.input.lock().mode
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
