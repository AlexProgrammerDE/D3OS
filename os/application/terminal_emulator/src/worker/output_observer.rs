use alloc::rc::Rc;
use stream::OutputStream;
use syscall::{SystemCall, syscall};

use crate::terminal::lfb_terminal::LFBTerminal;

use super::worker::Worker;

const BUFFER_SIZE: usize = 128;

pub struct OutputObserver {
    terminal: Rc<LFBTerminal>,
}

impl OutputObserver {
    pub const fn new(terminal: Rc<LFBTerminal>) -> Self {
        Self { terminal }
    }
}

impl Worker for OutputObserver {
    fn run(&mut self) -> bool {
        let mut buffer = [0u8; BUFFER_SIZE];

        let read_bytes = syscall(SystemCall::TerminalReadOutput, &[buffer.as_mut_ptr() as usize, buffer.len()]).expect("Unable to read output from tty");

        for &byte in &buffer[..read_bytes] {
            self.terminal.write_byte(byte);
        }
        read_bytes != 0
    }
}
