pub mod file;
pub mod print;
pub mod scan;

#[repr(C)]
pub struct FILE {
    handle: usize,
}

static TERMINAL_OUT: FILE = FILE { handle: usize::MAX };
static TERMINAL_ERR: FILE = FILE { handle: usize::MAX - 1 };

#[unsafe(no_mangle)]
static stdout: &FILE = &TERMINAL_OUT;
#[unsafe(no_mangle)]
static stderr: &FILE = &TERMINAL_ERR;
