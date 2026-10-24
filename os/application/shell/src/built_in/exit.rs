use concurrent::process;

use crate::built_in::built_in::BuiltIn;

pub struct ExitBuiltIn {}

impl BuiltIn for ExitBuiltIn {
    fn namespace(&self) -> &'static str {
        "exit"
    }

    fn run(&mut self, _args: &[&str]) -> usize {
        process::exit()
    }
}

impl ExitBuiltIn {
    pub fn new() -> Self {
        Self {}
    }
}
