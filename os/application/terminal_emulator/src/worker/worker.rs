pub trait Worker {
    /// Process pending work and report whether anything changed.
    fn run(&mut self) -> bool;
}
