//! Opt-in, serialized Qwen quantized projection diagnostics.
//!
//! The caller must enable device timing and own the device exclusively. Profile
//! snapshots add syscall work, so these measurements are never throughput runs.
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use d3os_compute::{Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{Kernel, ProfileEntry};
use spin::Mutex;

/// One selected specialization and its exact scalar launch packet.
#[derive(Clone, Debug)]
pub struct ProjectionEntry {
    pub entry: &'static str,
    pub parameters: Vec<u32>,
    pub calls: u64,
    pub prepare_cycles: u64,
    pub serial_wait_cycles: u64,
    pub device_ns: u64,
}

#[derive(Default)]
struct State {
    active: bool,
    entries: Vec<ProjectionEntry>,
}

impl State {
    fn record(&mut self, parameters: &[u32], before: ProfileEntry, after: ProfileEntry) -> Result<()> {
        let calls = after.calls.wrapping_sub(before.calls);
        let device_ns = after.device_ns.wrapping_sub(before.device_ns);
        // A different submitter or disabled device timing makes attribution
        // ambiguous. Reject it instead of manufacturing per-launch timings.
        if calls != 1 || device_ns == 0 {
            return Err(Error::InvalidParameter);
        }
        let entry = Kernel::ProjectQuantized.entry(parameters);
        let index = self.entries.iter().position(|row| row.parameters == parameters);
        let row = if let Some(index) = index {
            &mut self.entries[index]
        } else {
            self.entries.push(ProjectionEntry {
                entry,
                parameters: parameters.to_vec(),
                calls: 0,
                prepare_cycles: 0,
                serial_wait_cycles: 0,
                device_ns: 0,
            });
            self.entries.last_mut().expect("inserted projection entry")
        };
        row.calls += calls;
        row.prepare_cycles += after.cycles.wrapping_sub(before.cycles);
        row.serial_wait_cycles += after.wait_cycles.wrapping_sub(before.wait_cycles);
        row.device_ns += device_ns;
        Ok(())
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<State> = Mutex::new(State {
    active: false,
    entries: Vec::new(),
});

/// Exclusive process-local collector. Dropping it disables collection.
///
/// This does not enable GPU timestamps or synchronize unrelated submitters.
pub struct ProjectionProfiler(());

impl ProjectionProfiler {
    pub fn start() -> Result<Self> {
        let mut state = STATE.lock();
        if state.active {
            return Err(Error::InvalidParameter);
        }
        state.entries.clear();
        state.active = true;
        ENABLED.store(true, Ordering::Release);
        Ok(Self(()))
    }

    /// Drain a completed phase while keeping the collector enabled.
    pub fn take(&self) -> Vec<ProjectionEntry> {
        core::mem::take(&mut STATE.lock().entries)
    }
}

impl Drop for ProjectionProfiler {
    fn drop(&mut self) {
        let mut state = STATE.lock();
        ENABLED.store(false, Ordering::Release);
        state.active = false;
        state.entries.clear();
    }
}

pub(super) fn measure<T>(kernel: Kernel, parameters: &[usize], work: impl FnOnce() -> Result<T>) -> Result<T> {
    if kernel != Kernel::ProjectQuantized || !ENABLED.load(Ordering::Acquire) {
        return work();
    }
    let mut state = STATE.lock();
    if !state.active {
        return work();
    }
    let parameters = parameters
        .iter()
        .map(|&n| u32::try_from(n).map_err(|_| Error::SizeOverflow))
        .collect::<Result<Vec<_>>>()?;
    let before = Nvidia.profile()?[kernel as usize];
    let result = work()?;
    let after = Nvidia.profile()?[kernel as usize];
    state.record(&parameters, before, after)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(calls: u64, cycles: u64, wait_cycles: u64, device_ns: u64) -> ProfileEntry {
        ProfileEntry {
            calls,
            cycles,
            wait_cycles,
            device_ns,
        }
    }

    #[test]
    fn separates_launch_packets_and_accumulates_wrapping_counter_deltas() {
        let mut state = State::default();
        let before = counters(u64::MAX, u64::MAX - 2, u64::MAX - 4, u64::MAX - 8);
        let after = counters(0, 3, 5, 7);
        for packet in [
            &[32, 4096, 12288, 8, 0, 3][..],
            &[32, 4096, 12288, 8, 1, 3],
            &[32, 12288, 4096, 8, 1, 3],
            &[1, 4096, 12288, 8],
            &[4096, 8192, 4096, 32, 8],
            &[32, 4096, 12288, 8, 1, 3],
        ] {
            state.record(packet, before, after).unwrap();
        }
        assert_eq!(state.entries.len(), 5);
        assert_eq!(state.entries[0].entry, Kernel::ProjectQuantized.entry(&[32, 4096, 12288, 8, 0, 3]));
        let projection = &state.entries[1];
        assert_ne!(projection.entry, state.entries[0].entry);
        assert_eq!(
            (projection.calls, projection.prepare_cycles, projection.serial_wait_cycles, projection.device_ns),
            (2, 12, 20, 32)
        );
        assert_eq!(state.entries.iter().map(|entry| entry.calls).sum::<u64>(), 6);
    }

    #[test]
    fn rejects_missing_timestamps_and_concurrent_launch_attribution() {
        for after in [counters(0, 1, 1, 10), counters(2, 1, 1, 10), counters(1, 1, 1, 0)] {
            let mut state = State::default();
            assert_eq!(state.record(&[1, 4096, 4096, 8], ProfileEntry::default(), after), Err(Error::InvalidParameter));
            assert!(state.entries.is_empty());
        }
    }

    #[test]
    fn disabled_path_executes_once_without_accessing_the_device() {
        let collector = ProjectionProfiler::start().unwrap();
        assert!(ProjectionProfiler::start().is_err());
        assert!(collector.take().is_empty());
        drop(collector);
        let mut calls = 0;
        let result = measure(Kernel::ProjectQuantized, &[1, 4096, 4096, 8], || {
            calls += 1;
            Ok(7)
        });
        assert_eq!(result.unwrap(), 7);
        assert_eq!(calls, 1);
        drop(ProjectionProfiler::start().unwrap());
    }
}
