//! RunningState and record checks on real D3OS application threads.

use alloc::{format, string::String, sync::Arc, vec::Vec};
use burn::{
    module::{Flag, Module, Param, RunningState},
    store::ModuleRecord,
    tensor::{Device, Tensor},
};
use concurrent::thread;
use spin::Mutex;
use syscall::return_vals::Errno;

#[derive(Module, Debug)]
struct State {
    running: RunningState<Tensor<1>>,
    training: Param<Flag>,
}

fn tensor(value: f32, device: &Device) -> Tensor<1> {
    Tensor::from_floats([value], device).detach()
}

fn check(value: Tensor<1>, expected: f32) -> Result<(), String> {
    let actual = value.try_into_vec_as::<f32>().map_err(|error| format!("RunningState read: {error:?}"))?;
    if actual.as_slice() != [expected] {
        return Err(format!("RunningState value {actual:?}, expected [{expected}]"));
    }
    Ok(())
}

fn worker(state: RunningState<Tensor<1>>, device: Device, updates: Vec<f32>) -> Result<usize, String> {
    let id = thread::current().ok_or("cannot identify the RunningState worker")?.id();
    for _ in 0..64 {
        thread::switch();
        let current = thread::current().ok_or("RunningState worker identity disappeared")?.id();
        if current != id || id == 0 {
            return Err(format!("RunningState worker identity changed: {id} -> {current}"));
        }
    }
    for value in updates {
        state.update(tensor(value, &device));
    }
    Ok(id)
}

fn run_worker(state: &RunningState<Tensor<1>>, device: &Device, updates: Vec<f32>) -> Result<usize, String> {
    let output = Arc::new(Mutex::new(None));
    let result = Arc::clone(&output);
    let state = state.clone();
    let device = device.clone();
    let handle = thread::create(move || {
        *result.lock() = Some(worker(state, device, updates));
    })
    .ok_or("cannot create the RunningState worker")?;
    match handle.join() {
        Ok(_) | Err(Errno::ESRCH) => {}
        Err(error) => return Err(format!("cannot join the RunningState worker: {error:?}")),
    }
    let id = output.lock().take().ok_or("RunningState worker produced no result")??;
    if id != handle.id() {
        return Err(format!("RunningState worker reported {id}, created {}", handle.id()));
    }
    Ok(id)
}

/// Check retired-thread cohorts and an explicitly synchronized state record.
///
/// The caller supplies an autodiff device. This check uses exact F32 values and
/// runs after the application runtime installs the process identity provider.
pub fn run(device: &Device) -> Result<(), String> {
    if !device.is_autodiff() {
        return Err(String::from("RunningState validation requires an autodiff device"));
    }
    let main_id = thread::current().ok_or("cannot identify the RunningState main thread")?.id();
    let state = RunningState::new(tensor(0.0, device));
    let first = run_worker(&state, device, alloc::vec![4.0])?;
    let second = run_worker(&state, device, alloc::vec![8.0])?;
    // The caller has no pending update, so value_sync preserves the committed
    // value. The retired workers' two separate entries remain pending.
    check(state.value(), 0.0)?;
    check(state.value_sync(), 0.0)?;
    state.update(tensor(12.0, device));
    check(state.value_sync(), 8.0)?;
    let third = run_worker(&state, device, alloc::vec![16.0, 24.0])?;
    // A repeated update on one worker drains its previous cohort first.
    check(state.value(), 16.0)?;
    check(state.value_sync(), 16.0)?;
    state.update(tensor(40.0, device));
    check(state.value_sync(), 32.0)?;
    let ids = [main_id, first, second, third];
    for (position, id) in ids.iter().enumerate() {
        if *id == 0 || ids[..position].contains(id) {
            return Err(format!("RunningState thread identities are not distinct: {ids:?}"));
        }
    }

    let source = State {
        running: state,
        training: Param::from_bool(true),
    };
    // A record is not an implicit pending-update barrier. Synchronize on the
    // updating thread before exporting the first record.
    check(source.running.value_sync(), 32.0)?;
    let record = source.into_record();
    if record.len() != 1 {
        return Err(format!("RunningState record contains {} tensors, expected one", record.len()));
    }
    let bytes = record.into_bytes().map_err(|error| format!("RunningState record encode: {error:?}"))?;
    let record = ModuleRecord::from_bytes(bytes).map_err(|error| format!("RunningState record decode: {error:?}"))?;
    let destination = State {
        running: RunningState::new(tensor(-1.0, device)),
        training: Param::from_bool(false),
    };
    let flag_id = destination.training.id;
    let restored = destination
        .try_load_record(record)
        .map_err(|error| format!("RunningState record load: {error:?}"))?;
    check(restored.running.value(), 32.0)?;
    if restored.training.is_enabled() || restored.training.id != flag_id {
        return Err(String::from("RunningState record changed the destination training flag"));
    }
    let value = restored.running.value();
    if !value.is_autodiff() || value.is_require_grad() {
        return Err(String::from("RunningState record changed buffer autodiff metadata"));
    }
    restored.running.update(tensor(48.0, device));
    check(restored.running.value(), 32.0)?;
    check(restored.running.value_sync(), 48.0)?;
    Ok(())
}
