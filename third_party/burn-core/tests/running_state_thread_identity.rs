//! Public RunningState tests with real host threads and an explicit Flex device.

use burn_core::module::{Module, RunningState};
use burn_core::tensor::{Device, Tensor};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;

const CASES: &[&str] = &["cohort", "repeated", "clone", "retired", "yield", "parallel", "transitions"];

#[cfg(not(feature = "std"))]
mod host_identity {
    use std::collections::HashMap;
    use std::num::NonZeroU64;
    use std::sync::{Mutex, OnceLock};
    use std::thread::ThreadId;

    struct Identities {
        threads: HashMap<ThreadId, NonZeroU64>,
        next: u64,
    }

    pub fn current() -> NonZeroU64 {
        static IDENTITIES: OnceLock<Mutex<Identities>> = OnceLock::new();
        let identities = IDENTITIES.get_or_init(|| {
            Mutex::new(Identities {
                threads: HashMap::new(),
                next: 1,
            })
        });
        let mut identities = identities.lock().expect("host identity map is poisoned");
        let thread = std::thread::current().id();
        if let Some(identity) = identities.threads.get(&thread) {
            return *identity;
        }
        let identity = NonZeroU64::new(identities.next).expect("host identity is zero");
        identities.next = identities.next.checked_add(1).expect("host IDs exhausted");
        identities.threads.insert(thread, identity);
        identity
    }
}

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    match arguments.as_slice() {
        [] => supervise(),
        [flag, case] if flag == "--case" => {
            run_case(case.to_str().expect("case name is not UTF-8"));
        }
        _ => panic!("expected no arguments or --case <name>"),
    }
}

fn supervise() {
    use std::process::Command;
    use std::time::{Duration, Instant};

    let executable = std::env::current_exe().expect("test executable is unavailable");
    let mut completed = 0;
    for case in CASES {
        let mut child = Command::new(&executable)
            .args(["--case", *case])
            .spawn()
            .expect("cannot start isolated test case");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = child.try_wait().expect("cannot inspect test case") {
                assert!(status.success(), "case {case} failed with {status}");
                completed += 1;
                break;
            }
            if Instant::now() >= deadline {
                child.kill().expect("cannot stop timed-out test case");
                child.wait().expect("cannot reap timed-out test case");
                panic!("case {case} exceeded 30 seconds");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    assert_eq!(completed, CASES.len());
    eprintln!("{completed} isolated cases completed");
}

fn run_case(case: &str) {
    #[cfg(not(feature = "std"))]
    burn_std::sync::register_thread_id_provider(host_identity::current).expect("initial provider install failed");

    match case {
        "cohort" => contributor_sync(),
        "repeated" => repeated_update_drains_before_insertion(),
        "clone" => clones_share_pending_and_committed_values(),
        "retired" => retired_thread_keys_remain_distinct(),
        "yield" => yielding_preserves_contributor_identity(),
        "parallel" => parallel_cohorts_keep_every_contribution(),
        "transitions" => transitions_preserve_buffer_and_device_semantics(),
        _ => panic!("unknown case: {case}"),
    }
}

fn tensor(value: f32, device: &Device) -> Tensor<1> {
    Tensor::from_floats([value], device).detach()
}

fn assert_value(value: Tensor<1>, expected: f32) {
    assert_eq!(value.into_scalar::<f32>(), expected);
}

fn contributor_sync() {
    let device = Device::flex().autodiff();
    let state = RunningState::new(tensor(0.0, &device));
    let (ready_tx, ready_rx) = mpsc::channel();
    let (flush_tx, flush_rx) = mpsc::channel();
    let first = state.clone();
    let first_device = device.clone();
    let worker = thread::spawn(move || {
        first.update(tensor(2.0, &first_device));
        ready_tx.send(()).expect("coordinator disappeared");
        flush_rx.recv().expect("flush request missing");
        assert_value(first.value_sync(), 4.0);
    });
    ready_rx.recv().expect("first update missing");
    let second = state.clone();
    thread::spawn(move || second.update(tensor(6.0, &device)))
        .join()
        .expect("second worker panicked");
    assert_value(state.value(), 0.0);
    assert_value(state.value_sync(), 0.0);
    flush_tx.send(()).expect("first worker disappeared");
    worker.join().expect("first worker panicked");
    assert_value(state.value(), 4.0);
}

fn repeated_update_drains_before_insertion() {
    let device = Device::flex().autodiff();
    let state = RunningState::new(tensor(0.0, &device));
    let (ready_tx, ready_rx) = mpsc::channel();
    let (step_tx, step_rx) = mpsc::channel();
    let first = state.clone();
    let first_device = device.clone();
    let worker = thread::spawn(move || {
        first.update(tensor(2.0, &first_device));
        ready_tx.send(()).expect("coordinator disappeared");
        step_rx.recv().expect("second update request missing");
        first.update(tensor(10.0, &first_device));
        ready_tx.send(()).expect("coordinator disappeared");
        step_rx.recv().expect("flush request missing");
        assert_value(first.value_sync(), 10.0);
    });
    ready_rx.recv().expect("first update missing");
    let second = state.clone();
    thread::spawn(move || second.update(tensor(6.0, &device)))
        .join()
        .expect("second worker panicked");
    step_tx.send(()).expect("first worker disappeared");
    ready_rx.recv().expect("repeated update missing");
    assert_value(state.value(), 4.0);
    assert_value(state.value_sync(), 4.0);
    step_tx.send(()).expect("first worker disappeared");
    worker.join().expect("first worker panicked");
    assert_value(state.value(), 10.0);
}

fn clones_share_pending_and_committed_values() {
    let device = Device::flex().autodiff();
    let state = RunningState::new(tensor(0.0, &device));
    let shared = state.clone();
    state.update(tensor(2.0, &device));
    assert_value(shared.value(), 0.0);
    shared.update(tensor(6.0, &device));
    assert_value(state.value(), 2.0);
    assert_value(shared.value_sync(), 6.0);
    assert_value(state.value(), 6.0);
}

fn retired_thread_keys_remain_distinct() {
    let device = Device::flex().autodiff();
    let state = RunningState::new(tensor(0.0, &device));
    for value in [2.0, 6.0] {
        let state = state.clone();
        let device = device.clone();
        thread::spawn(move || state.update(tensor(value, &device)))
            .join()
            .expect("retiring worker panicked");
    }
    assert_value(state.value_sync(), 0.0);
    let valid = state.valid();
    assert_value(valid.value(), 4.0);
    assert_value(state.value(), 4.0);
    assert!(!valid.value().is_autodiff());
    assert!(state.value().is_autodiff());
}

fn yielding_preserves_contributor_identity() {
    let device = Device::flex().autodiff();
    let state = RunningState::new(tensor(0.0, &device));
    let shared = state.clone();
    thread::spawn(move || {
        shared.update(tensor(8.0, &device));
        for _ in 0..32 {
            thread::yield_now();
        }
        assert_value(shared.value_sync(), 8.0);
    })
    .join()
    .expect("yielding worker panicked");
    assert_value(state.value(), 8.0);
}

fn parallel_cohorts_keep_every_contribution() {
    let device = Device::flex().autodiff();
    let state = RunningState::new(tensor(0.0, &device));
    let barrier = Arc::new(Barrier::new(4));
    let workers: Vec<_> = (0..4)
        .map(|worker| {
            let state = state.clone();
            let device = device.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                for round in 0..8 {
                    state.update(tensor((round * 4 + worker) as f32, &device));
                    barrier.wait();
                    if worker == 0 {
                        assert_value(state.value_sync(), (round * 4) as f32 + 1.5);
                    }
                    barrier.wait();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().expect("cohort worker panicked");
    }
    assert_value(state.value(), 29.5);
}

fn transitions_preserve_buffer_and_device_semantics() {
    let device = Device::flex().autodiff();
    let state = RunningState::new(tensor(0.0, &device));
    state.update(tensor(7.0, &device));
    let valid = state.valid();
    assert_value(valid.value(), 7.0);
    assert!(!valid.value().is_autodiff());
    assert!(!valid.value().is_require_grad());
    assert!(state.value().is_autodiff());
    assert_eq!(state.devices(), vec![device.clone()]);
    let trained = valid.train();
    assert!(trained.value().is_autodiff());
    assert!(!trained.value().is_require_grad());
    assert_value(trained.value(), 7.0);
    let mapped = trained.set_require_grad(true);
    assert!(!mapped.value().is_require_grad());
    let moved = mapped.to_device(&Device::flex());
    assert!(moved.value().is_autodiff());
    assert!(!moved.value().is_require_grad());
    assert_value(moved.value(), 7.0);
}
