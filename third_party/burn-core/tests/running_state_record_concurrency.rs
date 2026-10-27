//! Public record semantics without an implicit flush or checkpoint barrier.

use burn::module::{Flag, Module, ModuleVisitor, Param, ParamId, RunningState};
use burn::store::ModuleRecord;
use burn::tensor::{Device, Tensor};
use burn_core as burn;
use std::sync::{Arc, Barrier};
use std::thread;

const CASES: &[&str] = &["c3", "unsynced", "pending-load", "from-record", "concurrent"];

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
        "c3" => same_thread_sync_precedes_first_record(),
        "unsynced" => direct_record_keeps_pending_updates(),
        "pending-load" => loading_keeps_destination_pending_updates(),
        "from-record" => fresh_record_state_preserves_identity(),
        "concurrent" => concurrent_records_observe_committed_values(),
        _ => panic!("unknown case: {case}"),
    }
}

#[derive(Module, Debug)]
struct State {
    running: RunningState<Tensor<1>>,
    training: Param<Flag>,
}

fn tensor(value: f32, device: &Device) -> Tensor<1> {
    Tensor::from_floats([value], device).detach()
}

fn state(value: f32, training: bool, device: &Device) -> State {
    State {
        running: RunningState::new(tensor(value, device)),
        training: Param::from_bool(training),
    }
}

fn assert_value(value: Tensor<1>, expected: f32) {
    assert_eq!(value.into_scalar::<f32>(), expected);
}

fn running_id(state: &RunningState<Tensor<1>>) -> ParamId {
    struct Identity(Option<ParamId>);
    impl ModuleVisitor for Identity {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            assert!(self.0.replace(param.id).is_none());
        }
    }
    let mut identity = Identity(None);
    state.visit(&mut identity);
    identity.0.expect("running buffer was not visited")
}

fn same_thread_sync_precedes_first_record() {
    let device = Device::flex().autodiff();
    let source = state(0.0, true, &device);
    source.running.update(tensor(7.0, &device));
    assert_value(source.running.value(), 0.0);
    assert_value(source.running.value_sync(), 7.0);
    let source_id = running_id(&source.running);
    let first_record = source.into_record();
    assert_eq!(first_record.len(), 1);
    let bytes = first_record.into_bytes().expect("record serialization failed");
    let record = ModuleRecord::from_bytes(bytes).expect("record decoding failed");
    let destination = state(-1.0, false, &device);
    let destination_id = running_id(&destination.running);
    let destination_flag_id = destination.training.id;
    assert_ne!(destination_id, source_id);
    let loaded = destination.load_record(record);
    assert_value(loaded.running.value(), 7.0);
    assert_eq!(running_id(&loaded.running), destination_id);
    assert_eq!(loaded.training.id, destination_flag_id);
    assert!(!loaded.training.is_enabled());
    assert!(loaded.running.value().is_autodiff());
    assert!(!loaded.running.value().is_require_grad());
}

fn direct_record_keeps_pending_updates() {
    let device = Device::flex().autodiff();
    let source = state(0.0, true, &device);
    source.running.update(tensor(7.0, &device));
    let unsynced = source.clone().into_record();
    let loaded = state(-1.0, false, &device).load_record(unsynced);
    assert_value(loaded.running.value(), 0.0);
    assert_value(source.running.value(), 0.0);
    assert_value(source.running.value_sync(), 7.0);
    assert!(!loaded.training.is_enabled());
    assert!(source.training.is_enabled());
}

fn loading_keeps_destination_pending_updates() {
    let device = Device::flex().autodiff();
    let source = state(5.0, true, &device);
    assert_value(source.running.value_sync(), 5.0);
    let record = source.into_record();
    let destination = state(0.0, false, &device);
    let alias = destination.running.clone();
    let identity = running_id(&destination.running);
    destination.running.update(tensor(9.0, &device));
    let loaded = destination.load_record(record);
    assert_value(alias.value(), 5.0);
    assert_eq!(running_id(&loaded.running), identity);
    assert!(!loaded.training.is_enabled());
    assert_value(loaded.running.value_sync(), 9.0);
    assert_value(alias.value(), 9.0);
}

fn fresh_record_state_preserves_identity() {
    let device = Device::flex().autodiff();
    let identity = ParamId::new();
    let state = RunningState::from_record(Param::initialized(identity, tensor(7.0, &device)));
    assert_eq!(running_id(&state), identity);
    assert_value(state.value_sync(), 7.0);
    state.update(tensor(9.0, &device));
    assert_value(state.value(), 7.0);
    assert_value(state.value_sync(), 9.0);
    let separate = RunningState::with_id(identity, tensor(3.0, &device));
    assert_eq!(running_id(&separate), identity);
    assert_value(separate.value_sync(), 3.0);
    assert_value(state.value(), 9.0);
}

fn concurrent_records_observe_committed_values() {
    let device = Device::flex().autodiff();
    let source = state(0.0, true, &device);
    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = barrier.clone();
    let worker_state = source.running.clone();
    let worker_device = device.clone();
    let worker = thread::spawn(move || {
        worker_barrier.wait();
        for value in 1..=16 {
            worker_state.update(tensor(value as f32, &worker_device));
            assert_value(worker_state.value_sync(), value as f32);
            thread::yield_now();
        }
    });
    barrier.wait();
    for _ in 0..32 {
        let snapshot = source.clone().into_record();
        let loaded = state(-1.0, false, &device).load_record(snapshot);
        let value = loaded.running.value().into_scalar::<f32>();
        assert!((0.0..=16.0).contains(&value));
        assert_eq!(value.fract(), 0.0);
        assert!(!loaded.training.is_enabled());
        thread::yield_now();
    }
    worker.join().expect("record writer panicked");
    assert_value(source.running.value(), 16.0);
}
