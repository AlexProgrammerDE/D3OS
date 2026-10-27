//! Process-isolated tests of the platform provider under both `std` selections.

use cubecl_environment::sync::{AlreadyInstalled, current_thread_id, register_thread_id_provider};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::{Arc, Barrier};
use std::thread;

const CASES: &[&str] = &["missing", "race", "one-shot", "identity", "callback-lock"];

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
    match case {
        "missing" => missing_provider(),
        "race" => concurrent_installers(),
        "one-shot" => one_shot_installation(),
        "identity" => thread_identity_lifetime(),
        "callback-lock" => callback_runs_without_registration_lock(),
        _ => panic!("unknown case: {case}"),
    }
}

fn other_host_provider() -> NonZeroU64 {
    host_identity::current()
}

fn missing_provider() {
    assert!(std::panic::catch_unwind(current_thread_id).is_err());
    register_thread_id_provider(host_identity::current).expect("initial install failed");
    assert_eq!(current_thread_id(), current_thread_id());
}

fn concurrent_installers() {
    let barrier = Arc::new(Barrier::new(3));
    let first_barrier = barrier.clone();
    let first = thread::spawn(move || {
        first_barrier.wait();
        register_thread_id_provider(host_identity::current)
    });
    let second_barrier = barrier.clone();
    let second = thread::spawn(move || {
        second_barrier.wait();
        register_thread_id_provider(other_host_provider)
    });
    barrier.wait();
    let outcomes = [
        first.join().expect("first installer panicked"),
        second.join().expect("second installer panicked"),
    ];
    assert_eq!(outcomes.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(outcomes.iter().filter(|result| **result == Err(AlreadyInstalled)).count(), 1,);
    let identity = current_thread_id();
    thread::yield_now();
    assert_eq!(current_thread_id(), identity);
}

fn one_shot_installation() {
    register_thread_id_provider(host_identity::current).expect("initial install failed");
    assert_eq!(register_thread_id_provider(host_identity::current), Err(AlreadyInstalled));
    assert_eq!(register_thread_id_provider(other_host_provider), Err(AlreadyInstalled));
    assert_eq!(current_thread_id(), current_thread_id());
}

fn thread_identity_lifetime() {
    register_thread_id_provider(host_identity::current).expect("initial install failed");
    let coordinator = current_thread_id();
    let barrier = Arc::new(Barrier::new(9));
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let barrier = barrier.clone();
            thread::spawn(move || {
                let identity = current_thread_id();
                barrier.wait();
                for _ in 0..32 {
                    thread::yield_now();
                    assert_eq!(current_thread_id(), identity);
                }
                identity
            })
        })
        .collect();
    barrier.wait();
    let retired: HashSet<_> = workers.into_iter().map(|worker| worker.join().expect("identity worker panicked")).collect();
    assert_eq!(retired.len(), 8);
    assert!(!retired.contains(&coordinator));
    let mut pending = HashMap::new();
    for identity in retired {
        assert!(pending.insert(identity, ()).is_none());
    }
    for _ in 0..8 {
        let identity = thread::spawn(current_thread_id).join().expect("later worker panicked");
        assert_ne!(identity, coordinator);
        assert!(pending.insert(identity, ()).is_none());
    }
    assert_eq!(pending.len(), 16);
    assert_eq!(current_thread_id(), coordinator);
}

fn provider_with_registration_check() -> NonZeroU64 {
    assert_eq!(register_thread_id_provider(host_identity::current), Err(AlreadyInstalled));
    host_identity::current()
}

fn callback_runs_without_registration_lock() {
    register_thread_id_provider(provider_with_registration_check).expect("initial install failed");
    assert_eq!(current_thread_id(), current_thread_id());
}
