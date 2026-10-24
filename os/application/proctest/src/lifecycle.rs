use alloc::{format, string::String, vec};
use concurrent::{process, shm, thread};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use syscall::return_vals::Errno;
use time::systime;

const WORKERS: usize = 4;

#[repr(align(64))]
struct Counter(AtomicUsize);

#[repr(C)]
struct Shared {
    started: AtomicUsize,
    stop: AtomicBool,
    counters: [Counter; WORKERS],
}

pub fn parent() -> Result<(), String> {
    let name = format!("process-lifecycle-{}", process::current().id());
    let size = size_of::<Shared>();
    let handle = shm::shm_open(&name, size, true).map_err(|e| format!("create shared state: {e:?}"))?;
    let address = shm::shm_attach(handle, false)
        .map_err(|e| format!("attach shared state: {e:?}"))?
        .cast::<Shared>();
    // The child opens the region only after these atomics are initialized.
    unsafe {
        address.write(Shared {
            started: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            counters: [const { Counter(AtomicUsize::new(0)) }; WORKERS],
        });
    }
    let shared = unsafe { &*address };
    let baseline = thread::count();
    let result = (|| {
        let child = thread::start_application("proctest", vec!["--lifecycle-child", &name]).ok_or("start lifecycle child")?;
        match child.join() {
            Ok(_) | Err(Errno::ESRCH) => {}
            Err(error) => return Err(format!("join child: {error:?}")),
        }
        // Remote kill messages must finish before either measurement window.
        thread::sleep(100);
        let before = core::array::from_fn::<_, WORKERS, _>(|i| shared.counters[i].0.load(Ordering::Acquire));
        thread::sleep(100);
        let after = core::array::from_fn::<_, WORKERS, _>(|i| shared.counters[i].0.load(Ordering::Acquire));
        if shared.started.load(Ordering::Acquire) != WORKERS || before.contains(&0) {
            return Err(format!("workers did not start: {before:?}"));
        }
        if before != after || thread::count() != baseline {
            return Err(format!(
                "workers survived process exit: before={before:?} after={after:?} baseline={baseline} active={}",
                thread::count()
            ));
        }
        Ok(())
    })();
    // A broken kernel must still let this test release its busy workers.
    shared.stop.store(true, Ordering::Release);
    let deadline = systime().num_milliseconds() + 2000;
    while thread::count() > baseline && systime().num_milliseconds() < deadline {
        thread::sleep(10);
    }
    shm::shm_detach(address.cast()).map_err(|e| format!("detach shared state: {e:?}"))?;
    shm::shm_unlink(&name).map_err(|e| format!("unlink shared state: {e:?}"))?;
    result
}

pub fn child(name: &str) {
    let handle = shm::shm_open(name, size_of::<Shared>(), false).expect("open lifecycle shared state");
    let address = shm::shm_attach(handle, false).expect("attach lifecycle shared state") as usize;
    for index in 0..WORKERS {
        thread::create(move || {
            // This mapping belongs to the child process, including its workers.
            let shared = unsafe { &*(address as *const Shared) };
            shared.counters[index].0.fetch_add(1, Ordering::Relaxed);
            shared.started.fetch_add(1, Ordering::Release);
            while !shared.stop.load(Ordering::Acquire) {
                shared.counters[index].0.fetch_add(1, Ordering::Relaxed);
                core::hint::spin_loop();
            }
        })
        .expect("start lifecycle worker");
    }
    let shared = unsafe { &*(address as *const Shared) };
    while shared.started.load(Ordering::Acquire) != WORKERS {
        thread::switch();
    }
    // Returning exits the entire process while all four workers are busy.
    // The mapping remains attached until the kernel tears the process down.
}
