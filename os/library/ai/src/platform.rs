//! Native clock and timer wakeups for Burn's portable runtime dependencies.
use crate::timer_queue::TimerQueue;
use core::task::Waker;
use embassy_time_driver::Driver;
use spin::Mutex;

struct State {
    queue: TimerQueue,
    running: bool,
}

struct Clock {
    state: Mutex<State>,
}

impl Driver for Clock {
    fn now(&self) -> u64 {
        time::systime().num_milliseconds().max(0) as u64 * 1000
    }

    fn schedule_wake(&self, at: u64, waker: &Waker) {
        if at == u64::MAX {
            return;
        }
        if at <= self.now() {
            waker.wake_by_ref();
            return;
        }
        let mut state = self.state.lock();
        state.queue.insert(at, waker);
        if !state.running {
            state.running = true;
            if concurrent::thread::create(run_timers).is_none() {
                state.running = false;
                drop(state);
                panic!("cannot create the Burn timer thread");
            }
        }
    }
}

embassy_time_driver::time_driver_impl!(static CLOCK: Clock = Clock {
    state: Mutex::new(State { queue: TimerQueue::new(), running: false }),
});

fn run_timers() {
    loop {
        let ready = {
            let mut state = CLOCK.state.lock();
            if state.queue.is_empty() {
                state.running = false;
                return;
            }
            state.queue.take_ready(CLOCK.now())
        };
        for waker in ready {
            waker.wake();
        }
        // A new earlier deadline cannot interrupt ThreadSleep. Limit each sleep
        // to one native clock tick, including when existing timers are far away.
        concurrent::thread::sleep(1);
    }
}

#[cfg(feature = "training")]
pub(crate) fn validate_timers() -> Result<(), alloc::string::String> {
    use alloc::{string::String, sync::Arc, task::Wake};
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct Count(AtomicUsize);
    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let start = CLOCK.now();
    let first = Arc::new(Count(AtomicUsize::new(0)));
    let second = Arc::new(Count(AtomicUsize::new(0)));
    let first_waker = Waker::from(first.clone());
    embassy_time_driver::schedule_wake(start + 40_000, &first_waker);
    embassy_time_driver::schedule_wake(start + 2_000, &first_waker);
    embassy_time_driver::schedule_wake(start + 4_000, &Waker::from(second.clone()));
    loop {
        let now = embassy_time_driver::now();
        if now < start {
            return Err(String::from("Burn clock moved backwards"));
        }
        if first.0.load(Ordering::SeqCst) == 1 && second.0.load(Ordering::SeqCst) == 1 {
            return Ok(());
        }
        if now - start >= 200_000 {
            return Err(alloc::format!(
                "Burn timers did not wake both tasks: first={} second={}",
                first.0.load(Ordering::SeqCst),
                second.0.load(Ordering::SeqCst),
            ));
        }
        concurrent::thread::sleep(1);
    }
}
