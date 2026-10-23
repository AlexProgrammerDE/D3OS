use alloc::vec::Vec;
use core::task::Waker;

struct Timer {
    deadline: u64,
    waker: Waker,
}

pub(crate) struct TimerQueue {
    timers: Vec<Timer>,
}

impl TimerQueue {
    pub(crate) const fn new() -> Self {
        Self { timers: Vec::new() }
    }

    pub(crate) fn insert(&mut self, deadline: u64, waker: &Waker) {
        if let Some(timer) = self.timers.iter_mut().find(|timer| timer.waker.will_wake(waker)) {
            timer.deadline = timer.deadline.min(deadline);
        } else {
            self.timers.push(Timer {
                deadline,
                waker: waker.clone(),
            });
        }
    }

    /// Return wakers without calling them while the driver's queue lock is held.
    pub(crate) fn take_ready(&mut self, now: u64) -> Vec<Waker> {
        let mut ready = Vec::new();
        let mut index = 0;
        while index < self.timers.len() {
            if self.timers[index].deadline <= now {
                ready.push(self.timers.swap_remove(index).waker);
            } else {
                index += 1;
            }
        }
        ready
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.timers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{sync::Arc, task::Wake};
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct Counter(AtomicUsize);

    impl Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn shared_waker_keeps_the_earliest_deadline_and_can_be_registered_again() {
        let counter = Arc::new(Counter(AtomicUsize::new(0)));
        let waker = Waker::from(counter.clone());
        let mut queue = TimerQueue::new();
        queue.insert(20, &waker);
        queue.insert(10, &waker);
        queue.insert(30, &waker);
        assert!(queue.take_ready(9).is_empty());
        let ready = queue.take_ready(10);
        assert!(queue.is_empty());
        assert_eq!(ready.len(), 1);
        ready.into_iter().for_each(Waker::wake);
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        queue.insert(30, &waker);
        assert!(queue.take_ready(29).is_empty());
        queue.take_ready(30).into_iter().for_each(Waker::wake);
        assert_eq!(counter.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn independent_timers_survive_removal_of_other_expired_timers() {
        let early = Arc::new(Counter(AtomicUsize::new(0)));
        let late = Arc::new(Counter(AtomicUsize::new(0)));
        let mut queue = TimerQueue::new();
        queue.insert(20, &Waker::from(late.clone()));
        queue.insert(5, &Waker::from(early.clone()));
        queue.take_ready(7).into_iter().for_each(Waker::wake);
        assert!(!queue.is_empty());
        assert_eq!(early.0.load(Ordering::SeqCst), 1);
        assert_eq!(late.0.load(Ordering::SeqCst), 0);
        queue.take_ready(21).into_iter().for_each(Waker::wake);
        assert!(queue.is_empty());
        assert_eq!(late.0.load(Ordering::SeqCst), 1);
    }
}
