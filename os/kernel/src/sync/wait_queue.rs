/* ╔═════════════════════════════════════════════════════════════════════════╗
   ║ Module: wait_queue                                                      ║
   ╟─────────────────────────────────────────────────────────────────────────╢
   ║ Wait queues for blocking i/o.                                           ║
   ║                                                                         ║
   ║ Public functions:                                                       ║
   ║   - wait:       Blocks calling thread if the given predicate is true.   ║
   ║   - notify_one: Deblocks one waiting thread (if any).                   ║
   ║   - notify_all: Deblocks all waiting threads (if any).                  ║
   ╟─────────────────────────────────────────────────────────────────────────╢
   ║ Author: Michael Schoettner, Univ. Duesseldorf, 16.02.2026               ║
   ╚═════════════════════════════════════════════════════════════════════════╝
*/

use alloc::{collections::VecDeque, vec::Vec};
use uuid::Uuid;

use crate::{process::core_local_storage::scheduler, sync::irqsave_spinlock::IrqSaveSpinlock};

pub struct WaitQueue {
    queue: IrqSaveSpinlock<VecDeque<(Uuid, usize)>>,
}

impl WaitQueue {
    pub fn new() -> WaitQueue {
        WaitQueue {
            queue: IrqSaveSpinlock::new(VecDeque::<(Uuid, usize)>::new()),
        }
    }

    /// Register one wait atomically with the predicate check. A wake completes
    /// this wait even if the predicate is unchanged, as required by futexes.
    pub fn wait_once<F>(&self, mut ready: F) -> bool
    where
        F: FnMut() -> bool,
    {
        let ids;
        {
            let mut guard = self.queue.lock();
            if ready() {
                return false;
            }
            ids = scheduler().park_current();
            guard.push_back(ids);
        }
        scheduler().block_if_parking(|| !ready());
        let mut guard = self.queue.lock();
        if let Some(index) = guard.iter().position(|&waiter| waiter == ids) {
            guard.remove(index);
        }
        true
    }

    /// Block until `pred()` becomes true.
    pub fn wait<F>(&self, mut pred: F, _message: &str)
    where
        F: FnMut() -> bool,
    {
        while !pred() {
            self.wait_once(&mut pred);
        }
    }

    /// Wake up exactly one waiter (if any). Returns true if someone was woken up.
    pub fn notify_one(&self) -> bool {
        let mut guard = self.queue.lock();

        let mut unblocked_success_idx = None;

        // check queue until one thread is successfully unblocked
        for (idx, (pid, tid)) in guard.iter().enumerate() {
            if scheduler().unblock(*pid, *tid) {
                unblocked_success_idx = Some(idx);
                break;
            }
        }

        match unblocked_success_idx {
            Some(idx) => guard.remove(idx).is_some(),
            None => false,
        }
    }

    /// Wake up all waiters currently queued.
    /// Returns the number of threads actually unblocked (stale entries are ignored).
    pub fn notify_all(&self) -> usize {
        let mut guard = self.queue.lock();

        let mut unblocked_success_idxs = Vec::new();

        // check queue until one thread is successfully unblocked
        for (idx, (pid, tid)) in guard.iter().enumerate() {
            if scheduler().unblock(*pid, *tid) {
                unblocked_success_idxs.push(idx);
            }
        }

        let mut woke = 0;

        for idx in unblocked_success_idxs.into_iter().rev() {
            if guard.remove(idx as usize).is_some() {
                woke += 1;
            }
        }

        woke
    }
}
