use alloc::collections::VecDeque;

use crate::sync::irqsave_spinlock::IrqSaveSpinlock;

/// One FIFO for thread transfers and control messages destined for a CPU.
///
/// Delivery must not wait for the destination scheduler to consume a slot:
/// producers can hold their own ready-state lock with interrupts disabled.
/// A separate overflow/control queue could overtake a thread transfer and
/// discard a wake or kill before the destination owns that thread.
pub struct SchedulerInbox<T> {
    queue: IrqSaveSpinlock<VecDeque<T>>,
}

impl<T> SchedulerInbox<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: IrqSaveSpinlock::new(VecDeque::with_capacity(capacity)),
        }
    }

    pub fn push(&self, item: T) {
        self.queue.lock().push_back(item);
    }

    /// Release the inbox lock before the caller handles the returned message.
    pub fn pop(&self) -> Option<T> {
        self.queue.lock().pop_front()
    }
}
