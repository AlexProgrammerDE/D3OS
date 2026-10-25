//! Host checks for the production FIFO and lock. Privileged interrupt masking
//! is replaced with thread-local state; VM regressions check real scheduling.
//! Run: rustc --edition=2024 --test tools/kernel/test-scheduler-inbox.rs -o
//! target/test-scheduler-inbox && target/test-scheduler-inbox
#![allow(dead_code)]

extern crate alloc;

mod device {
    pub mod cpu {
        use std::cell::Cell;

        thread_local! { static ENABLED: Cell<bool> = const { Cell::new(true) }; }

        pub fn disable_int_nested() -> bool {
            ENABLED.replace(false)
        }

        pub fn enable_int_nested(was_enabled: bool) {
            if was_enabled {
                ENABLED.set(true);
            }
        }

        pub fn pause() {
            std::hint::spin_loop();
        }

        pub fn enabled() -> bool {
            ENABLED.get()
        }
    }
}

#[path = "../../os/kernel/src/sync/irqsave_spinlock.rs"]
pub mod irqsave_spinlock;
mod sync {
    pub use crate::irqsave_spinlock;
}
#[path = "../../os/kernel/src/process/inbox.rs"]
mod inbox;

use inbox::SchedulerInbox;
use std::sync::{Arc, Barrier};

#[derive(Debug, PartialEq, Eq)]
enum Message {
    Thread(usize),
    Wake(usize),
    Kill(usize),
}

#[test]
fn saturation_preserves_thread_control_order_and_ownership() {
    let inbox = SchedulerInbox::with_capacity(100);
    let owner = Arc::new(());
    for tid in 0..4096 {
        inbox.push((Message::Thread(tid), owner.clone()));
        inbox.push((Message::Wake(tid), owner.clone()));
        inbox.push((Message::Kill(tid), owner.clone()));
    }
    assert_eq!(Arc::strong_count(&owner), 1 + 3 * 4096);
    for tid in 0..4096 {
        for expected in [Message::Thread(tid), Message::Wake(tid), Message::Kill(tid)] {
            let (actual, retained) = inbox.pop().unwrap();
            assert_eq!(actual, expected);
            assert!(Arc::ptr_eq(&owner, &retained));
        }
    }
    assert!(inbox.pop().is_none());
    assert_eq!(Arc::strong_count(&owner), 1);
}

#[test]
fn concurrent_producers_deliver_every_message_once_in_producer_order() {
    const PRODUCERS: usize = 4;
    const MESSAGES: usize = 16_384;
    let inbox = Arc::new(SchedulerInbox::with_capacity(1));
    let start = Arc::new(Barrier::new(PRODUCERS + 1));
    std::thread::scope(|scope| {
        for producer in 0..PRODUCERS {
            let inbox = inbox.clone();
            let start = start.clone();
            scope.spawn(move || {
                start.wait();
                for sequence in 0..MESSAGES {
                    inbox.push((producer, sequence));
                }
            });
        }
        start.wait();
        let mut next = [0; PRODUCERS];
        for _ in 0..PRODUCERS * MESSAGES {
            let (producer, sequence) = loop {
                if let Some(item) = inbox.pop() {
                    break item;
                }
                std::thread::yield_now();
            };
            assert_eq!(sequence, next[producer]);
            next[producer] += 1;
        }
        assert_eq!(next, [MESSAGES; PRODUCERS]);
    });
    assert!(inbox.pop().is_none());
}

#[test]
fn consumer_can_requeue_after_pop_and_preserves_nested_irq_state() {
    let inbox = SchedulerInbox::with_capacity(0);
    let outer = irqsave_spinlock::IrqSaveSpinlock::new(());
    assert!(device::cpu::enabled());
    {
        let _guard = outer.lock();
        inbox.push(Message::Kill(1));
        assert_eq!(inbox.pop(), Some(Message::Kill(1)));
        inbox.push(Message::Kill(1));
        assert_eq!(inbox.pop(), Some(Message::Kill(1)));
        assert!(inbox.pop().is_none());
        assert!(!device::cpu::enabled());
    }
    assert!(device::cpu::enabled());
}
