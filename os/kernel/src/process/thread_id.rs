use core::sync::atomic::{AtomicUsize, Ordering};

static THREAD_IDS: ThreadIdAllocator = ThreadIdAllocator::new();

struct ThreadIdAllocator {
    next: AtomicUsize,
}

impl ThreadIdAllocator {
    const fn new() -> Self {
        Self { next: AtomicUsize::new(1) }
    }

    fn allocate(&self) -> Option<ReservedThreadId> {
        self.next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                // The value after isize::MAX is a permanent exhaustion sentinel.
                (next <= isize::MAX as usize).then(|| next + 1)
            })
            .ok()
            .map(ReservedThreadId)
    }
}

// Ownership prevents a reserved ID from being used for multiple threads.
pub(crate) struct ReservedThreadId(usize);

impl ReservedThreadId {
    pub(crate) fn allocate() -> Option<Self> {
        THREAD_IDS.allocate()
    }

    pub(super) fn into_usize(self) -> usize {
        self.0
    }
}
