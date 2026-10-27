mod thread_id {
    include!("../src/process/thread_id.rs");

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::{sync::Barrier, thread};

        #[test]
        fn starts_at_one_and_does_not_reuse_dropped_reservations() {
            let allocator = ThreadIdAllocator::new();
            for expected in 1..=1024 {
                assert_eq!(allocator.allocate().unwrap().into_usize(), expected);
            }
        }

        #[test]
        fn last_id_is_positive_and_exhaustion_is_permanent() {
            let allocator = ThreadIdAllocator {
                next: AtomicUsize::new(isize::MAX as usize - 1),
            };
            assert_eq!(allocator.allocate().unwrap().into_usize(), isize::MAX as usize - 1);
            let last = allocator.allocate().unwrap().into_usize();
            assert_eq!(last as isize, isize::MAX);
            for _ in 0..1024 {
                assert!(allocator.allocate().is_none());
            }
            assert_eq!(allocator.next.load(Ordering::Relaxed), isize::MAX as usize + 1);
        }

        #[test]
        fn global_reservations_remain_unique_across_yields() {
            let barrier = Barrier::new(8);
            let mut ids = thread::scope(|scope| {
                let handles: Vec<_> = (0..8)
                    .map(|_| {
                        scope.spawn(|| {
                            barrier.wait();
                            (0..512)
                                .map(|_| {
                                    let reserved = ReservedThreadId::allocate().unwrap();
                                    let before = reserved.0;
                                    thread::yield_now();
                                    let id = reserved.into_usize();
                                    assert_eq!(id, before);
                                    assert!((1..=isize::MAX as usize).contains(&id));
                                    id
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles.into_iter().flat_map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
            });
            ids.sort_unstable();
            assert_eq!(ids.len(), 4096);
            assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
        }

        #[test]
        fn exhaustion_transition_is_unique_under_contention() {
            let first = isize::MAX as usize - 30;
            for _ in 0..32 {
                let allocator = ThreadIdAllocator { next: AtomicUsize::new(first) };
                let barrier = Barrier::new(8);
                let mut ids = thread::scope(|scope| {
                    let handles: Vec<_> = (0..8)
                        .map(|_| {
                            scope.spawn(|| {
                                barrier.wait();
                                let mut ids = Vec::new();
                                for _ in 0..64 {
                                    if let Some(reserved) = allocator.allocate() {
                                        ids.push(reserved.into_usize());
                                    }
                                    thread::yield_now();
                                }
                                assert!(allocator.allocate().is_none());
                                ids
                            })
                        })
                        .collect();
                    handles.into_iter().flat_map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
                });
                ids.sort_unstable();
                assert_eq!(ids, (first..=isize::MAX as usize).collect::<Vec<_>>());
                assert_eq!(allocator.next.load(Ordering::Relaxed), isize::MAX as usize + 1);
                assert!(allocator.allocate().is_none());
            }
        }
    }
}
