use d3os_nvidia::fence::reached;

#[test]
fn reused_reports_never_complete_future_work_or_block_old_consumers() {
    for first in [1u64, u64::MAX - 16] {
        let mut reports = [first.wrapping_sub(2), first.wrapping_sub(1)];
        for step in 0..64 {
            let expected = first.wrapping_add(step);
            let slot = step as usize % reports.len();
            let previous = reports[slot];
            assert!(!reached(previous, expected));
            reports[slot] = expected;
            assert!(reached(reports[slot], expected));
            assert!(reached(reports[slot], previous));
            assert!(!reached(reports[slot], expected.wrapping_add(1)));
        }
    }
}

#[test]
fn initial_zero_reports_do_not_complete_the_first_two_batches() {
    assert!(!reached(0, 1));
    assert!(!reached(0, 2));
    assert!(reached(1, 1));
    assert!(reached(2, 1));
    assert!(reached(0, u64::MAX));
    assert!(!reached(u64::MAX, 0));
}
