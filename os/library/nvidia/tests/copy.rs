use d3os_nvidia::copy::commands;

#[test]
fn linear_copy_preserves_full_virtual_addresses_and_orders_completion() {
    for bytes in [1, 4, 257, 1 << 20, 1 << 31] {
        let source = 0x0001_2345_6789_0004;
        let destination = 0x0001_abcd_1234_0008;
        let report = 0x0000_1234_5678_0040;
        let words = commands(destination, source, bytes, report).unwrap();
        let address = |upper, lower| (u64::from(upper) << 32) | u64::from(lower);
        assert_eq!(address(words[3], words[4]), source);
        assert_eq!(address(words[5], words[6]), destination);
        assert_eq!(words[8], bytes);
        assert_eq!(address(words[10], words[11]), report);
        assert_eq!(words[12], 1);
        assert_eq!(words[14] & 3, 2); // Non-pipelined transfer.
        assert_eq!((words[14] >> 2) & 1, 1); // System flush.
        assert_eq!((words[14] >> 3) & 3, 1); // Copy-engine release.
        assert_eq!((words[14] >> 7) & 3, 3); // Both layouts pitched.
        assert_eq!(address(words[17], words[16]), report + 8);
        assert_eq!((words[20] >> 20) & 1, 1); // FIFO idle before storage reuse.
    }
}

#[test]
fn copy_rejects_overlaps_overflow_and_unrepresentable_addresses() {
    for (destination, source, bytes, report) in [
        (0, 0x2000, 4, 0x3000),
        (0x1000, 0, 4, 0x3000),
        (0x1000, 0x2000, 4, 0),
        (0x1000, 0x2000, 0, 0x3000),
        (0x1000, 0x1000, 4, 0x3000),
        (0x1004, 0x1000, 8, 0x3000),
        (0x1000, 0x1004, 8, 0x3000),
        (0x1000, 0x2000, 8, 0x1000),
        (0x1000, 0x2000, 8, 0x2000),
        (0x1000, 0x2000, 8, 0x3001),
        (0x1000, 1 << 49, 4, 0x3000),
        ((1 << 49) - 4, 0x1000, 8, 0x3000),
        (0x1000, 0x2000, 4, (1 << 49) - 8),
        (u64::MAX, 0x1000, 4, 0x3000),
        (0x1000, 0x2000, 4, u64::MAX - 7),
        (1 << 40, 0x1000, (1 << 31) + 1, 0x3000),
    ] {
        assert!(commands(destination, source, bytes, report).is_err());
    }
    assert!(commands(0x1000, 0x2000, 4, 0x1008).is_ok());
    assert!(commands(0x1000, (1 << 49) - 4, 4, 0x3000).is_ok());
}
