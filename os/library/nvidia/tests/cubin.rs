use d3os_nvidia::cubin::stack_bytes;

#[test]
fn fixed_thread_dimensions_reject_malformed_or_oversized_blocks() {
    use d3os_nvidia::cubin::required_threads;
    let record = |dimensions: [u32; 3]| {
        let mut result = vec![4, 0x10, 12, 0];
        for axis in dimensions {
            result.extend_from_slice(&axis.to_le_bytes());
        }
        result
    };
    for dimensions in [[128, 1, 1], [256, 1, 1], [32, 4, 2], [1024, 1, 1]] {
        let bytes = record(dimensions);
        assert_eq!(required_threads(&bytes).unwrap(), Some(dimensions));
        for end in 1..bytes.len() {
            assert!(required_threads(&bytes[..end]).is_err());
        }
    }
    assert_eq!(required_threads(&[]).unwrap(), None);
    for dimensions in [[0, 1, 1], [128, 0, 1], [1024, 2, 1], [u32::MAX, u32::MAX, 2]] {
        assert!(required_threads(&record(dimensions)).is_err());
    }
    let mut duplicate = record([128, 1, 1]);
    duplicate.extend_from_slice(&duplicate.clone());
    assert!(required_threads(&duplicate).is_err());
    assert!(required_threads(&[1, 0x10, 12, 0]).is_err());
}

#[test]
fn module_stack_records_use_the_largest_entry_and_skip_immediate_records() {
    let records = [
        1, 0x20, 3, 0, 4, 0x12, 8, 0, 7, 0, 0, 0, 40, 0, 0, 0, 4, 0x12, 8, 0, 9, 0, 0, 0, 80, 0, 0, 0, 4, 0x12, 8, 0, 11, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(stack_bytes(&records).unwrap(), 80);
    assert_eq!(stack_bytes(&[]).unwrap(), 0);
    for end in [1, 3, 5, 9, 15, 17, 27, 35] {
        assert!(stack_bytes(&records[..end]).is_err());
    }
}

#[test]
fn malformed_stack_payloads_do_not_become_scratch_sizes() {
    assert!(stack_bytes(&[4, 0x12, 4, 0, 1, 0, 0, 0]).is_err());
    assert!(stack_bytes(&[1, 0x12, 8, 0]).is_err());
    assert!(stack_bytes(&[4, 0x12, 0, 0]).is_err());
}

#[test]
fn shared_allocation_accounts_for_reserved_space_and_partition_limit() {
    use d3os_nvidia::cubin::shared_bytes;
    assert_eq!(shared_bytes(0).unwrap(), 1024);
    assert_eq!(shared_bytes(2064).unwrap(), 3200);
    assert_eq!(shared_bytes(129).unwrap(), 1280);
    assert_eq!(shared_bytes(32 * 1024 - 1024).unwrap(), 32 * 1024);
    assert!(shared_bytes(100 * 1024 - 1024 + 1).is_err());
    assert!(shared_bytes(u64::MAX).is_err());
}

#[test]
fn ada_launch_rejects_register_exhaustion_and_counts_partial_warps() {
    use d3os_nvidia::cubin::validate_ada_registers;
    for (registers, threads) in [(0, 1024), (64, 1024), (80, 512), (128, 512), (255, 256)] {
        assert!(validate_ada_registers(registers, threads).is_ok());
    }
    for (registers, threads) in [(65, 1024), (80, 1024), (255, 257), (256, 1), (u64::MAX, 1), (1, 0), (1, 1025)] {
        assert!(validate_ada_registers(registers, threads).is_err());
    }
}

#[test]
fn ada_shared_partition_covers_every_rounded_allocation() {
    use d3os_nvidia::cubin::{shared_bytes, shared_partition};
    for bytes in [0, 1, 31744, 31745, 64512, 64513, 101376] {
        let total = shared_bytes(bytes).unwrap();
        let encoding = shared_partition(total).unwrap();
        let partition = (encoding - 1) * 4096;
        assert!(partition >= total);
        assert!([32768, 65536, 102400].contains(&partition));
    }
    assert_eq!(shared_partition(32768).unwrap(), 9);
    assert_eq!(shared_partition(32769).unwrap(), 17);
    assert_eq!(shared_partition(65537).unwrap(), 26);
    assert!(shared_partition(102401).is_err());
}
