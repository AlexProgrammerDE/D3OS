use d3os_nvidia::cubin::stack_bytes;

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
    assert!(shared_bytes(32 * 1024 - 1024 + 1).is_err());
    assert!(shared_bytes(u64::MAX).is_err());
}
