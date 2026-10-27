use d3os_nvidia::compute::Kernel;

#[test]
fn wide_rows_preserve_storage_and_launch_one_block_per_row() {
    let epsilon = 1e-6f32.to_bits();
    let ordinary = Kernel::RmsNorm.entry(&[1, 4095, epsilon, 1f32.to_bits()]);
    for rows in [1u32, 5, 32] {
        for offset in [0f32, 1f32] {
            let parameters = [rows, 4096, epsilon, offset.to_bits()];
            let work = Kernel::RmsNorm.work(&parameters).unwrap();
            assert_eq!(work.blocks, rows);
            assert_eq!((work.threads, work.pointers, work.output), (128, 3, 2));
            assert_eq!(&work.extents[..3], &[u64::from(rows) * 4096, 4096, u64::from(rows) * 4096]);
            assert_ne!(Kernel::RmsNorm.entry(&parameters), ordinary);
            assert!(!work.in_place);
            work.validate_aliases(&[0x100000, 0x200000, 0x300000]).unwrap();
            assert!(work.validate_aliases(&[0x100000, 0x200000, 0x100000]).is_err());
        }
    }
    assert_eq!(Kernel::RmsNorm.entry(&[1, 4097, epsilon, 1f32.to_bits()]), ordinary);
}

#[test]
fn wide_dispatch_retains_parameter_validation() {
    let epsilon = 1e-6f32.to_bits();
    for parameters in [
        vec![0, 4096, epsilon, 0],
        vec![u32::MAX, 4096, epsilon, 0],
        vec![1, 4096, 0, 0],
        vec![1, 4096, f32::NAN.to_bits(), 0],
        vec![1, 4096, epsilon, 2f32.to_bits()],
        vec![1, 4096, epsilon],
    ] {
        assert!(Kernel::RmsNorm.work(&parameters).is_err());
    }
}
