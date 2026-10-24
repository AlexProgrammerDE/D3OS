use d3os_nvidia::compute::Kernel;

#[test]
fn range_proves_exact_signed_endpoints_and_full_writer_extents() {
    for parameters in [[1, i32::MAX as u32, u32::MAX], [2, i32::MIN as u32, u32::MAX], [259, (-31i32) as u32, 3]] {
        let work = Kernel::Range.work(&parameters).unwrap();
        assert_eq!(work.extents[0], u64::from(parameters[0]));
        assert_eq!(
            (work.threads, work.blocks, work.pointers, work.output),
            (128, parameters[0].div_ceil(128), 1, 0)
        );
        assert!(Kernel::Range.allocates_output());
    }
    for parameters in [[0, 0, 1], [1, 0, 0], [2, i32::MAX as u32, 1], [3, i32::MIN as u32, u32::MAX], [u32::MAX, 0, 1]] {
        assert!(Kernel::Range.work(&parameters).is_err());
    }
    assert!(Kernel::Range.work(&[1, 0]).is_err());
    assert!(Kernel::Range.work(&[1, 0, 1, 0]).is_err());
}
