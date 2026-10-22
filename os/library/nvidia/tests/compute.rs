use d3os_nvidia::compute::Kernel;

#[test]
fn matrix_work_covers_partial_tiles_and_checks_products() {
    for kernel in [Kernel::Matmul, Kernel::Project] {
        let work = kernel.work(&[17, 33, 9]).unwrap();
        assert_eq!(work.extents, [17 * 33, 33 * 9, 17 * 9]);
        assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 1, 3, 2));
        assert!(!work.in_place);
        assert!(kernel.work(&[u32::MAX, 2, 1]).is_err());
        assert!(kernel.work(&[0, 1, 1]).is_err());
    }
}

#[test]
fn convolution_validates_padded_extents() {
    let work = Kernel::Im2col.work(&[3, 224, 224, 11, 11, 55, 55, 4, 4, 2, 2]).unwrap();
    assert_eq!(work.extents[..2], [3 * 224 * 224, 3 * 11 * 11 * 55 * 55]);
    assert!(!work.in_place);
    assert!(Kernel::Im2col.work(&[3, 224, 224, 11, 11, 56, 55, 4, 4, 2, 2]).is_err());
    assert!(Kernel::Im2col.work(&[1, u32::MAX, 1, 1, 1, 1, 1, 1, 1, 1, 0]).is_err());
}

#[test]
fn elementwise_work_rejects_incomplete_rows_and_wrapping_launches() {
    assert!(Kernel::BiasRow.work(&[10, 3]).is_err());
    assert!(Kernel::BiasChannel.work(&[13, 2, 3]).is_err());
    assert!(Kernel::Relu.work(&[u32::MAX]).is_err());
    let work = Kernel::BiasChannel.work(&[24, 4, 3]).unwrap();
    assert_eq!(work.extents, [24, 3, 24]);
    assert!(work.in_place);
    assert!(Kernel::MaxPool.work(&[2, 8, 8, 3, 3, 0, 2, 3, 3]).is_err());
    assert!(Kernel::Softmax.work(&[u32::MAX, 2]).is_err());
    assert!(Kernel::from_raw(8).is_err());
}
