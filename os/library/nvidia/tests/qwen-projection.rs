use d3os_nvidia::compute::Kernel;

#[test]
fn narrow_decode_uses_complete_rows_without_changing_storage() {
    let wide_entry = Kernel::ProjectQuantized.entry(&[1, 4096, 129, 8]);
    for (inner, outputs, columns) in [(4096u32, 32u32, 1u32), (4096, 128, 1), (1024, 1, 1), (992, 32, 4), (4096, 129, 4)] {
        let parameters = [1, inner, outputs, 8];
        let work = Kernel::ProjectQuantized.work(&parameters).unwrap();
        assert_eq!(work.blocks, outputs.div_ceil(columns));
        assert_eq!(
            &work.extents[..3],
            &[u64::from(inner), u64::from(inner) * u64::from(outputs) * 34 / 128, u64::from(outputs)]
        );
        assert_eq!((work.threads, work.pointers, work.output), (128, 3, 2));
        assert_eq!(Kernel::ProjectQuantized.entry(&parameters) != wide_entry, columns == 1);
        work.validate_aliases(&[0x100000, 0x200000, 0x300000]).unwrap();
        assert!(work.validate_aliases(&[0x100000, 0x200000, 0x100000]).is_err());
    }
    assert_eq!(Kernel::ProjectQuantized.work(&[2, 4096, 32, 8]).unwrap().blocks, 1);
    assert_eq!(Kernel::ProjectQuantized.work(&[1, 4096, 32, 12]).unwrap().blocks, 8);
}

#[test]
fn narrow_dispatch_rejects_incomplete_or_overflowing_projection_packets() {
    for parameters in [
        vec![1, 31, 32, 8],
        vec![0, 4096, 32, 8],
        vec![1, 4096, 0, 8],
        vec![1, 4096, 32, 8, 1],
        vec![1, 1 << 30, 4, 8],
    ] {
        assert!(Kernel::ProjectQuantized.work(&parameters).is_err());
    }
}

#[test]
fn signed_decode_preserves_packed_extents_and_small_projection_routes() {
    let ordinary = Kernel::ProjectQuantized.entry(&[1, 992, 256, 8]);
    let signed = Kernel::ProjectQuantized.entry(&[1, 4096, 4096, 8]);
    assert_ne!(signed, ordinary);
    for (inner, outputs) in [(1024u32, 256u32), (1056, 257), (4096, 4096), (5120, 5120), (12288, 4096)] {
        let parameters = [1, inner, outputs, 8];
        let work = Kernel::ProjectQuantized.work(&parameters).unwrap();
        let weight_words = (u64::from(inner / 32) * u64::from(outputs) * 34).div_ceil(4);
        assert_eq!(Kernel::ProjectQuantized.entry(&parameters), signed);
        assert_eq!(work.blocks, outputs.div_ceil(4));
        assert_eq!(&work.extents[..3], &[u64::from(inner), weight_words, u64::from(outputs)]);
        assert_eq!((work.threads, work.pointers, work.output, work.in_place), (128, 3, 2, false));
        work.validate_aliases(&[0x100000, 0x10000000, 0x40000000]).unwrap();
    }
    assert_eq!(Kernel::ProjectQuantized.entry(&[1, 4096, 255, 8]), ordinary);
    for parameters in [[2, 4096, 4096, 8], [17, 4096, 4096, 8], [1, 4096, 4096, 11], [1, 4096, 128, 8]] {
        assert_ne!(Kernel::ProjectQuantized.entry(&parameters), signed);
    }
}
