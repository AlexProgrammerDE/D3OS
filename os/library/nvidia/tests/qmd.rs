use d3os_nvidia::{Record, abi::QmdFields as Q, qmd::chain};

#[test]
fn dependent_qmd_encodes_virtual_addresses_and_preserves_launch_fields() {
    for next in [256, 4096, 0x0034_5678_9a00, 0x00ff_ffff_ff00] {
        let mut qmd = Record::<256>::default();
        qmd.set_bits(Q::RELEASE0_ENABLE, 1).unwrap();
        qmd.set_bits(Q::PROGRAM_ADDRESS_LOWER, 0xdead_beef).unwrap();
        qmd.set_bits(Q::CTA_RASTER_WIDTH, 257).unwrap();
        chain(&mut qmd, next).unwrap();
        assert_eq!(qmd.get_bits(Q::DEPENDENT_QMD0_POINTER).unwrap() << 8, next);
        assert_eq!(qmd.get_bits(Q::DEPENDENT_QMD0_ACTION).unwrap(), 1);
        assert_eq!(qmd.get_bits(Q::DEPENDENT_QMD0_ENABLE).unwrap(), 1);
        assert_eq!(qmd.get_bits(Q::DEPENDENT_QMD0_PREFETCH).unwrap(), 1);
        assert_eq!(qmd.get_bits(Q::RELEASE0_ENABLE).unwrap(), 0);
        assert_eq!(qmd.get_bits(Q::PROGRAM_ADDRESS_LOWER).unwrap(), 0xdead_beef);
        assert_eq!(qmd.get_bits(Q::CTA_RASTER_WIDTH).unwrap(), 257);
    }
}

#[test]
fn invalid_links_leave_the_qmd_unchanged() {
    for next in [0, 1, 255, 257, 1 << 40, u64::MAX] {
        let mut qmd = Record::<256>::from_bytes(&[0x5a; 256]).unwrap();
        let before = qmd.clone();
        assert!(chain(&mut qmd, next).is_err());
        assert_eq!(qmd.as_bytes(), before.as_bytes());
    }
}
