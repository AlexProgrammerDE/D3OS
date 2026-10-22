use d3os_nvidia::compute::Kernel;

#[test]
fn normalization_uses_one_full_block_per_row() {
    for (rows, width) in [(1, 1), (7, 33), (5, 1024), (2, 4097)] {
        let work = Kernel::RmsNorm.work(&[rows, width, 1e-6f32.to_bits(), 1f32.to_bits()]).unwrap();
        assert_eq!((work.blocks, work.threads), (rows, 128));
        assert_eq!(work.extents[..3], [u64::from(rows * width), u64::from(width), u64::from(rows * width)]);
    }
    assert!(Kernel::RmsNorm.work(&[u32::MAX / 128 + 1, 1, 1e-6f32.to_bits(), 0]).is_err());
}

#[test]
fn packed_weight_extents_cover_odd_tails_and_half_word_offsets() {
    let project = Kernel::ProjectBf16.work(&[17, 33, 9]).unwrap();
    assert_eq!(project.extents[..3], [17 * 33, 149, 17 * 9]);
    assert_eq!(Kernel::DequantizeBf16.work(&[10, 0]).unwrap().extents[..2], [5, 10]);
    assert_eq!(Kernel::DequantizeBf16.work(&[10, 1]).unwrap().extents[..2], [6, 10]);
    assert_eq!(Kernel::DequantizeBf16.work(&[11, 0]).unwrap().extents[..2], [6, 11]);
    assert!(Kernel::DequantizeBf16.work(&[11, 2]).is_err());
    assert!(Kernel::ProjectBf16.work(&[1, u32::MAX, 2]).is_err());
}

#[test]
fn binned_raster_checks_header_and_address_extents() {
    let work = Kernel::RasterBinned.work(&[320, 200, 1500, 4096, 5000]).unwrap();
    assert_eq!(work.extents[..5], [64000, 18000, 4096, 64000, 5000]);
    assert_eq!((work.pointers, work.output), (5, 3));
    assert!(Kernel::RasterBinned.work(&[320, 200, 1500, 4096, 520]).is_err());
    assert!(Kernel::RasterBinned.work(&[u32::MAX, 1, 1, 1, u32::MAX]).is_err());
    assert!(work.validate_aliases(&[0x100000, 0x200000, 0x300000, 0x100000, 0x400000]).is_err());
}

#[test]
fn matrix_work_covers_partial_tiles_and_checks_products() {
    for kernel in [Kernel::Matmul, Kernel::Project] {
        let work = kernel.work(&[17, 33, 9]).unwrap();
        assert_eq!(work.extents[..3], [17 * 33, 33 * 9, 17 * 9]);
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
    assert_eq!(work.extents[..3], [24, 3, 24]);
    assert!(work.in_place);
    assert!(Kernel::MaxPool.work(&[2, 8, 8, 3, 3, 0, 2, 3, 3]).is_err());
    assert!(Kernel::Softmax.work(&[u32::MAX, 2]).is_err());
    assert!(Kernel::from_raw(u32::MAX).is_err());
}

#[test]
fn stateful_dispatch_protects_every_mutated_range() {
    let work = Kernel::DeltaConvolution.work(&[14, 3]).unwrap();
    assert_eq!(work.extents[..4], [14, 42, 42, 14]);
    assert!(work.validate_aliases(&[0x1000, 0x2000, 0x3000, 0x4000]).is_ok());
    assert!(work.validate_aliases(&[0x3000, 0x2000, 0x3000, 0x4000]).is_err());
    assert!(work.validate_aliases(&[0x1000, 0x2000, 0x3000, 0x3004]).is_err());
    let update = Kernel::DeltaUpdate.work(&[2, 2, 3]).unwrap();
    assert_eq!(update.extents[..4], [14, 8, 12, 6]);
    assert!(update.validate_aliases(&[0x1000, 0x2000, 0x3000, 0x1004]).is_err());
    let relu = Kernel::Relu.work(&[16]).unwrap();
    assert!(relu.validate_aliases(&[0x1000, 0x1000]).is_ok());
    assert!(relu.validate_aliases(&[0x1000, 0x1004]).is_err());
}

#[test]
fn transformer_bounds_reject_invalid_groups_and_nonfinite_parameters() {
    assert!(Kernel::AttentionScores.work(&[3, 2, 4, 5]).is_err());
    assert!(Kernel::AttentionPrepare.work(&[2, 4, 3, 0, 10000f32.to_bits(), 1e-6f32.to_bits(), 2]).is_err());
    assert!(Kernel::RmsNorm.work(&[1, 11, f32::NAN.to_bits(), 1f32.to_bits()]).is_err());
    assert!(Kernel::DeltaPrepare.work(&[u32::MAX, 2, 3]).is_err());
    assert!(Kernel::DeltaOutput.work(&[2, 3, 0]).is_err());
}

#[test]
fn raster_dispatch_bounds_every_resource_and_disallows_frame_aliases() {
    for kernel in [Kernel::Raster, Kernel::Fuzz] {
        let work = kernel.work(&[320, 200, 73, 8192]).unwrap();
        assert_eq!(work.extents[..3], [64000, 73 * 12, 8192]);
        assert!(work.validate_aliases(&[0x100000, 0x200000, 0x300000, 0x400000][..work.pointers]).is_ok());
        let mut overlapping = [0x100000, 0x200000, 0x300000, 0x400000];
        overlapping[work.output] = overlapping[1] + 4;
        assert!(work.validate_aliases(&overlapping[..work.pointers]).is_err());
        assert!(kernel.work(&[u32::MAX, 2, 1, 1]).is_err());
        assert!(kernel.work(&[1, 1, u32::MAX, 1]).is_err());
        assert!(kernel.work(&[1, 1, 1, 0]).is_err());
    }
    let palette = Kernel::Palette.work(&[320, 200, 2]).unwrap();
    assert_eq!(palette.extents[..3], [64000, 256, 256000]);
    assert!(Kernel::Palette.work(&[1, 1, 5]).is_err());
    assert!(Kernel::Palette.work(&[u32::MAX / 8, 1, 4]).is_err());
}

#[test]
fn selection_covers_the_complete_input_without_wrapping_its_grid() {
    for n in [1, 33, 248320, u32::MAX] {
        let work = Kernel::ArgMax.work(&[n]).unwrap();
        assert_eq!(work.extents[..2], [u64::from(n), 64]);
        assert_eq!((work.threads, work.blocks), (128, 8));
        assert!(work.validate_aliases(&[0x1000, 0x1000]).is_err());
    }
    assert!(Kernel::ArgMax.work(&[0]).is_err());
}

#[test]
fn packed_raster_bounds_include_partial_words_without_widening_pixels() {
    let raster = Kernel::RasterBinnedBytes.work(&[13, 17, 3, 413, 70]).unwrap();
    assert_eq!(raster.extents[..5], [56, 36, 104, 56, 70]);
    assert!(raster.validate_aliases(&[0x1000, 0x2000, 0x3000, 0x1000, 0x4000]).is_err());
    assert!(Kernel::RasterBinnedBytes.work(&[13, 17, 3, 413, 30]).is_err());
    let fuzz = Kernel::FuzzBytes.work(&[13, 17, 3, 457]).unwrap();
    assert_eq!(fuzz.extents[..3], [56, 36, 115]);
    assert_eq!((fuzz.blocks, fuzz.threads), (4, 128));
    let palette = Kernel::PaletteBytes.work(&[13, 17, 3]).unwrap();
    assert_eq!(palette.extents[..3], [56, 256, 13 * 17 * 9]);
    assert!(Kernel::PaletteBytes.work(&[u32::MAX, 2, 1]).is_err());
}

#[test]
fn fused_delta_covers_all_writers_and_checks_cta_limits() {
    let work = Kernel::DeltaFused.work(&[16, 128, 128, 4, 1e-6f32.to_bits()]).unwrap();
    assert_eq!((work.threads, work.blocks, work.pointers, work.output), (1024, 16, 11, 10));
    assert_eq!(work.extents[..11], [6144, 2048, 16, 16, 24576, 16, 16, 128, 24576, 262144, 2048]);
    assert_eq!(work.mutated, (1 << 8) | (1 << 9));
    let mut addresses: Vec<_> = (1..=11).map(|i| i * 0x1000000).collect();
    assert!(work.validate_aliases(&addresses).is_ok());
    addresses[8] = addresses[0];
    assert!(work.validate_aliases(&addresses).is_err());
    addresses[8] = 9 * 0x1000000;
    addresses[9] = addresses[10];
    assert!(work.validate_aliases(&addresses).is_err());
    let odd = Kernel::DeltaFused.work(&[2, 5, 7, 3, 1e-6f32.to_bits()]).unwrap();
    assert_eq!((odd.threads, odd.blocks), (64, 2));
    for p in [
        [1, 129, 128, 4, 1],
        [1, 128, 129, 4, 1],
        [1, 128, 128, 9, 1],
        [1, 128, 128, 0, 1],
        [1, 0, 1, 1, 1],
    ] {
        assert!(Kernel::DeltaFused.work(&p).is_err());
    }
    assert!(Kernel::DeltaFused.work(&[1, 1, 1, 1, f32::NAN.to_bits()]).is_err());
}
