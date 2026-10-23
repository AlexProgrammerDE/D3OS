use d3os_nvidia::compute::Kernel;

#[test]
fn residual_projection_allows_only_exact_residual_aliasing() {
    let work = Kernel::ProjectResidualBf16.work(&[64, 17]).unwrap();
    assert_eq!(work.extents[..4], [17, 64, 544, 17]);
    assert_eq!((work.blocks, work.threads), (5, 128));
    assert!(Kernel::ProjectResidualBf16.allocates_output());
    assert!(work.validate_aliases(&[0x1000, 0x2000, 0x3000, 0x1000]).is_ok());
    assert!(work.validate_aliases(&[0x1000, 0x2000, 0x3000, 0x4000]).is_ok());
    for output in [0x1004, 0x2000, 0x2004, 0x3000, 0x3004] {
        assert!(work.validate_aliases(&[0x1000, 0x2000, 0x3000, output]).is_err());
    }
    for p in [[0, 17], [63, 17], [64, 0], [64, 8193], [u32::MAX - 1, 8192]] {
        assert!(Kernel::ProjectResidualBf16.work(&p).is_err());
    }
}

#[test]
fn residual_norm_checks_both_writers_and_all_read_ranges() {
    let work = Kernel::ResidualNorm.work(&[3, 129, 1e-6f32.to_bits()]).unwrap();
    assert_eq!(work.extents[..4], [387, 387, 129, 387]);
    assert_eq!((work.blocks, work.threads, work.mutated), (3, 128, 1));
    assert!(Kernel::ResidualNorm.allocates_output());
    let pointers = [0x10000, 0x20000, 0x30000, 0x40000];
    assert!(work.validate_aliases(&pointers).is_ok());
    for writer in [0, 3] {
        for other in 0..4 {
            if writer != other {
                let mut aliased = pointers;
                aliased[writer] = aliased[other] + 4;
                assert!(work.validate_aliases(&aliased).is_err());
            }
        }
    }
    for p in [
        [0, 129, 1],
        [1, 0, 1],
        [1, 1025, 1],
        [1, 129, 0],
        [1, 129, f32::NAN.to_bits()],
        [u32::MAX, 1024, 1],
    ] {
        assert!(Kernel::ResidualNorm.work(&p).is_err());
    }
}

#[test]
fn fused_attention_checks_all_rows_cache_prefixes_and_complete_output() {
    let work = Kernel::AttentionFusedBatch.work(&[3, 4, 2, 129, 127]).unwrap();
    assert_eq!((work.threads, work.blocks), (128, 12));
    assert_eq!(work.extents[..5], [1548, 33540, 33540, 3096, 1548]);
    let inputs = [0x10000, 0x100000, 0x200000, 0x300000, 0x400000];
    assert!(work.validate_aliases(&inputs).is_ok());
    for i in 0..4 {
        let mut aliased = inputs;
        aliased[4] = aliased[i] + 4;
        assert!(work.validate_aliases(&aliased).is_err());
    }
    assert!(Kernel::AttentionFusedBatch.allocates_output());
    for p in [
        [0, 4, 2, 129, 127],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0],
        [1, 3, 2, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 257, 0],
        [1, 1, 1, 1, u32::MAX],
        [u32::MAX, 1, 1, 1, 0],
    ] {
        assert!(Kernel::AttentionFusedBatch.work(&p).is_err());
    }
}

#[test]
fn fused_batch_checks_sequence_extents_and_state_aliases() {
    let p = [16, 128, 128, 4, 1e-6f32.to_bits(), 32];
    let work = Kernel::DeltaFusedBatch.work(&p).unwrap();
    assert_eq!(work.extents[..11], [196608, 65536, 512, 512, 24576, 16, 16, 128, 24576, 262144, 65536]);
    assert_eq!((work.threads, work.blocks), (256, 16));
    assert_eq!(work.mutated, (1 << 8) | (1 << 9));
    let pointers: Vec<_> = (1..=11).map(|i| i * 0x200000u64).collect();
    assert!(work.validate_aliases(&pointers).is_ok());
    for writer in [8, 9, 10] {
        for other in 0..11 {
            if writer != other {
                let mut aliased = pointers.clone();
                aliased[writer] = aliased[other] + 4;
                assert!(work.validate_aliases(&aliased).is_err());
            }
        }
    }
    assert!(Kernel::DeltaFusedBatch.allocates_output());
    let mut single = p;
    single[5] = 1;
    assert_eq!(
        Kernel::DeltaFusedBatch.work(&single).unwrap().extents,
        Kernel::DeltaFused.work(&single[..5]).unwrap().extents
    );
    for rows in [0, u32::MAX] {
        let mut invalid = p;
        invalid[5] = rows;
        assert!(Kernel::DeltaFusedBatch.work(&invalid).is_err());
    }
    assert!(Kernel::DeltaFusedBatch.work(&p[..5]).is_err());
    assert!(Kernel::DeltaFused.work(&p).is_err());
}

#[test]
fn split_projection_bounds_partial_tiles_and_reduction_storage() {
    let split = Kernel::ProjectBf16Split.work(&[17, 1025, 257, 16]).unwrap();
    assert_eq!(split.extents[..3], [17425, 131713, 69904]);
    assert_eq!((split.threads, split.blocks), (128, 288));
    assert!(split.validate_aliases(&[0x10000, 0x40000, 0x100000]).is_ok());
    assert!(split.validate_aliases(&[0x10000, 0x40000, 0x40004]).is_err());
    let reduction = Kernel::SumProjectionPartials.work(&[17, 257, 16]).unwrap();
    assert_eq!(reduction.extents[..2], [69904, 4369]);
    assert!(reduction.validate_aliases(&[0x10000, 0x10004]).is_err());
    for dimensions in [[17, 1025, 257, 0], [1, 1, 1, u32::MAX], [u32::MAX, 2, 1, 2]] {
        assert!(Kernel::ProjectBf16Split.work(&dimensions).is_err());
    }
    assert!(Kernel::SumProjectionPartials.work(&[u32::MAX, 1, 2]).is_err());
    assert!(Kernel::ProjectBf16Split.allocates_output());
    assert!(Kernel::SumProjectionPartials.allocates_output());
}

#[test]
fn batched_preparation_covers_tiles_and_checks_positions_and_aliases() {
    let work = Kernel::AttentionPrepareBatch
        .work(&[3, 65, 64, 5, 1e7f32.to_bits(), 1e-6f32.to_bits(), 2, 3])
        .unwrap();
    assert_eq!((work.threads, work.blocks), (128, 9));
    assert_eq!(work.extents[..3], [1170, 65, 585]);
    assert!(work.validate_aliases(&[0x10000, 0x20000, 0x30000]).is_ok());
    assert!(work.validate_aliases(&[0x10000, 0x20000, 0x20004]).is_err());
    for (dim, rotary, position, stride, rows) in [
        (65, 63, 0, 1, 1),
        (65, 66, 0, 1, 1),
        (65, 64, u32::MAX - 1, 1, 2),
        (65, 64, 0, 3, 1),
        (65, 64, 0, 1, 0),
    ] {
        assert!(
            Kernel::AttentionPrepareBatch
                .work(&[3, dim, rotary, position, 1e7f32.to_bits(), 1e-6f32.to_bits(), stride, rows])
                .is_err()
        );
    }
    assert!(Kernel::AttentionPrepareBatch.allocates_output());
}

#[test]
fn gather_bounds_packed_tables_tokens_and_the_complete_output() {
    for packed in [0, 1] {
        let work = Kernel::Embedding.work(&[7, 11, 5, packed]).unwrap();
        assert_eq!(work.extents[..3], [if packed == 1 { 39 } else { 77 }, 5, 55]);
        assert_eq!((work.pointers, work.output), (3, 2));
        assert!(work.validate_aliases(&[0x1000, 0x2000, 0x3000]).is_ok());
        assert!(work.validate_aliases(&[0x1000, 0x2000, 0x2004]).is_err());
    }
    for dimensions in [[7, 11, 5, 2], [0, 11, 5, 0], [7, 11, 0, 0], [u32::MAX, 2, 1, 0]] {
        assert!(Kernel::Embedding.work(&dimensions).is_err());
    }
    assert!(Kernel::Embedding.allocates_output());
    assert!(Kernel::Copy.allocates_output());
}

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
fn packed_vector_launch_covers_every_output_at_tile_boundaries() {
    for (n, outputs_per_block) in [(1u32, 4u32), (2048, 4), (2049, 4), (8192, 4), (8193, 32)] {
        let work = Kernel::ProjectBf16.work(&[1, 64, n]).unwrap();
        let covered = work.blocks * outputs_per_block;
        assert!(covered >= n);
        assert!(covered - n < outputs_per_block);
        assert_eq!(work.threads, 128);
        assert_eq!(work.extents[..3], [64, u64::from(32 * n), u64::from(n)]);
    }
    let batch = Kernel::ProjectBf16.work(&[17, 64, 2049]).unwrap();
    assert_eq!(batch.blocks, 2u32 * 2049u32.div_ceil(32));
}

#[test]
fn gated_projection_checks_both_weight_ranges_and_covers_odd_outputs() {
    for n in [17, 2051] {
        let work = Kernel::GatedProjectBf16.work(&[33, n]).unwrap();
        assert_eq!(
            work.extents[..4],
            [33, u64::from(33 * n).div_ceil(2), u64::from(33 * n).div_ceil(2), u64::from(n)]
        );
        assert_eq!((work.pointers, work.output), (4, 3));
        assert_eq!(work.blocks, n.div_ceil(4));
        for writer_alias in [0x1000, 0x20004, 0x40004] {
            assert!(work.validate_aliases(&[0x1000, 0x20000, 0x40000, writer_alias]).is_err());
        }
    }
    assert!(Kernel::GatedProjectBf16.allocates_output());
    assert!(Kernel::GatedProjectBf16.work(&[0, 1]).is_err());
    assert!(Kernel::GatedProjectBf16.work(&[u32::MAX, 2]).is_err());
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
        assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 2, 3, 2));
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
        assert_eq!((work.threads, work.blocks), (128, 32));
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
    assert_eq!((fuzz.blocks, fuzz.threads), (13, 32));
    let palette = Kernel::PaletteBytes.work(&[13, 17, 3]).unwrap();
    assert_eq!(palette.extents[..3], [56, 256, 13 * 17 * 9]);
    assert!(Kernel::PaletteBytes.work(&[u32::MAX, 2, 1]).is_err());
}

#[test]
fn fused_delta_covers_all_writers_and_checks_cta_limits() {
    let work = Kernel::DeltaFused.work(&[16, 128, 128, 4, 1e-6f32.to_bits()]).unwrap();
    assert_eq!((work.threads, work.blocks, work.pointers, work.output), (256, 16, 11, 10));
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
    assert_eq!((odd.threads, odd.blocks), (256, 2));
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
