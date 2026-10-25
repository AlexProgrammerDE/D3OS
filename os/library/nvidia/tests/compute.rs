use d3os_nvidia::compute::Kernel;

#[test]
fn argwhere_bounds_prefixes_and_full_coordinate_writes() {
    use d3os_nvidia::compute::argwhere::validate;
    let mut p = [0u32; 16];
    p[..4].copy_from_slice(&[1935, 3, 737, 11]);
    p[4..12].fill(1);
    p[4..7].copy_from_slice(&[3, 129, 5]);
    assert_eq!(validate(&p).unwrap(), [1935, 2211]);
    let work = Kernel::ArgWhere.work(&p).unwrap();
    assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 6, 2, 1));
    assert!(!work.in_place);
    assert!(Kernel::ArgWhere.allocates_output());
    assert!(work.validate_aliases(&[0x10000, 0x20000]).is_ok());
    assert!(work.validate_aliases(&[0x10000, 0x10004]).is_err());
    for (field, value) in [
        (0, 0),
        (0, 1934),
        (0, u32::MAX),
        (1, 0),
        (1, 9),
        (2, 0),
        (2, 1936),
        (3, 10),
        (4, 0),
        (5, 128),
        (7, 2),
        (12, 1),
    ] {
        let mut bad = p;
        bad[field] = value;
        assert!(validate(&bad).is_err());
    }
    p[..4].copy_from_slice(&[1 << 30, 3, 1 << 30, 31]);
    p[4..12].fill(1);
    p[4] = 1 << 30;
    assert!(validate(&p).is_err());
    assert!(validate(&p[..15]).is_err());
}

#[test]
fn sort_passes_bound_packed_outputs_and_validate_source_layouts() {
    use d3os_nvidia::compute::sort::{Mode, validate};
    let mut initial = [0u32; 32];
    initial[..8].copy_from_slice(&[1935, 3, 4, 1, 640, 1935, 129, 5]);
    initial[8..16].fill(1);
    initial[8..11].copy_from_slice(&[3, 129, 5]);
    initial[16..19].copy_from_slice(&[645, (-5i32) as u32, 1]);
    initial[24..26].copy_from_slice(&[1, 1]);
    let mut dense = initial;
    dense[4] = 0;
    dense[17] = 5;
    dense[24..26].copy_from_slice(&[128, 8]);
    for mode in Mode::ALL {
        let mut p = if mode.initial() { initial } else { dense };
        p[2] = mode as u32;
        let work = Kernel::TensorSort.work(&p).unwrap();
        assert_eq!(work.extents[..3], [1935, if mode.initial() { 0 } else { 1935 }, 3870]);
        assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 16, 3, 2));
        assert!(!work.in_place);
        assert!(Kernel::TensorSort.allocates_output());
        assert!(work.validate_aliases(&[0x10000, 0x20000, 0x30000]).is_ok());
        assert!(work.validate_aliases(&[0x10000, 0x20000, 0x10004]).is_err());
        if !mode.initial() {
            assert!(work.validate_aliases(&[0x10000, 0x20000, 0x20004]).is_err());
        }
    }
    for (field, value) in [
        (0, 1934),
        (1, 0),
        (1, 9),
        (2, 8),
        (3, 3),
        (4, 639),
        (5, 1934),
        (6, 0),
        (6, u32::MAX),
        (7, 4),
        (9, 128),
        (11, 2),
        (19, 1),
        (24, 0),
        (24, 3),
        (24, 2),
        (25, 2),
        (26, 1),
    ] {
        let mut p = initial;
        p[field] = value;
        assert!(validate(&p).is_err());
    }
    dense[2] = Mode::FloatAscending as u32;
    for (field, value) in [(4, 1), (5, 1936), (17, (-5i32) as u32), (24, 512), (25, 7)] {
        let mut p = dense;
        p[field] = value;
        assert!(validate(&p).is_err());
    }
    assert!(validate(&initial[..31]).is_err());
}

#[test]
fn cumulative_work_checks_signed_views_dense_output_and_chunk_rows() {
    use d3os_nvidia::compute::scan::Scan;
    let mut p = [0u32; 24];
    p[..8].copy_from_slice(&[15, 3, 0, 1, 640, 1935, 129, 5]);
    p[8..16].fill(1);
    p[8..11].copy_from_slice(&[3, 129, 5]);
    p[16..19].copy_from_slice(&[645, (-5i32) as u32, 1]);
    for operation in Scan::ALL {
        p[2] = operation as u32;
        let work = Kernel::TensorScan.work(&p).unwrap();
        assert_eq!(work.extents[..2], [1935, 1935]);
        assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 15, 2, 1));
        assert!(Kernel::TensorScan.allocates_output());
        assert!(!work.in_place);
        assert!(work.validate_aliases(&[0x10000, 0x20000]).is_ok());
        assert!(work.validate_aliases(&[0x10000, 0x10004]).is_err());
    }
    for (index, value) in [
        (0, 0),
        (0, 14),
        (1, 0),
        (1, 9),
        (2, 8),
        (3, 3),
        (4, 639),
        (5, 1934),
        (6, 0),
        (7, 4),
        (9, 128),
        (11, 2),
        (19, 1),
    ] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::TensorScan.work(&invalid).is_err());
    }
    assert!(Kernel::TensorScan.work(&p[..23]).is_err());
    // Broadcast source storage can be smaller than the dense output.
    p[16] = 0;
    p[5] = 645;
    assert_eq!(Kernel::TensorScan.work(&p).unwrap().extents[..2], [645, 1935]);
    // Even a broadcast source must respect the common 32-bit launch count.
    let mut wide = [0u32; 24];
    wide[..8].copy_from_slice(&[33_554_432, 2, 0, 1, 0, 1, 1, 1]);
    wide[8..16].fill(1);
    wide[8] = 33_554_432;
    assert!(Kernel::TensorScan.work(&wide).is_err());
}

#[test]
fn adam_bounds_every_state_component_and_rejects_inconsistent_flags() {
    use d3os_nvidia::compute::optimizer::{AMSGRAD, MAXIMUM, STATE};
    let mut p = [0u32; 9];
    p[0] = 257;
    for flags in [0, STATE, AMSGRAD, STATE | AMSGRAD, STATE | AMSGRAD | MAXIMUM] {
        p[1] = flags;
        let work = Kernel::Adam.work(&p).unwrap();
        let state = if flags & STATE == 0 { 0 } else { 257 };
        assert_eq!(
            work.extents[..6],
            [
                257,
                257,
                state,
                state,
                if flags & MAXIMUM == 0 { 0 } else { 257 },
                257 * if flags & AMSGRAD == 0 { 3 } else { 4 }
            ]
        );
        assert_eq!((work.blocks, work.pointers, work.output), (3, 6, 5));
        assert!(!work.in_place);
        let pointers = [0x10000, 0x20000, 0x30000, 0x40000, 0x50000, 0x60000];
        assert!(work.validate_aliases(&pointers).is_ok());
        for input in [0, 1] {
            let mut aliased = pointers;
            aliased[5] = aliased[input] + 4;
            assert!(work.validate_aliases(&aliased).is_err());
        }
    }
    for flags in [MAXIMUM, AMSGRAD | MAXIMUM, STATE | MAXIMUM, 8] {
        p[1] = flags;
        assert!(Kernel::Adam.work(&p).is_err());
    }
    p[1] = 0;
    for count in [0, u32::MAX] {
        p[0] = count;
        assert!(Kernel::Adam.work(&p).is_err());
    }
    assert!(Kernel::Adam.work(&p[..8]).is_err());
    assert!(Kernel::Adam.allocates_output());
}

#[test]
fn scatter_preserves_destination_ranges_and_rejects_writer_aliases() {
    use d3os_nvidia::compute::indexing::Scatter;
    let mut p = [0u32; 32];
    p[..8].copy_from_slice(&[2, 1, 6, 12, 3, 6, 5, 4]);
    p[8..16].fill(1);
    p[8..10].copy_from_slice(&[2, 3]);
    p[16..18].copy_from_slice(&[4, (-1i32) as u32]);
    p[24..26].copy_from_slice(&[(-3i32) as u32, (-1i32) as u32]);
    for operation in Scatter::ALL {
        p[0] = 2 | ((operation as u32) << 16);
        let work = Kernel::Scatter.work(&p).unwrap();
        assert_eq!(work.extents[..4], [6, 6, 1, 12]);
        assert_eq!((work.blocks, work.pointers, work.output, work.mutated), (1, 4, 3, 4));
        assert!(!Kernel::Scatter.allocates_output());
        assert!(!work.in_place);
        let addresses = [0x1000, 0x2000, 0x3000, 0x4000];
        assert!(work.validate_aliases(&addresses).is_ok());
        assert!(work.validate_aliases(&[0x1000, 0x1000, 0x3000, 0x4000]).is_ok());
        for (writer, other) in [(2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)] {
            let mut invalid = addresses;
            invalid[writer] = invalid[other];
            assert!(work.validate_aliases(&invalid).is_err());
        }
    }
    for (index, value) in [(0, 2 | (3 << 16)), (0, 2 | (1 << 8)), (3, 7), (4, 2), (5, 5), (6, 4), (7, 5), (18, 1), (26, 1)] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::Scatter.work(&invalid).is_err());
    }
    assert!(Kernel::Scatter.work(&p[..31]).is_err());
}

#[test]
fn gather_checks_both_strided_ranges_and_isolates_its_error_word() {
    let mut p = [0u32; 32];
    p[..8].copy_from_slice(&[2, 1, 6, 12, 3, 6, 5, 4]);
    p[8..16].fill(1);
    p[8..10].copy_from_slice(&[2, 3]);
    p[16..18].copy_from_slice(&[4, (-1i32) as u32]);
    p[24..26].copy_from_slice(&[(-3i32) as u32, (-1i32) as u32]);
    let work = Kernel::Gather.work(&p).unwrap();
    assert_eq!(work.extents[..4], [12, 6, 1, 6]);
    assert_eq!((work.blocks, work.pointers, work.output, work.mutated), (1, 4, 3, 4));
    assert!(Kernel::Gather.allocates_output());
    let addresses = [0x1000, 0x2000, 0x3000, 0x4000];
    assert!(work.validate_aliases(&addresses).is_ok());
    for (writer, other) in [(2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)] {
        let mut invalid = addresses;
        invalid[writer] = invalid[other];
        assert!(work.validate_aliases(&invalid).is_err());
    }
    for (index, value) in [
        (0, 0),
        (0, 9),
        (1, 2),
        (2, 0),
        (2, 5),
        (3, 0),
        (3, 7),
        (4, 2),
        (5, 0),
        (5, 5),
        (6, 4),
        (7, 5),
        (8, 0),
        (10, 2),
        (18, 1),
        (26, 1),
    ] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::Gather.work(&invalid).is_err());
    }
    assert!(Kernel::Gather.work(&p[..31]).is_err());
    p[3] = 1;
    p[4] = 0;
    p[7] = 0;
    p[16..24].fill(0);
    assert!(Kernel::Gather.work(&p).is_ok());
}

#[test]
fn typed_words_check_flags_scalar_domains_and_original_storage_bounds() {
    use d3os_nvidia::compute::{
        elementwise::SCALAR_RHS,
        words::{ValueType, WordOp},
    };
    let mut p = [0u32; 32];
    p[..8].copy_from_slice(&[6, 2, 0, 0, 5, 0, 6, 3]);
    p[8..16].fill(1);
    p[8..10].copy_from_slice(&[2, 3]);
    p[16..18].copy_from_slice(&[(-3i32) as u32, (-1i32) as u32]);
    p[25] = 1;
    for op in WordOp::ALL {
        p[2] = op as u32;
        let work = Kernel::TensorWords.work(&p).unwrap();
        assert_eq!(work.extents[..3], [6, 3, 6]);
        assert_eq!((work.blocks, work.pointers, work.output), (1, 3, 2));
        assert!(Kernel::TensorWords.allocates_output());
        assert!(work.validate_aliases(&[0x1000, 0x1000, 0x2000]).is_ok());
        assert!(work.validate_aliases(&[0x1000, 0x2000, 0x1004]).is_err());
        p[2] |= SCALAR_RHS;
        p[3] = 1;
        assert_eq!(Kernel::TensorWords.work(&p).is_ok(), op.binary());
        p[3] = 2;
        assert_eq!(Kernel::TensorWords.work(&p).is_ok(), op.binary() && op.input() != ValueType::Bool);
        p[2] = op as u32;
        assert_eq!(Kernel::TensorWords.work(&p).is_ok(), op == WordOp::IntMaskFill);
        p[3] = 0;
    }
    for (index, value) in [
        (0, 0),
        (0, 5),
        (1, 9),
        (2, WordOp::ALL.len() as u32),
        (2, 1 << 17),
        (4, 4),
        (5, 1),
        (6, 5),
        (7, 2),
        (10, 2),
        (18, 1),
        (26, 1),
    ] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::TensorWords.work(&invalid).is_err());
    }
    assert!(Kernel::TensorWords.work(&p[..31]).is_err());
}

#[test]
fn fill_is_a_full_writer_with_no_input_or_aliases() {
    for word in [0, 1, u32::MAX, 0x7fc0_1234] {
        let work = Kernel::Fill.work(&[257, word]).unwrap();
        assert_eq!(work.extents[0], 257);
        assert_eq!((work.blocks, work.pointers, work.output), (3, 1, 0));
        assert!(Kernel::Fill.allocates_output());
        assert!(!work.in_place);
    }
    for p in [&[0, 1][..], &[u32::MAX, 0], &[1], &[1, 0, 0]] {
        assert!(Kernel::Fill.work(p).is_err());
    }
}

#[test]
fn unary_math_rejects_binary_scalar_flags_and_unused_scalar_data() {
    use d3os_nvidia::compute::elementwise::{Elementwise, SCALAR_RHS};
    let mut p = [0; 32];
    p[..8].copy_from_slice(&[3, 1, 0, 0, 2, 2, 3, 3]);
    p[8..16].fill(1);
    p[8] = 3;
    p[16] = (-1i32) as u32;
    p[24] = (-1i32) as u32;
    for op in Elementwise::ALL.into_iter().filter(|op| op.unary()) {
        p[2] = op as u32;
        assert!(Kernel::TensorElementwise.work(&p).is_ok());
        p[2] |= SCALAR_RHS;
        assert!(Kernel::TensorElementwise.work(&p).is_err());
        p[2] = op as u32;
        p[3] = 1;
        assert!(Kernel::TensorElementwise.work(&p).is_err());
        p[3] = 0;
    }
    p[2] = Elementwise::Power as u32 | SCALAR_RHS;
    p[3] = 0.5f32.to_bits();
    assert!(Kernel::TensorElementwise.work(&p).is_ok());
}

#[test]
fn reductions_validate_strided_ranges_empty_axes_and_full_output_writes() {
    use d3os_nvidia::compute::reduction::Reduction;
    let mut p = [0u32; 24];
    p[..8].copy_from_slice(&[3, 3, Reduction::Sum as u32, 1, 11, 12, 4, 0]);
    p[8..16].fill(1);
    p[8..11].copy_from_slice(&[3, 4, 1]);
    p[16..19].copy_from_slice(&[(-4i32) as u32, (-1i32) as u32, 0]);
    for op in Reduction::ALL.into_iter().chain(Reduction::INTEGERS) {
        p[2] = op as u32;
        let work = Kernel::TensorReduction.work(&p).unwrap();
        assert_eq!(work.extents[..2], [12, 3]);
        assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 3, 2, 1));
        assert!(Kernel::TensorReduction.allocates_output());
        assert!(work.validate_aliases(&[0x1000, 0x2000]).is_ok());
        assert!(work.validate_aliases(&[0x1000, 0x1004]).is_err());
    }
    for (index, value) in [
        (0, 0),
        (0, 4),
        (1, 0),
        (1, 9),
        (2, 14),
        (3, 3),
        (4, 10),
        (4, 12),
        (5, 0),
        (5, 11),
        (5, u32::MAX),
        (6, 3),
        (7, 1),
        (8, 0),
        (8, u32::MAX),
        (11, 2),
        (19, 1),
        (16, i32::MIN as u32),
    ] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::TensorReduction.work(&invalid).is_err());
    }
    assert!(Kernel::TensorReduction.work(&p[..23]).is_err());
    p[4] = 0;
    p[5] = 1;
    p[6] = 0;
    p[9] = 0;
    p[16..24].fill(0);
    for op in Reduction::ALL.into_iter().chain(Reduction::INTEGERS) {
        p[2] = op as u32;
        assert_eq!(Kernel::TensorReduction.work(&p).is_ok(), op.allows_empty());
    }
    // Even broadcast storage cannot request a wrapped dispatch grid.
    p[..8].copy_from_slice(&[i32::MAX as u32, 2, 0, 1, 0, 1, 1, 0]);
    p[8] = i32::MAX as u32;
    p[9] = 1;
    assert!(Kernel::TensorReduction.work(&p).is_err());
}

#[test]
fn strided_elementwise_validates_every_reachable_address_before_dispatch() {
    use d3os_nvidia::compute::elementwise::{Elementwise, SCALAR_RHS};
    let mut p = [0u32; 32];
    p[..8].copy_from_slice(&[6, 2, Elementwise::Add as u32, 0, 5, 0, 6, 3]);
    p[8..16].fill(1);
    p[8..10].copy_from_slice(&[2, 3]);
    p[16..18].copy_from_slice(&[(-3i32) as u32, (-1i32) as u32]);
    p[25] = 1;
    let work = Kernel::TensorElementwise.work(&p).unwrap();
    assert_eq!(work.extents[..3], [6, 3, 6]);
    assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 1, 3, 2));
    assert!(Kernel::TensorElementwise.allocates_output());
    assert!(work.validate_aliases(&[0x1000, 0x1000, 0x2000]).is_ok());
    assert!(work.validate_aliases(&[0x1000, 0x2000, 0x1004]).is_err());
    assert!(work.validate_aliases(&[0x1000, 0x2000, 0x2004]).is_err());
    for (index, value) in [
        (0, 7),
        (1, 0),
        (1, 9),
        (2, Elementwise::ALL.len() as u32),
        (2, 1 << 17),
        (3, 1),
        (4, 4),
        (4, 6),
        (5, 1),
        (6, 0),
        (6, u32::MAX),
        (7, 2),
        (8, u32::MAX),
        (10, 2),
        (18, 1),
        (26, 1),
        (16, i32::MIN as u32),
    ] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::TensorElementwise.work(&invalid).is_err());
    }
    let mut scalar = p;
    scalar[2] |= SCALAR_RHS;
    scalar[3] = 0.25f32.to_bits();
    assert!(Kernel::TensorElementwise.work(&scalar).is_ok());
    scalar[2] = Elementwise::MaskFill as u32 | SCALAR_RHS;
    assert!(Kernel::TensorElementwise.work(&scalar).is_err());
    assert!(Kernel::TensorElementwise.work(&p[..31]).is_err());
}

#[test]
fn grouped_projection_checks_every_matrix_and_the_complete_packed_output() {
    let work = Kernel::GroupedProjectBf16.work(&[6, 4, 8, 4]).unwrap();
    assert_eq!(work.extents[..6], [6, 12, 24, 12, 12, 20]);
    assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 5, 6, 5));
    let addresses = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000];
    assert!(work.validate_aliases(&addresses).is_ok());
    for index in 0..5 {
        let mut invalid = addresses;
        invalid[5] = invalid[index] + 4;
        assert!(work.validate_aliases(&invalid).is_err());
    }
    for p in [
        [0, 4, 8, 4],
        [5, 4, 8, 4],
        [6, 0, 8, 4],
        [6, 4, 0, 4],
        [6, 4, 8, 0],
        [6, 5, 8, 4],
        [6, 4, 9, 4],
        [6, 4, 8, 5],
        [u32::MAX - 1, 4, 8, 4],
        [2, 1_500_000_000, 1_500_000_000, 1_000_000_000],
    ] {
        assert!(Kernel::GroupedProjectBf16.work(&p).is_err());
    }
}

#[test]
fn split_recurrence_checks_partial_columns_history_and_every_writer() {
    let convolution = Kernel::DeltaConvolution.work(&[33, 8, 3]).unwrap();
    assert_eq!(convolution.extents[..4], [99, 264, 264, 99]);
    assert_eq!((convolution.blocks, convolution.threads, convolution.mutated), (1, 128, 1 << 2));
    let recurrence = Kernel::DeltaUpdate.work(&[3, 33, 65, 7]).unwrap();
    assert_eq!(recurrence.extents[..7], [2751, 21, 21, 3, 3, 6435, 1365]);
    assert_eq!((recurrence.blocks, recurrence.threads, recurrence.mutated), (9, 128, 1 << 5));
    for work in [convolution, recurrence] {
        let addresses: Vec<_> = (0..work.pointers).map(|i| (i as u64 + 1) * 0x10000).collect();
        assert!(work.validate_aliases(&addresses).is_ok());
        for writer in 0..work.pointers {
            if (work.mutated | (1 << work.output)) & (1 << writer) == 0 {
                continue;
            }
            for other in 0..work.pointers {
                if writer == other {
                    continue;
                }
                let mut invalid = addresses.clone();
                invalid[writer] = invalid[other];
                assert!(work.validate_aliases(&invalid).is_err());
            }
        }
    }
    for p in [[33, 0, 3], [33, 9, 3], [33, 4, 0], [u32::MAX, 4, 3]] {
        assert!(Kernel::DeltaConvolution.work(&p).is_err());
    }
    for p in [[3, 129, 65, 7], [3, 33, 129, 7], [3, 33, 65, 0], [u32::MAX, 128, 128, 7]] {
        assert!(Kernel::DeltaUpdate.work(&p).is_err());
    }
}

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
fn split_attention_checks_partial_storage_merge_ranges_and_overflow() {
    let split = Kernel::AttentionFusedBatch.work(&[3, 4, 2, 129, 127, 16]).unwrap();
    assert_eq!((split.threads, split.blocks, split.pointers, split.output), (128, 192, 5, 4));
    assert_eq!(split.extents[..5], [1548, 33540, 33540, 3096, 25152]);
    let merge = Kernel::AttentionOutput.work(&[12, 129, 16]).unwrap();
    assert_eq!((merge.threads, merge.blocks, merge.pointers, merge.output), (128, 12, 3, 2));
    assert_eq!(merge.extents[..3], [25152, 3096, 1548]);
    for work in [split, merge] {
        let pointers: Vec<_> = (1..=work.pointers).map(|i| i as u64 * 0x100000).collect();
        assert!(work.validate_aliases(&pointers).is_ok());
        for source in 0..work.output {
            let mut aliased = pointers.clone();
            aliased[work.output] = aliased[source] + 4;
            assert!(work.validate_aliases(&aliased).is_err());
        }
    }
    for p in [
        [0, 4, 2, 129, 127, 16],
        [1, 3, 2, 129, 127, 16],
        [1, 4, 0, 129, 127, 16],
        [1, 4, 2, 257, 127, 16],
        [1, 4, 2, 129, 127, 0],
        [1, 4, 2, 129, 127, 33],
        [1, 1, 1, 1, u32::MAX - 1, 32],
        [u32::MAX, 1, 1, 1, 0, 16],
    ] {
        assert!(Kernel::AttentionFusedBatch.work(&p).is_err());
    }
    for p in [[0, 129, 16], [1, 0, 16], [1, 257, 16], [1, 129, 0], [1, 129, 33], [u32::MAX, 256, 32]] {
        assert!(Kernel::AttentionOutput.work(&p).is_err());
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
fn integer_matrix_bounds_every_buffer_and_rejects_aliased_outputs() {
    let kernel = Kernel::IntMatmul;
    let work = kernel.work(&[17, 33, 129]).unwrap();
    assert_eq!(work.extents[..3], [17 * 33, 33 * 129, 17 * 129]);
    assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 34, 3, 2));
    assert!(kernel.allocates_output());
    assert!(!work.in_place);
    assert!(work.validate_aliases(&[0x10000, 0x20000, 0x30000]).is_ok());
    assert!(work.validate_aliases(&[0x10000, 0x10000, 0x30000]).is_ok());
    for addresses in [[0x10000, 0x20000, 0x10004], [0x10000, 0x20000, 0x20004]] {
        assert!(work.validate_aliases(&addresses).is_err());
    }
    for dimensions in [[0, 1, 1], [1, 0, 1], [1, 1, 0], [u32::MAX, 2, 1], [1, 2, u32::MAX], [65536, 1, 65535]] {
        assert!(kernel.work(&dimensions).is_err());
    }
    assert!(kernel.work(&[2, 3]).is_err());
    assert!(kernel.work(&[2, 3, 4, 1]).is_err());
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
fn split_matrix_bounds_all_fp32_partials_and_rejects_aliases() {
    let work = Kernel::Matmul.work(&[17, 4099, 33, 32]).unwrap();
    assert_eq!(work.extents[..3], [17 * 4099, 4099 * 33, 32 * 17 * 33]);
    assert_eq!((work.threads, work.blocks, work.pointers, work.output), (128, 128, 3, 2));
    assert!(!work.in_place);
    assert!(work.validate_aliases(&[0x100000, 0x200000, 0x300000]).is_ok());
    assert!(work.validate_aliases(&[0x100000, 0x200000, 0x200004]).is_err());
    for dimensions in [[17, 4099, 33, 0], [0, 4099, 33, 32], [1, 1, 1, u32::MAX], [u32::MAX, 2, 1, 2]] {
        assert!(Kernel::Matmul.work(&dimensions).is_err());
    }
    assert!(Kernel::Project.work(&[17, 4099, 33, 32]).is_err());
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
    assert!(Kernel::Relu.allocates_output());
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
#[test]
fn quantized_weights_validate_block_rows_and_exact_word_extents() {
    for (format, block, bytes) in [(8, 32, 34u32), (11, 256, 110), (12, 256, 144), (13, 256, 176), (14, 256, 210)] {
        let inner = block * 3;
        let work = Kernel::ProjectQuantized.work(&[17, inner, 33, format]).unwrap();
        assert_eq!(work.extents[..3], [u64::from(17 * inner), (u64::from(99 * bytes)).div_ceil(4), 17 * 33]);
        assert_eq!(work.blocks, 2);
        assert_eq!(Kernel::ProjectQuantized.work(&[33, inner, 33, format]).unwrap().blocks, 4);
        assert_eq!(Kernel::ProjectQuantized.work(&[1, inner, 33, format]).unwrap().blocks, 9);
        let decode = Kernel::Dequantize.work(&[block, format]).unwrap();
        assert_eq!(decode.extents[..2], [u64::from(bytes).div_ceil(4), u64::from(block)]);
        let gather = Kernel::EmbeddingQuantized.work(&[5, block, 3, format]).unwrap();
        assert_eq!(gather.extents[..3], [u64::from(5 * bytes).div_ceil(4), 3, u64::from(3 * block)]);
        assert!(Kernel::ProjectQuantized.work(&[1, block - 1, 5, format]).is_err());
        assert!(Kernel::ProjectQuantized.work(&[1, block, u32::MAX, format]).is_err());
        assert!(Kernel::Dequantize.work(&[block - 1, format]).is_err());
        assert!(Kernel::EmbeddingQuantized.work(&[0, block, 1, format]).is_err());
        assert!(Kernel::EmbeddingQuantized.work(&[5, block - 1, 3, format]).is_err());
    }
    for format in [0, 1, 9, 10, 15, u32::MAX] {
        assert!(Kernel::ProjectQuantized.work(&[1, 256, 1, format]).is_err());
        assert!(Kernel::Dequantize.work(&[256, format]).is_err());
        assert!(Kernel::EmbeddingQuantized.work(&[1, 256, 1, format]).is_err());
    }
}
#[test]
fn grouped_recurrence_uses_shared_key_channels_and_separate_value_states() {
    for heads in [32u32, 48] {
        let work = Kernel::DeltaUpdate.work(&[heads, 128, 128, 3, 16]).unwrap();
        let channels = 2 * 16 * 128 + heads * 128;
        assert_eq!(
            work.extents[..7],
            [
                u64::from(3 * channels),
                u64::from(3 * heads),
                u64::from(3 * heads),
                u64::from(heads),
                u64::from(heads),
                u64::from(heads * 128 * 128),
                u64::from(3 * heads * 128)
            ]
        );
        assert_eq!(work.mutated, 1 << 5);
        assert_eq!(work.output, 6);
        assert_eq!(work.blocks, heads * 4);
    }
    for key_heads in [0, 7, 64] {
        assert!(Kernel::DeltaUpdate.work(&[32, 128, 128, 3, key_heads]).is_err());
    }
    assert!(Kernel::DeltaUpdate.work(&[32, 128, 128, u32::MAX, 16]).is_err());
}
