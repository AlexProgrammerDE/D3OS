use d3os_nvidia::compute::{
    Kernel,
    packed_gather::{validate, work_items},
    storage::Width,
};

fn strided_parameters(width: Width) -> [u32; 32] {
    let mut p = [0u32; 32];
    p[..8].copy_from_slice(&[3 | ((width as u32) << 8), 1, 3 * 137 * 7, 117, 24, 3 * 137 * 7, 6, 5]);
    p[8..16].fill(1);
    p[8..11].copy_from_slice(&[3, 137, 7]);
    p[16..19].copy_from_slice(&[(-2i32) as u32, 23, (-3i32) as u32]);
    p[24..27].copy_from_slice(&[7, 21, (-1i32) as u32]);
    p
}

#[test]
fn signed_noncontiguous_views_keep_logical_geometry_and_packed_extents() {
    for (width, source, output, work) in [
        (Width::One, 30, 720, 720),
        (Width::Two, 59, 1439, 1439),
        (Width::Four, 117, 2877, 2877),
        (Width::Eight, 234, 5754, 2877),
    ] {
        let p = strided_parameters(width);
        assert_eq!(validate(&p).unwrap(), [source, 2877, 1, output]);
        assert_eq!(work_items(&p).unwrap(), work);
    }
}

#[test]
fn geometry_rejects_bounds_errors_even_inside_the_same_physical_word() {
    for width in Width::ALL {
        let p = strided_parameters(width);
        for (field, value) in [
            (1, 3),
            (2, 2876),
            (3, 116),
            (4, 21),
            (5, 2876),
            (6, 5),
            (7, 6),
            (8, 0),
            (11, 2),
            (19, 1),
            (27, 1),
        ] {
            let mut invalid = p;
            invalid[field] = value;
            assert!(validate(&invalid).is_err());
            assert!(work_items(&invalid).is_err());
        }
    }
}

#[test]
fn rank_width_flags_and_parameter_count_are_canonical() {
    let p = strided_parameters(Width::One);
    for raw in [0, 9, 0x10300, 3 | (4 << 8), 3 | (1 << 10), 3 | (1 << 16), 3 | (1 << 31)] {
        let mut invalid = p;
        invalid[0] = raw;
        assert!(validate(&invalid).is_err());
        assert!(work_items(&invalid).is_err());
    }
    assert!(validate(&p[..31]).is_err());
    assert!(work_items(&p[..31]).is_err());
    let mut extended = p.to_vec();
    extended.push(0);
    assert!(validate(&extended).is_err());
    assert!(work_items(&extended).is_err());
}

#[test]
fn maximum_logical_extent_can_use_unsigned_physical_word_addresses() {
    let mut p = [0u32; 32];
    p[..8].copy_from_slice(&[1, 0, i32::MAX as u32, i32::MAX as u32, 0, 1, 0, i32::MAX as u32]);
    p[8..16].fill(1);
    p[8] = i32::MAX as u32;
    p[16] = 1;
    for (width, words, work) in [
        (Width::One, 536870912, 536870912),
        (Width::Two, 1073741824, 1073741824),
        (Width::Four, 2147483647, 2147483647),
        (Width::Eight, 4294967294, 2147483647),
    ] {
        p[0] = 1 | ((width as u32) << 8);
        assert_eq!(validate(&p).unwrap(), [words, 1, 1, words]);
        assert_eq!(work_items(&p).unwrap(), work);
    }
    p[8] = 1 << 31;
    p[2] = 1 << 31;
    assert!(validate(&p).is_err());
}

#[test]
fn empty_source_axis_is_valid_for_runtime_index_rejection() {
    for width in Width::ALL {
        let mut p = strided_parameters(width);
        p[3] = 1;
        p[4] = 1;
        p[7] = 0;
        p[16..24].fill(0);
        assert_eq!(validate(&p).unwrap(), [width.words(1), 2877, 1, width.words(2877)]);
        p[2] = 0;
        assert!(validate(&p).is_err());
        assert!(work_items(&p).is_err());
    }
}

#[test]
fn all_axes_check_signed_source_and_index_spans() {
    for axis in 0..3 {
        for source_stride in -4i32..=4 {
            for index_stride in -4i32..=4 {
                let mut p = [0u32; 32];
                p[..8].copy_from_slice(&[3, axis, 27, 25, 12, 25, 12, 5]);
                p[8..16].fill(1);
                p[8..11].fill(3);
                p[16..19].fill(1);
                p[24..27].fill(index_stride as u32);
                p[16 + axis as usize] = source_stride as u32;
                let mut source_fits = true;
                let mut indices_fit = true;
                for x in 0..3i64 {
                    for y in 0..3i64 {
                        for z in 0..3i64 {
                            let coordinates = [x, y, z];
                            let index = 12 + (x + y + z) * i64::from(index_stride);
                            indices_fit &= (0..25).contains(&index);
                            for selected in 0..5i64 {
                                let source = 12
                                    + coordinates
                                        .iter()
                                        .enumerate()
                                        .filter(|&(a, _)| a != axis as usize)
                                        .map(|(_, &c)| c)
                                        .sum::<i64>()
                                    + selected * i64::from(source_stride);
                                source_fits &= (0..25).contains(&source);
                            }
                        }
                    }
                }
                assert_eq!(validate(&p).is_ok(), source_fits && indices_fit);
            }
        }
    }
}

#[test]
fn registered_work_owns_output_words_and_isolates_both_writers() {
    for width in Width::ALL {
        let p = strided_parameters(width);
        let work = Kernel::PackedGather.work(&p).unwrap();
        assert_eq!(work.extents[..4], validate(&p).unwrap());
        assert_eq!((work.blocks, work.threads), (work_items(&p).unwrap().div_ceil(128), 128));
        assert_eq!((work.pointers, work.output, work.mutated, work.in_place), (4, 3, 1 << 2, false));
        assert!(Kernel::PackedGather.allocates_output());
        let addresses = [0x10000, 0x20000, 0x30000, 0x40000];
        assert!(work.validate_aliases(&addresses).is_ok());
        assert!(work.validate_aliases(&[0x10000, 0x10000, 0x30000, 0x40000]).is_ok());
        for writer in [2, 3] {
            for other in (0..4).filter(|&other| other != writer) {
                let mut overlapping = addresses;
                overlapping[writer] = addresses[other] + (work.extents[other] - 1) * 4;
                assert!(work.validate_aliases(&overlapping).is_err());
                overlapping[writer] = addresses[other] + work.extents[other] * 4;
                assert!(work.validate_aliases(&overlapping).is_ok());
            }
        }
    }
}
