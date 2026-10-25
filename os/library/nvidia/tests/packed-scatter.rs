use d3os_nvidia::compute::{Kernel, packed_scatter, storage::Width};

fn parameters(width: Width, axis: usize) -> [u32; 32] {
    let mut p = [0u32; 32];
    let shape = [3, 5, 7];
    let mut updates = shape;
    updates[axis] = 137;
    p[..8].copy_from_slice(&[
        3 | ((width as u32) << 8),
        axis as u32,
        updates.iter().product(),
        107,
        2,
        3000,
        2990,
        shape[axis],
    ]);
    p[8..16].fill(1);
    p[8..11].copy_from_slice(&updates);
    p[16..19].copy_from_slice(&[35, 7, 1]);
    p[24..27].copy_from_slice(&[(-3i32) as u32, (-5i32) as u32, (-7i32) as u32]);
    p
}

#[test]
fn all_widths_and_axes_use_logical_row_ownership_and_physical_extents() {
    for width in Width::ALL {
        for axis in 0..3 {
            let p = parameters(width, axis);
            let sizes = packed_scatter::validate(&p).unwrap();
            assert_eq!(sizes, [width.words(p[2]), 3000, 1, width.words(107)]);
            let rows = p[2] / 137;
            assert_eq!(packed_scatter::work_items(&p).unwrap(), rows);
            let work = Kernel::PackedScatterAssign.work(&p).unwrap();
            assert_eq!(work.extents[..4], sizes);
            assert_eq!((work.blocks, work.threads), (rows.div_ceil(128), 128));
            assert_eq!((work.pointers, work.output, work.mutated, work.in_place), (4, 3, 4, false));
            assert!(!Kernel::PackedScatterAssign.allocates_output());
        }
    }
}

#[test]
fn destination_density_rejects_overlapping_rows_and_strided_or_reversed_outputs() {
    for width in Width::ALL {
        for axis in 0..3 {
            let p = parameters(width, axis);
            for (field, value) in [(16, 34), (17, 6), (18, 0), (18, (-1i32) as u32), (3, 106), (4, 3), (7, 0)] {
                let mut invalid = p;
                invalid[field] = value;
                assert!(packed_scatter::validate(&invalid).is_err(), "width={width:?} axis={axis} field={field}");
                assert!(packed_scatter::work_items(&invalid).is_err());
            }
        }
    }
}

#[test]
fn index_bounds_include_signed_spans_and_logical_endpoints() {
    for axis in 0..3 {
        let mut p = parameters(Width::One, axis);
        // The index view's maximal address is exactly its offset.
        p[5] = p[6];
        assert!(packed_scatter::validate(&p).is_err());
        p[5] += 1;
        assert!(packed_scatter::validate(&p).is_ok());
        p[6] = 0;
        assert!(packed_scatter::validate(&p).is_err());
        p[24..27].fill(0);
        assert!(packed_scatter::validate(&p).is_ok());
    }
}

#[test]
fn encoding_rejects_reserved_flags_rank_tail_and_mismatched_counts() {
    let p = parameters(Width::Eight, 1);
    for (field, value) in [
        (0, 0),
        (0, 9),
        (0, 3 | (4 << 8)),
        (0, p[0] | (1 << 16)),
        (1, 3),
        (2, 0),
        (2, p[2] - 1),
        (11, 2),
        (19, 1),
        (27, 1),
    ] {
        let mut invalid = p;
        invalid[field] = value;
        assert!(packed_scatter::validate(&invalid).is_err());
    }
    assert!(packed_scatter::validate(&p[..31]).is_err());
    let mut long = p.to_vec();
    long.push(0);
    assert!(packed_scatter::validate(&long).is_err());
}

#[test]
fn writers_cannot_alias_inputs_status_or_each_other() {
    for width in Width::ALL {
        let p = parameters(width, 1);
        let work = Kernel::PackedScatterAssign.work(&p).unwrap();
        let bases = [0x10000, 0x20000, 0x30000, 0x40000];
        assert!(work.validate_aliases(&bases).is_ok());
        assert!(work.validate_aliases(&[0x10000, 0x10000, 0x30000, 0x40000]).is_ok());
        for writer in [2, 3] {
            for other in (0..4).filter(|&i| i != writer) {
                let mut touching = bases;
                touching[writer] = bases[other] + work.extents[other] * 4;
                assert!(work.validate_aliases(&touching).is_ok());
                touching[writer] -= 4;
                assert!(work.validate_aliases(&touching).is_err());
            }
        }
    }
}

#[test]
fn wide_physical_extent_need_not_fit_signed_logical_addresses() {
    let mut p = [0u32; 32];
    p[..8].copy_from_slice(&[1 | ((Width::Eight as u32) << 8), 0, i32::MAX as u32, i32::MAX as u32, 0, 1, 0, i32::MAX as u32]);
    p[8..16].fill(1);
    p[8] = i32::MAX as u32;
    p[16] = 1;
    assert_eq!(packed_scatter::validate(&p).unwrap(), [4294967294, 1, 1, 4294967294]);
    assert_eq!(packed_scatter::work_items(&p).unwrap(), 1);
}
