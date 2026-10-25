use d3os_nvidia::compute::{
    Kernel,
    packed_updates::{MASK_U8, REMAINDER_SHIFT, SCALAR, mask_work_items, validate_assign, validate_bool_scatter, validate_mask},
    storage::Width,
};

fn views(width: Width) -> [u32; 32] {
    let mut p = [0; 32];
    p[..8].copy_from_slice(&[129, 2, width as u32, 0, 128, 2, 133, 89]);
    p[8..16].fill(1);
    p[8..10].copy_from_slice(&[3, 43]);
    p[16..18].copy_from_slice(&[(-43i32) as u32, (-1i32) as u32]);
    p[24..26].copy_from_slice(&[1, 2]);
    p
}

#[test]
fn masks_round_physical_extents_after_adding_the_initial_byte_remainder() {
    for width in Width::ALL {
        for remainder in 0..4 {
            let mut p = views(width);
            p[2] |= MASK_U8 | remainder << REMAINDER_SHIFT;
            assert_eq!(
                validate_mask(&p).unwrap(),
                [width.words(133), width.words(89), (129u64 + u64::from(remainder)).div_ceil(4), width.words(129)]
            );
            assert_eq!(mask_work_items(&p).unwrap(), if width == Width::Eight { 129 } else { width.words(129) as u32 });
        }
        let p = views(width);
        assert_eq!(validate_mask(&p).unwrap()[2], 129);
        for remainder in 1..4 {
            let mut invalid = p;
            invalid[2] |= remainder << REMAINDER_SHIFT;
            assert!(validate_mask(&invalid).is_err());
        }
    }
}

#[test]
fn scalar_union_preserves_both_halves_and_requires_canonical_unused_geometry() {
    for width in Width::ALL {
        let bits: u64 = match width {
            Width::One => 0xef,
            Width::Two => 0xcdef,
            Width::Four => 0x89ab_cdef,
            Width::Eight => 0xfff8_5678_89ab_cdef,
        };
        let mut p = views(width);
        p[2] |= SCALAR | MASK_U8 | 3 << REMAINDER_SHIFT;
        p[3] = bits as u32;
        p[5] = (bits >> 32) as u32;
        p[7] = 0;
        p[24..32].fill(0);
        assert_eq!(validate_mask(&p).unwrap(), [width.words(133), 1, 33, width.words(129)]);
        for (field, value) in [(7, 1), (24, 1), (31, 1), (2, p[2] | 1 << 12)] {
            let mut invalid = p;
            invalid[field] = value;
            assert!(validate_mask(&invalid).is_err());
        }
        if width != Width::Eight {
            let mut invalid = p;
            if width == Width::Four {
                invalid[5] = 1;
            } else {
                invalid[3] |= 1 << (8 * width.bytes());
            }
            assert!(validate_mask(&invalid).is_err());
        }
    }
}

#[test]
fn logical_bounds_are_checked_before_packed_rounding() {
    for width in Width::ALL {
        let p = views(width);
        assert!(validate_assign(&p).is_ok());
        for (field, value) in [(4, 127), (5, 3), (6, 128), (7, 88), (2, 4), (3, 1), (10, 2), (18, 1), (26, 1)] {
            let mut invalid = p;
            invalid[field] = value;
            assert!(validate_assign(&invalid).is_err());
        }
        assert!(validate_assign(&p[..31]).is_err());
        assert!(validate_mask(&p[..31]).is_err());
        let mut extended = p.to_vec();
        extended.push(0);
        assert!(validate_assign(&extended).is_err());
        assert!(validate_mask(&extended).is_err());
    }
}

#[test]
fn assignment_acceptance_proves_unique_logical_fields_even_when_words_are_shared() {
    for width in Width::ALL {
        for x in -5i32..=5 {
            for y in -5i32..=5 {
                let mut p = [0; 32];
                p[..8].copy_from_slice(&[6, 2, width as u32, 0, 20, 0, 41, 6]);
                p[8..16].fill(1);
                p[8..10].copy_from_slice(&[2, 3]);
                p[16..18].copy_from_slice(&[x as u32, y as u32]);
                p[24..26].copy_from_slice(&[3, 1]);
                let addresses: Vec<_> = (0..2).flat_map(|row| (0..3).map(move |col| 20 + row * x + col * y)).collect();
                if validate_assign(&p).is_ok() {
                    for (i, address) in addresses.iter().enumerate() {
                        assert!((0..41).contains(address));
                        assert!(!addresses[..i].contains(address));
                    }
                }
            }
        }
        // Adjacent fields are valid despite sharing physical words.
        let mut p = views(width);
        p[16..18].copy_from_slice(&[43, 1]);
        p[4] = 0;
        assert_eq!(validate_assign(&p).unwrap(), [width.words(133), width.words(89)]);
        p[16] = 1;
        assert!(validate_assign(&p).is_err());
        p[16] = 0;
        assert!(validate_assign(&p).is_err());
    }
}

fn scatter(width: Width, remainder: u32) -> [u32; 32] {
    let mut p = [0; 32];
    p[..8].copy_from_slice(&[8 | (width as u32) << 8 | remainder << REMAINDER_SHIFT, 4, 258, 14, 6, 258, 128, 7]);
    p[8..16].copy_from_slice(&[1, 2, 1, 1, 129, 1, 1, 1]);
    p[17] = 7;
    p[20] = (-1i32) as u32;
    p[25] = 129;
    p[28] = (-1i32) as u32;
    p
}

#[test]
fn boolean_scatter_decodes_rank_separately_from_width_and_byte_remainder() {
    for remainder in 0..4 {
        let p = scatter(Width::One, remainder);
        assert_eq!(validate_bool_scatter(&p).unwrap(), [(258u64 + u64::from(remainder)).div_ceil(4), 258, 1, 4]);
    }
    assert_eq!(validate_bool_scatter(&scatter(Width::Four, 0)).unwrap(), [258, 258, 1, 14]);
    for width in [Width::Two, Width::Eight] {
        assert!(validate_bool_scatter(&scatter(width, 0)).is_err());
    }
    for remainder in 1..4 {
        assert!(validate_bool_scatter(&scatter(Width::Four, remainder)).is_err());
    }
    let p = scatter(Width::One, 3);
    for (field, value) in [(0, p[0] | 1 << 12), (0, p[0] | 1 << 31), (1, 8), (3, 13), (4, 5), (5, 257), (6, 127), (2, 257)] {
        let mut invalid = p;
        invalid[field] = value;
        assert!(validate_bool_scatter(&invalid).is_err());
    }
    assert!(validate_bool_scatter(&p[..31]).is_err());
}

fn check_writer_aliases(kernel: Kernel, p: &[u32]) {
    let work = kernel.work(p).unwrap();
    let addresses = [0x10000, 0x20000, 0x30000, 0x40000];
    assert!(work.validate_aliases(&addresses[..work.pointers]).is_ok());
    for writer in (0..work.pointers).filter(|&i| i == work.output || work.mutated & (1 << i) != 0) {
        for other in (0..work.pointers).filter(|&i| i != writer) {
            let mut overlap = addresses;
            overlap[writer] = addresses[other] + (work.extents[other] - 1) * 4;
            assert!(work.validate_aliases(&overlap[..work.pointers]).is_err());
            overlap[writer] = addresses[other] + work.extents[other] * 4;
            assert!(work.validate_aliases(&overlap[..work.pointers]).is_ok());
        }
    }
}

#[test]
fn registered_writers_enforce_physical_alias_bounds_and_initialization_contracts() {
    for width in Width::ALL {
        let p = views(width);
        let assignment = Kernel::PackedAssign.work(&p).unwrap();
        assert_eq!(assignment.extents[..2], validate_assign(&p).unwrap());
        assert_eq!(
            (assignment.pointers, assignment.output, assignment.mutated, assignment.in_place),
            (2, 0, 0, false)
        );
        assert_eq!((assignment.blocks, assignment.threads), (2, 128));
        assert!(!Kernel::PackedAssign.allocates_output());
        check_writer_aliases(Kernel::PackedAssign, &p);
        let mask = Kernel::PackedMaskWhere.work(&p).unwrap();
        assert_eq!(mask.extents[..4], validate_mask(&p).unwrap());
        assert_eq!((mask.pointers, mask.output, mask.mutated, mask.in_place), (4, 3, 0, false));
        assert_eq!(mask.blocks, mask_work_items(&p).unwrap().div_ceil(128));
        assert!(Kernel::PackedMaskWhere.allocates_output());
        check_writer_aliases(Kernel::PackedMaskWhere, &p);
    }
    for width in [Width::One, Width::Four] {
        let p = scatter(width, 0);
        let work = Kernel::PackedBoolScatter.work(&p).unwrap();
        assert_eq!(work.extents[..4], validate_bool_scatter(&p).unwrap());
        assert_eq!((work.pointers, work.output, work.mutated, work.in_place), (4, 3, 1 << 2, false));
        assert_eq!((work.blocks, work.threads), (3, 128));
        assert!(!Kernel::PackedBoolScatter.allocates_output());
        check_writer_aliases(Kernel::PackedBoolScatter, &p);
    }
}
