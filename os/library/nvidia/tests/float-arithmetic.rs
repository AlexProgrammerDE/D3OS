use d3os_nvidia::compute::{
    Kernel,
    cast::Format,
    elementwise::Elementwise,
    float_arithmetic::{self, FORMATS, Operation, SCALAR_RHS},
    storage::Width,
};

fn parameters(op: Operation, format: Format, count: u32) -> [u32; 32] {
    let mut p = [0; 32];
    p[..8].copy_from_slice(&[
        count,
        1,
        op as u32 | ((format as u32) << 8),
        0,
        0,
        0,
        count,
        if op.binary() { count } else { 0 },
    ]);
    p[8..16].fill(1);
    p[8] = count;
    p[16] = 1;
    if op.binary() {
        p[24] = 1;
    }
    p
}

fn scalar(mut p: [u32; 32], bits: u64) -> [u32; 32] {
    p[2] |= SCALAR_RHS;
    p[3] = bits as u32;
    p[5] = (bits >> 32) as u32;
    p[7] = 0;
    p[24..32].fill(0);
    p
}

#[test]
fn operation_registry_accepts_exactly_the_sixteen_float_primitives() {
    for elementwise in Elementwise::ALL {
        let selected = matches!(elementwise as u32, 1..=4 | 11..=16 | 21..=25 | 34);
        let operation = Operation::from_elementwise(elementwise);
        assert_eq!(operation.is_ok(), selected);
        if let Ok(op) = operation {
            assert_eq!(op as u32, elementwise as u32);
            assert_eq!(op.unary(), matches!(elementwise as u32, 11..=16 | 21..=25));
            assert_eq!(op.binary(), !op.unary());
        }
    }
    assert_eq!(Operation::ALL.len(), 16);
    for raw in 0..=256 {
        let selected = matches!(raw, 1..=4 | 11..=16 | 21..=25 | 34);
        assert_eq!(Operation::from_raw(raw).is_ok(), selected);
    }
    for op in Operation::ALL {
        for format in Format::ALL {
            let supported = FORMATS.contains(&format);
            assert_eq!(op.supports(format), supported);
            assert_eq!(float_arithmetic::validate(&parameters(op, format, 129)).is_ok(), supported);
        }
    }
}

#[test]
fn transposed_reversed_and_broadcast_views_retain_packed_storage_extents() {
    for op in Operation::ALL {
        for format in FORMATS {
            let mut p = parameters(op, format, 411);
            p[1] = 2;
            p[4] = 415;
            p[6] = 421;
            p[8..10].copy_from_slice(&[3, 137]);
            p[16..18].copy_from_slice(&[(-1i32) as u32, (-3i32) as u32]);
            if op.binary() {
                p[5] = 3;
                p[7] = 141;
                p[24..26].copy_from_slice(&[0, 1]);
            }
            let extents = [
                format.width().words(421),
                if op.binary() { format.width().words(141) } else { 1 },
                format.width().words(411),
            ];
            assert_eq!(float_arithmetic::validate(&p).unwrap(), extents);
            let mut invalid = p;
            invalid[4] = 409;
            assert!(float_arithmetic::validate(&invalid).is_err());
            if op.binary() {
                invalid = p;
                invalid[7] = 139;
                assert!(float_arithmetic::validate(&invalid).is_err());
            }
        }
    }
}

#[test]
fn signed_geometry_matches_enumerated_addresses_for_either_reader() {
    for format in FORMATS {
        for op in [Operation::Divide, Operation::Remainder, Operation::Exp, Operation::Log] {
            for input in 0..if op.binary() { 2 } else { 1 } {
                for offset in 0..9 {
                    for outer in -3i32..=3 {
                        for inner in -3i32..=3 {
                            let mut p = parameters(op, format, 6);
                            p[1] = 2;
                            p[4 + input] = offset;
                            p[6 + input] = 9;
                            p[8..10].copy_from_slice(&[2, 3]);
                            p[16..18].copy_from_slice(&[3, 1]);
                            if op.binary() {
                                p[24..26].copy_from_slice(&[3, 1]);
                            }
                            p[16 + input * 8..18 + input * 8].copy_from_slice(&[outer as u32, inner as u32]);
                            let reachable = (0..2).all(|row| (0..3).all(|column| (0..9).contains(&(offset as i32 + row * outer + column * inner))));
                            assert_eq!(float_arithmetic::validate(&p).is_ok(), reachable);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn physical_singleton_and_rank_eight_support_expansion_and_negative_strides() {
    for format in FORMATS {
        for op in Operation::ALL {
            // Native rank-zero values use this physical singleton geometry.
            assert!(float_arithmetic::validate(&parameters(op, format, 1)).is_ok());
            let mut p = parameters(op, format, 24);
            p[1] = 8;
            p[4] = 15;
            p[6] = 24;
            p[8..16].copy_from_slice(&[1, 2, 1, 2, 1, 2, 1, 3]);
            p[16..24].copy_from_slice(&[0, 0, 0, (-3i32) as u32, 0, 6, 0, 1]);
            if op.binary() {
                p[5] = 1;
                p[7] = 4;
                p[24..32].copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
            }
            assert!(float_arithmetic::validate(&p).is_ok());
            // Reducing physical rank makes the active final axis invalid.
            p[1] = 7;
            assert!(float_arithmetic::validate(&p).is_err());
        }
    }
}

#[test]
fn unused_rhs_geometry_and_scalar_widths_are_canonical() {
    for format in FORMATS {
        let width = format.width().bytes() * 8;
        let bits = u64::MAX >> (64 - width);
        for op in Operation::ALL {
            let p = parameters(op, format, 257);
            if op.unary() {
                assert!(float_arithmetic::validate(&p).is_ok());
                for field in [3, 5, 7, 24, 25, 26, 27, 28, 29, 30, 31] {
                    let mut invalid = p;
                    invalid[field] = 1;
                    assert!(float_arithmetic::validate(&invalid).is_err());
                }
                for bit in 16..32 {
                    let mut invalid = p;
                    invalid[2] |= 1 << bit;
                    assert!(float_arithmetic::validate(&invalid).is_err());
                    assert!(Kernel::FloatArithmetic.work(&invalid).is_err());
                }
            } else {
                // All destination bit patterns, including NaNs and infinities,
                // are valid scalar payloads. They are not integer ranges.
                let encoded = scalar(p, bits);
                assert_eq!(
                    float_arithmetic::validate(&encoded).unwrap(),
                    [format.width().words(257), 1, format.width().words(257)]
                );
                for field in [7, 24, 25, 26, 27, 28, 29, 30, 31] {
                    let mut invalid = encoded;
                    invalid[field] = 1;
                    assert!(float_arithmetic::validate(&invalid).is_err());
                }
                for bit in width..64 {
                    assert!(float_arithmetic::validate(&scalar(p, 1u64 << bit)).is_err());
                }
            }
        }
    }
}

#[test]
fn full_writers_count_packed_owners_and_forbid_every_output_overlap() {
    assert!(Kernel::FloatArithmetic.allocates_output());
    for format in FORMATS {
        for count in [1, 2, 3, 4, 5, 127, 128, 129, 255, 256, 257, 511, 512, 513] {
            let parameters = Operation::ALL.into_iter().flat_map(|op| {
                let p = parameters(op, format, count);
                [Some(p), op.binary().then(|| scalar(p, 1))].into_iter().flatten()
            });
            for p in parameters {
                let work = Kernel::FloatArithmetic.work(&p).unwrap();
                let owners = if format.width() == Width::Eight {
                    count
                } else {
                    format.width().words(count) as u32
                };
                assert_eq!(float_arithmetic::work_items(&p).unwrap(), owners);
                assert_eq!((work.blocks, work.threads, work.pointers, work.output), (owners.div_ceil(128), 128, 3, 2));
                assert_eq!(work.extents[..3], float_arithmetic::validate(&p).unwrap());
                assert_eq!(work.extents[2], format.width().words(count));
                assert!(!work.in_place);
                assert_eq!(work.mutated, 0);
                assert!(work.validate_aliases(&[0x10000, 0x10000, 0x30000]).is_ok());
                for input in 0..2 {
                    let addresses = [0x10000u64, 0x20000, 0x30000];
                    let start = addresses[input];
                    let end = start + work.extents[input] * 4;
                    for output in [start, end - 4, start - work.extents[2] * 4 + 4] {
                        let mut aliased = addresses;
                        aliased[2] = output;
                        assert!(work.validate_aliases(&aliased).is_err());
                    }
                    let mut adjacent = addresses;
                    adjacent[2] = end;
                    assert!(work.validate_aliases(&adjacent).is_ok());
                }
            }
        }
    }
}

#[test]
fn wide_physical_words_use_unsigned_bounds_while_logical_addresses_stay_signed() {
    for format in FORMATS {
        for op in Operation::ALL.into_iter().filter(|op| op.binary()) {
            let mut p = parameters(op, format, 1);
            p[4] = i32::MAX as u32 - 1;
            p[5] = p[4];
            p[6] = i32::MAX as u32;
            p[7] = p[6];
            let extents = float_arithmetic::validate(&p).unwrap();
            assert_eq!(extents[..2], [format.width().words(i32::MAX as u32); 2]);
            if format == Format::F64 {
                assert_eq!(extents, [u64::from(u32::MAX) - 1, u64::from(u32::MAX) - 1, 2]);
            }
            for field in [4, 5, 6, 7] {
                let mut invalid = p;
                invalid[field] = i32::MAX as u32 + 1;
                assert!(float_arithmetic::validate(&invalid).is_err());
            }
            let p = parameters(op, format, i32::MAX as u32);
            assert!(Kernel::FloatArithmetic.work(&p).is_ok());
        }
    }
}

#[test]
fn malformed_fields_empty_launches_and_geometry_overflow_are_rejected() {
    for format in FORMATS {
        for op in Operation::ALL.into_iter().filter(|op| op.binary()) {
            let p = parameters(op, format, 129);
            for (field, value) in [
                (0, 0),
                (0, 128),
                (0, i32::MAX as u32 + 1),
                (1, 0),
                (1, 9),
                (3, 1),
                (4, u32::MAX),
                (5, u32::MAX),
                (6, 0),
                (6, 128),
                (7, 0),
                (7, 128),
                (8, 0),
                (9, 2),
                (17, 1),
                (25, 1),
            ] {
                let mut invalid = p;
                invalid[field] = value;
                assert!(float_arithmetic::validate(&invalid).is_err(), "field {field}, value {value}");
                assert!(float_arithmetic::work_items(&invalid).is_err());
            }
            for bit in (16..24).chain(25..32) {
                let mut invalid = p;
                invalid[2] |= 1 << bit;
                assert!(float_arithmetic::validate(&invalid).is_err());
            }
            let mut invalid = p;
            invalid[2] = Operation::Add as u32 | (255 << 8);
            assert!(float_arithmetic::validate(&invalid).is_err());
            invalid = p;
            invalid[2] = 255 | ((Format::F16 as u32) << 8);
            assert!(float_arithmetic::validate(&invalid).is_err());
            let mut overflow = p;
            overflow[1] = 8;
            overflow[8..16].fill(65536);
            assert!(float_arithmetic::validate(&overflow).is_err());
            assert!(float_arithmetic::validate(&p[..31]).is_err());
            assert!(float_arithmetic::work_items(&p[..31]).is_err());
            assert!(float_arithmetic::validate(&[0; 33]).is_err());
        }
    }
}

#[test]
fn unary_boundaries_retain_signed_element_offsets_and_reject_empty_launches() {
    for format in FORMATS {
        for op in Operation::ALL.into_iter().filter(|op| op.unary()) {
            assert!(float_arithmetic::validate(&parameters(op, format, 0)).is_err());
            for count in [1, 2, 3, 127, 128, 129, 255, 256, 257] {
                // An odd element offset also tests halfword input alignment.
                let mut p = parameters(op, format, count);
                p[4] = 1;
                p[6] = 2 * count;
                p[16] = 2;
                let extents = [format.width().words(2 * count), 1, format.width().words(count)];
                assert_eq!(float_arithmetic::validate(&p).unwrap(), extents);
                p[4] = 2 * count - 1;
                p[16] = (-2i32) as u32;
                assert_eq!(float_arithmetic::validate(&p).unwrap(), extents);
                p[4] = 0;
                if count > 1 {
                    assert!(float_arithmetic::validate(&p).is_err());
                }
            }
            let mut p = parameters(op, format, 1);
            p[4] = i32::MAX as u32 - 1;
            p[6] = i32::MAX as u32;
            assert_eq!(
                float_arithmetic::validate(&p).unwrap(),
                [format.width().words(i32::MAX as u32), 1, format.width().words(1)]
            );
            p[4] += 1;
            assert!(float_arithmetic::validate(&p).is_err());
        }
    }
}

#[test]
fn remainder_tensor_and_scalar_boundaries_retain_signed_halfword_offsets() {
    for format in FORMATS {
        assert!(float_arithmetic::validate(&parameters(Operation::Remainder, format, 0)).is_err());
        for count in [1, 2, 3, 127, 128, 129, 255, 256, 257] {
            let mut p = parameters(Operation::Remainder, format, count);
            p[4] = 2 * count - 1;
            p[6] = 2 * count;
            p[16] = (-2i32) as u32;
            p[5] = 1;
            p[7] = 2 * count;
            p[24] = 2;
            assert_eq!(
                float_arithmetic::validate(&p).unwrap(),
                [format.width().words(2 * count), format.width().words(2 * count), format.width().words(count)]
            );
            let encoded = scalar(p, 1);
            assert_eq!(
                float_arithmetic::validate(&encoded).unwrap(),
                [format.width().words(2 * count), 1, format.width().words(count)]
            );
            for field in [4, 5] {
                let mut invalid = p;
                invalid[field] = 2 * count;
                assert!(float_arithmetic::validate(&invalid).is_err());
            }
        }
    }
}
