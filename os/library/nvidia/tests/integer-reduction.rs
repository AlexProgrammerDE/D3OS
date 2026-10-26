use d3os_nvidia::compute::{
    Kernel,
    cast::Format,
    integer_reduction::{self as reduction, ALL_AXES, FORMATS, Operation, Schedule},
    reduction::Reduction,
    storage::Width,
};

fn packet(op: Operation, input: Format, output: Format, dims: &[u32], axis: u32) -> [u32; 24] {
    let mut p = [0; 24];
    p[1] = dims.len() as u32;
    p[2] = reduction::control(op, input, output).unwrap();
    p[3] = axis;
    p[8..16].fill(1);
    p[8..8 + dims.len()].copy_from_slice(dims);
    let mut stride = 1;
    for i in (0..dims.len()).rev() {
        p[16 + i] = stride;
        stride *= dims[i];
    }
    p[5] = stride;
    p[0] = if axis == ALL_AXES {
        1
    } else {
        dims.iter().enumerate().filter_map(|(i, &d)| (i != axis as usize).then_some(d)).product()
    };
    p[6] = if axis == ALL_AXES { stride } else { dims[axis as usize] };
    p
}

#[test]
fn control_checks_every_format_operation_and_accumulator() {
    for legacy in Reduction::ALL.into_iter().chain(Reduction::INTEGERS) {
        assert_eq!(reduction::Operation::from_reduction(legacy).unwrap() as u32, legacy as u32 % 7);
    }
    for op in Operation::ALL {
        for input in Format::ALL {
            for output in Format::ALL {
                let supported = FORMATS.contains(&input) && FORMATS.contains(&output) && (op.index() || input == output);
                let control = reduction::control(op, input, output);
                assert_eq!(control.is_ok(), supported);
                if let Ok(control) = control {
                    let accumulator = if op as u32 >= 3 {
                        input
                    } else {
                        match input {
                            Format::I64 | Format::U64 => input,
                            Format::I32 | Format::I16 | Format::I8 => Format::I32,
                            _ => Format::U32,
                        }
                    };
                    assert_eq!(control >> 24, accumulator as u32);
                    let mut p = packet(op, input, output, &[3, 129], 1);
                    assert!(reduction::validate(&p).is_ok());
                    p[2] ^= 1 << 24;
                    assert!(reduction::geometry(&p).is_err());
                }
            }
        }
    }
    assert!(Operation::from_raw(7).is_err());
}

#[test]
fn geometry_switches_schedules_without_multiplying_by_length() {
    for format in FORMATS {
        for count in [1, 3, 5, 127, 129] {
            for length in [0, 1, 127, 128, 129, 4096] {
                let p = packet(Operation::Product, format, format, &[count, length], 1);
                let g = reduction::geometry(&p).unwrap();
                let owners = if format.width() == Width::Eight {
                    count
                } else {
                    format.width().words(count) as u32
                };
                let tree = length >= 128;
                assert_eq!(g.owners, owners);
                assert_eq!(g.schedule, if tree { Schedule::Tree } else { Schedule::Short });
                assert_eq!(reduction::work_items(&p).unwrap(), owners * if tree { 128 } else { 1 });
                assert_eq!(g.output_words, format.width().words(count));
                assert_eq!(g.input_words, if length == 0 { 1 } else { format.width().words(count * length) });
            }
        }
    }
    // This geometry is valid without allocating its declared storage.
    let mut p = packet(Operation::Sum, Format::U64, Format::U64, &[1], 0);
    p[5] = i32::MAX as u32;
    p[4] = p[5] - 1;
    assert_eq!(reduction::validate(&p).unwrap(), [4_294_967_294, 2]);
    p = packet(Operation::Sum, Format::I8, Format::I8, &[i32::MAX as u32], 0);
    assert_eq!(reduction::work_items(&p).unwrap(), 128);
}

#[test]
fn empty_wide_inputs_use_one_dummy_word_and_validate_before_returning() {
    for format in FORMATS {
        for op in Operation::ALL {
            let p = packet(op, format, format, &[2, 0], 1);
            assert_eq!(reduction::geometry(&p).is_ok(), op.allows_empty());
            if op.allows_empty() {
                assert_eq!(reduction::validate(&p).unwrap(), [1, format.width().words(2)]);
                let mut view = p;
                view[4] = 7;
                view[5] = 7;
                assert_eq!(reduction::validate(&view).unwrap()[0], 1);
                view[4] = 8;
                assert!(reduction::geometry(&view).is_err());
            }
            let p = packet(op, format, format, &[0, 3], 1);
            let g = reduction::geometry(&p).unwrap();
            assert_eq!((g.input_words, g.output_words, g.owners), (1, 1, 0));
            assert!(reduction::validate(&p).is_err());
            let p = packet(op, format, format, &[0, 0], 1);
            assert_eq!(reduction::geometry(&p).is_ok(), op.allows_empty());
        }
    }
}

#[test]
fn signed_reverse_broadcast_and_rank_eight_views_keep_declared_byte_bounds() {
    for format in FORMATS {
        let mut p = packet(Operation::Max, format, format, &[2, 3], 1);
        p[4] = 5;
        p[16..18].copy_from_slice(&[(-3i32) as u32, (-1i32) as u32]);
        assert_eq!(reduction::validate(&p).unwrap(), [format.width().words(6), format.width().words(2)]);
        p[4] = 4;
        assert!(reduction::validate(&p).is_err());
        p[4] = 0;
        p[5] = 1;
        p[16..18].fill(0);
        assert_eq!(reduction::validate(&p).unwrap()[0], format.width().words(1));
        let p = packet(Operation::ArgMax, format, Format::I8, &[2; 8], 7);
        assert_eq!(reduction::geometry(&p).unwrap().output_count, 128);
        assert!(reduction::validate(&p).is_ok());
    }
    let mut p = packet(Operation::Sum, Format::U64, Format::U64, &[1], 0);
    p[4] = 3;
    p[5] = 8;
    assert_eq!(reduction::validate(&p).unwrap(), [16, 2]);
}

#[test]
fn zero_aware_products_distinguish_empty_input_from_overflowing_output() {
    let mut p = packet(Operation::Product, Format::I64, Format::I64, &[0, 1, 1], 1);
    p[9..11].fill(i32::MAX as u32);
    p[6] = i32::MAX as u32;
    assert!(reduction::geometry(&p).is_ok());
    p[3] = 0;
    p[6] = 0;
    assert!(reduction::geometry(&p).is_err());
    p[3] = ALL_AXES;
    p[0] = 1;
    assert_eq!(reduction::validate(&p).unwrap(), [1, 2]);
    let mut p = packet(Operation::Sum, Format::I8, Format::I8, &[1], 0);
    p[1] = 8;
    p[8..16].fill(65536);
    assert!(reduction::geometry(&p).is_err());
    let mut p = packet(Operation::Sum, Format::U64, Format::U64, &[1, 1, 0], 0);
    p[8..10].fill(i32::MAX as u32);
    p[6] = i32::MAX as u32;
    // The trailing zero makes both counts zero despite the preceding product.
    assert_eq!(reduction::geometry(&p).unwrap().output_count, 0);
}

#[test]
fn canonical_fields_all_axis_rules_and_signed_limits_are_checked() {
    let p = packet(Operation::Sum, Format::I64, Format::I64, &[3, 129], 1);
    for (field, value) in [
        (0, 2),
        (1, 0),
        (1, 9),
        (3, 2),
        (4, u32::MAX),
        (5, 0),
        (5, i32::MAX as u32 + 1),
        (6, 128),
        (7, 1),
        (7, 2),
        (10, 2),
        (18, 1),
        (8, u32::MAX),
    ] {
        let mut invalid = p;
        invalid[field] = value;
        assert!(reduction::geometry(&invalid).is_err(), "field {field}: {value}");
    }
    assert!(reduction::geometry(&p[..23]).is_err());
    assert!(reduction::geometry(&[0; 25]).is_err());
    for op in Operation::ALL {
        let p = packet(op, Format::I8, Format::I8, &[2, 129], ALL_AXES);
        assert_eq!(reduction::validate(&p).is_ok(), !op.index());
    }
}

#[test]
fn registered_launch_owns_full_words_and_rejects_partial_aliases() {
    assert!(Kernel::IntegerReduction.allocates_output());
    for format in FORMATS {
        for count in [1, 3, 5, 127, 128, 129, 257] {
            for length in [0, 1, 127, 128, 4096] {
                let p = packet(Operation::Product, format, format, &[count, length], 1);
                let g = reduction::geometry(&p).unwrap();
                let work = Kernel::IntegerReduction.work(&p).unwrap();
                assert_eq!(work.extents[..2], [g.input_words, g.output_words]);
                assert_eq!((work.pointers, work.output, work.in_place, work.mutated), (2, 1, false, 0));
                assert_eq!(
                    (work.blocks, work.threads),
                    (if length >= 128 { g.owners } else { g.owners.div_ceil(128) }, 128)
                );
                assert_eq!(Kernel::IntegerReduction.entry(&p), reduction::symbol(&p).unwrap());
                let start = 0x100000u64;
                let end = start + g.input_words * 4;
                // Four-byte alignment also permits the wide load/store contract.
                assert!(work.validate_aliases(&[start + 4, end + 4]).is_ok());
                assert!(work.validate_aliases(&[start, end]).is_ok());
                for output in [start, end - 4, start - g.output_words * 4 + 4] {
                    assert!(work.validate_aliases(&[start, output]).is_err());
                }
                let per_owner = if format.width() == Width::Eight { 1 } else { 4 / format.width().bytes() };
                let payload_bytes = u64::from(count) * u64::from(format.width().bytes());
                let mut bytes_owned = 0;
                for owner in 0..g.owners {
                    let valid_rows = (count - owner * per_owner).min(per_owner);
                    bytes_owned += u64::from(valid_rows) * u64::from(format.width().bytes());
                }
                assert_eq!(bytes_owned, payload_bytes);
                assert!(g.output_words * 4 - bytes_owned < 4);
            }
        }
    }
    let p = packet(Operation::Min, Format::U64, Format::U64, &[0, 3], 1);
    assert!(Kernel::IntegerReduction.work(&p).is_err());
}
