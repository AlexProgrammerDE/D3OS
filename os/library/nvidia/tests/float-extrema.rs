use d3os_nvidia::compute::{
    Kernel,
    cast::Format,
    float_extrema::{self as extrema, ALL_AXES, FORMATS, Operation, Schedule},
    integer_reduction,
    reduction::Reduction,
    storage::Width,
};

fn packet(op: Operation, input: Format, output: Format, dims: &[u32], axis: u32) -> [u32; 24] {
    let mut p = [0; 24];
    p[1] = dims.len() as u32;
    p[2] = extrema::control(op, input, output).unwrap();
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
fn exact_control_rejects_arithmetic_integer_and_noncanonical_policies() {
    for legacy in Reduction::ALL.into_iter().chain(Reduction::INTEGERS) {
        let op = Operation::from_reduction(legacy);
        assert_eq!(op.is_ok(), !legacy.is_integer() && legacy as u32 >= 3);
        if let Ok(op) = op {
            assert_eq!(op as u32, legacy as u32);
        }
    }
    for raw in 0..256 {
        assert_eq!(Operation::from_raw(raw).is_ok(), (3..=6).contains(&raw));
    }
    assert!(Operation::from_raw(u32::MAX).is_err());
    for op in Operation::ALL {
        for input in Format::ALL {
            for output in Format::ALL {
                let supported = FORMATS.contains(&input)
                    && if op.index() {
                        integer_reduction::FORMATS.contains(&output)
                    } else {
                        input == output
                    };
                let control = extrema::control(op, input, output);
                assert_eq!(control.is_ok(), supported);
                if let Ok(control) = control {
                    assert_eq!(control, op as u32 | (input as u32) << 8 | (output as u32) << 16 | (input as u32) << 24);
                    let p = packet(op, input, output, &[3, 129], 1);
                    assert!(extrema::validate(&p).is_ok());
                    for bit in 24..32 {
                        let mut invalid = p;
                        invalid[2] ^= 1 << bit;
                        assert!(extrema::geometry(&invalid).is_err());
                    }
                }
            }
        }
    }
}

#[test]
fn registered_work_owns_whole_words_for_each_schedule_and_output_width() {
    assert!(Kernel::FloatExtrema.allocates_output());
    for input in FORMATS {
        for op in Operation::ALL {
            let outputs: &[Format] = if op.index() {
                &integer_reduction::FORMATS
            } else {
                core::slice::from_ref(&input)
            };
            for &output in outputs {
                for count in [1, 3, 5, 127, 128, 129, 257] {
                    for length in [1, 127, 128, 129, 4096] {
                        let p = packet(op, input, output, &[count, length], 1);
                        let g = extrema::geometry(&p).unwrap();
                        let tree = length >= 128;
                        let owners = if output.width() == Width::Eight {
                            count
                        } else {
                            output.width().words(count) as u32
                        };
                        assert_eq!(g.owners, owners);
                        assert_eq!(g.output_count, count);
                        assert_eq!(g.reduction_length, length);
                        assert_eq!(g.schedule, if tree { Schedule::Tree } else { Schedule::Short });
                        assert_eq!(extrema::work_items(&p).unwrap(), owners * if tree { 128 } else { 1 });
                        let work = Kernel::FloatExtrema.work(&p).unwrap();
                        assert_eq!(work.extents[..2], [input.width().words(count * length), output.width().words(count)]);
                        assert_eq!((work.pointers, work.output, work.in_place, work.mutated), (2, 1, false, 0));
                        assert_eq!((work.blocks, work.threads), (if tree { owners } else { owners.div_ceil(128) }, 128));
                        assert_eq!(Kernel::FloatExtrema.entry(&p), extrema::symbol(&p).unwrap());
                        let start = 0x100000u64;
                        let end = start + g.input_words * 4;
                        assert!(work.validate_aliases(&[start + 4, end + 4]).is_ok());
                        assert!(work.validate_aliases(&[start, end]).is_ok());
                        for dst in [start, end - 4, start - g.output_words * 4 + 4] {
                            assert!(work.validate_aliases(&[start, dst]).is_err());
                        }
                        let per_owner = if output.width() == Width::Eight { 1 } else { 4 / output.width().bytes() };
                        let mut bytes_owned = 0;
                        for owner in 0..owners {
                            bytes_owned += u64::from((count - owner * per_owner).min(per_owner)) * u64::from(output.width().bytes());
                        }
                        assert_eq!(bytes_owned, u64::from(count) * u64::from(output.width().bytes()));
                        assert!(g.output_words * 4 - bytes_owned < 4);
                    }
                }
            }
        }
    }
}

#[test]
fn positive_length_precedes_empty_output_and_no_read_uses_one_dummy_word() {
    for input in FORMATS {
        for op in Operation::ALL {
            let output = if op.index() { Format::U64 } else { input };
            for dims in [[2, 0], [0, 0]] {
                let p = packet(op, input, output, &dims, 1);
                assert!(extrema::geometry(&p).is_err());
            }
            let mut p = packet(op, input, output, &[0, 3], 1);
            p[4] = 7;
            p[5] = 7;
            p[16..18].fill(i32::MIN as u32);
            let g = extrema::geometry(&p).unwrap();
            assert_eq!((g.input_words, g.output_words, g.owners), (1, 1, 0));
            assert!(extrema::validate(&p).is_err());
            assert!(extrema::work_items(&p).is_err());
            assert!(extrema::symbol(&p).is_err());
            assert!(Kernel::FloatExtrema.work(&p).is_err());
            p[4] = 8;
            assert!(extrema::geometry(&p).is_err());
        }
        let p = packet(Operation::Min, input, input, &[0], ALL_AXES);
        assert!(extrema::geometry(&p).is_err());
    }
}

#[test]
fn signed_views_and_symbolic_wide_limits_keep_full_declared_bounds() {
    for format in FORMATS {
        let mut p = packet(Operation::Max, format, format, &[2, 3], 1);
        p[4] = 5;
        p[16..18].copy_from_slice(&[(-3i32) as u32, (-1i32) as u32]);
        assert_eq!(extrema::validate(&p).unwrap(), [format.width().words(6), format.width().words(2)]);
        p[4] = 4;
        assert!(extrema::validate(&p).is_err());
        p[4] = 0;
        p[5] = 1;
        p[16..18].fill(0);
        assert_eq!(extrema::validate(&p).unwrap()[0], format.width().words(1));
        // Overlapping read-only windows and a nonzero offset remain legal.
        p[4] = 2;
        p[5] = 6;
        p[16..18].fill(1);
        assert!(extrema::validate(&p).is_ok());
        let mut p = packet(Operation::ArgMax, format, Format::I8, &[2; 8], 7);
        p[4] = 255;
        for stride in &mut p[16..24] {
            *stride = (-(*stride as i32)) as u32;
        }
        assert_eq!(extrema::geometry(&p).unwrap().output_count, 128);
        assert!(extrema::validate(&p).is_ok());
    }
    let mut p = packet(Operation::Min, Format::F64, Format::F64, &[1], ALL_AXES);
    p[5] = i32::MAX as u32;
    p[4] = p[5] - 1;
    assert_eq!(extrema::validate(&p).unwrap(), [4_294_967_294, 2]);
    let p = packet(Operation::Min, Format::F16, Format::F16, &[i32::MAX as u32], 0);
    assert_eq!(extrema::work_items(&p).unwrap(), 128);
    let mut p = packet(Operation::ArgMax, Format::BF16, Format::U64, &[1], 0);
    p[0] = i32::MAX as u32;
    p[8] = i32::MAX as u32;
    p[9] = 1;
    p[1] = 2;
    p[3] = 1;
    p[5] = 1;
    p[16..18].fill(0);
    assert_eq!(extrema::geometry(&p).unwrap().output_words, 4_294_967_294);
    assert_eq!(Kernel::FloatExtrema.work(&p).unwrap().blocks, (i32::MAX as u32).div_ceil(128));
}

#[test]
fn independent_zero_aware_products_do_not_divide_by_reduction_length() {
    let mut p = packet(Operation::Min, Format::F64, Format::F64, &[0, 1, 1], 1);
    p[9..11].fill(i32::MAX as u32);
    p[6] = i32::MAX as u32;
    let g = extrema::geometry(&p).unwrap();
    assert_eq!((g.output_count, g.input_words, g.output_words), (0, 1, 1));
    p[3] = 0;
    p[6] = 0;
    assert!(extrema::geometry(&p).is_err());
    p[3] = ALL_AXES;
    p[0] = 1;
    assert!(extrema::geometry(&p).is_err());
    let mut p = packet(Operation::Min, Format::F16, Format::F16, &[1], 0);
    p[1] = 8;
    p[8..16].fill(65536);
    assert!(extrema::geometry(&p).is_err());
    let mut p = packet(Operation::Max, Format::F64, Format::F64, &[1, 1, 0], 0);
    p[8..10].fill(i32::MAX as u32);
    p[6] = i32::MAX as u32;
    assert_eq!(extrema::geometry(&p).unwrap().output_count, 0);
}

#[test]
fn canonical_packet_scalar_promotion_and_all_axis_rules_are_checked() {
    let p = packet(Operation::Min, Format::F64, Format::F64, &[3, 129], 1);
    for (field, value) in [
        (0, 2),
        (1, 0),
        (1, 9),
        (3, 2),
        (4, u32::MAX),
        (5, 0),
        (5, i32::MAX as u32 + 1),
        (6, 128),
        (6, 0),
        (7, 1),
        (10, 2),
        (18, 1),
        (8, u32::MAX),
    ] {
        let mut invalid = p;
        invalid[field] = value;
        assert!(extrema::geometry(&invalid).is_err(), "field {field}: {value}");
    }
    assert!(extrema::geometry(&p[..23]).is_err());
    assert!(extrema::geometry(&[0; 25]).is_err());
    for op in Operation::ALL {
        let output = if op.index() { Format::I32 } else { Format::F32 };
        let p = packet(op, Format::F32, output, &[2, 129], ALL_AXES);
        assert_eq!(extrema::validate(&p).is_ok(), !op.index());
    }
    // A logical scalar is promoted by the host to a physical [1] packet.
    let p = packet(Operation::Min, Format::F64, Format::F64, &[1], ALL_AXES);
    assert_eq!(extrema::validate(&p).unwrap(), [2, 2]);
    let mut raw_rank_zero = p;
    raw_rank_zero[1] = 0;
    assert!(extrema::geometry(&raw_rank_zero).is_err());
}
