use d3os_nvidia::compute::{
    Kernel,
    cast::Format,
    float_reduction::{self as numeric, FORMATS, Operation, SERIAL_POLICY},
    reduction::Reduction,
    storage::Width,
};

fn packet(operation: Operation, format: Format, shape: &[u32], axis: usize) -> [u32; 24] {
    let mut p = [0; 24];
    p[1] = shape.len() as u32;
    p[2] = numeric::control(operation, format).unwrap();
    p[3] = axis as u32;
    p[6] = shape[axis];
    p[7] = SERIAL_POLICY;
    p[8..16].fill(1);
    p[8..8 + shape.len()].copy_from_slice(shape);
    let mut stride = 1;
    for index in (0..shape.len()).rev() {
        p[16 + index] = stride;
        stride *= shape[index];
    }
    p[5] = stride;
    p[0] = shape.iter().enumerate().filter_map(|(i, &n)| (i != axis).then_some(n)).product();
    p
}

#[test]
fn registered_packed_writers_cover_each_output_once_including_read_free_identities() {
    assert_eq!(Kernel::from_raw(Kernel::FloatReduction as u32).unwrap(), Kernel::FloatReduction);
    for format in FORMATS {
        for operation in Operation::ALL {
            for count in [0, 1, 2, 3, 31, 127, 128, 129, 257] {
                for length in [0, 1, 3, 31, 128, 129] {
                    let p = packet(operation, format, &[count, length], 1);
                    let geometry = numeric::geometry(&p).unwrap();
                    let input = if count == 0 || length == 0 { 1 } else { format.width().words(count * length) };
                    let output = format.width().words(count).max(1);
                    let owners = if format.width() == Width::Eight {
                        count
                    } else {
                        format.width().words(count) as u32
                    };
                    assert_eq!((geometry.input_words, geometry.output_words, geometry.owners), (input, output, owners));
                    assert_eq!((geometry.output_count, geometry.reduction_length), (count, length));
                    if count == 0 {
                        assert!(numeric::validate(&p).is_err());
                        assert!(Kernel::FloatReduction.work(&p).is_err());
                        continue;
                    }
                    let work = Kernel::FloatReduction.work(&p).unwrap();
                    assert_eq!(numeric::work_items(&p).unwrap(), owners);
                    assert_eq!((work.blocks, work.threads), (owners.div_ceil(128), 128));
                    assert_eq!(work.extents[..2], [input, output]);
                    assert_eq!((work.pointers, work.output, work.mutated, work.in_place), (2, 1, 0, false));
                    assert!(Kernel::FloatReduction.allocates_output());
                    assert_eq!(Kernel::FloatReduction.entry(&p), "burn_float_reduce");
                    let slots = if format.width() == Width::Eight { 1 } else { 4 / format.width().bytes() };
                    let mut visits = vec![0; count as usize];
                    for owner in 0..owners {
                        for slot in 0..slots {
                            if let Some(visit) = visits.get_mut((owner * slots + slot) as usize) {
                                *visit += 1;
                            }
                        }
                    }
                    assert!(visits.into_iter().all(|visits| visits == 1));
                    assert!(output * 4 - u64::from(count) * u64::from(format.width().bytes()) < 4);
                }
            }
        }
    }
}

#[test]
fn arithmetic_control_is_distinct_from_extrema_and_legacy_formats() {
    for operation in Operation::ALL {
        assert_eq!(Operation::from_raw(operation as u32).unwrap(), operation);
        assert_eq!(Operation::from_reduction(Reduction::from_raw(operation as u32).unwrap()).unwrap(), operation);
        for format in FORMATS {
            let control = numeric::control(operation, format).unwrap();
            let accumulator = if format == Format::F64 { Format::F64 } else { Format::F32 };
            assert_eq!(
                control,
                operation as u32 | (format as u32) << 8 | (format as u32) << 16 | (accumulator as u32) << 24
            );
            for shift in [0, 8, 16, 24] {
                let mut p = packet(operation, format, &[2, 3], 1);
                p[2] ^= 0xff << shift;
                assert!(numeric::geometry(&p).is_err());
            }
        }
        for format in [
            Format::F32,
            Format::I64,
            Format::I32,
            Format::I16,
            Format::I8,
            Format::U64,
            Format::U32,
            Format::U16,
            Format::U8,
            Format::Bool8,
            Format::Bool32,
        ] {
            assert!(numeric::control(operation, format).is_err());
        }
    }
    for operation in [
        Reduction::Min,
        Reduction::Max,
        Reduction::ArgMin,
        Reduction::ArgMax,
        Reduction::IntSum,
        Reduction::IntMean,
        Reduction::IntProduct,
    ] {
        assert!(Operation::from_reduction(operation).is_err());
    }
}

#[test]
fn signed_overlapping_and_broadcast_views_validate_their_actual_storage() {
    for format in FORMATS {
        for (shape, strides, offset, extent) in [([2, 3], [-3, 1], 3, 6), ([2, 3], [1, 2], 0, 6), ([3, 3], [0, 1], 0, 3), ([2, 3], [1, 1], 1, 5)] {
            let mut p = packet(Operation::Sum, format, &shape, 1);
            p[4] = offset;
            p[5] = extent;
            p[16] = strides[0] as u32;
            p[17] = strides[1] as u32;
            for row in 0..shape[0] {
                for column in 0..shape[1] {
                    let address = i64::from(offset) + i64::from(row) * i64::from(strides[0]) + i64::from(column) * i64::from(strides[1]);
                    assert!((0..i64::from(extent)).contains(&address));
                }
            }
            assert_eq!(numeric::validate(&p).unwrap()[0], format.width().words(extent));
            let mut invalid = p;
            invalid[4] = extent;
            assert!(numeric::geometry(&invalid).is_err());
        }
    }
    let mut rank_eight = packet(Operation::Product, Format::F16, &[1, 1, 1, 1, 1, 1, 3, 2], 7);
    rank_eight[16..22].fill(i32::MIN as u32);
    assert!(numeric::geometry(&rank_eight).is_ok());
}

#[test]
fn empty_packets_validate_metadata_but_never_require_a_wide_dummy_buffer() {
    for format in FORMATS {
        for shape in [[0, 3], [3, 0], [0, 0]] {
            let mut p = packet(Operation::Mean, format, &shape, 1);
            p[4] = 7;
            p[5] = 7;
            p[16..18].fill(i32::MIN as u32);
            assert_eq!(numeric::geometry(&p).unwrap().input_words, 1);
            for (index, value) in [(1, 0), (1, 9), (3, 2), (3, u32::MAX), (4, 8), (7, 0), (7, 2), (10, 2), (18, 1)] {
                let mut invalid = p;
                invalid[index] = value;
                assert!(numeric::geometry(&invalid).is_err(), "index {index}={value}");
            }
        }
    }
    let p = packet(Operation::Mean, Format::F64, &[3, 0], 1);
    assert_eq!(numeric::validate(&p).unwrap(), [1, 6]);
    assert_eq!(numeric::work_items(&p).unwrap(), 3);
    assert!(numeric::geometry(&p[..23]).is_err());
}

#[test]
fn full_u32_word_indices_and_zero_aware_products_do_not_narrow_through_i32() {
    let limit = i32::MAX as u32;
    let p = packet(Operation::Sum, Format::F64, &[limit], 0);
    assert_eq!(numeric::validate(&p).unwrap(), [u64::from(limit) * 2, 2]);
    assert!(numeric::validate(&p).unwrap()[0] > i32::MAX as u64);
    let mut invalid = p;
    invalid[8] = limit + 1;
    invalid[6] = limit + 1;
    assert!(numeric::geometry(&invalid).is_err());
    let mut empty = packet(Operation::Sum, Format::F16, &[0, 1, 1], 2);
    empty[9] = limit;
    assert_eq!(numeric::geometry(&empty).unwrap().output_count, 0);
    empty[8] = 2;
    assert!(numeric::geometry(&empty).is_err());
}

#[test]
fn physical_overlap_rejects_even_for_no_read_input_and_exact_output_bounds() {
    for format in FORMATS {
        for length in [0, 3, 129] {
            let p = packet(Operation::Product, format, &[3, length], 1);
            let work = Kernel::FloatReduction.work(&p).unwrap();
            let start = 0x10_0000;
            let end = start + work.extents[0] * 4;
            assert!(work.validate_aliases(&[start, end]).is_ok());
            assert!(work.validate_aliases(&[start, start - work.extents[1] * 4]).is_ok());
            for output in [start, end - 4, start - work.extents[1] * 4 + 4] {
                assert!(work.validate_aliases(&[start, output]).is_err());
            }
        }
    }
}
