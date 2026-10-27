use d3os_nvidia::compute::{
    Kernel,
    cast::Format,
    indexed_accumulation::{self, FORMATS, Operation},
    storage::Width,
};

fn parameters(shape: &[u32], axis: usize, updates: u32, format: Format, operation: Operation) -> [u32; 32] {
    let mut p = [0; 32];
    p[0] = indexed_accumulation::control(shape.len() as u32, format, operation).unwrap();
    p[1] = axis as u32;
    p[3] = shape.iter().product();
    p[7] = shape[axis];
    p[8..16].fill(1);
    p[8..8 + shape.len()].copy_from_slice(shape);
    p[8 + axis] = updates;
    p[2] = p[8..16].iter().product();
    p[5] = p[2];
    let mut output_stride = 1;
    let mut index_stride = 1;
    for a in (0..shape.len()).rev() {
        p[16 + a] = output_stride;
        p[24 + a] = index_stride;
        output_stride *= shape[a];
        index_stride *= p[8 + a];
    }
    p
}

#[test]
fn controls_preserve_numeric_tags_and_reject_other_formats_and_reserved_fields() {
    for rank in 1..=8 {
        for format in Format::ALL {
            for operation in Operation::ALL {
                let encoded = indexed_accumulation::control(rank, format, operation);
                assert_eq!(encoded.is_ok(), FORMATS.contains(&format));
                if let Ok(encoded) = encoded {
                    assert_eq!(encoded, rank | ((format as u32) << 8) | ((operation as u32) << 16));
                }
            }
        }
    }
    for rank in [0, 9, 256, u32::MAX] {
        assert!(indexed_accumulation::control(rank, Format::F64, Operation::Add).is_err());
    }
    for raw in 0..=256 {
        assert_eq!(Operation::from_raw(raw).is_ok(), raw < 4);
    }
    let p = parameters(&[3, 5], 1, 7, Format::F16, Operation::Add);
    for raw_format in 0..=255 {
        let mut candidate = p;
        candidate[0] = 2 | (raw_format << 8);
        assert_eq!(indexed_accumulation::geometry(&candidate).is_ok(), raw_format < 13);
    }
    for bit in 24..32 {
        let mut candidate = p;
        candidate[0] |= 1 << bit;
        assert!(indexed_accumulation::geometry(&candidate).is_err());
    }
    let mut candidate = p;
    candidate[0] |= 4 << 16;
    assert!(indexed_accumulation::geometry(&candidate).is_err());
}

#[test]
fn dense_geometry_uses_physical_words_and_exclusive_wide_pairs() {
    for format in FORMATS {
        for operation in Operation::ALL {
            for rank in 1..=8 {
                let mut shape = vec![1; rank];
                shape[rank - 1] = 5;
                for axis in 0..rank {
                    let p = parameters(&shape, axis, 7, format, operation);
                    let geometry = indexed_accumulation::geometry(&p).unwrap();
                    let expected = [format.width().words(p[2]), u64::from(p[5]), 1, format.width().words(p[3])];
                    assert_eq!(indexed_accumulation::validate(&p).unwrap(), expected);
                    assert_eq!(
                        [geometry.values_words, geometry.indices_words, geometry.status_words, geometry.output_words],
                        expected
                    );
                    let owners = if format.width() == Width::Eight { p[3] } else { expected[3] as u32 };
                    assert_eq!(geometry.owners, owners);
                    assert_eq!(indexed_accumulation::work_items(&p).unwrap(), owners);
                }
            }
        }
    }
}

#[test]
fn exact_destination_layout_rejects_offsets_padding_and_overlapping_rows() {
    for axis in 0..3 {
        let p = parameters(&[3, 5, 7], axis, 137, Format::I64, Operation::Mul);
        assert!(indexed_accumulation::geometry(&p).is_ok());
        for (field, value) in [(3, p[3] - 1), (3, p[3] + 1), (4, 1), (16, 34), (17, 6), (18, 0), (18, (-1i32) as u32)] {
            let mut candidate = p;
            candidate[field] = value;
            assert!(indexed_accumulation::geometry(&candidate).is_err());
        }
    }
    let mut singleton = parameters(&[1, 5], 1, 7, Format::U8, Operation::Max);
    singleton[16] = 0;
    assert!(indexed_accumulation::geometry(&singleton).is_err());
}

#[test]
fn signed_index_span_agrees_with_independent_address_enumeration() {
    let p = parameters(&[2, 5], 1, 3, Format::F64, Operation::Min);
    for extent in 1..=12 {
        for offset in 0..=12 {
            for outer in -4i32..=4 {
                for inner in -4i32..=4 {
                    let mut candidate = p;
                    candidate[5] = extent;
                    candidate[6] = offset;
                    candidate[24] = outer as u32;
                    candidate[25] = inner as u32;
                    let reachable = (0..2).all(|row| {
                        (0..3).all(|column| {
                            let address = i64::from(offset) + i64::from(row * outer + column * inner);
                            (0..i64::from(extent)).contains(&address)
                        })
                    });
                    assert_eq!(indexed_accumulation::geometry(&candidate).is_ok(), reachable);
                }
            }
        }
    }
    for stride in [i32::MIN, i32::MAX] {
        let mut candidate = p;
        candidate[24] = stride as u32;
        assert!(indexed_accumulation::geometry(&candidate).is_err());
    }
}

#[test]
fn malformed_and_empty_packets_never_produce_work() {
    let p = parameters(&[3, 5], 1, 7, Format::BF16, Operation::Add);
    for (field, value) in [
        (0, 0),
        (0, 9),
        (1, 2),
        (2, 0),
        (2, p[2] - 1),
        (2, 1 << 31),
        (3, 0),
        (3, 1 << 31),
        (5, 0),
        (5, 1 << 31),
        (6, p[5]),
        (6, u32::MAX),
        (7, 0),
        (7, 1 << 31),
        (8, 0),
        (8, 1 << 31),
        (10, 2),
        (18, 1),
        (26, 1),
    ] {
        let mut candidate = p;
        candidate[field] = value;
        assert!(indexed_accumulation::geometry(&candidate).is_err());
        assert!(indexed_accumulation::validate(&candidate).is_err());
        assert!(indexed_accumulation::work_items(&candidate).is_err());
        assert!(Kernel::IndexedAccumulation.work(&candidate).is_err());
    }
    for length in 0..32 {
        assert!(indexed_accumulation::geometry(&p[..length]).is_err());
    }
    let mut long = p.to_vec();
    long.push(0);
    assert!(indexed_accumulation::geometry(&long).is_err());
    let mut overflow = p;
    overflow[8] = i32::MAX as u32;
    overflow[9] = i32::MAX as u32;
    assert!(indexed_accumulation::geometry(&overflow).is_err());
}

#[test]
fn registered_work_preserves_initialized_output_and_rounds_physical_owners() {
    assert!(!Kernel::IndexedAccumulation.allocates_output());
    for format in FORMATS {
        let elements_per_owner = if format.width() == Width::Eight { 1 } else { 4 / format.width().bytes() };
        for owners in [1u32, 127, 128, 129] {
            for tail in 1..=elements_per_owner {
                let elements = (owners - 1) * elements_per_owner + tail;
                let p = parameters(&[elements], 0, 3, format, Operation::Add);
                let work = Kernel::IndexedAccumulation.work(&p).unwrap();
                assert_eq!((work.blocks, work.threads), (owners.div_ceil(128), 128));
                assert_eq!((work.pointers, work.output, work.mutated, work.in_place), (4, 3, 1 << 2, false));
                assert_eq!(work.extents[..4], indexed_accumulation::validate(&p).unwrap());
            }
        }
    }
}

#[test]
fn registered_alias_rules_allow_readers_but_exclude_every_writer_overlap() {
    for format in FORMATS {
        let p = parameters(&[3, 5, 7], 1, 137, format, Operation::Max);
        let work = Kernel::IndexedAccumulation.work(&p).unwrap();
        let bases = [0x10000u64, 0x20000, 0x30000, 0x40000];
        assert!(work.validate_aliases(&bases).is_ok());
        let mut readers = bases;
        readers[1] = readers[0] + 4;
        assert!(work.validate_aliases(&readers).is_ok());
        for writer in [2, 3] {
            for other in (0..4).filter(|&other| other != writer) {
                let mut candidate = bases;
                candidate[writer] = bases[other] + 4 * work.extents[other];
                assert!(work.validate_aliases(&candidate).is_ok());
                candidate[writer] -= 4;
                assert!(work.validate_aliases(&candidate).is_err());
            }
        }
        for pointer in 0..4 {
            let mut overflow = bases;
            overflow[pointer] = u64::MAX - 3;
            assert!(work.validate_aliases(&overflow).is_err());
        }
    }
}

#[test]
fn wide_logical_maximum_retains_the_final_unsigned_high_word() {
    for format in [Format::F64, Format::I64, Format::U64] {
        let mut p = parameters(&[i32::MAX as u32], 0, i32::MAX as u32, format, Operation::Add);
        p[5] = 1;
        p[24] = 0;
        let geometry = indexed_accumulation::geometry(&p).unwrap();
        assert_eq!(
            (geometry.values_words, geometry.output_words, geometry.owners),
            (0xfffffffe, 0xfffffffe, 0x7fffffff)
        );
        let work = Kernel::IndexedAccumulation.work(&p).unwrap();
        assert_eq!(work.extents[..4], [0xfffffffe, 1, 1, 0xfffffffe]);
        let bases = [0x10000000000u64, 0x20000000000, 0x30000000000, 0x40000000000];
        let mut high_word_overlap = bases;
        high_word_overlap[1] = bases[3] + 4 * (geometry.output_words - 1);
        assert!(work.validate_aliases(&high_word_overlap).is_err());
        high_word_overlap[1] += 4;
        assert!(work.validate_aliases(&high_word_overlap).is_ok());
    }
}
