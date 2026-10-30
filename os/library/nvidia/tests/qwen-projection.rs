use d3os_nvidia::compute::Kernel;

#[test]
fn block_scaled_prefill_matches_packer_storage_and_rejects_unmeasured_cells() {
    let pack = Kernel::ProjectQuantized.work(&[5, 5120, 17408, 8, 3, 3]).unwrap();
    let project = Kernel::ProjectQuantized.work(&[5, 5120, 17408, 11, 8, 3]).unwrap();
    let values = 5 * 5120;
    let words = 3 * values / 4 + values / 256;
    assert_eq!(pack.extents[..2], [values, words]);
    assert_eq!(project.extents[..3], [words, 17408 * 20 * 28, 5 * 17408]);
    assert_eq!((pack.blocks, project.blocks, pack.threads, project.threads), (100, 544, 128, 128));
    assert_eq!((pack.pointers, pack.output, project.pointers, project.output), (2, 1, 3, 2));
    pack.validate_aliases(&[0x100000, 0x200000]).unwrap();
    project.validate_aliases(&[0x200000, 0x10000000, 0x40000000]).unwrap();
    assert!(pack.validate_aliases(&[0x100000, 0x100000 + 4 * (values - 1)]).is_err());
    for source in 0..2 {
        let mut addresses = [0x200000, 0x10000000, 0x40000000];
        addresses[2] = addresses[source] + 4 * (project.extents[source] - 1);
        assert!(project.validate_aliases(&addresses).is_err());
    }
    for parameters in [[5, 5120, 17408, 8, 3, 3], [5, 5120, 17408, 11, 8, 3]] {
        for field in [0, 1, 2, 5] {
            for value in [0, 1, 4, 6, 32, u32::MAX] {
                if value == parameters[field] {
                    continue;
                }
                let mut changed = parameters;
                changed[field] = value;
                assert!(Kernel::ProjectQuantized.work(&changed).is_err());
            }
        }
    }
}

#[test]
fn paired_tail_rows_cover_odd_rows_without_changing_storage_or_decode() {
    let pair = Kernel::ProjectQuantized.entry(&[5, 4096, 12288, 8, 2]);
    let single = Kernel::ProjectQuantized.entry(&[1, 4096, 4096, 8, 2]);
    assert_ne!(pair, single);
    for rows in 2u32..=8 {
        for (inner, columns) in [(4096u32, 4096u32), (4096, 8192), (4096, 12288), (12288, 4096)] {
            let parameters = [rows, inner, columns, 8, 2];
            let work = Kernel::ProjectQuantized.work(&parameters).unwrap();
            assert_eq!(Kernel::ProjectQuantized.entry(&parameters), pair);
            assert_eq!(work.blocks, rows.div_ceil(2) * (columns / 4));
            assert_eq!(
                work.extents[..3],
                [
                    u64::from(rows * inner),
                    u64::from(inner / 32) * u64::from(columns) * 34 / 4,
                    u64::from(rows * columns)
                ]
            );
            assert_eq!((work.threads, work.pointers, work.output, work.in_place), (128, 3, 2, false));
            let pointers = [0x100000, 0x10000000, 0x60000000];
            work.validate_aliases(&pointers).unwrap();
            for source in 0..2 {
                let mut overlap = pointers;
                overlap[2] = overlap[source] + (work.extents[source] - 1) * 4;
                assert!(work.validate_aliases(&overlap).is_err());
            }
        }
    }
    for [rows, inner, columns] in [[1, 4096, 4096], [5, 4128, 12288], [5, 4096, 12256], [5, 4096, 248320]] {
        let parameters = [rows, inner, columns, 8, 2];
        assert_eq!(Kernel::ProjectQuantized.entry(&parameters), single);
        assert_eq!(Kernel::ProjectQuantized.work(&parameters).unwrap().blocks, rows * (columns / 4));
    }
    for rows in [0, 9] {
        assert!(Kernel::ProjectQuantized.work(&[rows, 4096, 12288, 8, 2]).is_err());
    }
}

#[test]
fn vocabulary_head_preserves_complete_projection_storage_and_neighboring_routes() {
    let parameters = [1, 4096, 248320, 8, 2];
    let work = Kernel::ProjectQuantized.work(&parameters).unwrap();
    assert_eq!(work.extents[..3], [4096, 1080688640 / 4, 248320]);
    assert_eq!((work.blocks, work.threads, work.pointers, work.output), (248320 / 4, 128, 3, 2));
    assert_eq!(
        Kernel::ProjectQuantized.entry(&parameters),
        Kernel::ProjectQuantized.entry(&[1, 4096, 12288, 8, 2])
    );
    let ordinary = Kernel::ProjectQuantized.entry(&[1, 4096, 4096, 8, 2]);
    for parameters in [
        [1, 4096, 248288, 8, 2],
        [1, 4096, 248352, 8, 2],
        [1, 4128, 248320, 8, 2],
        [2, 4096, 248320, 8, 2],
    ] {
        assert_eq!(Kernel::ProjectQuantized.entry(&parameters), ordinary);
    }
    let pointers = [0x100000, 0x10000000, 0x60000000];
    work.validate_aliases(&pointers).unwrap();
    for source in 0..2 {
        let mut aliases = pointers;
        aliases[2] = aliases[source] + 4 * (work.extents[source] - 1);
        assert!(work.validate_aliases(&aliases).is_err());
    }
}

#[test]
fn narrow_decode_uses_complete_rows_without_changing_storage() {
    let wide_entry = Kernel::ProjectQuantized.entry(&[1, 4096, 129, 8]);
    for (inner, outputs, columns) in [(4096u32, 32u32, 1u32), (4096, 128, 1), (1024, 1, 1), (992, 32, 4), (4096, 129, 4)] {
        let parameters = [1, inner, outputs, 8];
        let work = Kernel::ProjectQuantized.work(&parameters).unwrap();
        assert_eq!(work.blocks, outputs.div_ceil(columns));
        assert_eq!(
            &work.extents[..3],
            &[u64::from(inner), u64::from(inner) * u64::from(outputs) * 34 / 128, u64::from(outputs)]
        );
        assert_eq!((work.threads, work.pointers, work.output), (128, 3, 2));
        assert_eq!(Kernel::ProjectQuantized.entry(&parameters) != wide_entry, columns == 1);
        work.validate_aliases(&[0x100000, 0x200000, 0x300000]).unwrap();
        assert!(work.validate_aliases(&[0x100000, 0x200000, 0x100000]).is_err());
    }
    assert_eq!(Kernel::ProjectQuantized.work(&[2, 4096, 32, 8]).unwrap().blocks, 64);
    assert_eq!(Kernel::ProjectQuantized.work(&[1, 4096, 32, 12]).unwrap().blocks, 8);
}

#[test]
fn narrow_dispatch_rejects_incomplete_or_overflowing_projection_packets() {
    for parameters in [
        vec![1, 31, 32, 8],
        vec![0, 4096, 32, 8],
        vec![1, 4096, 0, 8],
        vec![1, 4096, 32, 8, 1],
        vec![1, 1 << 30, 4, 8],
    ] {
        assert!(Kernel::ProjectQuantized.work(&parameters).is_err());
    }
}

#[test]
fn signed_decode_preserves_packed_extents_and_small_projection_routes() {
    let ordinary = Kernel::ProjectQuantized.entry(&[1, 992, 256, 8]);
    let signed = Kernel::ProjectQuantized.entry(&[1, 4096, 4096, 8]);
    assert_ne!(signed, ordinary);
    for (inner, outputs) in [(1024u32, 256u32), (1056, 257), (4096, 4096), (5120, 5120), (12288, 4096)] {
        let parameters = [1, inner, outputs, 8];
        let work = Kernel::ProjectQuantized.work(&parameters).unwrap();
        let weight_words = (u64::from(inner / 32) * u64::from(outputs) * 34).div_ceil(4);
        assert_eq!(Kernel::ProjectQuantized.entry(&parameters), signed);
        assert_eq!(work.blocks, outputs.div_ceil(4));
        assert_eq!(&work.extents[..3], &[u64::from(inner), weight_words, u64::from(outputs)]);
        assert_eq!((work.threads, work.pointers, work.output, work.in_place), (128, 3, 2, false));
        work.validate_aliases(&[0x100000, 0x10000000, 0x40000000]).unwrap();
    }
    assert_eq!(Kernel::ProjectQuantized.entry(&[1, 4096, 255, 8]), ordinary);
    for parameters in [[2, 4096, 4096, 8], [17, 4096, 4096, 8], [1, 4096, 4096, 11], [1, 4096, 128, 8]] {
        assert_ne!(Kernel::ProjectQuantized.entry(&parameters), signed);
    }
}

#[test]
fn grouped_q8_checks_each_packed_matrix_and_combined_output_range() {
    for [inner, qkv, gate, heads] in [[1024u32, 256, 256, 1], [1056, 257, 259, 3], [4096, 8192, 4096, 32], [5120, 12288, 4096, 128]] {
        let work = Kernel::ProjectQuantized.work(&[inner, qkv, gate, heads, 8]).unwrap();
        let words = |rows| (u64::from(inner / 32) * u64::from(rows) * 34).div_ceil(4);
        let total = qkv + gate + 2 * heads;
        assert_eq!(
            work.extents[..6],
            [u64::from(inner), words(qkv), words(gate), words(heads), words(heads), u64::from(total)]
        );
        assert_eq!(
            (work.blocks, work.threads, work.pointers, work.output, work.mutated),
            (qkv.div_ceil(4) + gate.div_ceil(4) + 2 * heads, 128, 6, 5, 0)
        );
        let pointers = [0x100000, 0x10000000, 0x20000000, 0x30000000, 0x40000000, 0x50000000];
        work.validate_aliases(&pointers).unwrap();
        for source in 0..5 {
            let mut aliases = pointers;
            aliases[5] = aliases[source] + 4 * (work.extents[source] - 1);
            assert!(work.validate_aliases(&aliases).is_err());
        }
        assert_ne!(
            Kernel::ProjectQuantized.entry(&[inner, qkv, gate, heads, 8]),
            Kernel::ProjectQuantized.entry(&[1, inner, qkv, 8])
        );
        let ordinary = Kernel::ProjectQuantized.work(&[1, inner, qkv, 8]).unwrap();
        assert_eq!((ordinary.blocks, ordinary.pointers, ordinary.output), (qkv.div_ceil(4), 3, 2));
    }
    for parameters in [
        [992, 256, 256, 1, 8],
        [1055, 256, 256, 1, 8],
        [1024, 255, 256, 1, 8],
        [1024, 256, 255, 1, 8],
        [1024, 256, 256, 0, 8],
        [1024, 256, 256, 129, 8],
        [1024, 256, 256, 32, 9],
        [1024, u32::MAX, 256, 32, 8],
        [1024, 256, u32::MAX, 32, 8],
        [1 << 30, 256, 256, 32, 8],
    ] {
        assert!(Kernel::ProjectQuantized.work(&parameters).is_err());
    }
}

#[test]
fn component_projection_covers_byte_planes_scale_tail_and_each_source() {
    for [rows, inner, columns] in [[1u32, 32, 1], [17, 1056, 259], [31, 1024, 257], [32, 4096, 12288], [32, 12288, 4096]] {
        let elements = rows * inner;
        let scratch = u64::from(3 * elements / 4 + elements / 32);
        let packed = (u64::from(inner / 32) * u64::from(columns) * 34).div_ceil(4);
        let pack = Kernel::ProjectQuantized.work(&[rows, inner, columns, 8, 0, 3]).unwrap();
        let project = Kernel::ProjectQuantized.work(&[rows, inner, columns, 8, 1, 3]).unwrap();
        assert_eq!(&pack.extents[..2], &[u64::from(elements), scratch]);
        assert_eq!(&project.extents[..3], &[scratch, packed, u64::from(rows * columns)]);
        assert_eq!((pack.blocks, pack.pointers, pack.output), (rows.div_ceil(4) * (inner / 32), 2, 1));
        assert_eq!(
            (project.blocks, project.pointers, project.output),
            (rows.div_ceil(32) * columns.div_ceil(32), 3, 2)
        );
        assert_eq!((pack.threads, project.threads, pack.in_place, project.in_place), (128, 128, false, false));
        let pointers = [0x100000, 0x10000000, 0x40000000];
        pack.validate_aliases(&pointers[..2]).unwrap();
        assert!(pack.validate_aliases(&[pointers[0], pointers[0] + 4 * (pack.extents[0] - 1)]).is_err());
        project.validate_aliases(&pointers).unwrap();
        for source in 0..2 {
            let mut overlapping = pointers;
            overlapping[2] = pointers[source] + 4 * (project.extents[source] - 1);
            assert!(project.validate_aliases(&overlapping).is_err());
        }
        assert_ne!(
            Kernel::ProjectQuantized.entry(&[rows, inner, columns, 8, 0, 3]),
            Kernel::ProjectQuantized.entry(&[rows, inner, columns, 8, 1, 3])
        );
    }
}

#[test]
fn component_projection_rejects_partial_invalid_and_overflowing_packets() {
    for parameters in [
        vec![0, 4096, 12288, 8, 0, 3],
        vec![32, 0, 12288, 8, 0, 3],
        vec![32, 4095, 12288, 8, 1, 3],
        vec![32, 4096, 0, 8, 0, 3],
        vec![32, 4096, 12288, 9, 1, 3],
        vec![32, 4096, 12288, 8, 3, 3],
        vec![32, 4096, 12288, 8, 1, 2],
        vec![32, 4096, 12288, 8, 0, 4],
        vec![32, 4096, 12288, 8, 1],
        vec![32, 1 << 27, 1, 8, 0, 3],
        vec![1, 1 << 30, 1, 8, 0, 3],
        vec![32, 4096, u32::MAX, 8, 1, 3],
        vec![32, 4096, 12288, 8, 0, 3, 0],
    ] {
        assert!(Kernel::ProjectQuantized.work(&parameters).is_err());
    }
}

#[test]
fn lossless_column_layout_dispatches_complete_storage_and_preserves_alias_guards() {
    for [rows, inner, columns] in [[1u32, 32, 32], [5, 1056, 96], [32, 4096, 12288], [64, 12288, 4096]] {
        let phase = if rows <= 8 {
            vec![rows, inner, columns, 8, 2]
        } else {
            vec![rows, inner, columns, 8, 2, 3]
        };
        let work = Kernel::ProjectQuantized.work(&phase).unwrap();
        let input = if rows <= 8 { rows * inner } else { rows * inner * 3 / 4 + rows * inner / 32 };
        let blocks = if rows <= 8 { rows * columns / 4 } else { rows.div_ceil(32) * columns / 32 };
        assert_eq!(
            &work.extents[..3],
            &[
                u64::from(input),
                (u64::from(inner / 32) * u64::from(columns) * 34).div_ceil(4),
                u64::from(rows * columns)
            ]
        );
        assert_eq!((work.blocks, work.pointers, work.output), (blocks, 3, 2));
        let pointers = [0x100000, 0x10000000, 0x40000000];
        work.validate_aliases(&pointers).unwrap();
        for source in 0..2 {
            let mut alias = pointers;
            alias[2] = alias[source] + 4 * (work.extents[source] - 1);
            assert!(work.validate_aliases(&alias).is_err());
        }
        let embedding = Kernel::EmbeddingQuantized.work(&[columns, inner, rows, 8, 2]).unwrap();
        assert_eq!(&embedding.extents[..3], &[work.extents[1], u64::from(rows), u64::from(rows * inner)]);
        assert!(Kernel::ProjectQuantized.work(&[rows, inner, columns, 8, 0, 3]).is_ok());
    }
    for p in [
        vec![1, 32, 33, 8, 2],
        vec![9, 32, 32, 8, 2],
        vec![0, 32, 32, 8, 2],
        vec![32, 32, 33, 8, 2, 3],
        vec![32, 31, 32, 8, 2, 3],
        vec![32, 1 << 27, 32, 8, 2, 3],
    ] {
        assert!(Kernel::ProjectQuantized.work(&p).is_err());
    }
    for p in [[33, 32, 1, 8, 2], [32, 33, 1, 8, 2], [32, 32, 0, 8, 2], [32, u32::MAX - 31, 1, 8, 2]] {
        assert!(Kernel::EmbeddingQuantized.work(&p).is_err());
    }
}

#[test]
fn component_projection_checks_packed_byte_addresses_before_word_extents() {
    // The last complete 34-byte Q8 block in the shader's u32 address space.
    for phase in [0, 1] {
        assert!(Kernel::ProjectQuantized.work(&[1, 32, 126322567, 8, phase, 3]).is_ok());
        assert!(Kernel::ProjectQuantized.work(&[1, 32, 126322568, 8, phase, 3]).is_err());
        // Its element and dispatch counts fit u32, but packed byte indices do not.
        assert!(Kernel::ProjectQuantized.work(&[1, 32, 130000000, 8, phase, 3]).is_err());
    }
}

#[test]
fn batched_narrow_projections_cover_every_row_column_pair_without_repacking() {
    let selected = Kernel::ProjectQuantized.entry(&[32, 4096, 32, 8]);
    for [rows, inner, columns] in [[2, 1024, 1], [5, 4096, 32], [32, 4096, 32], [17, 1056, 33], [32, 4096, 128]] {
        let parameters = [rows, inner, columns, 8];
        let work = Kernel::ProjectQuantized.work(&parameters).unwrap();
        assert_eq!(Kernel::ProjectQuantized.entry(&parameters), selected);
        assert_eq!((work.blocks, work.threads, work.pointers, work.output), (rows * columns, 128, 3, 2));
        assert_eq!(
            &work.extents[..3],
            &[
                u64::from(rows * inner),
                (u64::from(inner / 32) * u64::from(columns) * 34).div_ceil(4),
                u64::from(rows * columns)
            ]
        );
        let mut seen = vec![false; (rows * columns) as usize];
        for block in 0..work.blocks {
            let row = block / columns;
            let column = block % columns;
            assert!(row < rows);
            assert!(row * inner + inner - 1 < rows * inner);
            let output = (row * columns + column) as usize;
            assert!(!seen[output]);
            seen[output] = true;
        }
        assert!(seen.into_iter().all(|value| value));
        let pointers = [0x100000, 0x10000000, 0x40000000];
        work.validate_aliases(&pointers).unwrap();
        for source in 0..2 {
            let mut overlap = pointers;
            overlap[2] = overlap[source] + (work.extents[source] - 1) * 4;
            assert!(work.validate_aliases(&overlap).is_err());
        }
    }
    for parameters in [[1, 4096, 32, 8], [33, 4096, 32, 8], [32, 992, 32, 8], [32, 4096, 129, 8], [32, 4096, 32, 12]] {
        assert_ne!(Kernel::ProjectQuantized.entry(&parameters), selected);
        assert!(Kernel::ProjectQuantized.work(&parameters).is_ok());
    }
}

#[test]
fn batched_narrow_rejects_incomplete_blocks_and_packed_byte_address_overflow() {
    let largest_inner = (u32::MAX / (128 * 34)) * 32;
    assert!(Kernel::ProjectQuantized.work(&[32, largest_inner, 128, 8]).is_ok());
    assert!(Kernel::ProjectQuantized.work(&[32, largest_inner + 32, 128, 8]).is_err());
    for parameters in [[0, 4096, 32, 8], [32, 4095, 32, 8], [32, 4096, 0, 8], [32, u32::MAX - 31, 32, 8]] {
        assert!(Kernel::ProjectQuantized.work(&parameters).is_err());
    }
}

#[test]
fn batched_wide_projections_cover_column_tails_and_independent_rows() {
    let selected = Kernel::ProjectQuantized.entry(&[5, 4096, 12288, 8]);
    for [rows, inner, columns] in [[2, 1024, 129], [5, 4096, 12288], [5, 12288, 4096], [8, 1056, 259]] {
        let parameters = [rows, inner, columns, 8];
        let work = Kernel::ProjectQuantized.work(&parameters).unwrap();
        assert_eq!(Kernel::ProjectQuantized.entry(&parameters), selected);
        assert_eq!((work.blocks, work.threads, work.pointers, work.output), (rows * columns.div_ceil(4), 128, 3, 2));
        assert_eq!(
            &work.extents[..3],
            &[
                u64::from(rows * inner),
                (u64::from(inner / 32) * u64::from(columns) * 34).div_ceil(4),
                u64::from(rows * columns)
            ]
        );
        let mut seen = vec![false; (rows * columns) as usize];
        for block in 0..work.blocks {
            let row = block % rows;
            let first = (block / rows) * 4;
            assert!(row * inner + inner - 1 < rows * inner);
            for column in first..(first + 4).min(columns) {
                let output = (row * columns + column) as usize;
                assert!(!seen[output]);
                seen[output] = true;
            }
        }
        assert!(seen.into_iter().all(|value| value));
        let pointers = [0x100000, 0x10000000, 0x40000000];
        work.validate_aliases(&pointers).unwrap();
        for source in 0..2 {
            let mut overlap = pointers;
            overlap[2] = overlap[source] + (work.extents[source] - 1) * 4;
            assert!(work.validate_aliases(&overlap).is_err());
        }
    }
    for parameters in [
        [1, 4096, 4096, 8],
        [9, 4096, 4096, 8],
        [32, 4096, 4096, 8],
        [5, 992, 4096, 8],
        [5, 4096, 128, 8],
        [5, 4096, 4096, 12],
    ] {
        assert_ne!(Kernel::ProjectQuantized.entry(&parameters), selected);
        assert!(Kernel::ProjectQuantized.work(&parameters).is_ok());
    }
}

#[test]
fn batched_wide_checks_packed_byte_indices_before_word_extents() {
    let columns = 129;
    let largest_inner = (u32::MAX / (columns * 34)) * 32;
    assert!(Kernel::ProjectQuantized.work(&[2, largest_inner, columns, 8]).is_ok());
    assert!(Kernel::ProjectQuantized.work(&[2, largest_inner + 32, columns, 8]).is_err());
    for parameters in [[0, 4096, 129, 8], [5, 4095, 129, 8], [5, 4096, 0, 8], [5, 4096, u32::MAX, 8]] {
        assert!(Kernel::ProjectQuantized.work(&parameters).is_err());
    }
}

#[test]
fn d4_activations_use_transposed_blocks_and_disjoint_buffers() {
    for [rows, inner, columns] in [[1u32, 512, 32], [5, 4096, 12288], [32, 12288, 4096], [65, 512, 288]] {
        let pack = Kernel::ProjectQuantized.work(&[rows, inner, columns, 8, 0, 1]).unwrap();
        let project = Kernel::ProjectQuantized.work(&[rows, inner, columns, 8, 2, 1]).unwrap();
        let row_major = Kernel::ProjectQuantized.work(&[rows, inner, columns, 8, 1, 1]).unwrap();
        assert_eq!(
            (row_major.blocks, row_major.extents, row_major.pointers, row_major.output),
            (project.blocks, project.extents, project.pointers, project.output)
        );
        assert_ne!(
            Kernel::ProjectQuantized.entry(&[rows, inner, columns, 8, 1, 1]),
            Kernel::ProjectQuantized.entry(&[rows, inner, columns, 8, 2, 1])
        );
        let blocks128 = u64::from(rows) * u64::from(inner / 128);
        assert_eq!(&pack.extents[..2], &[u64::from(rows) * u64::from(inner), blocks128 * 36]);
        assert_eq!(pack.blocks, rows.div_ceil(4) * (inner / 32));
        assert_eq!(
            &project.extents[..3],
            &[
                blocks128 * 36,
                (u64::from(inner / 32) * u64::from(columns) * 34).div_ceil(4),
                u64::from(rows) * u64::from(columns)
            ]
        );
        assert_eq!(project.blocks, rows.div_ceil(32) * (columns / 32));
        assert_eq!((pack.pointers, pack.output, project.pointers, project.output), (2, 1, 3, 2));
        pack.validate_aliases(&[0x100000, 0x10000000]).unwrap();
        project.validate_aliases(&[0x100000, 0x10000000, 0x40000000]).unwrap();
        assert!(pack.validate_aliases(&[0x100000, 0x100000]).is_err());
        assert!(project.validate_aliases(&[0x100000, 0x10000000, 0x100000]).is_err());
        assert_ne!(
            Kernel::ProjectQuantized.entry(&[rows, inner, columns, 8, 0, 1]),
            Kernel::ProjectQuantized.entry(&[rows, inner, columns, 8, 0, 3])
        );
    }
}

#[test]
fn d4_small_tiles_preserve_dispatch_geometry_and_row_boundaries() {
    let small = Kernel::ProjectQuantized.entry(&[5, 4096, 12288, 8, 2, 1]);
    let regular = Kernel::ProjectQuantized.entry(&[32, 4096, 12288, 8, 2, 1]);
    assert_ne!(small, regular);
    for rows in 1u32..=65 {
        for [inner, columns] in [[512, 32], [4096, 12288], [12288, 4096]] {
            let packet = [rows, inner, columns, 8, 2, 1];
            assert_eq!(Kernel::ProjectQuantized.entry(&packet), if (2..=8).contains(&rows) { small } else { regular });
            let work = Kernel::ProjectQuantized.work(&packet).unwrap();
            assert_eq!(work.blocks, rows.div_ceil(32) * columns.div_ceil(32));
            assert_eq!((work.threads, work.pointers, work.output), (128, 3, 2));
            let row_major = Kernel::ProjectQuantized.work(&[rows, inner, columns, 8, 1, 1]).unwrap();
            assert_eq!(work.extents, row_major.extents);
        }
    }
}

#[test]
fn d4_rejects_wrong_block_geometry_phase_and_address_overflow() {
    for packet in [
        [0, 4096, 12288, 8, 0, 1],
        [5, 0, 12288, 8, 0, 1],
        [5, 1056, 12288, 8, 0, 1],
        [5, 4096, 0, 8, 2, 1],
        [5, 4096, 12287, 8, 2, 1],
        [5, 4096, 12288, 8, 3, 1],
        [5, 4096, 12288, 8, 2, 2],
        [1 << 23, 512, 32, 8, 0, 1],
        [1, 1 << 29, 32, 8, 2, 1],
    ] {
        assert!(Kernel::ProjectQuantized.work(&packet).is_err());
    }
}

#[test]
fn tiled_grouped_decode_preserves_extents_and_rejects_partial_tiles() {
    for [k, q, z, h] in [[1024, 256, 256, 1], [4096, 8192, 4096, 32]] {
        let ordinary = Kernel::ProjectQuantized.work(&[k, q, z, h, 8]).unwrap();
        let tiled = Kernel::ProjectQuantized.work(&[k, q, z, h, 8, 2]).unwrap();
        assert_eq!(tiled.extents, ordinary.extents);
        assert_eq!(
            (tiled.blocks, tiled.pointers, tiled.output),
            (ordinary.blocks, ordinary.pointers, ordinary.output)
        );
        assert_ne!(
            Kernel::ProjectQuantized.entry(&[k, q, z, h, 8, 2]),
            Kernel::ProjectQuantized.entry(&[k, q, z, h, 8])
        );
        assert!(Kernel::ProjectQuantized.work(&[k, q - 1, z, h, 8, 2]).is_err());
        assert!(Kernel::ProjectQuantized.work(&[k, q, z - 1, h, 8, 2]).is_err());
        assert!(Kernel::ProjectQuantized.work(&[k, q, z, h, 8, 3]).is_err());
    }
}

#[test]
fn weight_reordering_checks_complete_blocks_and_rejects_aliases() {
    for [columns, inner] in [[32u32, 32], [64, 96], [96, 608], [12288, 4096], [4096, 12288]] {
        let work = Kernel::ProjectQuantized.work(&[columns, inner, 8, 3]).unwrap();
        let words = u64::from(columns) * u64::from(inner / 32) * 34 / 4;
        assert_eq!(&work.extents[..2], &[words, words]);
        assert_eq!((work.pointers, work.output, work.threads, work.in_place), (2, 1, 128, false));
        assert_eq!(u64::from(work.blocks), words.div_ceil(128));
        work.validate_aliases(&[0x100000, 0x10000000]).unwrap();
        assert!(work.validate_aliases(&[0x100000, 0x100000]).is_err());
        assert!(work.validate_aliases(&[0x100000, 0x100004]).is_err());
    }
    for parameters in [
        [0, 32, 8, 3],
        [32, 0, 8, 3],
        [31, 32, 8, 3],
        [32, 31, 8, 3],
        [32, 33, 8, 3],
        [1 << 30, 32, 8, 3],
    ] {
        assert!(Kernel::ProjectQuantized.work(&parameters).is_err());
    }
}

#[test]
fn component_vectors_bound_scratch_packed_words_grid_and_aliases() {
    for (format, block_bytes) in [(11, 110u64), (12, 144)] {
        for inner in [256u32, 512, 5120, 6144, 17408] {
            for columns in [1u32, 3, 4, 31, 32, 33, 17408] {
                let packet = [1, inner, columns, format, 1, 3];
                let work = Kernel::ProjectQuantized.work(&packet).unwrap();
                assert_eq!(
                    &work.extents[..3],
                    &[
                        u64::from(inner) * 3 / 4 + u64::from(inner) / 32,
                        (u64::from(inner / 256) * u64::from(columns) * block_bytes).div_ceil(4),
                        u64::from(columns),
                    ]
                );
                assert_eq!((work.blocks, work.threads, work.pointers, work.output), (columns.div_ceil(4), 128, 3, 2));
                work.validate_aliases(&[0x100000, 0x10000000, 0x40000000]).unwrap();
                for output in [0x100000, 0x100004, 0x10000000] {
                    assert!(work.validate_aliases(&[0x100000, 0x10000000, output]).is_err());
                }
                assert_ne!(Kernel::ProjectQuantized.entry(&packet), Kernel::ProjectQuantized.entry(&packet[..4]));
            }
        }
        for packet in [
            [0, 5120, 17408, format, 1, 3],
            [2, 5120, 17408, format, 1, 3],
            [1, 0, 17408, format, 1, 3],
            [1, 5119, 17408, format, 1, 3],
            [1, 5120, 0, format, 1, 3],
            [1, 5120, 17408, format, 0, 3],
            [1, 5120, 17408, format, 1, 2],
            [1, 5120, u32::MAX, format, 1, 3],
            [1, 1 << 31, 1, format, 1, 3],
        ] {
            assert!(Kernel::ProjectQuantized.work(&packet).is_err());
        }
    }
}

#[test]
fn q3_register_prefill_bounds_all_allocations_and_rejects_unmeasured_packets() {
    let packet = [32, 5120, 17408, 11, 2, 3];
    let work = Kernel::ProjectQuantized.work(&packet).unwrap();
    assert_eq!(&work.extents[..3], &[128000, 9574400, 557056]);
    assert_eq!((work.blocks, work.threads, work.pointers, work.output), (1088, 128, 3, 2));
    work.validate_aliases(&[0x100000, 0x10000000, 0x40000000]).unwrap();
    for output in [0x100000, 0x100004, 0x10000000, 0x10000004] {
        assert!(work.validate_aliases(&[0x100000, 0x10000000, output]).is_err());
    }
    for (field, value) in [(0, 31), (0, 33), (1, 5119), (1, 5376), (2, 17407), (2, 17440), (3, 12), (4, 0), (4, 1), (5, 1)] {
        let mut invalid = packet;
        invalid[field] = value;
        assert!(Kernel::ProjectQuantized.work(&invalid).is_err());
    }
}

#[test]
fn padded_q3_consumers_bound_words_tails_alignment_and_aliases() {
    for rows in [1u32, 2, 5, 15, 16, 17, 31, 32, 33] {
        for components in [0, 3] {
            let packet = [rows, 5120, 17408, 11, 3, components];
            if rows > 1 && components == 0 {
                assert!(Kernel::ProjectQuantized.work(&packet).is_err());
                continue;
            }
            let work = Kernel::ProjectQuantized.work(&packet).unwrap();
            let input = if components == 0 { rows * 5120 } else { rows * 4000 };
            assert_eq!(&work.extents[..3], &[u64::from(input), 9748480, u64::from(rows * 17408)]);
            assert_eq!(work.blocks, if rows == 1 { 4352 } else { rows.div_ceil(16) * 544 });
            assert_eq!((work.threads, work.pointers, work.output, work.in_place), (128, 3, 2, false));
            let addresses = [0x100000, 0x10000000, 0x40000000];
            work.validate_aliases(&addresses).unwrap();
            let alignment = if rows == 1 && components == 3 { 8 } else { 4 };
            assert_eq!(work.alignments[1], alignment);
            assert_eq!(work.validate_aliases(&[addresses[0], addresses[1] + 4, addresses[2]]).is_ok(), alignment == 4);
            for source in 0..2 {
                let mut overlap = addresses;
                overlap[2] = addresses[source] + (work.extents[source] - 1) * 4;
                assert!(work.validate_aliases(&overlap).is_err());
            }
        }
    }
    for rows in [1, 8, 33] {
        let work = Kernel::EmbeddingQuantized.work(&[17408, 5120, rows, 11, 3]).unwrap();
        assert_eq!(&work.extents[..3], &[9748480, u64::from(rows), u64::from(rows * 5120)]);
        assert_eq!(work.blocks, (rows * 5120).div_ceil(128));
        work.validate_aliases(&[0x10000000, 0x100000, 0x40000000]).unwrap();
        assert!(work.validate_aliases(&[0x10000000, 0x100000, 0x10000000]).is_err());
    }
    for (field, value) in [(0, 0), (0, u32::MAX), (1, 5119), (1, 5376), (2, 17407), (2, 17440), (3, 12), (4, 5), (5, 1)] {
        let mut packet = [1, 5120, 17408, 11, 3, 3];
        packet[field] = value;
        assert!(Kernel::ProjectQuantized.work(&packet).is_err());
    }
    for packet in [
        [17408, 5120, 0, 11, 3],
        [17408, 5120, u32::MAX, 11, 3],
        [17407, 5120, 8, 11, 3],
        [17408, 5119, 8, 11, 3],
        [17408, 5120, 8, 12, 3],
    ] {
        assert!(Kernel::EmbeddingQuantized.work(&packet).is_err());
    }
}

#[test]
fn aligned_q4_checks_full_storage_and_rejects_unmeasured_packets() {
    for inner in [6144u32, 17408] {
        let packet = [1, inner, 5120, 12, 4, 3];
        let work = Kernel::ProjectQuantized.work(&packet).unwrap();
        let ordinary = Kernel::ProjectQuantized.work(&[1, inner, 5120, 12, 1, 3]).unwrap();
        assert_eq!(work.extents, ordinary.extents);
        assert_eq!((work.blocks, work.threads, work.pointers, work.output, work.in_place), (1280, 128, 3, 2, false));
        assert_eq!(work.alignments[1], 8);
        let addresses = [0x100000, 0x10000000, 0x40000000];
        work.validate_aliases(&addresses).unwrap();
        assert!(work.validate_aliases(&[addresses[0], addresses[1] + 4, addresses[2]]).is_err());
        ordinary.validate_aliases(&[addresses[0], addresses[1] + 4, addresses[2]]).unwrap();
        for source in 0..2 {
            let mut overlap = addresses;
            overlap[2] = addresses[source] + (work.extents[source] - 1) * 4;
            assert!(work.validate_aliases(&overlap).is_err());
        }
        for (field, value) in [(0, 0), (0, 2), (1, 5120), (1, u32::MAX), (2, 5119), (2, u32::MAX), (3, 11), (4, 0), (5, 1)] {
            let mut invalid = packet;
            invalid[field] = value;
            assert!(Kernel::ProjectQuantized.work(&invalid).is_err());
        }
    }
}

#[test]
fn aligned_q3_keeps_padded_extents_and_rejects_unmeasured_packets() {
    let packet = [1, 5120, 17408, 11, 4, 3];
    let work = Kernel::ProjectQuantized.work(&packet).unwrap();
    let ordinary = Kernel::ProjectQuantized.work(&[1, 5120, 17408, 11, 3, 3]).unwrap();
    assert_eq!(work.extents, ordinary.extents);
    assert_eq!((work.blocks, work.threads, work.pointers, work.output, work.in_place), (4352, 128, 3, 2, false));
    assert_eq!(work.alignments, ordinary.alignments);
    let addresses = [0x100000, 0x10000000, 0x40000000];
    work.validate_aliases(&addresses).unwrap();
    assert!(work.validate_aliases(&[addresses[0], addresses[1] + 4, addresses[2]]).is_err());
    for source in 0..2 {
        let mut overlap = addresses;
        overlap[2] = addresses[source] + (work.extents[source] - 1) * 4;
        assert!(work.validate_aliases(&overlap).is_err());
    }
    for (field, value) in [
        (0, 0),
        (0, 2),
        (1, 5119),
        (1, u32::MAX),
        (2, 17407),
        (2, u32::MAX),
        (3, 12),
        (4, 5),
        (5, 0),
        (5, 1),
    ] {
        let mut invalid = packet;
        invalid[field] = value;
        assert!(Kernel::ProjectQuantized.work(&invalid).is_err());
    }
}

#[test]
fn q4_prefill_uses_exact_raw_and_packed_extents() {
    for rows in [5u32, 32] {
        for inner in [6144u32, 17408] {
            let packet = [rows, inner, 5120, 12, 2, 3];
            let work = Kernel::ProjectQuantized.work(&packet).unwrap();
            assert_eq!(
                &work.extents[..3],
                &[
                    u64::from(rows * (3 * inner / 4 + inner / 32)),
                    u64::from(inner / 256 * 5120 * 36),
                    u64::from(rows * 5120)
                ]
            );
            assert_eq!(
                (work.blocks, work.threads, work.pointers, work.output, work.in_place),
                (rows.div_ceil(16) * 160, 128, 3, 2, false)
            );
            let addresses = [0x100000, 0x10000000, 0x40000000];
            work.validate_aliases(&addresses).unwrap();
            assert!(work.validate_aliases(&[addresses[0], addresses[1] + 2, addresses[2]]).is_err());
            for source in 0..2 {
                let mut overlap = addresses;
                overlap[2] = addresses[source] + (work.extents[source] - 1) * 4;
                assert!(work.validate_aliases(&overlap).is_err());
            }
            for (field, value) in [(0, 1), (0, 16), (0, 33), (1, 5120), (1, u32::MAX), (2, 5119), (3, 13), (4, 5), (5, 1)] {
                let mut invalid = packet;
                invalid[field] = value;
                assert!(Kernel::ProjectQuantized.work(&invalid).is_err());
            }
        }
    }
}

#[test]
fn shared_prefill_preserves_extents_and_covers_each_row_tile_once() {
    for (format, phase, inner, columns) in [(11, 5, 5120, 17408), (11, 7, 5120, 17408), (12, 3, 6144, 5120), (12, 3, 17408, 5120)] {
        for rows in [5u32, 32] {
            let packet = [rows, inner, columns, format, phase, 3];
            let work = Kernel::ProjectQuantized.work(&packet).unwrap();
            let baseline = Kernel::ProjectQuantized
                .work(&[rows, inner, columns, format, if format == 11 { 3 } else { 2 }, 3])
                .unwrap();
            assert_eq!(work.extents, baseline.extents);
            assert_eq!(
                (work.blocks, work.threads, work.pointers, work.output, work.in_place),
                (columns / 32, 128, 3, 2, false)
            );
            let pointers = [0x100000, 0x10000000, 0x40000000];
            work.validate_aliases(&pointers).unwrap();
            for source in 0..2 {
                let mut overlap = pointers;
                overlap[2] = pointers[source] + (work.extents[source] - 1) * 4;
                assert!(work.validate_aliases(&overlap).is_err());
            }
            for (field, value) in [(0, 1), (0, 16), (0, 33), (1, u32::MAX), (2, columns - 1), (3, 13), (5, 1)] {
                let mut invalid = packet;
                invalid[field] = value;
                assert!(Kernel::ProjectQuantized.work(&invalid).is_err());
            }
        }
    }
}

#[test]
fn q4_halfwarp_preserves_storage_bounds_and_alignment_with_half_the_ctas() {
    for inner in [6144u32, 17408] {
        let packet = [1, inner, 5120, 12, 5, 3];
        let work = Kernel::ProjectQuantized.work(&packet).unwrap();
        let baseline = Kernel::ProjectQuantized.work(&[1, inner, 5120, 12, 4, 3]).unwrap();
        assert_eq!(work.extents, baseline.extents);
        assert_eq!(work.alignments, baseline.alignments);
        assert_eq!((work.blocks, work.threads, work.pointers, work.output, work.in_place), (640, 128, 3, 2, false));
        assert_eq!(baseline.blocks, work.blocks * 2);
        let addresses = [0x100000, 0x10000000, 0x40000000];
        work.validate_aliases(&addresses).unwrap();
        assert!(work.validate_aliases(&[addresses[0], addresses[1] + 4, addresses[2]]).is_err());
        for source in 0..2 {
            let mut overlap = addresses;
            overlap[2] = addresses[source] + (work.extents[source] - 1) * 4;
            assert!(work.validate_aliases(&overlap).is_err());
        }
        for (field, value) in [
            (0, 0),
            (0, 2),
            (1, 5120),
            (1, u32::MAX),
            (2, 5119),
            (2, u32::MAX),
            (3, 11),
            (4, 6),
            (5, 0),
            (5, 1),
        ] {
            let mut invalid = packet;
            invalid[field] = value;
            assert!(Kernel::ProjectQuantized.work(&invalid).is_err());
        }
    }
}

#[test]
fn halfwarp_decode_halves_ctas_without_changing_extents_or_aliases() {
    let packet = [1, 5120, 17408, 11, 6, 3];
    let work = Kernel::ProjectQuantized.work(&packet).unwrap();
    let baseline = Kernel::ProjectQuantized.work(&[1, 5120, 17408, 11, 4, 3]).unwrap();
    assert_eq!(work.extents, baseline.extents);
    assert_eq!(work.alignments, baseline.alignments);
    assert_eq!((work.blocks, work.threads, work.pointers, work.output, work.in_place), (2176, 128, 3, 2, false));
    assert_eq!(baseline.blocks, work.blocks * 2);
    let addresses = [0x100000, 0x10000000, 0x40000000];
    work.validate_aliases(&addresses).unwrap();
    assert!(work.validate_aliases(&[addresses[0], addresses[1] + 4, addresses[2]]).is_err());
    for source in 0..2 {
        let mut overlap = addresses;
        overlap[2] = addresses[source] + (work.extents[source] - 1) * 4;
        assert!(work.validate_aliases(&overlap).is_err());
    }
    for (field, value) in [(0, 0), (0, 2), (1, 5119), (1, u32::MAX), (2, 17407), (3, 12), (4, 7), (5, 0), (5, 1)] {
        let mut invalid = packet;
        invalid[field] = value;
        assert!(Kernel::ProjectQuantized.work(&invalid).is_err());
    }
}
