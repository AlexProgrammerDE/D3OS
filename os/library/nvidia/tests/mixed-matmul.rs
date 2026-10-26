use d3os_nvidia::compute::{
    Kernel,
    mixed_matmul::{self, Kind, PARAMETERS, THREADS},
};

#[test]
fn registered_tiles_cover_each_output_once_and_keep_packed_slots_in_bounds() {
    for kind in Kind::ALL {
        let kernel = kind.kernel();
        assert_eq!(Kernel::from_raw(kernel as u32).unwrap(), kernel);
        assert!(kernel.allocates_output());
        assert!(![Kernel::Matmul, Kernel::IntMatmul].contains(&kernel));
        for [m, k, n] in [[1, 1, 1], [3, 5, 7], [16, 32, 32], [17, 33, 35], [31, 65, 63]] {
            for lhs_slot in 0..=u32::from(kind != Kind::Flex32) {
                for rhs_slot in 0..=u32::from(kind != Kind::Flex32) {
                    let p = [m, k, n, lhs_slot, rhs_slot];
                    let extents = mixed_matmul::validate(kind, &p).unwrap();
                    let work = kernel.work(&p).unwrap();
                    let blocks = m.div_ceil(16) * n.div_ceil(32);
                    assert_eq!((work.blocks, work.threads), (blocks, THREADS));
                    assert_eq!(mixed_matmul::work_items(kind, &p).unwrap(), blocks * THREADS);
                    assert_eq!(work.extents[..3], extents);
                    assert_eq!((work.pointers, work.output, work.mutated, work.in_place), (3, 2, 0, false));
                    for (count, slot, words) in [(m * k, lhs_slot, extents[0]), (k * n, rhs_slot, extents[1])] {
                        let expected = if kind == Kind::Flex32 {
                            u64::from(count)
                        } else {
                            u64::from(count + slot).div_ceil(2)
                        };
                        assert_eq!(words, expected);
                        assert!(u64::from((count - 1 + slot) / if kind == Kind::Flex32 { 1 } else { 2 }) < words);
                    }
                    let mut visits = vec![0; (m * n) as usize];
                    for block in 0..blocks {
                        let row = block / n.div_ceil(32) * 16;
                        let column = block % n.div_ceil(32) * 32;
                        for r in row..row + 16 {
                            for c in column..column + 32 {
                                if r < m && c < n {
                                    visits[(r * n + c) as usize] += 1;
                                }
                            }
                        }
                    }
                    assert_eq!(extents[2], u64::from(m * n));
                    assert!(visits.into_iter().all(|count| count == 1));
                }
            }
        }
    }
}

#[test]
fn raw_zero_k_uses_only_canonical_one_word_placeholders() {
    for kind in Kind::ALL {
        for [m, n] in [[1, 1], [3, 7], [17, 33]] {
            let p = [m, 0, n, 0, 0];
            assert_eq!(mixed_matmul::validate(kind, &p).unwrap(), [1, 1, u64::from(m * n)]);
            assert!(kind.kernel().work(&p).is_ok());
            for slots in [[1, 0], [0, 1], [1, 1]] {
                assert!(kind.kernel().work(&[m, 0, n, slots[0], slots[1]]).is_err());
            }
        }
    }
}

#[test]
fn malformed_metadata_rejects_before_extent_or_dispatch_arithmetic() {
    let limit = i32::MAX as u32;
    for kind in Kind::ALL {
        let p = [3, 5, 7, 0, 0];
        assert_eq!(p.len(), PARAMETERS);
        for length in 0..PARAMETERS {
            assert!(kind.kernel().work(&p[..length]).is_err());
        }
        assert!(kind.kernel().work(&[3, 5, 7, 0, 0, 0]).is_err());
        for bad in [
            [0, 5, 7, 0, 0],
            [3, 5, 0, 0, 0],
            [limit + 1, 0, 1, 0, 0],
            [1, limit + 1, 1, 0, 0],
            [1, 0, limit + 1, 0, 0],
            [limit, 2, 1, 0, 0],
            [1, 2, limit, 0, 0],
            [limit, 0, 2, 0, 0],
            [1, 1, 1, 2, 0],
            [1, 1, 1, 0, 2],
            [1, 1, 1, 1 << 31, 0],
            [1, 1, 1, 0, u32::MAX],
            // Live counts fit I32, but padded dispatch does not fit U32.
            [1, 0, limit, 0, 0],
            [1, 1, 1 << 30, 0, 0],
        ] {
            assert!(mixed_matmul::validate(kind, &bad).is_err());
            assert!(mixed_matmul::work_items(kind, &bad).is_err());
            assert!(kind.kernel().work(&bad).is_err());
        }
    }
    for slots in [[1, 0], [0, 1], [1, 1]] {
        assert!(Kind::Flex32.kernel().work(&[3, 5, 7, slots[0], slots[1]]).is_err());
    }
}

#[test]
fn symbolic_huge_k_and_asymmetric_padding_preserve_the_dense_count_contract() {
    // Only geometry is evaluated. These symbolic buffers are never allocated.
    for kind in Kind::ALL {
        let slot = u32::from(kind != Kind::Flex32);
        for [m, k, n] in [[1, i32::MAX as u32, 1], [3, 715_827_881, 1], [1, 1, 134_217_729], [3, 1, 715_827_881]] {
            let p = [m, k, n, slot, slot];
            let work = kind.kernel().work(&p).unwrap();
            let words = |count| {
                if kind == Kind::Flex32 {
                    u64::from(count)
                } else {
                    (u64::from(count) + u64::from(slot)).div_ceil(2)
                }
            };
            assert_eq!(work.extents[..3], [words(m * k), words(k * n), u64::from(m * n)]);
            assert_eq!((work.blocks, work.threads), (m.div_ceil(16) * n.div_ceil(32), THREADS));
            assert_eq!(mixed_matmul::work_items(kind, &p).unwrap(), work.blocks * THREADS);
            for (rows, columns, padded_rows, padded_columns) in [
                (m, k, m.div_ceil(16) * 16, k.div_ceil(32) * 32),
                (k, n, k.div_ceil(32) * 32, n.div_ceil(32) * 32),
                (m, n, m.div_ceil(16) * 16, n.div_ceil(32) * 32),
            ] {
                for row in [0, rows - 1, padded_rows - 1] {
                    for column in [0, columns - 1, padded_columns - 1] {
                        let valid = row < rows && column < columns;
                        let (safe_row, safe_column) = if valid { (row, column) } else { (0, 0) };
                        let address = safe_row.checked_mul(columns).unwrap().checked_add(safe_column).unwrap();
                        assert!(address < rows * columns);
                        assert!(address.checked_add(slot).is_some());
                    }
                }
            }
        }
        let work = kind.kernel().work(&[3, 1, 715_827_881, slot, slot]).unwrap();
        assert!(work.blocks * THREADS > i32::MAX as u32);
        assert!(kind.kernel().work(&[1, i32::MAX as u32, 1, 0, 0]).is_ok());
    }
}

#[test]
fn input_aliases_are_allowed_but_any_output_overlap_or_address_wrap_rejects() {
    for kind in Kind::ALL {
        for k in [0, 5] {
            let work = kind.kernel().work(&[3, k, 7, 0, 0]).unwrap();
            let a = 0x10_0000;
            let b = 0x20_0000;
            let out = 0x30_0000;
            assert!(work.validate_aliases(&[a, a, out]).is_ok());
            assert!(work.validate_aliases(&[a, a + 4, out]).is_ok());
            assert!(work.validate_aliases(&[a, b, out]).is_ok());
            for (source, extent) in [(a, work.extents[0]), (b, work.extents[1])] {
                for output in [source, source + extent * 4 - 4, source - work.extents[2] * 4 + 4] {
                    assert!(work.validate_aliases(&[a, b, output]).is_err());
                }
            }
            assert!(work.validate_aliases(&[a, b, u64::MAX - 3]).is_err());
            assert!(work.validate_aliases(&[u64::MAX - 3, b, out]).is_err());
            assert!(work.validate_aliases(&[a, u64::MAX - 3, out]).is_err());
            assert!(work.validate_aliases(&[a, b]).is_err());
            assert!(work.validate_aliases(&[a, b, out, out + 4]).is_err());
        }
    }
}
