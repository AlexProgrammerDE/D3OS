#![cfg(feature = "training")]
use burn::{
    backend::DType,
    tensor::{bf16, f16},
};
use core::cmp::Ordering;
use d3os_ai::validation::float_extrema::{
    self as extrema, DTYPES, Decoded, ExecutionScope, INDEX_DTYPES, Input, Operation as Op, Options, ReferenceError as Error, Report, Scope, Values, View,
    compare, corpus, decode, decode_index, expected_summary, pack, pool, project_index, reference, success_footer, unpack,
};

fn select(dtype: DType, index_dtype: DType, values: &[u64], op: Op) -> extrema::Expected {
    reference(dtype, index_dtype, &Values::new(&[values.len()], values.to_vec()), op, Scope::Axis(0)).unwrap()
}

#[test]
fn ieee_decoder_retains_classes_and_exact_binary_components() {
    assert_eq!(
        decode(DType::F64, 0x3ff0_0000_0000_0001),
        Decoded::Finite {
            negative: false,
            significand: (1 << 52) + 1,
            exponent: -52
        }
    );
    assert_eq!(
        decode(DType::F64, 1),
        Decoded::Finite {
            negative: false,
            significand: 1,
            exponent: -1074
        }
    );
    assert_eq!(
        decode(DType::F32, 0x8000_0000),
        Decoded::Finite {
            negative: true,
            significand: 0,
            exponent: -149
        }
    );
    assert_eq!(decode(DType::F16, 0xfc00), Decoded::Infinity { negative: true });
    assert_eq!(decode(DType::BF16, 0xff81), Decoded::Nan { negative: true, payload: 1 });
    for dtype in DTYPES {
        let p = pool(dtype);
        assert_eq!(compare(dtype, p[0], p[1]), Some(Ordering::Equal));
        assert_eq!(compare(dtype, p[2], p[5]), Some(Ordering::Less));
        assert_eq!(compare(dtype, p[6], p[7]), Some(Ordering::Less));
        assert_eq!(compare(dtype, p[14], p[0]), None);
    }
}

#[test]
fn exact_comparison_matches_primitive_order_across_wide_and_all_half_encodings() {
    // Primitive comparisons provide an independent check on exponent alignment,
    // negative order, subnormals, infinities, and unordered NaNs.
    for dtype in DTYPES {
        let p = pool(dtype);
        for &a in &p {
            for &b in &p {
                let expected = match dtype {
                    DType::F64 => f64::from_bits(a).partial_cmp(&f64::from_bits(b)),
                    DType::F32 | DType::Flex32 => f32::from_bits(a as u32).partial_cmp(&f32::from_bits(b as u32)),
                    DType::F16 => f16::from_bits(a as u16).to_f64().partial_cmp(&f16::from_bits(b as u16).to_f64()),
                    DType::BF16 => bf16::from_bits(a as u16).to_f64().partial_cmp(&bf16::from_bits(b as u16).to_f64()),
                    _ => unreachable!(),
                };
                assert_eq!(compare(dtype, a, b), expected);
            }
        }
    }
    for bits in 0..=u16::MAX {
        for dtype in [DType::F16, DType::BF16] {
            let next = bits.wrapping_add(1);
            let as_f64 = |bits| {
                if dtype == DType::F16 {
                    f16::from_bits(bits).to_f64()
                } else {
                    bf16::from_bits(bits).to_f64()
                }
            };
            for other in [next, bits ^ 0x8000, 0, 1, 0x7bff, 0x7c00, 0x7f80] {
                assert_eq!(compare(dtype, bits.into(), other.into()), as_f64(bits).partial_cmp(&as_f64(other)));
            }
        }
    }
    let mut seed = 0x5379_18a2_61fd_9e43u64;
    for _ in 0..8192 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let a = seed;
        let b = seed.rotate_left(29);
        assert_eq!(compare(DType::F64, a, b), f64::from_bits(a).partial_cmp(&f64::from_bits(b)));
        assert_eq!(
            compare(DType::F32, a as u32 as u64, b as u32 as u64),
            f32::from_bits(a as u32).partial_cmp(&f32::from_bits(b as u32))
        );
    }
}

#[test]
fn nan_payloads_and_numeric_ties_keep_first_logical_source_bits() {
    for dtype in DTYPES {
        let p = pool(dtype);
        for op in Op::PAIRS {
            for values in [
                vec![p[14], p[15], p[12]],
                vec![p[15], p[14], p[13]],
                vec![p[0], p[1]],
                vec![p[1], p[0]],
                vec![p[12]; 129],
                vec![p[13]; 129],
            ] {
                let result = select(dtype, DType::I64, &values, op);
                assert_eq!(result.output.values, [values[0]]);
                assert_eq!(result.indices.unwrap().values, [0]);
                assert_eq!(result.logical_indices, [0]);
            }
            let mut values = vec![p[6]; 131];
            values[1] = p[14];
            values[129] = p[15];
            assert_eq!(select(dtype, DType::I32, &values, op).output.values, [p[14]]);
            values.reverse();
            let reversed = select(dtype, DType::I32, &values, op);
            assert_eq!(reversed.output.values, [p[15]]);
            assert_eq!(reversed.logical_indices, [1]);
        }
        for (op, expected) in [(Op::MinWithIndices, p[6]), (Op::MaxWithIndices, p[8])] {
            let result = select(dtype, DType::U64, &p[6..9], op);
            assert_eq!(result.output.values, [expected]);
        }
    }
}

#[test]
fn projection_never_changes_the_selected_pair_value() {
    for (winner, dtype, signed) in [(129, DType::I8, -127), (256, DType::U8, 0), (32768, DType::I16, -32768), (65536, DType::U16, 0)] {
        let mut values = vec![0x3ff0_0000_0000_0000; winner + 1];
        values[winner] = 0x3ff0_0000_0000_0001;
        let result = select(DType::F64, dtype, &values, Op::MaxWithIndices);
        assert_eq!(result.output.values, [values[winner]]);
        assert_eq!(result.logical_indices, [winner]);
        assert_eq!(decode_index(dtype, result.indices.unwrap().values[0]), signed);
    }
    for dtype in INDEX_DTYPES {
        for coordinate in [0, 1, 129, 256, 32768, 65536, i32::MAX as usize] {
            let projected = project_index(dtype, coordinate);
            let bytes = coordinate.to_le_bytes();
            assert_eq!(pack(dtype, &[projected]), bytes[..dtype.size()]);
            assert_eq!(unpack(dtype, &bytes[..dtype.size()]), [projected]);
        }
    }
}

#[test]
fn invalid_axes_and_empty_reductions_reject_before_empty_output() {
    for dtype in DTYPES {
        for op in Op::AXIS.into_iter().chain(Op::PAIRS) {
            assert_eq!(
                reference(dtype, DType::I32, &Values::new(&[0, 3], vec![]), op, Scope::Axis(1)).unwrap().output,
                Values::new(&[0, 1], vec![])
            );
            for shape in [[2, 0], [0, 0]] {
                assert_eq!(
                    reference(dtype, DType::I32, &Values::new(&shape, vec![]), op, Scope::Axis(1)),
                    Err(Error::EmptyAxis)
                );
            }
            assert_eq!(reference(dtype, DType::I32, &Values::new(&[], vec![0]), op, Scope::Axis(0)), Err(Error::Axis));
            assert_eq!(
                reference(dtype, DType::I32, &Values::new(&[0, 3], vec![]), op, Scope::Axis(2)),
                Err(Error::Axis)
            );
        }
        for op in Op::GLOBAL {
            assert_eq!(
                reference(dtype, DType::I32, &Values::new(&[0], vec![]), op, Scope::Global),
                Err(Error::EmptyAxis)
            );
            let scalar = Values::new(&[], vec![pool(dtype)[14]]);
            assert_eq!(
                reference(dtype, DType::I32, &scalar, op, Scope::Global).unwrap().output,
                Values::new(&[1], scalar.values)
            );
        }
        assert_eq!(
            reference(dtype, DType::I32, &Values::new(&[1], vec![0]), Op::ArgMin, Scope::Global),
            Err(Error::GlobalIndex)
        );
        assert_eq!(
            reference(dtype, dtype, &Values::new(&[1], vec![0]), Op::Min, Scope::Global),
            Err(Error::IndexDtype)
        );
    }
    assert_eq!(
        reference(DType::I32, DType::I32, &Values::new(&[1], vec![0]), Op::Min, Scope::Global),
        Err(Error::Dtype)
    );
    assert_eq!(
        reference(
            DType::F64,
            DType::I32,
            &Values {
                shape: vec![0, i32::MAX as usize, i32::MAX as usize],
                values: vec![]
            },
            Op::Min,
            Scope::Axis(0)
        ),
        Err(Error::EmptyAxis)
    );
    let outer = reference(
        DType::F64,
        DType::I32,
        &Values {
            shape: vec![0, i32::MAX as usize, i32::MAX as usize],
            values: vec![],
        },
        Op::Min,
        Scope::Axis(1),
    )
    .unwrap();
    assert!(outer.output.values.is_empty());
}

#[test]
fn signed_permuted_offset_and_broadcast_views_keep_source_identity() {
    let p = pool(DType::F64);
    let root = Values::new(&[3, 5], (0..15).map(|i| p[6] + i).collect());
    let input = Input {
        root: root.clone(),
        views: vec![
            View::Slice {
                starts: vec![1, 1],
                steps: vec![1, 2],
                shape: vec![2, 2],
            },
            View::Flip(vec![1]),
            View::Permute(vec![1, 0]),
        ],
    };
    assert_eq!(input.logical(), Values::new(&[2, 2], [8, 13, 6, 11].map(|i| root.values[i]).to_vec()));
    let expected = reference(DType::F64, DType::U8, &input.logical(), Op::MaxWithIndices, Scope::Axis(0)).unwrap();
    assert_eq!(expected.output.values, [root.values[8], root.values[13]]);
    assert_eq!(expected.logical_indices, [0, 0]);
    assert_eq!(input.root, root);
    let broadcast = Input {
        root: Values::new(&[1, 2], vec![p[0], p[1]]),
        views: vec![View::Expand(vec![3, 2]), View::Flip(vec![1])],
    };
    let expected = reference(DType::F64, DType::I16, &broadcast.logical(), Op::MinWithIndices, Scope::Axis(0)).unwrap();
    assert_eq!(expected.output.values, [p[1], p[0]]);
    assert_eq!(expected.logical_indices, [0, 0]);
}

#[test]
fn reverse_axis_staging_matches_direct_global_bits_for_every_four_element_pattern() {
    fn global(dtype: DType, values: &[u64], op: Op) -> u64 {
        reference(dtype, DType::I32, &Values::new(&[2, 2], values.to_vec()), op, Scope::Global)
            .unwrap()
            .output
            .values[0]
    }
    for dtype in DTYPES {
        let p = pool(dtype);
        let choices = [p[0], p[1], p[6], p[7], p[14], p[15]];
        for a in choices {
            for b in choices {
                for c in choices {
                    for d in choices {
                        let values = [a, b, c, d];
                        for op in Op::GLOBAL {
                            let mut staged = Values::new(&[2, 2], values.to_vec());
                            for axis in [1, 0] {
                                staged = reference(dtype, DType::I32, &staged, op, Scope::Axis(axis)).unwrap().output;
                            }
                            assert_eq!(staged.values, [global(dtype, &values, op)]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bounded_corpus_covers_requested_types_lengths_layouts_and_retains_full_winners() {
    let cases = corpus(&Options::default());
    let mut ids = std::collections::BTreeSet::new();
    for case in &cases {
        assert!(ids.insert(&case.id));
    }
    assert_eq!(cases.iter().filter(|c| c.steps.len() == 1).count(), 126);
    assert_eq!(cases.iter().filter(|c| c.steps.len() > 1).count(), 8);
    for dtype in DTYPES {
        for index in INDEX_DTYPES {
            assert!(
                cases
                    .iter()
                    .any(|c| c.dtype == dtype && c.steps[0].index_dtype == index && c.steps[0].operation.is_pair() && c.expected().is_ok())
            );
        }
        for op in Op::GLOBAL {
            assert!(
                cases
                    .iter()
                    .any(|c| c.dtype == dtype && c.steps[0].operation == op && c.steps[0].scope == Scope::Global && c.expected().is_ok())
            );
        }
        for op in Op::AXIS {
            assert!(
                cases
                    .iter()
                    .any(|c| c.dtype == dtype && c.steps[0].operation == op && matches!(c.steps[0].scope, Scope::Axis(_)) && c.expected().is_ok())
            );
        }
        for op in Op::PAIRS {
            assert!(
                cases
                    .iter()
                    .any(|c| c.dtype == dtype && c.steps[0].operation == op && c.steps[0].index_dtype == DType::I32 && c.expected().is_ok())
            );
        }
    }
    for rank in 0..=8 {
        assert!(cases.iter().any(|c| c.input.logical().shape.len() == rank && c.expected().is_ok()));
    }
    for length in extrema::LENGTHS {
        assert!(cases.iter().any(|c| match c.steps[0].scope {
            Scope::Axis(axis) => c.input.logical().shape.get(axis) == Some(&length),
            Scope::Global => false,
        }));
    }
    for case in &cases {
        if let Ok(outputs) = case.expected() {
            assert_eq!(outputs.len(), case.steps.len());
            for (step, expected) in case.steps.iter().zip(outputs) {
                if let Some(indices) = expected.indices {
                    assert_eq!(indices.shape, expected.output.shape);
                    assert_eq!(
                        indices.values,
                        expected.logical_indices.iter().map(|&i| project_index(step.index_dtype, i)).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn native_error_scope_explicitly_plans_input_source_alias_and_recovery_checks() {
    for skip_rank_zero in [false, true] {
        for include_empty in [false, true] {
            let linux_options = Options {
                skip_rank_zero,
                include_empty,
                ..Options::default()
            };
            let native_options = Options {
                native_error_extensions: true,
                ..linux_options
            };
            let linux = expected_summary(&linux_options);
            let native = expected_summary(&native_options);
            assert_eq!(linux.scope, ExecutionScope::LinuxDefault);
            assert_eq!(native.scope, ExecutionScope::NativeErrorExtensions);
            assert_eq!(linux.options, linux_options);
            assert_eq!(native.options, native_options);
            assert_eq!(linux.checks.input, linux.valid_cases);
            assert_eq!(linux.checks.source, linux.valid_cases);
            assert_eq!(linux.checks.alias, linux.valid_cases);
            assert_eq!(native.checks.input, native.cases);
            assert_eq!(native.checks.source, native.cases);
            assert_eq!(native.checks.alias, native.cases);
            assert_eq!(native.checks.recovery, native.diagnostic_cases);
            assert_eq!(linux.checks.recovery, 0);
            assert_eq!(native.checks.output, linux.checks.output);
            assert_eq!(native.checks.chain_index, linux.checks.chain_index);
            let selected: Vec<_> = DTYPES
                .into_iter()
                .map(|dtype| {
                    expected_summary(&Options {
                        dtype: Some(dtype),
                        ..native_options
                    })
                })
                .collect();
            assert_eq!(selected.iter().map(|s| s.cases).sum::<usize>(), native.cases);
            assert_eq!(selected.iter().map(|s| s.checks.input).sum::<usize>(), native.checks.input);
            assert_eq!(selected.iter().map(|s| s.checks.source).sum::<usize>(), native.checks.source);
            assert_eq!(selected.iter().map(|s| s.checks.alias).sum::<usize>(), native.checks.alias);
        }
    }
}

#[test]
fn footer_requires_matching_scope_complete_cases_and_observed_checks() {
    for native_error_extensions in [false, true] {
        let options = Options {
            native_error_extensions,
            ..Options::default()
        };
        let expected = expected_summary(&options);
        let observed_errors = if native_error_extensions { expected.diagnostic_cases } else { 0 };
        let skipped_diagnostics = expected.diagnostic_cases - observed_errors;
        let report = Report {
            attempted: expected.valid_cases + observed_errors,
            passed: expected.valid_cases + observed_errors,
            skipped_diagnostics,
            observed_errors,
            failed: 0,
            checks: expected.checks.clone(),
            expected,
            failures: vec![],
        };
        assert!(report.is_success());
        assert!(success_footer(&options, &report).is_ok());
        for which in 0..6 {
            let mut incomplete = report.clone();
            let checks = &mut incomplete.checks;
            let count = match which {
                0 => &mut checks.input,
                1 => &mut checks.source,
                2 => &mut checks.alias,
                3 => &mut checks.output,
                4 => &mut checks.chain_index,
                _ => &mut checks.recovery,
            };
            *count += 1;
            assert!(!incomplete.is_success());
            assert!(success_footer(&options, &incomplete).is_err());
        }
        assert!(
            success_footer(
                &Options {
                    native_error_extensions: !native_error_extensions,
                    ..options
                },
                &report
            )
            .is_err()
        );
        let mut incomplete = report.clone();
        incomplete.passed -= 1;
        assert!(!incomplete.is_success());
        let mut bad_plan = report;
        bad_plan.expected.values += 1;
        assert!(success_footer(&options, &bad_plan).is_err());
    }
}
