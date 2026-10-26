#![cfg(feature = "training")]
extern crate alloc;

use burn::backend::DType;
use d3os_ai::validation::integer_reduction::{
    self, DTYPES, Operation as Op, Options, ReferenceError as Error, Report, Scope, Summary, Values, corpus, decode, expected_summary, mask, reference,
    success_footer,
};

fn reduce(dtype: DType, values: &[u64], op: Op, scope: Scope) -> u64 {
    reference(dtype, &Values::new(&[values.len()], values.to_vec()), op, scope)
        .unwrap()
        .output
        .values[0]
}

#[test]
fn widened_axis_mean_and_stored_global_mean_are_distinct() {
    assert_eq!(reduce(DType::I8, &[127, 127], Op::Mean, Scope::Axis(0)), 127);
    assert_eq!(decode(DType::I8, reduce(DType::I8, &[127, 127], Op::Mean, Scope::Global)), -1);
    assert_eq!(reduce(DType::U8, &[255, 255], Op::Mean, Scope::Axis(0)), 255);
    assert_eq!(reduce(DType::U8, &[255, 255], Op::Mean, Scope::Global), 127);
    assert_eq!(decode(DType::I32, reduce(DType::I32, &[(-5i64) as u64, 0], Op::Mean, Scope::Axis(0))), -2);
    // These constants distinguish wrap-before-division from an exact average.
    assert_eq!(reduce(DType::I32, &[i32::MAX as u64; 2], Op::Mean, Scope::Axis(0)), u64::from(u32::MAX));
    assert_eq!(reduce(DType::U32, &[u32::MAX as u64; 2], Op::Mean, Scope::Axis(0)), 2_147_483_647);
    assert_eq!(reduce(DType::I64, &[i64::MAX as u64; 2], Op::Mean, Scope::Axis(0)), u64::MAX);
    assert_eq!(reduce(DType::U64, &[u64::MAX; 2], Op::Mean, Scope::Axis(0)), i64::MAX as u64);
}

#[test]
fn byte_pairs_match_primitive_stores_and_widened_signed_division() {
    // A small complete byte domain checks sign extension independently of the
    // oracle's residue arithmetic, without duplicating its reduction loop.
    for a in 0..=255u64 {
        for b in 0..=255u64 {
            let unsigned = [a, b];
            assert_eq!(reduce(DType::U8, &unsigned, Op::Sum, Scope::Global), u64::from((a as u8).wrapping_add(b as u8)));
            assert_eq!(
                reduce(DType::U8, &unsigned, Op::Product, Scope::Global),
                u64::from((a as u8).wrapping_mul(b as u8))
            );
            let signed = i16::from(a as i8) + i16::from(b as i8);
            assert_eq!(reduce(DType::I8, &unsigned, Op::Mean, Scope::Axis(0)), (signed / 2) as u8 as u64);
        }
    }
}

#[test]
fn products_wrap_in_accumulator_before_division_and_preserve_high_words() {
    assert_eq!(reduce(DType::I8, &[127; 5], Op::Product, Scope::Axis(0)), 127);
    assert_eq!(reduce(DType::U64, &[u64::MAX, u64::MAX], Op::Product, Scope::Global), 1);
    assert_eq!(reduce(DType::I64, &[i64::MIN as u64, u64::MAX], Op::Product, Scope::Global), i64::MIN as u64);
    for dtype in DTYPES {
        let values = [mask(dtype), 1, 1u64 << (integer_reduction::width(dtype) - 1)];
        let data = integer_reduction::pack(dtype, &values);
        assert_eq!(integer_reduction::unpack(dtype, &data), values);
        assert_eq!(data.len(), values.len() * dtype.size());
    }
}

#[test]
fn extrema_choose_first_logical_winner_without_float_or_narrowed_gather() {
    for (dtype, values, minimum, maximum) in [
        (DType::U64, vec![u64::MAX - 1, u64::MAX, u64::MAX], 0, 1),
        (DType::I64, vec![(1 << 53) + 1, (1 << 53) + 2, i64::MIN as u64], 2, 1),
        (DType::I8, vec![128, 255, 127, 128], 0, 2),
    ] {
        for (op, winner) in [(Op::MinWithIndices, minimum), (Op::MaxWithIndices, maximum)] {
            let expected = reference(dtype, &Values::new(&[values.len()], values.clone()), op, Scope::Axis(0)).unwrap();
            assert_eq!(expected.logical_indices, Some(vec![winner]));
            assert_eq!(expected.output.values, [values[winner]]);
            assert_eq!(expected.indices.unwrap().values, [winner as u64]);
        }
    }
    for (dtype, length, winner, value, exposed) in [(DType::I8, 130, 129, 127, 129), (DType::U8, 257, 256, 255, 0)] {
        let mut values = vec![0; length];
        values[winner] = value;
        let expected = reference(dtype, &Values::new(&[length], values), Op::MaxWithIndices, Scope::Axis(0)).unwrap();
        assert_eq!(expected.output.values, [value]);
        assert_eq!(expected.logical_indices, Some(vec![winner]));
        assert_eq!(expected.indices.unwrap().values, [exposed]);
    }
}

#[test]
fn empty_axis_validation_precedes_outer_empty_and_global_count_checks() {
    for dtype in DTYPES {
        for (op, identity) in [(Op::Sum, 0), (Op::Product, 1)] {
            let expected = reference(dtype, &Values::new(&[2, 0], vec![]), op, Scope::Axis(1)).unwrap();
            assert_eq!(expected.output, Values::new(&[2, 1], vec![identity; 2]));
            assert_eq!(
                reference(dtype, &Values::new(&[0], vec![]), op, Scope::Global).unwrap().output,
                Values::new(&[1], vec![identity])
            );
        }
        for op in Op::AXIS.into_iter().chain(Op::PAIRS) {
            let positive_axis = reference(dtype, &Values::new(&[0, 3], vec![]), op, Scope::Axis(1)).unwrap();
            assert_eq!(positive_axis.output, Values::new(&[0, 1], vec![]));
            if !matches!(op, Op::Sum | Op::Product) {
                assert_eq!(reference(dtype, &Values::new(&[0, 0], vec![]), op, Scope::Axis(1)), Err(Error::EmptyAxis));
            }
        }
        assert_eq!(reference(dtype, &Values::new(&[0], vec![]), Op::Mean, Scope::Global), Err(Error::ZeroCount));
        assert_eq!(reference(dtype, &Values::new(&[0], vec![]), Op::Sum, Scope::Axis(1)), Err(Error::Axis));
        assert_eq!(reference(dtype, &Values::new(&[], vec![1]), Op::Sum, Scope::Axis(0)), Err(Error::Axis));
        assert_eq!(
            reference(dtype, &Values::new(&[], vec![1]), Op::Sum, Scope::Global).unwrap().output,
            Values::new(&[1], vec![1])
        );
    }
    for (dtype, length) in [(DType::I8, 128), (DType::U8, 256), (DType::I16, 32768), (DType::U16, 65536)] {
        let input = Values::new(&[length], vec![1; length]);
        assert_eq!(reference(dtype, &input, Op::Mean, Scope::Global), Err(Error::CountNotRepresentable));
        assert_eq!(reference(dtype, &input, Op::Mean, Scope::Axis(0)).unwrap().output.values, [1]);
    }
}

#[test]
fn multidimensional_rows_and_materialized_order_have_independent_answers() {
    let input = Values::new(&[2, 2], [-4i64, -4, -3, -1].into_iter().map(|v| v as u64 & 0xffff_ffff).collect());
    for (axes, expected) in [([0, 1], -2), ([1, 0], -3)] {
        let mut data = input.clone();
        for axis in axes {
            data = reference(DType::I32, &data, Op::Mean, Scope::Axis(axis)).unwrap().output;
        }
        assert_eq!(decode(DType::I32, data.values[0]), expected);
    }
    assert_eq!(
        decode(DType::I32, reference(DType::I32, &input, Op::Mean, Scope::Global).unwrap().output.values[0]),
        -3
    );
    let input = Values::new(&[2, 3], vec![5, 1, 5, 7, 0, 2]);
    assert_eq!(
        reference(DType::U64, &input, Op::Sum, Scope::Axis(0)).unwrap().output,
        Values::new(&[1, 3], vec![12, 1, 7])
    );
    let expected = reference(DType::U64, &input, Op::MaxWithIndices, Scope::Axis(1)).unwrap();
    assert_eq!(expected.output, Values::new(&[2, 1], vec![5, 7]));
    assert_eq!(expected.indices.unwrap().values, [0, 0]);
}

#[test]
fn selected_corpus_covers_formats_axes_views_and_separates_diagnostics() {
    let options = Options::default();
    let cases = corpus(&options);
    let mut ids = std::collections::BTreeSet::new();
    for case in &cases {
        assert!(ids.insert(&case.id));
    }
    for dtype in DTYPES {
        for op in Op::AXIS.into_iter().chain(Op::PAIRS) {
            assert!(
                cases
                    .iter()
                    .any(|c| c.dtype == dtype && c.steps[0].operation == op && matches!(c.steps[0].scope, Scope::Axis(_)) && c.expected().is_ok())
            );
        }
        for op in Op::GLOBAL {
            assert!(
                cases
                    .iter()
                    .any(|c| c.dtype == dtype && c.steps[0].operation == op && c.steps[0].scope == Scope::Global && c.expected().is_ok())
            );
        }
    }
    for rank in 1..=8 {
        for axis in 0..rank {
            assert!(
                cases
                    .iter()
                    .any(|c| c.input.logical().shape == vec![2; rank] && c.steps[0].scope == Scope::Axis(axis))
            );
        }
    }
    for rank in 1..=8 {
        assert!(
            cases
                .iter()
                .any(|c| c.public_api && c.input.logical().shape.len() == rank && c.expected().is_ok())
        );
    }
    assert!(cases.len() <= 512);
    for case in &cases {
        if let Ok(outputs) = case.expected() {
            assert_eq!(outputs.len(), case.steps.len());
            for expected in outputs {
                assert!(expected.output.values.iter().all(|&v| v & !mask(case.dtype) == 0));
                if let Some(indices) = expected.indices {
                    assert_eq!(indices.shape, expected.output.shape);
                    for (&exposed, logical) in indices.values.iter().zip(expected.logical_indices.unwrap()) {
                        assert_eq!(exposed, logical as u64 & mask(case.dtype));
                    }
                }
            }
        }
    }
    let s = expected_summary(&options);
    assert_eq!(s.cases, cases.len());
    assert_eq!(s.valid_cases + s.diagnostic_cases, s.cases);
    assert!(s.diagnostic_cases > 0 && s.chain_steps > s.chain_cases);
    assert_eq!(
        s,
        Summary {
            dtypes: 8,
            cases: 365,
            valid_cases: 323,
            diagnostic_cases: 42,
            steps: 347,
            public_cases: 155,
            pair_steps: 52,
            chain_cases: 16,
            chain_steps: 40,
            values: 2360,
            source_checks: 323,
            alias_checks: 323,
        }
    );
    let selected = Options {
        dtype: Some(DType::U64),
        skip_rank_zero: true,
        include_empty: false,
        ..options
    };
    assert!(
        corpus(&selected)
            .iter()
            .all(|c| c.dtype == DType::U64 && !c.input.logical().shape.is_empty() && !c.input.logical().shape.contains(&0))
    );
}

#[test]
fn summaries_partition_selected_corpus_and_keep_error_checks_separate() {
    for skip_rank_zero in [false, true] {
        for include_empty in [false, true] {
            let options = Options {
                skip_rank_zero,
                include_empty,
                ..Options::default()
            };
            let total = expected_summary(&options);
            let mut parts = Summary::default();
            for dtype in DTYPES {
                let selected = expected_summary(&Options { dtype: Some(dtype), ..options });
                parts.dtypes += selected.dtypes;
                parts.cases += selected.cases;
                parts.valid_cases += selected.valid_cases;
                parts.diagnostic_cases += selected.diagnostic_cases;
                parts.steps += selected.steps;
                parts.public_cases += selected.public_cases;
                parts.pair_steps += selected.pair_steps;
                parts.chain_cases += selected.chain_cases;
                parts.chain_steps += selected.chain_steps;
                parts.values += selected.values;
                parts.source_checks += selected.source_checks;
                parts.alias_checks += selected.alias_checks;
            }
            assert_eq!(parts, total);
            let native = expected_summary(&Options {
                native_error_extensions: true,
                ..options
            });
            assert_eq!(native.source_checks, total.source_checks + total.diagnostic_cases);
            assert_eq!(native.alias_checks, total.alias_checks + total.diagnostic_cases);
            assert_eq!(
                Summary {
                    source_checks: total.source_checks,
                    alias_checks: total.alias_checks,
                    ..native
                },
                total
            );
        }
    }
}

#[test]
fn success_requires_complete_observations_in_the_selected_error_scope() {
    for native_error_extensions in [false, true] {
        let options = Options {
            native_error_extensions,
            ..Options::default()
        };
        let expected = expected_summary(&options);
        let skipped_diagnostics = if native_error_extensions { 0 } else { expected.diagnostic_cases };
        let observed_errors = expected.diagnostic_cases - skipped_diagnostics;
        let report = Report {
            attempted: expected.valid_cases + observed_errors,
            passed: expected.valid_cases + observed_errors,
            skipped_diagnostics,
            observed_errors,
            failed: 0,
            failures: vec![],
            expected,
        };
        assert!(report.is_success());
        assert!(success_footer(&options, &report).is_ok());
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
        let mut missing_diagnostic = report.clone();
        if native_error_extensions {
            missing_diagnostic.observed_errors -= 1;
        } else {
            missing_diagnostic.skipped_diagnostics -= 1;
        }
        assert!(!missing_diagnostic.is_success());
        let mut mismatched_plan = report;
        mismatched_plan.expected.values += 1;
        assert!(success_footer(&options, &mismatched_plan).is_err());
    }
}

#[test]
fn signed_permuted_offset_and_broadcast_views_follow_logical_order() {
    use integer_reduction::{Input, View};
    let input = Input {
        root: Values::new(&[2, 3], (0..6).collect()),
        views: vec![View::Permute(vec![1, 0]), View::Flip(vec![0])],
    }
    .logical();
    assert_eq!(input, Values::new(&[3, 2], vec![2, 5, 1, 4, 0, 3]));
    assert_eq!(
        reference(DType::U64, &input, Op::Sum, Scope::Axis(1)).unwrap().output,
        Values::new(&[3, 1], vec![7, 5, 3])
    );
    let input = Input {
        root: Values::new(&[3, 5], (0..15).collect()),
        views: vec![View::Slice {
            starts: vec![1, 1],
            steps: vec![1, 2],
            shape: vec![2, 2],
        }],
    }
    .logical();
    assert_eq!(input, Values::new(&[2, 2], vec![6, 8, 11, 13]));
    let input = Input {
        root: Values::new(&[1, 2], vec![7, 9]),
        views: vec![View::Expand(vec![3, 2]), View::Flip(vec![1])],
    }
    .logical();
    let expected = reference(DType::U64, &input, Op::MaxWithIndices, Scope::Axis(0)).unwrap();
    assert_eq!(expected.output, Values::new(&[1, 2], vec![9, 7]));
    assert_eq!(expected.indices.unwrap().values, [0, 0]);
}
