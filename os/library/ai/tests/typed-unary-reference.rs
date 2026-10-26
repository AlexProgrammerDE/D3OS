#![cfg(feature = "training")]
extern crate alloc;

#[path = "../src/validation/typed_unary.rs"]
#[allow(dead_code)]
mod typed_unary;
use burn::backend::{DType, bf16, f16};
use typed_unary::{
    Accuracy, Class, DTYPES, Failures, OPERATIONS, Operation, Options, Reference, classify, compare, exact_special, expected_bits, expected_summary, is_finite,
    is_nan, ordered_rank, pack, rows, selected_rows, unpack,
};

fn unrank(dtype: DType, rank: u64) -> u64 {
    let sign = 1u64 << (dtype.size() * 8 - 1);
    if rank & sign != 0 { rank & (sign - 1) } else { !rank & (sign | (sign - 1)) }
}
fn half_round(dtype: DType, bits: u64) -> u64 {
    let value = f32::from_bits(bits as u32);
    match dtype {
        DType::F16 => u64::from(f16::from_f32(value).to_bits()),
        DType::BF16 => u64::from(bf16::from_f32(value).to_bits()),
        _ => unreachable!(),
    }
}
fn nz(class: Class) -> bool {
    matches!(class, Class::Normal | Class::Subnormal)
}
fn one(dtype: DType) -> u64 {
    expected_bits(dtype, Operation::Cosh, 0).unwrap().staged
}
fn infinity(dtype: DType) -> u64 {
    match dtype {
        DType::F64 => 0x7ff0_0000_0000_0000,
        DType::F16 => 0x7c00,
        DType::BF16 => 0x7f80,
        _ => 0x7f80_0000,
    }
}
fn successors(dtype: DType, op: Operation, input: u64) -> Vec<u64> {
    let row = expected_bits(dtype, op, input).expect("every legal stored input has a frozen reference");
    if is_nan(dtype, row.staged) || exact_special(dtype, op, input) || !nz(classify(dtype, row.staged)) {
        return vec![row.staged];
    }
    let ranks = if matches!(dtype, DType::F16 | DType::BF16) {
        ordered_rank(dtype, row.allowed_low)..=ordered_rank(dtype, row.allowed_high)
    } else {
        let center = ordered_rank(dtype, row.staged);
        center - op.budget(dtype)..=center + op.budget(dtype)
    };
    ranks
        .map(|r| unrank(dtype, r))
        .filter(|&actual| compare(dtype, op, &row, actual).passed)
        .collect()
}

#[test]
fn every_half_encoding_has_exact_staging_image_and_independent_class_census() {
    for dtype in [DType::F16, DType::BF16] {
        for op in OPERATIONS {
            let references = rows(dtype, op);
            assert_eq!(references.len(), 65536);
            let inputs: Vec<_> = references.iter().map(|r| r.input).collect();
            assert_eq!(inputs, (0..65536).collect::<Vec<_>>());
            assert_eq!(unpack(dtype, &pack(dtype, &inputs)), inputs);
            let mut differences = 0;
            let mut class_disagreements = 0;
            let mut status_disagreements = 0;
            let mut accuracy = Accuracy::default();
            for row in references {
                assert_eq!(expected_bits(dtype, op, row.input), Some(row));
                assert!(accuracy.record(dtype, op, &row, row.staged).passed);
                differences += usize::from(row.ideal != row.staged);
                class_disagreements += usize::from(classify(dtype, row.ideal) != classify(dtype, row.staged));
                status_disagreements += usize::from(nz(classify(dtype, row.ideal)) != nz(classify(dtype, row.staged)));
                if is_nan(dtype, row.staged) {
                    assert_eq!(row.intermediate_f32, 0x7fc0_0000);
                    continue;
                }
                assert_eq!(half_round(dtype, u64::from(row.intermediate_f32)), row.staged);
                if exact_special(dtype, op, row.input) {
                    assert_eq!(row.allowed_low, row.staged);
                    assert_eq!(row.allowed_high, row.staged);
                    continue;
                }
                let center = ordered_rank(DType::F32, u64::from(row.intermediate_f32));
                let budget = op.budget(DType::F32);
                let mut image: Vec<_> = (center - budget..=center + budget)
                    .map(|r| unrank(DType::F32, r))
                    .filter(|&b| !is_nan(DType::F32, b))
                    .map(|b| half_round(dtype, b))
                    .collect();
                image.sort_unstable_by_key(|&b| ordered_rank(dtype, b));
                image.dedup();
                let low = ordered_rank(dtype, row.allowed_low);
                let high = ordered_rank(dtype, row.allowed_high);
                assert_eq!(image, (low..=high).map(|r| unrank(dtype, r)).collect::<Vec<_>>());
                for actual in image {
                    let c = compare(dtype, op, &row, actual);
                    assert!(c.envelope_member);
                    assert_eq!(c.passed, !c.classification_failure && !c.exact_special_failure && !c.codomain_failure);
                }
            }
            assert_eq!(
                differences,
                if dtype == DType::BF16 {
                    0
                } else if op == Operation::Tan {
                    6
                } else {
                    2
                }
            );
            // These are exhaustive observations about this fixture, not generic
            // invariants of staged half arithmetic or actual backend outputs.
            assert_eq!((class_disagreements, status_disagreements), (0, 0));
            let cell = &accuracy.cells[DTYPES.iter().position(|&d| d == dtype).unwrap() * 4 + op.index()];
            assert_eq!(cell.reference_class_disagreements, class_disagreements);
            assert_eq!(cell.reference_nonzero_finite_disagreements, status_disagreements);
            assert!(accuracy.is_coherent());
        }
    }
}

#[test]
fn specials_domains_nonzero_sign_and_cosh_codomain_gate_independently() {
    for dtype in DTYPES {
        let sign = 1u64 << (dtype.size() * 8 - 1);
        for op in OPERATIONS {
            for input in [0, sign, infinity(dtype), infinity(dtype) | sign] {
                let row = expected_bits(dtype, op, input).unwrap();
                assert!(compare(dtype, op, &row, row.staged).passed);
                if !is_nan(dtype, row.staged) {
                    assert!(compare(dtype, op, &row, row.staged ^ sign).classification_failure);
                }
            }
            if op != Operation::Cosh {
                let row = expected_bits(dtype, op, 1).unwrap();
                assert!(compare(dtype, op, &row, 0).classification_failure);
            }
        }
        let singular = expected_bits(dtype, Operation::Log1p, sign | one(dtype)).unwrap();
        assert_eq!(singular.staged, sign | infinity(dtype));
        assert!(compare(dtype, Operation::Log1p, &singular, singular.staged - 1).classification_failure);
        let domain = expected_bits(dtype, Operation::Log1p, sign | (one(dtype) + 1)).unwrap();
        assert!(is_nan(dtype, domain.staged));
        assert!(!compare(dtype, Operation::Log1p, &domain, 0).passed);
        let floor = expected_bits(dtype, Operation::Cosh, 0).unwrap();
        assert!(compare(dtype, Operation::Cosh, &floor, one(dtype) - 1).codomain_failure);
        let normal = match dtype {
            DType::F64 => 1 << 52,
            DType::F16 => 1 << 10,
            DType::BF16 => 1 << 7,
            _ => 1 << 23,
        };
        let row = expected_bits(dtype, Operation::Sinh, normal).unwrap();
        let c = compare(dtype, Operation::Sinh, &row, normal - 1);
        assert!(!c.classification_failure);
        if !matches!(dtype, DType::F16 | DType::BF16) {
            assert!(c.passed);
        }
    }
}

#[test]
fn counterexample_stays_outside_one_step_log1p_target_and_f64_overflow_is_strict() {
    let row = expected_bits(DType::F32, Operation::Log1p, 0x3e7f_fffb).unwrap();
    assert_eq!(row.ideal, 0x3e64_7fba);
    let c = compare(DType::F32, Operation::Log1p, &row, 0x3e64_7fbc);
    assert!(!c.passed && c.numerical_failure);
    assert_eq!(c.staged_ulp, Some(2));
    for op in [Operation::Sinh, Operation::Cosh] {
        let finite = expected_bits(DType::F64, op, 0x4086_33ce_8fb9_f87d).unwrap();
        let overflow = expected_bits(DType::F64, op, 0x4086_33ce_8fb9_f87e).unwrap();
        assert_eq!(finite.ideal, 0x7fef_ffff_ffff_fd3b);
        assert_eq!(overflow.ideal, infinity(DType::F64));
        assert!(compare(DType::F64, op, &finite, infinity(DType::F64)).classification_failure);
        assert!(compare(DType::F64, op, &overflow, infinity(DType::F64) - 1).classification_failure);
    }
    assert!(expected_bits(DType::F64, Operation::Sinh, 0x0123_4567_89ab_cdef).is_none());
    let nan = expected_bits(DType::F64, Operation::Sinh, 0xfff5_4321_89ab_cdef).unwrap();
    assert!(compare(DType::F64, Operation::Sinh, &nan, 0x7ff0_0000_0000_0001).passed);
}

#[test]
fn wide_targets_preserve_each_operation_budget_and_genuine_destination_precision() {
    for dtype in [DType::F64, DType::F32, DType::Flex32] {
        let budgets = if dtype == DType::F64 { [1, 2, 1, 2] } else { [1, 3, 2, 4] };
        for op in OPERATIONS {
            let row = expected_bits(dtype, op, one(dtype)).unwrap();
            let budget = budgets[op.index()];
            assert!(compare(dtype, op, &row, row.staged + budget).passed);
            assert!(compare(dtype, op, &row, row.staged - budget).passed);
            assert!(compare(dtype, op, &row, row.staged + budget + 1).numerical_failure);
            assert!(compare(dtype, op, &row, row.staged - budget - 1).numerical_failure);
            if dtype == DType::F64 {
                let narrowed = f64::from_bits(row.staged) as f32 as f64;
                assert!(!compare(dtype, op, &row, narrowed.to_bits()).passed);
            }
        }
    }
}

#[test]
fn dual_transition_matrices_derive_distinct_pair_counts_in_both_crossing_directions() {
    let dtype = DType::F16;
    let baseline = expected_bits(dtype, Operation::Sinh, one(dtype)).unwrap();
    let mut accuracy = Accuracy::default();
    // Synthetic references exercise counter logic for possible status crossings.
    // They do not assert that this exact mathematical fixture has such a row.
    for (ideal, staged, actual) in [
        (infinity(dtype), one(dtype), one(dtype)),
        (0, 1, 1),
        (one(dtype), infinity(dtype), infinity(dtype)),
    ] {
        let row = Reference {
            ideal,
            staged,
            allowed_low: staged,
            allowed_high: staged,
            ..baseline
        };
        assert!(accuracy.record(dtype, Operation::Sinh, &row, actual).passed);
    }
    let cell = &accuracy.cells[3 * 4 + Operation::Sinh.index()];
    assert_eq!(cell.finite_pairs, 2);
    assert_eq!(cell.ideal_finite_pairs, 0);
    assert_eq!(cell.reference_class_disagreements, 3);
    assert_eq!(cell.reference_nonzero_finite_disagreements, 3);
    assert_eq!(cell.max_ideal_ulp, None);
    assert!(accuracy.is_coherent());
    accuracy.record_unreferenced(dtype, Operation::Sinh, one(dtype));
    assert!(accuracy.is_coherent());
    assert_eq!(accuracy.failures, 1);
    let cell = &mut accuracy.cells[3 * 4 + Operation::Sinh.index()];
    assert_eq!(cell.unreferenced_observed_classes[Class::Normal as usize], 1);
    cell.ideal_transitions[Class::Nan as usize][Class::Normal as usize] += 1;
    assert!(!accuracy.is_coherent());
}

const RECIPES: [[Operation; 3]; 8] = [
    [Operation::Log1p; 3],
    [Operation::Sinh; 3],
    [Operation::Cosh; 3],
    [Operation::Tan; 3],
    [Operation::Sinh, Operation::Log1p, Operation::Tan],
    [Operation::Cosh, Operation::Log1p, Operation::Sinh],
    [Operation::Tan, Operation::Cosh, Operation::Log1p],
    [Operation::Log1p, Operation::Sinh, Operation::Cosh],
];

fn independent_chain_finiteness(dtype: DType, operations: &[Operation], seed: u64) -> bool {
    let mut level = vec![seed];
    let mut finite = true;
    let mut nodes = 1;
    for &op in operations {
        let mut next = Vec::new();
        for bits in level {
            let row = expected_bits(dtype, op, bits).unwrap();
            finite &= is_finite(dtype, bits) && is_finite(dtype, row.ideal);
            next.extend(successors(dtype, op, bits));
        }
        next.sort_unstable();
        next.dedup();
        nodes += next.len();
        assert!(nodes <= 820);
        level = next;
    }
    finite
}

#[test]
fn pure_expected_matrix_counts_each_actual_operation_in_selected_mixed_chains() {
    for finite in [false, true] {
        for filter in [None, Some(Operation::Log1p), Some(Operation::Sinh), Some(Operation::Cosh), Some(Operation::Tan)] {
            let options = Options {
                finite,
                op: filter,
                ..Options::default()
            };
            let mut expected = [[0; 4]; 5];
            let mut chain_counts = [[0; 4]; 5];
            let mut chains = 0;
            for (fi, dtype) in DTYPES.into_iter().enumerate() {
                for op in OPERATIONS {
                    let rows = selected_rows(dtype, op, &options);
                    if !rows.is_empty() {
                        // Independent geometric count: both API routes, four
                        // length families, rank 0, seven 21-value permutations,
                        // a 28-value expand and a 24-value rank-8 expand.
                        let layout = 4 * [0, 1, 2, 3, 127, 128, 129, 255, 256, 257].iter().sum::<usize>() + 1 + 7 * 21 + 28 + 24;
                        expected[fi][op.index()] += 2 * (rows.len() + layout);
                    }
                }
                let sign = 1 << (dtype.size() * 8 - 1);
                for recipe in RECIPES {
                    if filter.is_some_and(|op| !recipe.contains(&op)) {
                        continue;
                    }
                    for seed in [sign, sign | one(dtype), 1, one(dtype), infinity(dtype) - 1] {
                        let all_finite = independent_chain_finiteness(dtype, &recipe, seed);
                        if finite && !all_finite {
                            continue;
                        }
                        chains += 1;
                        for op in recipe {
                            expected[fi][op.index()] += 1;
                            chain_counts[fi][op.index()] += 1;
                        }
                    }
                }
            }
            let summary = expected_summary(&options);
            assert_eq!(summary.expected_value_counts, expected);
            assert_eq!(summary.chain_value_counts, chain_counts);
            assert_eq!(summary.chain_cases, chains);
            assert_eq!(summary.chain_steps, chains * 3);
            assert_eq!(summary.values, expected.iter().flatten().sum::<usize>());
            let skip = expected_summary(&Options {
                skip_rank_zero: true,
                ..options
            });
            for (fi, counts) in expected.iter().enumerate() {
                for op in OPERATIONS {
                    let primitive = filter.is_none_or(|selected| selected == op);
                    assert_eq!(skip.expected_value_counts[fi][op.index()], counts[op.index()] - 2 * usize::from(primitive));
                }
            }
        }
    }
    let all = expected_summary(&Options::default());
    assert_eq!(all.tensor_rows, 700819);
    assert_eq!(all.chain_cases, 200);
    let only_log = expected_summary(&Options {
        op: Some(Operation::Log1p),
        ..Options::default()
    });
    assert!(only_log.expected_value_counts.iter().all(|counts| counts[Operation::Sinh.index()] > 0));
    let no_flex = expected_summary(&Options {
        no_flex32: true,
        ..Options::default()
    });
    assert_eq!(no_flex.expected_value_counts[2], [0; 4]);
}

#[test]
fn mandatory_reference_class_populations_bind_layout_repetitions_and_exclude_chains() {
    let selections = [
        Options::default(),
        Options {
            skip_rank_zero: true,
            ..Options::default()
        },
        Options {
            finite: true,
            ..Options::default()
        },
        Options {
            finite: true,
            skip_rank_zero: true,
            no_flex32: true,
            ..Options::default()
        },
        Options {
            dtype: Some(DType::F64),
            op: Some(Operation::Log1p),
            ..Options::default()
        },
        Options {
            dtype: Some(DType::F16),
            op: Some(Operation::Sinh),
            ..Options::default()
        },
        Options {
            dtype: Some(DType::BF16),
            op: Some(Operation::Cosh),
            finite: true,
            ..Options::default()
        },
        Options {
            dtype: Some(DType::Flex32),
            no_flex32: true,
            ..Options::default()
        },
    ];
    for options in selections {
        let summary = expected_summary(&options);
        let mut staged = [[[0; 5]; 4]; 5];
        let mut ideal = [[[0; 5]; 4]; 5];
        for (fi, dtype) in DTYPES.into_iter().enumerate() {
            for op in OPERATIONS {
                let selected = selected_rows(dtype, op, &options);
                if selected.is_empty() {
                    continue;
                }
                // Independently count which selected rows appear in each
                // logical geometry. Source guards and empty expands contribute
                // no operation outputs. Both API routes have identical weights.
                let mut weights = vec![2; selected.len()];
                let select = |i: usize| (i * 4051 + i / 7) % selected.len();
                for length in [0, 1, 2, 3, 127, 128, 129, 255, 256, 257] {
                    for i in 0..length {
                        weights[select(i)] += 4; // contiguous and reversed
                        weights[select(i + 1)] += 2; // offset
                        weights[select(1 + 2 * i)] += 2; // stepped and reversed
                    }
                }
                if !options.skip_rank_zero {
                    weights[select(0)] += 2;
                }
                for i in 0..21 {
                    weights[select(i)] += 14;
                } // ranks 2..8
                for i in 0..7 {
                    weights[select(i)] += 8;
                } // four-way expand
                for i in 0..12 {
                    weights[select(i)] += 4;
                } // rank-8 expand
                for (row, weight) in selected.iter().zip(weights) {
                    staged[fi][op.index()][classify(dtype, row.staged) as usize] += weight;
                    ideal[fi][op.index()][classify(dtype, row.ideal) as usize] += weight;
                }
            }
        }
        assert_eq!(summary.expected_staged_class_counts, staged);
        assert_eq!(summary.expected_ideal_class_counts, ideal);
        for fi in 0..5 {
            for oi in 0..4 {
                let nonchain = summary.expected_value_counts[fi][oi] - summary.chain_value_counts[fi][oi];
                assert_eq!(staged[fi][oi].iter().sum::<usize>(), nonchain);
                assert_eq!(ideal[fi][oi].iter().sum::<usize>(), nonchain);
            }
        }
    }
    let all = expected_summary(&Options::default());
    assert_eq!(all.chain_value_counts, [[35, 30, 30, 25]; 5]);
    // Relabeling both matrices as all-NaN can satisfy their internal sums, but
    // cannot satisfy the mandatory frozen reference populations in any cell.
    for fi in 0..5 {
        for oi in 0..4 {
            let mut relabeled = [0; 5];
            relabeled[Class::Nan as usize] = all.expected_value_counts[fi][oi];
            for counts in [&all.expected_staged_class_counts, &all.expected_ideal_class_counts] {
                assert!(
                    (0..5)
                        .any(|class| { relabeled[class] < counts[fi][oi][class] || relabeled[class] > counts[fi][oi][class] + all.chain_value_counts[fi][oi] })
                );
            }
        }
    }
}

#[test]
fn metrics_keep_failure_maxima_separate_and_examples_remain_bounded() {
    let row = expected_bits(DType::F32, Operation::Log1p, 0x3e7f_fffb).unwrap();
    let mut accuracy = Accuracy::default();
    for offset in [0, 1, 2, 3, 16, 1024] {
        accuracy.record(DType::F32, Operation::Log1p, &row, row.staged + offset);
    }
    accuracy.record(DType::F32, Operation::Log1p, &row, 0);
    let cell = &accuracy.cells[4];
    assert_eq!((cell.values, cell.finite_pairs, cell.passing_finite_pairs), (7, 6, 2));
    assert_eq!(cell.max_ulp, Some(1024));
    assert_eq!(cell.passing_max_ulp, Some(1));
    assert!(accuracy.is_coherent());
    let mut failures = Failures::default();
    let mut rendered = 0;
    for _ in 0..10000 {
        failures.record(|| {
            rendered += 1;
            String::new()
        });
    }
    assert_eq!((failures.count, failures.examples.len(), rendered), (10000, 12, 12));
}

#[test]
#[ignore = "requires MPFR 4.2.2; reproduces baseline streams, full closure, binary, index, and manifest"]
fn pinned_generator_reproduces_all_artifacts() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let status = std::process::Command::new("python3")
        .arg("-B")
        .arg(root.join("tools/nvidia/typed-unary-generate.py"))
        .arg("--check")
        .current_dir(root)
        .status()
        .unwrap();
    assert!(status.success());
}
