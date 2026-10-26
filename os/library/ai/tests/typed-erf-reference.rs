#![cfg(feature = "training")]
use burn::backend::{DType, bf16, f16};
use d3os_ai::validation::typed_erf;
use typed_erf::{
    Accuracy, Class, DTYPES, Failures, OPERATIONS, Operation, Options, Report, classify, compare, exact_special, expected_bits, expected_summary, is_finite,
    is_nan, ordered_rank, pack, rows, selected_rows,
};

const ERF: Operation = Operation::Erf;
fn sign(dtype: DType) -> u64 {
    1 << (dtype.size() * 8 - 1)
}
fn one(dtype: DType) -> u64 {
    match dtype {
        DType::F64 => 0x3ff0_0000_0000_0000,
        DType::F16 => 0x3c00,
        DType::BF16 => 0x3f80,
        _ => 0x3f80_0000,
    }
}
fn infinity(dtype: DType) -> u64 {
    match dtype {
        DType::F64 => 0x7ff0_0000_0000_0000,
        DType::F16 => 0x7c00,
        DType::BF16 => 0x7f80,
        _ => 0x7f80_0000,
    }
}
fn unrank(dtype: DType, rank: u64) -> u64 {
    let sign = sign(dtype);
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
fn successors(dtype: DType, input: u64) -> Vec<u64> {
    let row = expected_bits(dtype, ERF, input).expect("immutable chain reference");
    if exact_special(dtype, ERF, input) {
        return vec![row.staged];
    }
    let center = ordered_rank(dtype, row.staged);
    let ranks = if matches!(dtype, DType::F16 | DType::BF16) {
        ordered_rank(dtype, row.allowed_low)..=ordered_rank(dtype, row.allowed_high)
    } else {
        center - 2..=center + 2
    };
    ranks.map(|r| unrank(dtype, r)).filter(|&b| compare(dtype, ERF, &row, b).passed).collect()
}

#[test]
fn exhaustive_half_images_staging_and_oddness_have_independent_populations() {
    for dtype in [DType::F16, DType::BF16] {
        let references = rows(dtype, ERF);
        assert_eq!(references.len(), 65536);
        assert!(references.iter().enumerate().all(|(i, r)| r.input == i as u64));
        let inputs: Vec<_> = references.iter().map(|r| r.input).collect();
        assert_eq!(typed_erf::unpack(dtype, &pack(dtype, &inputs)), inputs);
        let mut accuracy = Accuracy::default();
        let mut differences = 0;
        let mut nonsingle = 0;
        let mut input_classes = [0; 5];
        for row in &references {
            assert_eq!(expected_bits(dtype, ERF, row.input), Some(*row));
            assert!(accuracy.record(dtype, ERF, row, row.staged).passed);
            input_classes[classify(dtype, row.input) as usize] += 1;
            differences += usize::from(row.ideal != row.staged);
            nonsingle += usize::from(row.allowed_low != row.allowed_high);
            if is_nan(dtype, row.staged) {
                assert_eq!(row.intermediate_f32, 0x7fc0_0000);
                continue;
            }
            assert_eq!(half_round(dtype, u64::from(row.intermediate_f32)), row.staged);
            if exact_special(dtype, ERF, row.input) {
                assert_eq!((row.allowed_low, row.allowed_high), (row.staged, row.staged));
                continue;
            }
            let s = row.intermediate_f32 & 0x8000_0000;
            let magnitude = row.intermediate_f32 & 0x7fff_ffff;
            let mut image: Vec<_> = (magnitude.saturating_sub(2).max(1)..=magnitude.saturating_add(2).min(0x3f80_0000))
                .map(|b| half_round(dtype, u64::from(s | b)))
                .collect();
            image.sort_unstable_by_key(|&b| ordered_rank(dtype, b));
            image.dedup();
            let encoded_image: Vec<_> = (ordered_rank(dtype, row.allowed_low)..=ordered_rank(dtype, row.allowed_high))
                .map(|r| unrank(dtype, r))
                .collect();
            assert_eq!(image, encoded_image);
            assert!(image.iter().all(|&b| compare(dtype, ERF, row, b).passed));
        }
        for b in 0..32768 {
            if is_finite(dtype, b) {
                assert!(accuracy.record_oddness(dtype, references[b as usize].staged, references[(b | 0x8000) as usize].staged));
            }
        }
        assert_eq!((differences, nonsingle), if dtype == DType::F16 { (2, 10) } else { (0, 0) });
        assert_eq!(
            input_classes,
            if dtype == DType::F16 {
                [2, 2046, 61440, 2, 2046]
            } else {
                [2, 254, 65024, 2, 254]
            }
        );
        assert!(accuracy.is_coherent());
    }
    for (input, ideal, staged) in [(0x1612, 0x16d9, 0x16da), (0x9612, 0x96d9, 0x96da)] {
        let row = expected_bits(DType::F16, ERF, input).unwrap();
        assert_eq!((row.ideal, row.staged), (ideal, staged));
        let mut accuracy = Accuracy::default();
        accuracy.record(DType::F16, ERF, &row, staged);
        let cell = &accuracy.cells[3];
        assert_eq!((cell.max_ulp, cell.max_ideal_ulp), (Some(0), Some(1)));
        assert_eq!((cell.worst_input_bits, cell.worst_ideal_input_bits), (Some(input), Some(input)));
        assert!(accuracy.is_coherent());
    }
}

#[test]
fn special_sign_nonzero_codomain_and_step_gates_remain_separate() {
    for dtype in DTYPES {
        for input in [0, sign(dtype), infinity(dtype), infinity(dtype) | sign(dtype)] {
            let row = expected_bits(dtype, ERF, input).unwrap();
            assert!(compare(dtype, ERF, &row, row.staged).passed);
            assert!(compare(dtype, ERF, &row, row.staged ^ sign(dtype)).exact_special_failure);
        }
        let row = expected_bits(dtype, ERF, 1).unwrap();
        assert!(compare(dtype, ERF, &row, 0).classification_failure);
        assert!(compare(dtype, ERF, &row, row.staged | sign(dtype)).classification_failure);
        let row = expected_bits(dtype, ERF, infinity(dtype) - 1).unwrap();
        let outside = compare(dtype, ERF, &row, one(dtype) + 1);
        assert!(outside.codomain_failure);
        assert!(!outside.passed);
        assert!(compare(dtype, ERF, &row, infinity(dtype)).classification_failure);
        if dtype != DType::F64 {
            assert!(compare(dtype, ERF, &row, row.staged | (1 << (dtype.size() * 8))).classification_failure);
        }
        let nan = expected_bits(dtype, ERF, infinity(dtype) | 1).unwrap();
        assert!(compare(dtype, ERF, &nan, infinity(dtype) | sign(dtype) | 2).passed);
        assert!(!compare(dtype, ERF, &nan, one(dtype)).passed);
        if !matches!(dtype, DType::F16 | DType::BF16) {
            let row = expected_bits(dtype, ERF, one(dtype)).unwrap();
            assert!(compare(dtype, ERF, &row, row.staged + 2).passed);
            assert!(compare(dtype, ERF, &row, row.staged + 3).numerical_failure);
            let boundary = rows(dtype, ERF)
                .into_iter()
                .find(|r| classify(dtype, r.staged) == Class::Subnormal && classify(dtype, r.staged + 2) == Class::Normal)
                .unwrap();
            let c = compare(dtype, ERF, &boundary, boundary.staged + 2);
            assert!(c.passed);
            assert!(!c.classification_failure);
        }
    }
}

#[test]
fn wide_saturation_adjacency_and_genuine_f64_bits_are_frozen() {
    for (dtype, first) in [
        (DType::F64, 0x4017_afb4_8dc9_6627),
        (DType::F32, 0x407a_d445),
        (DType::Flex32, 0x407a_d445),
        (DType::F16, 0x4131),
        (DType::BF16, 0x400d),
    ] {
        for s in [0, sign(dtype)] {
            let before = expected_bits(dtype, ERF, (first - 1) | s).unwrap();
            let at = expected_bits(dtype, ERF, first | s).unwrap();
            assert_eq!(before.ideal & !sign(dtype), one(dtype) - 1);
            assert_eq!(at.ideal, one(dtype) | s);
        }
    }
    let a = expected_bits(DType::F64, ERF, one(DType::F64)).unwrap();
    let b = expected_bits(DType::F64, ERF, one(DType::F64) + 1).unwrap();
    assert_ne!(a.ideal, b.ideal);
}

fn seeds(dtype: DType) -> Vec<u64> {
    let witness = if dtype == DType::F16 {
        0x1612
    } else {
        match dtype {
            DType::F64 => 0x3fc0_0000_0000_0000,
            DType::BF16 => 0x3e00,
            _ => 0x3e00_0000,
        }
    };
    let mut values = Vec::new();
    for b in [
        0,
        1,
        one(dtype),
        infinity(dtype) - 1,
        infinity(dtype),
        infinity(dtype) | ((infinity(dtype) & infinity(dtype).wrapping_neg()) >> 1),
        witness,
    ] {
        values.extend([b, b | sign(dtype)]);
    }
    values
}
fn closure_is_finite(dtype: DType, seed: u64) -> bool {
    let mut level = vec![seed];
    let mut finite = true;
    for _ in 0..3 {
        finite &= level.iter().all(|&b| is_finite(dtype, b));
        level = level.into_iter().flat_map(|b| successors(dtype, b)).collect();
        level.sort_unstable();
        level.dedup();
        assert!(level.len() <= 125);
    }
    finite
}
#[test]
fn expected_summary_is_pure_and_binds_every_cell_and_mandatory_class_population() {
    for options in [
        Options::default(),
        Options {
            finite: true,
            ..Options::default()
        },
        Options {
            no_flex32: true,
            skip_rank_zero: true,
            ..Options::default()
        },
        Options {
            dtype: Some(DType::F16),
            ..Options::default()
        },
        Options {
            dtype: Some(DType::Flex32),
            no_flex32: true,
            ..Options::default()
        },
    ] {
        let summary = expected_summary(&options);
        let mut expected = [[0; 1]; 5];
        let mut chain_count = 0;
        for (fi, dtype) in DTYPES.into_iter().enumerate() {
            let selected = selected_rows(dtype, ERF, &options);
            if selected.is_empty() {
                continue;
            }
            let layout = 4 * [0, 1, 2, 3, 127, 128, 129, 255, 256, 257].iter().sum::<usize>() + usize::from(!options.skip_rank_zero) + 7 * 21 + 28 + 28 + 24;
            expected[fi][0] = 2 * (selected.len() + layout);
            let mut weights = vec![2; selected.len()];
            let select = |i: usize| (i * 4051 + i / 7) % selected.len();
            for length in [0, 1, 2, 3, 127, 128, 129, 255, 256, 257] {
                for i in 0..length {
                    weights[select(i)] += 4;
                    weights[select(i + 1)] += 2;
                    weights[select(1 + 2 * i)] += 2;
                }
            }
            if !options.skip_rank_zero {
                weights[select(0)] += 2;
            }
            for i in 0..21 {
                weights[select(i)] += 14;
            }
            for i in 0..7 {
                weights[select(i)] += 16;
            }
            for i in 0..12 {
                weights[select(i)] += 4;
            }
            let mut staged = [0; 5];
            let mut ideal = [0; 5];
            for (row, weight) in selected.iter().zip(weights) {
                staged[classify(dtype, row.staged) as usize] += weight;
                ideal[classify(dtype, row.ideal) as usize] += weight;
            }
            assert_eq!(summary.expected_staged_class_counts[fi][0], staged);
            assert_eq!(summary.expected_ideal_class_counts[fi][0], ideal);
            for seed in seeds(dtype) {
                if options.finite && !closure_is_finite(dtype, seed) {
                    continue;
                }
                if !options.finite {
                    closure_is_finite(dtype, seed);
                }
                expected[fi][0] += 3;
                chain_count += 1;
            }
        }
        assert_eq!(summary.expected_value_counts, expected);
        assert_eq!(summary.chain_cases, chain_count);
        assert_eq!(summary.chain_steps, chain_count * 3);
        assert_eq!(summary.values, expected.iter().flatten().sum::<usize>());
        let empty = summary.values == 0;
        assert_eq!(Report { summary, ..Report::default() }.is_success(), empty);
    }
    assert_eq!(OPERATIONS, [ERF]);
    assert_eq!(ERF.opcode(), 33);
}

#[test]
fn observed_invariants_reject_wrong_witness_metrics_unknowns_and_nonodd_results() {
    let row = expected_bits(DType::F32, ERF, one(DType::F32)).unwrap();
    let mut accuracy = Accuracy::default();
    for steps in [0, 1, 2, 3, 16, 1024] {
        accuracy.record(DType::F32, ERF, &row, row.staged + steps);
    }
    accuracy.record(DType::F32, ERF, &row, 0);
    assert_eq!((accuracy.cells[1].finite_pairs, accuracy.cells[1].passing_finite_pairs), (6, 3));
    assert_eq!((accuracy.cells[1].max_ulp, accuracy.cells[1].passing_max_ulp), (Some(1024), Some(2)));
    assert!(accuracy.is_coherent());
    for metric in [false, true] {
        let mut forged = accuracy.clone();
        if metric {
            forged.cells[1].worst_ideal_actual_bits = Some(row.staged);
        } else {
            forged.cells[1].worst_actual_bits = Some(row.staged);
        }
        assert!(!forged.is_coherent());
    }
    assert!(!accuracy.record_oddness(DType::F32, row.staged, (row.staged + 1) | sign(DType::F32)));
    assert_eq!(accuracy.cells[1].oddness_failures, 1);
    assert!(accuracy.is_coherent());
    accuracy.record_unreferenced(DType::F32, ERF, one(DType::F32));
    assert!(accuracy.is_coherent());
    let summary = expected_summary(&Options {
        dtype: Some(DType::F64),
        ..Options::default()
    });
    assert!(!accuracy.matches_summary(&summary));
    let mut failures = Failures::default();
    let mut rendered = 0;
    for _ in 0..1000 {
        failures.record(|| {
            rendered += 1;
            String::new()
        });
    }
    assert_eq!((failures.count, failures.examples.len(), rendered), (1000, 12, 12));
}

#[test]
fn internally_coherent_relabeling_cannot_replace_planned_reference_classes() {
    let summary = expected_summary(&Options {
        dtype: Some(DType::F32),
        ..Options::default()
    });
    let mut accuracy = Accuracy::default();
    let cell = &mut accuracy.cells[1];
    cell.values = summary.expected_value_counts[1][0];
    cell.oddness_checks = summary.expected_oddness_checks[1][0];
    cell.observed_classes[Class::Nan as usize] = cell.values;
    cell.transitions[Class::Nan as usize][Class::Nan as usize] = cell.values;
    cell.ideal_transitions[Class::Nan as usize][Class::Nan as usize] = cell.values;
    assert!(accuracy.is_coherent());
    assert!(!accuracy.matches_summary(&summary));
    assert!(
        !Report {
            summary,
            accuracy,
            ..Report::default()
        }
        .is_success()
    );
    let empty = expected_summary(&Options {
        dtype: Some(DType::Flex32),
        no_flex32: true,
        ..Options::default()
    });
    assert!(Accuracy::default().matches_summary(&empty));
    let row = expected_bits(DType::F64, ERF, one(DType::F64)).unwrap();
    let mut extra = Accuracy::default();
    extra.record(DType::F64, ERF, &row, row.staged);
    assert!(extra.is_coherent());
    assert!(!extra.matches_summary(&empty));
}

#[test]
#[ignore = "requires MPFR 4.2.2 and reproduces every fixture and closure input"]
fn pinned_generator_reproduces_all_artifacts() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let status = std::process::Command::new("python3")
        .arg("-B")
        .arg(root.join("tools/nvidia/typed-erf-generate.py"))
        .arg("--check")
        .current_dir(root)
        .status()
        .unwrap();
    assert!(status.success());
}
