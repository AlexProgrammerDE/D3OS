#![cfg(feature = "training")]
extern crate alloc;

#[path = "../src/validation/typed_trig.rs"]
#[allow(dead_code)]
mod typed_trig;
use burn::backend::{DType, bf16, f16};
use typed_trig::{
    Accuracy, Class, DTYPES, Failures, OPERATIONS, Operation, Options, Report, classify, compare, expected_bits, expected_summary, is_finite, ordered_rank,
    pack, rows, selected_rows, unpack,
};

fn unrank(dtype: DType, rank: u64) -> u64 {
    let sign = 1u64 << (dtype.size() * 8 - 1);
    if rank & sign != 0 { rank & (sign - 1) } else { !rank & (2 * sign - 1) }
}
fn half_round(dtype: DType, bits: u64) -> u64 {
    let value = f32::from_bits(bits as u32);
    match dtype {
        DType::F16 => u64::from(f16::from_f32(value).to_bits()),
        DType::BF16 => u64::from(bf16::from_f32(value).to_bits()),
        _ => unreachable!(),
    }
}

#[test]
fn all_half_encodings_have_independent_schedules_and_exact_neighbor_images() {
    let mut differences = 0;
    let mut nonsingletons = 0;
    for dtype in [DType::F16, DType::BF16] {
        for op in OPERATIONS {
            let references = rows(dtype, op);
            assert_eq!(references.len(), 65536);
            let inputs: Vec<_> = references.iter().map(|r| r.input).collect();
            assert_eq!(inputs, (0..65536).collect::<Vec<_>>());
            assert_eq!(unpack(dtype, &pack(dtype, &inputs)), inputs);
            for row in references {
                assert_eq!(expected_bits(dtype, op, row.input), Some(row));
                assert!(compare(dtype, op, &row, row.staged).passed);
                differences += usize::from(row.ideal != row.staged);
                if classify(dtype, row.staged) == Class::Nan {
                    assert_eq!(classify(DType::F32, u64::from(row.intermediate_f32)), Class::Nan);
                }
                if matches!(classify(dtype, row.input), Class::Zero | Class::Nan | Class::Infinity) {
                    continue;
                }
                assert_eq!(half_round(dtype, u64::from(row.intermediate_f32)), row.staged);
                let center = ordered_rank(DType::F32, u64::from(row.intermediate_f32));
                let mut image: Vec<_> = (center - 2..=center + 2).map(|rank| half_round(dtype, unrank(DType::F32, rank))).collect();
                image.sort_unstable_by_key(|&bits| ordered_rank(dtype, bits));
                image.dedup();
                let low = ordered_rank(dtype, row.allowed_low);
                let high = ordered_rank(dtype, row.allowed_high);
                assert_eq!(image, (low..=high).map(|rank| unrank(dtype, rank)).collect::<Vec<_>>());
                nonsingletons += usize::from(image.len() > 1);
                for actual in image {
                    let result = compare(dtype, op, &row, actual);
                    assert!(result.envelope_member);
                    // Codomain and class/sign constraints apply independently of membership.
                    assert_eq!(result.passed, !result.codomain_failure && !result.classification_failure);
                }
                for rank in [low - 1, high + 1] {
                    let result = compare(dtype, op, &row, unrank(dtype, rank));
                    assert!(!result.passed && !result.envelope_member && result.numerical_failure);
                }
            }
        }
    }
    assert!(differences > 0 && nonsingletons > differences);
}

#[test]
fn class_sign_exact_specials_and_codomain_are_independent_gates() {
    for dtype in DTYPES {
        let sign = 1u64 << (dtype.size() * 8 - 1);
        for op in OPERATIONS {
            for input in [0, sign] {
                let row = expected_bits(dtype, op, input).unwrap();
                assert!(compare(dtype, op, &row, row.staged).passed);
                assert!(compare(dtype, op, &row, row.staged ^ sign).classification_failure);
                let one_step_wrong = row.staged + 1;
                assert!(compare(dtype, op, &row, one_step_wrong).exact_special_failure);
            }
            for row in rows(dtype, op).into_iter().filter(|row| classify(dtype, row.staged) == Class::Nan) {
                assert!(compare(dtype, op, &row, row.staged ^ sign).passed);
                assert!(compare(dtype, op, &row, 0).classification_failure);
            }
            if op != Operation::Cos {
                let row = expected_bits(dtype, op, 1).unwrap();
                let ftz = compare(dtype, op, &row, 0);
                assert!(!ftz.passed && ftz.classification_failure);
                assert_eq!(ftz.staged_ulp, None);
            }
        }
        let one = expected_bits(dtype, Operation::Cos, 0).unwrap().staged;
        // A saturated finite input is not an exact-special input. Its allowed
        // numerical neighbor still fails if it exceeds the unit codomain.
        let large = match dtype {
            DType::F64 => 0x4050_0000_0000_0000,
            DType::F16 => 0x5400,
            DType::BF16 => 0x4280,
            _ => 0x4280_0000,
        };
        let row = expected_bits(dtype, Operation::Tanh, large).unwrap();
        let result = compare(dtype, Operation::Tanh, &row, one + 1);
        assert!(!result.passed && result.codomain_failure && !result.classification_failure);
    }
    let row = expected_bits(DType::F32, Operation::Sin, 0x0080_0000).unwrap();
    let result = compare(DType::F32, Operation::Sin, &row, 0x007f_ffff);
    assert!(result.passed && !result.classification_failure);
    assert_eq!(result.staged_ulp, Some(1));
    let mut accuracy = Accuracy::default();
    accuracy.record(DType::F32, Operation::Sin, &row, 0x007f_ffff);
    assert_eq!(accuracy.cells[4].transitions[Class::Normal as usize][Class::Subnormal as usize], 1);
    assert!(accuracy.is_coherent());
}

#[test]
fn wide_comparators_enforce_per_operation_steps_at_destination_precision() {
    for dtype in [DType::F64, DType::F32, DType::Flex32] {
        let one = match dtype {
            DType::F64 => 0x3ff0_0000_0000_0000,
            _ => 0x3f80_0000,
        };
        for op in OPERATIONS {
            let row = expected_bits(dtype, op, one).unwrap();
            let budget = if dtype == DType::F64 && op == Operation::Tanh { 1 } else { 2 };
            for delta in 0..=budget {
                assert!(compare(dtype, op, &row, row.staged + delta).passed);
                assert!(compare(dtype, op, &row, row.staged - delta).passed);
            }
            assert!(!compare(dtype, op, &row, row.staged + budget + 1).passed);
            if dtype == DType::F64 {
                let narrowed = (f64::from_bits(row.staged) as f32) as f64;
                assert!(!compare(dtype, op, &row, narrowed.to_bits()).passed);
            }
        }
    }
    assert_eq!(
        ordered_rank(DType::F64, 0xbff0_0000_0000_0000).abs_diff(ordered_rank(DType::F64, 0xbfef_ffff_ffff_ffff)),
        1
    );
}

#[test]
fn counts_and_filters_describe_work_without_observed_accuracy() {
    let all = expected_summary(&Options::default());
    assert_eq!(all.tensor_rows, 439735);
    assert_eq!(
        all.selected_dtype_operation_counts,
        [[10099, 10117, 10115], [2692, 2702, 2700], [2692, 2702, 2700], [65536; 3], [65536; 3]]
    );
    assert_eq!(all.chain_cases, 45);
    assert_eq!(all.chain_steps, 135);
    let skip = expected_summary(&Options {
        skip_rank_zero: true,
        ..Options::default()
    });
    assert_eq!(all.cases - skip.cases, 30);
    assert_eq!(all.tensor_rows, skip.tensor_rows);
    assert_eq!(all.values - skip.values, 30);
    let no_flex = expected_summary(&Options {
        no_flex32: true,
        ..Options::default()
    });
    assert_eq!(all.tensor_rows - no_flex.tensor_rows, 8094);
    for dtype in DTYPES {
        for op in OPERATIONS {
            let options = Options {
                dtype: Some(dtype),
                op: Some(op),
                finite: true,
                ..Options::default()
            };
            let selected = selected_rows(dtype, op, &options);
            assert!(selected.iter().all(|r| is_finite(dtype, r.input) && is_finite(dtype, r.ideal)));
            assert_eq!(expected_summary(&options).tensor_rows, selected.len());
        }
    }
    assert_eq!(
        expected_summary(&Options {
            dtype: Some(DType::Flex32),
            no_flex32: true,
            ..Options::default()
        })
        .cases,
        0
    );
    let mut report = Report {
        summary: all.clone(),
        ..Report::default()
    };
    let row = expected_bits(DType::F64, Operation::Sin, 0x3ff0_0000_0000_0000).unwrap();
    report.accuracy.record(DType::F64, Operation::Sin, &row, row.staged + 1);
    assert_eq!(report.summary, all);
    report.accuracy.record(DType::F64, Operation::Sin, &row, 0);
    assert!(!report.is_success());
    assert!(report.accuracy.is_coherent());
}

#[test]
fn metrics_include_all_failures_and_passing_histograms_have_independent_maxima() {
    let row = expected_bits(DType::F64, Operation::Sin, 0x3ff0_0000_0000_0000).unwrap();
    let mut accuracy = Accuracy::default();
    for offset in [0, 1, 2, 3, 4, 16, 1024] {
        accuracy.record(DType::F64, Operation::Sin, &row, row.staged + offset);
    }
    accuracy.record(DType::F64, Operation::Sin, &row, 0);
    let cell = accuracy.cells[1];
    assert_eq!(cell.values, 8);
    assert_eq!(cell.failures, 5);
    assert_eq!(cell.finite_pairs, 7);
    assert_eq!(cell.max_ulp, Some(1024));
    assert_eq!(cell.passing_finite_pairs, 3);
    assert_eq!(cell.passing_max_ulp, Some(2));
    assert_eq!(cell.passing_histogram, [1, 1, 1, 0, 0, 0, 0]);
    assert_eq!(cell.histogram, [1, 1, 1, 2, 0, 1, 1]);
    assert!(accuracy.is_coherent());
    accuracy.cells[1].max_ulp = Some(2);
    assert!(!accuracy.is_coherent());
    let mut failures = Failures::default();
    let mut rendered = 0;
    for _ in 0..10000 {
        failures.record(|| {
            rendered += 1;
            String::new()
        });
    }
    assert_eq!(failures.count, 10000);
    assert_eq!(failures.examples.len(), 12);
    assert_eq!(rendered, 12);
}

#[test]
#[ignore = "requires MPFR 4.2.2; reproduces every reference, baseline stream, closure, binary index and manifest"]
fn fixture_generator_reproduces_every_checked_in_artifact() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let status = std::process::Command::new("python3")
        .arg("-B")
        .arg(root.join("tools/nvidia/typed-trig-generate.py"))
        .arg("--check")
        .current_dir(root)
        .status()
        .unwrap();
    assert!(status.success());
}
