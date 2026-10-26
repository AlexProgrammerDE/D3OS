#![cfg(feature = "training")]
extern crate alloc;

#[path = "../src/validation/typed_exp_log.rs"]
#[allow(dead_code)]
mod typed_exp_log;
use burn::backend::DType;
use typed_exp_log::{
    Accuracy, Class, DTYPES, Failures, OPERATIONS, Operation, Options, Report, classify, compare, expected_bits, expected_summary, is_finite, ordered_rank,
    pack, rows, selected_rows, unpack,
};

#[test]
fn every_half_encoding_has_both_independent_schedules_and_byte_exact_storage() {
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
            }
        }
    }
    let differences: Vec<_> = OPERATIONS
        .into_iter()
        .flat_map(|op| {
            rows(DType::F16, op)
                .into_iter()
                .filter(|r| r.ideal != r.staged)
                .map(move |r| (op, r.input, r.ideal, r.staged))
        })
        .collect();
    assert_eq!(
        differences,
        vec![
            (Operation::Exp, 0x1f79, 0x3c07, 0x3c08),
            (Operation::Exp, 0x25cf, 0x3c17, 0x3c18),
            (Operation::Log, 0x1d78, 0xc53b, 0xc53c)
        ]
    );
    for op in OPERATIONS {
        assert!(rows(DType::BF16, op).iter().all(|r| r.ideal == r.staged));
    }
}

#[test]
fn half_neighbor_image_accepts_both_roundings_but_rejects_next_destination() {
    for (op, input) in [(Operation::Exp, 0x1f79), (Operation::Exp, 0x25cf), (Operation::Log, 0x1d78)] {
        let row = expected_bits(DType::F16, op, input).unwrap();
        assert!(compare(DType::F16, op, &row, row.ideal).passed);
        assert!(compare(DType::F16, op, &row, row.staged).passed);
        for endpoint in [row.allowed_low, row.allowed_high] {
            assert!(compare(DType::F16, op, &row, endpoint).passed);
        }
        let positive = row.allowed_low & 0x8000 == 0;
        for outside in [
            if positive { row.allowed_low - 1 } else { row.allowed_low + 1 },
            if positive { row.allowed_high + 1 } else { row.allowed_high - 1 },
        ] {
            let comparison = compare(DType::F16, op, &row, outside);
            assert!(!comparison.passed && comparison.numerical_failure && !comparison.envelope_member);
        }
    }
}

#[test]
fn special_values_signs_and_ftz_fail_independently_of_the_ulp_budget() {
    for dtype in DTYPES {
        let sign = 1u64 << (dtype.size() * 8 - 1);
        for op in OPERATIONS {
            let references = rows(dtype, op);
            for row in references
                .iter()
                .filter(|r| matches!(classify(dtype, r.staged), Class::Nan | Class::Infinity | Class::Zero))
            {
                assert!(compare(dtype, op, row, row.staged).passed);
                let wrong = if classify(dtype, row.staged) == Class::Nan { 0 } else { row.staged ^ sign };
                assert!(compare(dtype, op, row, wrong).classification_failure);
            }
            if let Some(row) = references.iter().find(|r| classify(dtype, r.staged) == Class::Subnormal) {
                let comparison = compare(dtype, op, row, 0);
                assert!(!comparison.passed && comparison.classification_failure);
                assert_eq!(comparison.staged_ulp, None);
                let mut accuracy = Accuracy::default();
                accuracy.record(dtype, op, row, 0);
                let cell = &accuracy.cells[DTYPES.iter().position(|&d| d == dtype).unwrap() * 2 + op.index()];
                assert_eq!(cell.nonzero_to_zero, 1);
                assert_eq!(cell.finite_pairs, 0);
                assert_eq!(accuracy.failures, 1);
            }
        }
        let row = expected_bits(dtype, Operation::Exp, 0).unwrap();
        assert!(compare(dtype, Operation::Exp, &row, row.staged + 1).exact_special_failure);
    }
    let mut row = expected_bits(DType::F32, Operation::Exp, 1).unwrap();
    row.staged = 0x0080_0000;
    let boundary = compare(DType::F32, Operation::Exp, &row, 0x007f_ffff);
    assert!(boundary.passed);
    assert_eq!(boundary.staged_ulp, Some(1));
}

#[test]
fn wide_comparators_use_integer_steps_at_full_destination_precision() {
    let exp = expected_bits(DType::F64, Operation::Exp, 0x3ff0_0000_0000_0000).unwrap();
    assert_eq!(exp.ideal, 0x4005_bf0a_8b14_5769);
    for actual in [exp.ideal - 1, exp.ideal, exp.ideal + 1] {
        assert!(compare(DType::F64, Operation::Exp, &exp, actual).passed);
    }
    assert!(!compare(DType::F64, Operation::Exp, &exp, exp.ideal + 2).passed);
    assert!(!compare(DType::F64, Operation::Exp, &exp, f64::from(f32::from_bits(0x402d_f854)).to_bits()).passed);
    assert_eq!(
        expected_bits(DType::F64, Operation::Exp, 0x3ca0_0000_0000_0000).unwrap().ideal,
        0x3ff0_0000_0000_0001
    );
    let log = expected_bits(DType::F64, Operation::Log, 0x3ff0_0000_0000_0001).unwrap();
    assert_eq!(log.ideal, 0x3caf_ffff_ffff_ffff);
    assert!(!compare(DType::F64, Operation::Log, &log, 0).passed);
    for dtype in [DType::F32, DType::Flex32] {
        for op in OPERATIONS {
            let row = expected_bits(dtype, op, 0x4000_0000).unwrap();
            let budget = if op == Operation::Exp { 2 } else { 1 };
            assert!(compare(dtype, op, &row, row.staged + budget).passed);
            assert!(!compare(dtype, op, &row, row.staged + budget + 1).passed);
        }
    }
    assert_eq!(
        ordered_rank(DType::F64, 0x3ff0_0000_0000_0000).abs_diff(ordered_rank(DType::F64, 0x3fef_ffff_ffff_ffff)),
        1
    );
    assert_eq!(
        ordered_rank(DType::F64, 0xbff0_0000_0000_0000).abs_diff(ordered_rank(DType::F64, 0xbfef_ffff_ffff_ffff)),
        1
    );
}

#[test]
fn filters_preserve_independent_counts_without_observed_accuracy() {
    let all = expected_summary(&Options::default());
    assert_eq!(all.tensor_rows, 306572);
    assert_eq!(
        all.selected_dtype_operation_counts,
        [[16972, 16972], [2621, 2621], [2621, 2621], [65536, 65536], [65536, 65536]]
    );
    let finite = expected_summary(&Options {
        finite: true,
        ..Options::default()
    });
    assert_eq!(finite.tensor_rows, 192604);
    let skip = expected_summary(&Options {
        skip_rank_zero: true,
        ..Options::default()
    });
    assert_eq!(all.cases - skip.cases, 20);
    assert_eq!(all.tensor_rows, skip.tensor_rows);
    assert_eq!(
        all.tensor_rows
            - expected_summary(&Options {
                no_flex32: true,
                ..Options::default()
            })
            .tensor_rows,
        5242
    );
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
    let row = expected_bits(DType::F64, Operation::Exp, 0x3ff0_0000_0000_0000).unwrap();
    report.accuracy.record(DType::F64, Operation::Exp, &row, row.staged + 1);
    assert_eq!(report.summary, all);
    report.accuracy.record(DType::F64, Operation::Exp, &row, 0);
    assert!(!report.is_success());
}

#[test]
fn mismatch_counters_remain_complete_while_examples_are_bounded() {
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
    let mut accuracy = Accuracy::default();
    let row = expected_bits(DType::F16, Operation::Exp, 0x1f79).unwrap();
    for _ in 0..10000 {
        accuracy.record(DType::F16, Operation::Exp, &row, 0);
    }
    assert_eq!(accuracy.failures, 10000);
    assert_eq!(accuracy.cells[6].nonzero_to_zero, 10000);
    assert_eq!(accuracy.cells[6].transitions[Class::Normal as usize][Class::Zero as usize], 10000);
}

#[test]
#[ignore = "requires MPFR 4.2.2; reproduces every checked-in reference from directed intervals"]
fn fixture_generator_reproduces_the_binary_index_and_manifest() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let result = std::process::Command::new("python3")
        .arg(root.join("tools/nvidia/typed-exp-log-generate.py"))
        .arg("--check")
        .current_dir(root)
        .status()
        .unwrap();
    assert!(result.success());
}
