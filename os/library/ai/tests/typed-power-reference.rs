#![cfg(feature = "training")]
extern crate alloc;
#[path = "../src/validation/typed_power.rs"]
#[allow(dead_code)]
mod typed_power;
use burn::backend::{DType, Scalar};
use typed_power::*;

fn bits(dtype: DType, value: f64) -> u64 {
    match dtype {
        DType::F64 => value.to_bits(),
        DType::F16 => u64::from(burn::backend::f16::from_f32(value as f32).to_bits()),
        DType::BF16 => u64::from(burn::backend::bf16::from_f32(value as f32).to_bits()),
        _ => u64::from((value as f32).to_bits()),
    }
}
#[test]
fn every_fixture_is_sorted_unique_lookupable_and_accepts_its_contract_center() {
    for dtype in DTYPES {
        let mut previous = None;
        for row in rows(dtype) {
            let key = (row.lhs, row.rhs);
            assert!(previous.is_none_or(|p| p < key));
            previous = Some(key);
            assert_eq!(expected_bits(dtype, row.schedule, row.lhs, row.rhs), Some(row));
            assert!(compare(dtype, &row, row.staged).passed);
        }
        assert!(expected_bits(dtype, Schedule::Square, 0x123, 0x456).is_none());
        let half = matches!(dtype, DType::F16 | DType::BF16);
        if half {
            for exponent in [0.5, 3.0] {
                for lhs in 0..65536 {
                    assert!(expected_bits(dtype, Schedule::Mathematical, lhs, bits(dtype, exponent)).is_some());
                }
            }
        }
    }
}
#[test]
fn stored_minus_two_overflow_is_not_a_mathematical_power_failure() {
    for (dtype, lhs) in [
        (DType::F64, 0x6070_0000_0000_0000),
        (DType::F32, 0x6000_0000),
        (DType::Flex32, 0x6000_0000),
        (DType::F16, 0x5c00),
        (DType::BF16, 0x6000),
    ] {
        let rhs = bits(dtype, -2.0);
        let mathematical = expected_bits(dtype, Schedule::Mathematical, lhs, rhs).unwrap();
        let wrapper = expected_bits(dtype, Schedule::SquareReciprocal, lhs, rhs).unwrap();
        assert_eq!(classify(dtype, mathematical.staged), Class::Subnormal);
        assert_eq!(wrapper.ideal, mathematical.ideal);
        assert_eq!(wrapper.staged, 0);
        assert!(compare(dtype, &wrapper, 0).passed);
        assert!(!compare(dtype, &mathematical, 0).passed);
        assert!(!compare(dtype, &wrapper, mathematical.staged).passed);
    }
}
#[test]
fn budgets_never_excuse_domain_sign_zero_or_overflow_errors() {
    for dtype in DTYPES {
        let row = expected_bits(dtype, Schedule::Mathematical, bits(dtype, 3.0), bits(dtype, 3.0)).unwrap();
        let limit = if dtype == DType::F64 { 2 } else { 4 };
        if !matches!(dtype, DType::F16 | DType::BF16) {
            assert!(compare(dtype, &row, row.staged + limit).passed);
            assert!(!compare(dtype, &row, row.staged + limit + 1).passed);
        }
        let sign = 1 << (dtype.size() * 8 - 1);
        for actual in [0, sign, row.staged ^ sign] {
            assert!(compare(dtype, &row, actual).classification_failure);
        }
        let nan = expected_bits(dtype, Schedule::Mathematical, bits(dtype, -2.0), bits(dtype, 0.5)).unwrap();
        assert_eq!(classify(dtype, nan.staged), Class::Nan);
        assert!(!compare(dtype, &nan, 0).passed);
        let identity = expected_bits(dtype, Schedule::Mathematical, bits(dtype, -1.0), bits(dtype, 3.0)).unwrap();
        assert!(!compare(dtype, &identity, identity.staged + 1).passed);
    }
}
#[test]
fn identity_alias_preserves_nan_payload_but_other_outputs_require_only_nan_class() {
    for dtype in DTYPES {
        let fixture = scalar_fixtures(dtype).into_iter().find(|s| matches!(s.scalar, Scalar::Int(1))).unwrap();
        let row = fixture.wrapper.iter().find(|r| is_nan(dtype, r.lhs)).unwrap();
        assert!(compare(dtype, row, row.lhs).passed);
        assert!(!compare(dtype, row, row.lhs ^ 2).passed);
        let mathematical = expected_bits(dtype, Schedule::Mathematical, row.lhs, bits(dtype, 3.0)).unwrap();
        assert!(compare(dtype, &mathematical, row.lhs ^ 2).passed);
        assert_eq!(unpack(dtype, &pack(dtype, &[row.lhs, row.lhs ^ 2])), [row.lhs, row.lhs ^ 2]);
    }
}
#[test]
fn public_float_powi_and_powf_both_normalize_scalar_to_f64_without_i64_truncation() {
    for dtype in DTYPES {
        assert!(matches!(Scalar::new(2.5, &dtype), Scalar::Float(v) if v == 2.5));
        assert!(matches!(Scalar::new(u64::MAX, &dtype), Scalar::Float(v) if v == (u64::MAX as f64)));
        assert!(matches!(Scalar::new(f64::INFINITY, &dtype), Scalar::Float(v) if v.is_infinite()));
        for fixture in scalar_fixtures(dtype) {
            let normalized = match fixture.scalar {
                Scalar::Float(v) => Scalar::new(v, &dtype),
                Scalar::Int(v) => Scalar::new(v, &dtype),
                Scalar::UInt(v) => Scalar::new(v, &dtype),
                Scalar::Bool(v) => Scalar::new(v, &dtype),
            };
            let (Scalar::Float(actual), Scalar::Float(wanted)) = (normalized, fixture.public_powi_scalar) else {
                panic!("public float scalar variant");
            };
            assert_eq!(actual.to_bits(), wanted.to_bits());
            assert_eq!(fixture.public_powi_bits, fixture.public_bits);
            assert_eq!(fixture.public_powi_wrapper, fixture.public_wrapper);
            if let Scalar::Float(v) = fixture.scalar {
                assert_eq!(fixture.valid_powi, v.is_finite() && v.fract() == 0.0);
            }
        }
    }
    // Integer tensor APIs use the tensor dtype, which is a separate contract.
    assert!(matches!(Scalar::new(2.5, &DType::I64), Scalar::Int(2)));
}
#[test]
fn scalar_conversion_and_f64_parity_witnesses_survive_original_and_public_routes() {
    let odd = ((1u64 << 24) + 1) as f64;
    let row = expected_bits(DType::F64, Schedule::Mathematical, (-1.0f64).to_bits(), odd.to_bits()).unwrap();
    assert_eq!(row.staged, (-1.0f64).to_bits());
    let narrowed = (odd as f32) as u64;
    assert_eq!(narrowed % 2, 0);
    assert_eq!((odd as u64) % 2, 1);
    let sensitive = expected_bits(DType::F64, Schedule::Mathematical, 0x3fef_ffff_ffff_ffff, (1e18f64).to_bits()).unwrap();
    assert_eq!(sensitive.staged, 0x35ec_6ab6_c6b3_f2f9);
    for dtype in [DType::F32, DType::Flex32] {
        let row = scalar_fixtures(dtype)
            .into_iter()
            .find(|s| matches!(s.scalar, Scalar::Int(n) if n == (1i64 << 62)+(1i64 << 38)+1))
            .unwrap();
        assert_eq!(row.scalar_bits, 0x5e80_0000);
        let integer = integer_fixtures(dtype)
            .into_iter()
            .find(|r| r.exponent == (1i64 << 62) + (1i64 << 38) + 1)
            .unwrap();
        assert_eq!(integer.stored, 0x5e80_0001);
        assert_ne!(integer.stored, row.scalar_bits);
    }
}
#[test]
fn accuracy_tracks_both_operands_schedule_and_failed_values_without_truncation() {
    let dtype = DType::F64;
    let row = expected_bits(dtype, Schedule::Mathematical, bits(dtype, 3.0), bits(dtype, 3.0)).unwrap();
    let mut accuracy = Accuracy::default();
    for offset in 0..100 {
        accuracy.record(dtype, Route::Tensor, &row, row.staged + offset);
    }
    assert!(accuracy.is_coherent());
    let c = &accuracy.cells[0].accuracy;
    assert_eq!((c.values, c.finite_pairs, c.failures, c.max_ulp), (100, 100, 97, Some(99)));
    let worst = c.worst.unwrap();
    assert_eq!((worst.lhs, worst.rhs, worst.schedule), (row.lhs, row.rhs, Schedule::Mathematical));
    let mut failures = Failures::default();
    for _ in 0..100 {
        failures.record(|| alloc::string::String::from("example"));
    }
    assert_eq!((failures.count, failures.examples.len()), (100, MAX_EXAMPLES));
}
#[test]
fn pure_plan_populations_account_for_routes_filters_and_chains_without_measurements() {
    let all = expected_summary(&Options::default());
    assert_eq!(all.values, all.nonchain_values + all.chain_values);
    assert_eq!(all.chain_values, all.chain_steps);
    assert_eq!(all.cells.iter().map(|c| c.values).sum::<usize>(), all.values);
    for c in &all.cells {
        assert_eq!(c.values, c.reference_classes.iter().sum::<usize>());
        assert_eq!(c.values, c.ideal_classes.iter().sum::<usize>());
    }
    let common = expected_summary(&Options {
        common_linux: true,
        ..Options::default()
    });
    assert_eq!((common.rejected_cases, common.extension_cases), (0, 0));
    assert!(common.cases < all.cases && all.rejected_cases > 0 && all.extension_cases > 0);
    let finite = expected_summary(&Options {
        finite: true,
        ..Options::default()
    });
    assert!(
        finite
            .cells
            .iter()
            .all(|c| c.reference_classes[Class::Nan as usize] == 0 && c.reference_classes[Class::Infinity as usize] == 0)
    );
    let no_zero = expected_summary(&Options {
        skip_rank_zero: true,
        ..Options::default()
    });
    assert_eq!(all.cases - no_zero.cases, 35);
    for dtype in DTYPES {
        let s = expected_summary(&Options {
            dtype: Some(dtype),
            ..Options::default()
        });
        assert_eq!(s.dtypes, 1);
        assert!(s.cells.iter().all(|c| c.dtype == dtype));
    }
}
#[test]
fn report_rejects_route_shift_even_with_coherent_accuracy_and_unchanged_global_count() {
    let dtype = DType::F64;
    let row = expected_bits(dtype, Schedule::Mathematical, bits(dtype, 3.0), bits(dtype, 3.0)).unwrap();
    let cells = [Route::Tensor, Route::ScalarImpl].map(|route| ReferenceCell {
        dtype,
        route,
        schedule: Schedule::Mathematical,
        chain: false,
        values: 1,
        reference_classes: [0, 0, 1, 0, 0],
        ideal_classes: [0, 0, 1, 0, 0],
    });
    let mut report = Report {
        summary: Summary {
            values: 2,
            nonchain_values: 2,
            cells: cells.to_vec(),
            ..Summary::default()
        },
        ..Report::default()
    };
    for route in [Route::Tensor, Route::ScalarImpl] {
        report.accuracy.record(dtype, route, &row, row.staged);
    }
    assert!(report.is_success());
    report.accuracy = Accuracy::default();
    for _ in 0..2 {
        report.accuracy.record(dtype, Route::Tensor, &row, row.staged);
    }
    assert!(report.accuracy.is_coherent());
    assert_eq!(report.accuracy.cells.iter().map(|c| c.accuracy.values).sum::<usize>(), report.summary.values);
    assert!(!report.observed_counts_match_plan());
    assert!(!report.is_success());
    report.accuracy = Accuracy::default();
    for route in [Route::Tensor, Route::ScalarImpl] {
        report.accuracy.record(dtype, route, &row, row.staged);
    }
    report.accuracy.cells.push(Cell {
        dtype,
        route: Route::PublicTensor,
        schedule: Schedule::Mathematical,
        chain: false,
        accuracy: AccuracyCell::default(),
    });
    assert!(!report.is_success());
}
