use d3os_ai::{
    burn::backend::DType,
    validation::float_reduction::{self as fr, Arithmetic as A, Case, Checks, Evidence, Operation as Op, Options, ReferenceError as E, Report, Scope, Values},
};

fn reduced(dtype: DType, shape: &[usize], values: &[u64], op: Op, scope: Scope) -> Values {
    fr::reference(dtype, &Values::new(shape, values.to_vec()), op, scope).unwrap()
}
fn row<'a>(cases: &'a [Case], id: &str) -> &'a Case {
    cases.iter().find(|c| c.id == id).unwrap()
}

#[test]
fn exact_ieee_steps_cover_ties_underflow_overflow_and_invalid_arithmetic() {
    for (dtype, one, half, inf, sign, subnormal) in [
        (DType::F64, 0x3ff0000000000000, 0x3fe0000000000000, 0x7ff0000000000000, 0x8000000000000000, 1),
        (DType::F32, 0x3f800000, 0x3f000000, 0x7f800000, 0x80000000, 1),
        (DType::Flex32, 0x3f800000, 0x3f000000, 0x7f800000, 0x80000000, 1),
        (DType::F16, 0x3c00, 0x3800, 0x7c00, 0x8000, 1),
        (DType::BF16, 0x3f80, 0x3f00, 0x7f80, 0x8000, 1),
    ] {
        assert_eq!(fr::ieee_binary(dtype, one, one | sign, A::Add), 0);
        assert_eq!(fr::ieee_binary(dtype, sign, sign, A::Add), sign);
        assert_eq!(fr::ieee_binary(dtype, subnormal, half, A::Multiply), 0);
        assert_eq!(fr::ieee_binary(dtype, subnormal | sign, half, A::Multiply), sign);
        assert_eq!(fr::ieee_binary(dtype, subnormal * 3, half, A::Multiply), 2);
        assert_eq!(fr::ieee_binary(dtype, inf - 1, fr::integer_bits(dtype, 2), A::Multiply), inf);
        assert_eq!(fr::ieee_binary(dtype, 0, inf, A::Multiply), fr::canonical_nan(dtype));
        assert_eq!(fr::ieee_binary(dtype, inf, inf | sign, A::Add), fr::canonical_nan(dtype));
        assert_eq!(fr::ieee_binary(dtype, 0, 0, A::Divide), fr::canonical_nan(dtype));
        assert_eq!(fr::ieee_binary(dtype, one, inf | sign, A::Divide), sign);
    }
    assert_eq!(fr::ieee_binary(DType::F64, 0x3ff0000000000000, 1, A::Add), 0x3ff0000000000000);
    assert_eq!(fr::ieee_binary(DType::F64, 0x0010000000000000, 0x800fffffffffffff, A::Add), 1);
    assert_eq!(fr::cast_bits(DType::F64, DType::F16, 0x3ff0020000000000), 0x3c00);
    assert_eq!(fr::cast_bits(DType::F64, DType::F16, 0x3ff0060000000000), 0x3c02);
}

#[test]
fn cancellation_uses_the_declared_accumulator_and_serial_order() {
    for (dtype, huge, one, negative) in [
        (DType::F64, 0x4340000000000000, 0x3ff0000000000000, 0xc340000000000000),
        (DType::Flex32, 0x4b800000, 0x3f800000, 0xcb800000),
    ] {
        assert_eq!(fr::axis_reference(dtype, &[huge, one, negative], Op::Sum), 0);
        assert_eq!(fr::axis_reference(dtype, &[huge, negative, one], Op::Sum), one);
    }
    assert_eq!(fr::axis_reference(DType::F16, &[0x3c00, 0x1000, 0x1000], Op::Sum), 0x3c01);
    assert_eq!(fr::axis_reference(DType::BF16, &[0x3f80, 0x3b80, 0x3b80], Op::Sum), 0x3f81);
    assert_eq!(fr::axis_reference(DType::F16, &[0x7bff, 0x7bff], Op::Mean), 0x7bff);
    assert_eq!(
        fr::axis_reference(DType::F64, &[0x7fefffffffffffff, 0x7fefffffffffffff], Op::Mean),
        0x7ff0000000000000
    );
}

#[test]
fn globals_sort_original_lengths_and_store_every_axis_mean() {
    let input = Values::new(&[2, 2], vec![0x6800, 0x3c00, 0xe800, 0]);
    let forward = fr::reference(DType::F16, &input, Op::Mean, Scope::Global).unwrap();
    assert_eq!(forward, Values::new(&[1], vec![0x3400]));
    let a = fr::reference(DType::F16, &input, Op::Mean, Scope::Axis(1)).unwrap();
    assert_eq!(fr::reference(DType::F16, &a, Op::Mean, Scope::Axis(0)).unwrap().values, [0]);
    let maximum = Values::new(&[2, 2], vec![0x7bff; 4]);
    assert_eq!(fr::reference(DType::F16, &maximum, Op::Mean, Scope::Global).unwrap().values, [0x7bff]);
    let stored_sum = fr::reference(DType::F16, &maximum, Op::Sum, Scope::Global).unwrap().values[0];
    assert_eq!(fr::ieee_binary(DType::F16, stored_sum, fr::scalar_count(DType::F16, 4), A::Divide), 0x7c00);
    let cases = fr::corpus(&Options::default());
    let input = row(&cases, "P37").input.logical();
    let first = fr::reference(DType::F64, &input, Op::Product, Scope::Axis(1)).unwrap();
    let second = fr::reference(DType::F64, &first, Op::Product, Scope::Axis(0)).unwrap();
    assert_eq!(fr::reference(DType::F64, &input, Op::Product, Scope::Global).unwrap().values, second.values);
    assert_eq!(second.values, [0x7fdfffffffffffff]);
    let reverse = fr::reference(DType::F64, &input, Op::Product, Scope::Axis(0)).unwrap();
    assert_eq!(
        fr::reference(DType::F64, &reverse, Op::Product, Scope::Axis(1)).unwrap().values,
        [0x7ff0000000000000]
    );
}

#[test]
fn count_conversion_and_symbolic_rows_do_not_fold_large_tensors() {
    for (n, b) in [(65519, 0x7bff), (65520, 0x7c00), (65536, 0x7c00)] {
        assert_eq!(fr::scalar_count(DType::F16, n), b);
        let raw = fr::cast_bits(
            DType::F32,
            DType::F16,
            fr::ieee_binary(DType::F32, 0x3f800000, fr::integer_bits(DType::F32, n), A::Divide),
        );
        assert_eq!(raw, 0x100);
        let stored_count_alternative = fr::ieee_binary(DType::F16, 0x3c00, b, A::Divide);
        assert_eq!(stored_count_alternative, if n == 65519 { 0x100 } else { 0 });
    }
    assert_eq!(fr::integer_bits(DType::F64, 16777217), 0x4170000010000000);
    assert_eq!(fr::integer_bits(DType::F32, 16777219), 0x4b800002);
    assert_eq!(fr::integer_bits(DType::F32, 2147483647), 0x4f000000);
    let cases = fr::corpus(&Options::default());
    assert_eq!(row(&cases, "P53").expected().unwrap().output.values, [0x100]);
    assert_eq!(row(&cases, "P54").expected().unwrap().output.values, [0x3c00]);
    assert_eq!(row(&cases, "P60").expected().unwrap().output.values, [0x3c00, 0x3c01]);
    for c in cases.iter().filter(|c| c.model.is_some()) {
        assert!(c.input.root.values.len() <= 1);
    }
}

#[test]
fn empty_rank_zero_and_invalid_axes_preserve_separate_semantics() {
    for dtype in [DType::F64, DType::Flex32, DType::F16, DType::BF16] {
        for op in [Op::Sum, Op::Mean, Op::Product] {
            let identity = match op {
                Op::Sum => 0,
                Op::Mean => fr::canonical_nan(dtype),
                Op::Product => fr::integer_bits(dtype, 1),
            };
            assert_eq!(reduced(dtype, &[2, 0], &[], op, Scope::Axis(1)), Values::new(&[2, 1], vec![identity; 2]));
            assert_eq!(reduced(dtype, &[0, 3], &[], op, Scope::Axis(1)), Values::new(&[0, 1], vec![]));
            assert_eq!(reduced(dtype, &[0, 0], &[], op, Scope::Axis(1)), Values::new(&[0, 1], vec![]));
            assert_eq!(reduced(dtype, &[0, 0], &[], op, Scope::Global), Values::new(&[1], vec![identity]));
            let bits = fr::mask(dtype) ^ 7;
            assert_eq!(reduced(dtype, &[], &[bits], op, Scope::Global), Values::new(&[1], vec![bits]));
            assert_eq!(fr::reference(dtype, &Values::new(&[], vec![bits]), op, Scope::Axis(0)), Err(E::Axis));
            assert_eq!(fr::reference(dtype, &Values::new(&[0], vec![]), op, Scope::Axis(1)), Err(E::Axis));
        }
    }
}

#[test]
fn axis_output_count_is_checked_independently_of_zero_input_count() {
    let input = Values::new(&[0, i32::MAX as usize, i32::MAX as usize], vec![]);
    assert_eq!(fr::reference(DType::F64, &input, Op::Sum, Scope::Axis(0)), Err(E::Count));
    assert_eq!(fr::reference(DType::F64, &input, Op::Sum, Scope::Global).unwrap(), Values::new(&[1], vec![0]));
}

#[test]
fn legacy_f32_requires_a_sufficient_all_order_exactness_proof() {
    for bits in [0x4b800001, 0x7f7fffff, 0x5f800000, 0xbf800000, 0x3f000000] {
        assert_eq!(
            fr::reference(DType::F32, &Values::new(&[1], vec![bits]), Op::Sum, Scope::Axis(0)),
            Err(E::LegacyOrderUnspecified)
        );
    }
    let values = Values::new(&[2, 3], vec![0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]);
    assert_eq!(fr::reference(DType::F32, &values, Op::Mean, Scope::Global).unwrap().values, [0x40600000]);
    assert_eq!(
        fr::reference(
            DType::F32,
            &Values::new(&[3], vec![0xbf800000, 0x3f800000, 0xbf800000]),
            Op::Product,
            Scope::Global
        )
        .unwrap()
        .values,
        [0x3f800000]
    );
}

#[test]
fn selected_corpus_is_bounded_and_views_preserve_logical_multiplicity() {
    let cases = fr::corpus(&Options::default());
    assert_eq!(cases.len(), 80);
    let mut ids = std::collections::BTreeSet::new();
    for case in &cases {
        assert!(ids.insert(case.id.clone()));
        assert!(case.input.root.values.len() <= 258);
        if case.expected().is_err() {
            assert!(case.model.is_none());
        }
    }
    assert_eq!(cases.iter().filter(|c| c.model.is_some()).count(), 7);
    assert_eq!(cases.iter().filter(|c| c.expected().is_err()).count(), 2);
    assert_eq!(cases.iter().filter(|c| c.chain).count(), 8);
    for rank in 0..=8 {
        assert!(cases.iter().any(|c| c.input.logical().shape.len() == rank));
    }
    let logical = row(&cases, "P31").input.logical();
    assert_eq!(logical.values, vec![0x4600, 0xc500, 0x4400, 0xc200, 0x4000, 0xbc00]);
    assert_eq!(row(&cases, "P32").expected().unwrap().output.values, vec![0x4000, 0x4080, 0x40c0]);
    assert_eq!(row(&cases, "P35").input.logical().values, vec![0x4500, 0x4600, 0x4880, 0x4900, 0x4a80, 0x4b00]);
    let expected = row(&cases, "C06").expected().unwrap();
    assert_eq!(expected.dtype, DType::F64);
    assert_eq!(expected.output.values, vec![0x4018000000000000; 129]);
    assert_eq!(row(&cases, "C07").expected().unwrap().output.values, [0x4040]);
    assert_eq!(row(&cases, "C08").expected(), Err(E::Axis));
    assert_eq!(row(&cases, "P61").evidence, Evidence::LegacyAllOrderExact);
}

#[test]
fn expected_summary_partitions_options_and_observed_checks_cannot_be_inferred() {
    for skip_rank_zero in [false, true] {
        for include_empty in [false, true] {
            let options = Options {
                skip_rank_zero,
                include_empty,
                ..Options::default()
            };
            let total = fr::expected_summary(&options);
            assert_eq!(total.cases, total.device_cases + total.model_cases + total.diagnostic_cases);
            let mut counts = [0; 6];
            for dtype in fr::DTYPES {
                let s = fr::expected_summary(&Options { dtype: Some(dtype), ..options });
                for (a, b) in counts
                    .iter_mut()
                    .zip([s.cases, s.device_cases, s.model_cases, s.diagnostic_cases, s.standalone_cases, s.chain_cases])
                {
                    *a += b;
                }
            }
            assert_eq!(
                counts,
                [
                    total.cases,
                    total.device_cases,
                    total.model_cases,
                    total.diagnostic_cases,
                    total.standalone_cases,
                    total.chain_cases
                ]
            );
            let mut report = Report {
                expected: total.clone(),
                attempted: total.device_cases,
                passed: total.device_cases,
                failed: 0,
                skipped_diagnostics: total.diagnostic_cases,
                observed_errors: 0,
                model_passed: total.model_cases,
                checks: total.checks.clone(),
                failures: vec![],
            };
            assert!(fr::success_footer(&options, &report).is_ok());
            report.checks.output -= 1;
            assert!(!report.is_success());
            report.checks = total.checks.clone();
            report.model_passed -= 1;
            assert!(!report.is_success());
            report.model_passed = total.model_cases;
            report.expected.options.include_empty = !include_empty;
            assert!(fr::success_footer(&options, &report).is_err());
            let native = fr::expected_summary(&Options {
                native_error_extensions: true,
                ..options
            });
            assert_eq!(native.checks.error, native.diagnostic_cases);
            assert_eq!(native.checks.recovery, native.diagnostic_cases);
            assert_eq!(native.checks.input, total.checks.input + total.diagnostic_cases);
        }
    }
    let s = fr::expected_summary(&Options::default());
    assert_eq!(
        (s.standalone_cases, s.chain_cases, s.device_cases, s.model_cases, s.diagnostic_cases),
        (72, 8, 71, 7, 2)
    );
    assert_eq!(
        s.checks,
        Checks {
            input: 71,
            source: 71,
            alias: 71,
            output: 71,
            error: 0,
            recovery: 0,
            model: 7
        }
    );
}

#[test]
fn packing_preserves_original_bits_including_half_payloads() {
    for dtype in fr::DTYPES {
        let values = [0, 1, fr::mask(dtype), fr::mask(dtype) ^ 7];
        assert_eq!(fr::unpack(dtype, &fr::pack(dtype, &values)), values);
    }
}
