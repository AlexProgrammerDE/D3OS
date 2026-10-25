#![cfg(feature = "training")]
extern crate alloc;

#[path = "../src/validation/typed_remainder.rs"]
#[allow(dead_code)]
mod typed_remainder;
use burn::backend::{DType, Scalar, bf16, f16};
use typed_remainder::{DTYPES, Options, expected_bits, expected_summary, is_nan, pack, pairs, scalar_fixtures, storage_scalar_fixtures, unpack};

fn equivalent(dtype: DType, actual: u64, expected: u64) -> bool {
    actual == expected || (is_nan(dtype, actual) && is_nan(dtype, expected))
}
fn widen(dtype: DType, bits: u64) -> f32 {
    match dtype {
        DType::F16 => f16::from_bits(bits as u16).to_f32(),
        DType::BF16 => bf16::from_bits(bits as u16).to_f32(),
        _ => f32::from_bits(bits as u32),
    }
}
fn store(dtype: DType, value: f32) -> u64 {
    match dtype {
        DType::F16 => u64::from(f16::from_f32(value).to_bits()),
        DType::BF16 => u64::from(bf16::from_f32(value).to_bits()),
        _ => u64::from(value.to_bits()),
    }
}
fn host_remainder(dtype: DType, a: u64, b: u64) -> u64 {
    // Secondary host cross-check only. Neither fixture generation nor runtime
    // expectations call this compiler/libm-backed remainder implementation.
    if dtype == DType::F64 {
        let (x, y) = (f64::from_bits(a), f64::from_bits(b));
        let r = x % y;
        (if r != 0. && r.is_sign_negative() != y.is_sign_negative() { r + y } else { r }).to_bits()
    } else {
        let (x, y) = (widen(dtype, a), widen(dtype, b));
        let r = x % y;
        store(dtype, if r != 0. && r.is_sign_negative() != y.is_sign_negative() { r + y } else { r })
    }
}
#[test]
fn host_fmod_and_target_precision_correction_cross_check_every_numerical_row() {
    for dtype in DTYPES {
        for row in pairs(dtype) {
            let actual = host_remainder(dtype, row.lhs, row.rhs);
            assert!(
                equivalent(dtype, actual, row.expected),
                "{dtype:?} {} {:x} % {:x}: {actual:x} != {:x}",
                row.group,
                row.lhs,
                row.rhs,
                row.expected
            );
        }
        for row in scalar_fixtures(dtype).into_iter().chain(storage_scalar_fixtures(dtype)) {
            let value = match row.scalar {
                Scalar::Float(v) => v,
                Scalar::Int(v) => v as f64,
                Scalar::UInt(v) => v as f64,
                Scalar::Bool(v) => u8::from(v) as f64,
            };
            let rhs = if dtype == DType::F64 { value.to_bits() } else { store(dtype, value as f32) };
            assert!(equivalent(dtype, rhs, row.scalar_bits));
            for (&lhs, &expected) in row.inputs.iter().zip(&row.expected) {
                assert!(
                    equivalent(dtype, host_remainder(dtype, lhs, rhs), expected),
                    "{dtype:?} {:?} lhs={lhs:x}",
                    row.scalar
                );
            }
        }
    }
}
#[test]
fn scalar_staging_precision_range_and_storage_witnesses_remain_observable() {
    for dtype in [DType::F32, DType::Flex32] {
        let row = scalar_fixtures(dtype)
            .into_iter()
            .find(|r| matches!(r.scalar, Scalar::Int(v) if v == (1i64 << 62) + (1i64 << 38) + 1))
            .unwrap();
        assert_eq!(row.scalar_bits, 0x5e80_0000);
        assert_eq!(row.expected[row.inputs.iter().position(|&v| v == 0xc040_0000).unwrap()], 0x5e80_0000);
    }
    assert_eq!(
        expected_bits(DType::F64, 0x3ff0_0000_0000_0001, 0x3ff0_0000_0000_0000),
        Some(0x3cb0_0000_0000_0000)
    );
    assert_eq!(expected_bits(DType::F64, 1, 0x4008_0000_0000_0000), Some(1));
    for (dtype, source, stored_one) in [(DType::F16, 0x3ff0_0200_0000_0001, 0x3c00), (DType::BF16, 0x3ff0_1000_0000_0001, 0x3f80)] {
        let row = scalar_fixtures(dtype)
            .into_iter()
            .find(|r| matches!(r.scalar, Scalar::Float(v) if v.to_bits() == source))
            .unwrap();
        assert_eq!(row.scalar_bits, stored_one);
        assert_eq!(row.expected[row.inputs.iter().position(|&v| v == stored_one).unwrap()], 0);
    }
}
#[test]
fn strict_policy_distinguishes_signed_zero_and_preserves_input_nan_payload_bytes() {
    for dtype in DTYPES {
        let sign = 1u64 << (dtype.size() * 8 - 1);
        let rows = pairs(dtype);
        assert!(rows.iter().any(|r| r.expected == sign));
        assert!(rows.iter().any(|r| r.expected == 0));
        let payloads: Vec<_> = rows.iter().filter(|r| is_nan(dtype, r.lhs)).map(|r| r.lhs).collect();
        assert!(!payloads.is_empty());
        assert_eq!(unpack(dtype, &pack(dtype, &payloads)), payloads);
        assert!(payloads.iter().any(|v| v & 1 != 0));
    }
}
#[test]
fn device_free_totals_include_the_complete_default_numerical_corpus() {
    let s = expected_summary(&Options::default());
    assert_eq!((s.dtypes, s.tensor_rows, s.scalar_specs, s.scalar_results), (5, 1608, 221, 2210));
    let wide = expected_summary(&Options {
        group: Some("f64-precision-range"),
        ..Options::default()
    });
    assert_eq!((wide.dtypes, wide.tensor_cases, wide.tensor_rows), (1, 1, 28));
    assert_eq!(
        expected_summary(&Options {
            dtype: Some(DType::F32),
            group: Some("f64-precision-range"),
            ..Options::default()
        })
        .dtypes,
        0
    );
    assert_eq!(s.values, s.exact_values + s.nan_values);
    assert_eq!(s.chain_steps, 45);
    assert_eq!(
        expected_summary(&Options {
            finite: true,
            ..Options::default()
        })
        .nan_values,
        0
    );
    assert_eq!(
        s.cases()
            - expected_summary(&Options {
                skip_rank_zero: true,
                ..Options::default()
            })
            .cases(),
        10
    );
}
