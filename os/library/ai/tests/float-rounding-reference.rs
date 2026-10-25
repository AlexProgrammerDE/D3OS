#![cfg(feature = "training")]
extern crate alloc;

#[path = "../src/validation/float_rounding.rs"]
#[allow(dead_code)]
mod float_rounding;
use burn::backend::{DType, bf16, f16};
use float_rounding::{
    DTYPES, Operation as Op, Values, chain_fixtures, expected_bits, half_expected_bits, is_nan, pack, pool, raw_integral_nan_bits, staged_sqrt_bits, unpack,
};

fn matches(dtype: DType, actual: u64, expected: u64) -> bool {
    actual == expected || (is_nan(dtype, actual) && is_nan(dtype, expected))
}

#[test]
fn target_precision_host_operations_cross_check_the_independent_fixtures() {
    // Secondary host checks only. Fixture generation never uses these methods.
    for dtype in [DType::F64, DType::F32, DType::Flex32] {
        for input in pool(dtype) {
            for op in Op::ALL {
                let actual = if dtype == DType::F64 {
                    let x = f64::from_bits(input);
                    match op {
                        Op::Sqrt => x.sqrt(),
                        Op::Floor => x.floor(),
                        Op::Ceil => x.ceil(),
                        Op::Round => x.round_ties_even(),
                        Op::Trunc => x.trunc(),
                    }
                    .to_bits()
                } else {
                    let x = f32::from_bits(input as u32);
                    u64::from(
                        match op {
                            Op::Sqrt => x.sqrt(),
                            Op::Floor => x.floor(),
                            Op::Ceil => x.ceil(),
                            Op::Round => x.round_ties_even(),
                            Op::Trunc => x.trunc(),
                        }
                        .to_bits(),
                    )
                };
                let expected = expected_bits(dtype, op, input).unwrap();
                assert!(matches(dtype, actual, expected), "{dtype:?} {op:?} input={input:x}: {actual:x} != {expected:x}");
            }
        }
    }
}

#[test]
fn all_half_codes_cross_check_integrals_and_the_separate_staged_sqrt_model() {
    // Half inputs widen exactly. Integral results are exactly representable in
    // their original half format. Sqrt uses the explicit staged model, compared
    // separately with the ideal table, never substituted for its expectations.
    for dtype in [DType::F16, DType::BF16] {
        for bits in 0..=u16::MAX {
            let x = if dtype == DType::F16 {
                f16::from_bits(bits).to_f32()
            } else {
                bf16::from_bits(bits).to_f32()
            };
            for op in Op::ALL {
                let y = match op {
                    Op::Sqrt => x.sqrt(),
                    Op::Floor => x.floor(),
                    Op::Ceil => x.ceil(),
                    Op::Round => x.round_ties_even(),
                    Op::Trunc => x.trunc(),
                };
                let actual = if dtype == DType::F16 {
                    f16::from_f32(y).to_bits()
                } else {
                    bf16::from_f32(y).to_bits()
                };
                let ideal = half_expected_bits(dtype, op, bits).unwrap();
                if op == Op::Sqrt {
                    let staged = staged_sqrt_bits(dtype, bits).unwrap();
                    assert!(matches(dtype, u64::from(actual), u64::from(staged)), "staged {dtype:?} {bits:04x}");
                    assert!(matches(dtype, u64::from(staged), u64::from(ideal)), "ideal/staged {dtype:?} {bits:04x}");
                } else {
                    assert!(
                        matches(dtype, u64::from(actual), u64::from(ideal)),
                        "{dtype:?} {op:?} {bits:04x}: {actual:04x} != {ideal:04x}"
                    );
                    if is_nan(dtype, u64::from(bits)) {
                        assert_eq!(raw_integral_nan_bits(dtype, op, u64::from(bits)), Some(u64::from(bits)));
                        assert_eq!(ideal, bits);
                    }
                }
                assert_eq!(expected_bits(dtype, op, u64::from(bits)), Some(u64::from(ideal)));
            }
        }
    }
}

#[test]
fn signed_zero_domain_nan_and_halfway_parity_are_explicit() {
    for dtype in DTYPES {
        let p = pool(dtype);
        let sign = p[1];
        let get = |op, bits| expected_bits(dtype, op, bits).unwrap();
        for op in Op::ALL {
            assert_eq!(get(op, 0), 0);
            assert_eq!(get(op, sign), sign);
        }
        assert_eq!(get(Op::Sqrt, p[2]), p[2]);
        assert!(is_nan(dtype, get(Op::Sqrt, p[3])));
        assert!(is_nan(dtype, get(Op::Sqrt, sign | 1)));
        for nan in &p[4..8] {
            for op in Op::ALL {
                assert!(is_nan(dtype, get(op, *nan)));
            }
            assert_eq!(raw_integral_nan_bits(dtype, Op::Sqrt, *nan), None);
        }
        assert_eq!(get(Op::Floor, 1), 0);
        assert_eq!(get(Op::Ceil, sign | 1), sign);
        assert_eq!(get(Op::Round, sign | 1), sign);
        assert_eq!(get(Op::Trunc, sign | 1), sign);
        // Pool entries 18..42 are three neighbors of four ties, both signs.
        let encode_integer = |integer: f64| match dtype {
            DType::F64 => integer.to_bits(),
            DType::F32 | DType::Flex32 => u64::from((integer as f32).to_bits()),
            DType::F16 => u64::from(f16::from_f64(integer).to_bits()),
            DType::BF16 => u64::from(bf16::from_f64(integer).to_bits()),
            _ => unreachable!(),
        };
        for (tie, below, at, above) in [(0, 0., 0., 1.), (1, 1., 2., 2.), (2, 2., 2., 3.), (3, 3., 4., 4.)] {
            for (neighbor, integer) in [below, at, above].into_iter().enumerate() {
                let input = p[18 + tie * 6 + neighbor * 2];
                let expected = encode_integer(integer);
                assert_eq!(get(Op::Round, input), expected);
                assert_eq!(get(Op::Round, input | sign), expected | sign);
            }
        }
    }
}

#[test]
fn f64_extremes_retain_wide_precision_and_no_integer_conversion_limit() {
    let get = |op, x| expected_bits(DType::F64, op, x).unwrap();
    assert_eq!(get(Op::Sqrt, 1), (2.0f64).powi(-537).to_bits());
    assert_eq!(get(Op::Sqrt, (2.0f64).powi(200).to_bits()), (2.0f64).powi(100).to_bits());
    assert_eq!(get(Op::Sqrt, 0x3ff0_0000_0000_0002), 0x3ff0_0000_0000_0001);
    assert_eq!(get(Op::Ceil, ((1u64 << 30) as f64 + 0.5).to_bits()), ((1u64 << 30) as f64 + 1.).to_bits());
    assert_eq!(get(Op::Round, ((1u64 << 30) as f64 + 1.5).to_bits()), ((1u64 << 30) as f64 + 2.).to_bits());
    for input in [f64::MAX.to_bits(), (2.0f64).powi(64).to_bits()] {
        for op in [Op::Floor, Op::Ceil, Op::Round, Op::Trunc] {
            assert_eq!(get(op, input), input);
        }
    }
}

#[test]
fn materialized_sqrt_round_witness_and_retained_zero_chains_are_exact() {
    for dtype in DTYPES {
        let fixtures = chain_fixtures(dtype);
        assert_eq!(fixtures.len(), 3);
        let witness = &fixtures[0];
        assert_eq!(witness.steps[0].operation, Op::Sqrt);
        assert_eq!(witness.steps[1].operation, Op::Round);
        let (input, root, rounded) = match dtype {
            DType::F64 => (0x4019_0000_0000_0001, 0x4004_0000_0000_0000, 0x4000_0000_0000_0000),
            DType::F32 | DType::Flex32 => (0x40c8_0001, 0x4020_0000, 0x4000_0000),
            DType::F16 => (0x4641, 0x4100, 0x4000),
            DType::BF16 => (0x40c9, 0x4020, 0x4000),
            _ => unreachable!(),
        };
        assert_eq!(witness.initial, input);
        assert_eq!(witness.steps[0].expected, root);
        assert_eq!(witness.steps[1].expected, rounded);
        assert!(fixtures[1].steps.iter().all(|s| s.expected == pool(dtype)[1]));
    }
}

#[test]
fn fixture_storage_and_view_mapping_preserve_identity_and_empty_shapes() {
    let values = Values::new(&[3, 7], (0..21).collect());
    assert_eq!(
        values.slice(&[0, 1], &[1, 2], &[3, 3]).permute(&[1, 0]).flip(&[0, 1]).values,
        [19, 12, 5, 17, 10, 3, 15, 8, 1]
    );
    assert!(Values::new(&[2, 0, 7], vec![]).permute(&[2, 0, 1]).values.is_empty());
    for dtype in DTYPES {
        let p = pool(dtype);
        assert_eq!(unpack(dtype, &pack(dtype, &p)), p);
        assert!(unpack(dtype, &pack(dtype, &[])).is_empty());
        assert_eq!(expected_bits(dtype, Op::Sqrt, u64::MAX), None);
    }
    for dtype in DTYPES {
        let nan = pool(dtype).into_iter().find(|&bits| is_nan(dtype, bits)).unwrap();
        for op in [Op::Floor, Op::Ceil, Op::Round, Op::Trunc] {
            assert_eq!(raw_integral_nan_bits(dtype, op, nan), Some(nan));
        }
    }
    assert_eq!(half_expected_bits(DType::F64, Op::Sqrt, 0), None);
    assert_eq!(staged_sqrt_bits(DType::F32, 0), None);
}
