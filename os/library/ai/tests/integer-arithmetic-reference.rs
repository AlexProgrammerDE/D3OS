#![cfg(feature = "training")]
extern crate alloc;

// Allow host-only verification before the parent registers the new module.
#[path = "../src/validation/integer_arithmetic.rs"]
#[allow(dead_code)]
mod integer_arithmetic;
use burn::backend::{DType, Scalar};
use integer_arithmetic::{
    DTYPES, Operation as Op, Values, broadcast_shape, decode, exponent_bits, mask, normalize_scalar, pack, portable, power_reference, reference,
    scalar_exponent, unpack,
};

#[test]
fn byte_binary_domain_matches_primitive_arithmetic_and_floor_modulo() {
    for a in 0..=255u64 {
        for b in 0..=255u64 {
            let (x, y) = (a as u8, b as u8);
            for (op, expected) in [
                (Op::Add, x.wrapping_add(y)),
                (Op::Sub, x.wrapping_sub(y)),
                (Op::Mul, x.wrapping_mul(y)),
                (Op::And, x & y),
                (Op::Or, x | y),
                (Op::Xor, x ^ y),
            ] {
                assert_eq!(reference(DType::U8, op, a, DType::U8, b), u64::from(expected));
            }
            let (x, y) = (i16::from(a as i8), i16::from(b as i8));
            if y != 0 {
                let q = x / y;
                let mut r = x - q * y;
                if r != 0 && (r < 0) != (y < 0) {
                    r += y;
                }
                assert_eq!(reference(DType::I8, Op::Div, a, DType::I8, b), (q as u8) as u64);
                assert_eq!(reference(DType::I8, Op::FloorRem, a, DType::I8, b), (r as u8) as u64);
                assert!(r == 0 || r.signum() == y.signum());
                assert!(r.abs() < y.abs());
            }
            let bits = exponent_bits(DType::I8, b);
            assert_eq!(power_reference(DType::I8, a, bits), (a as i8).wrapping_pow(bits) as u8 as u64);
            assert_eq!(power_reference(DType::U8, a, b as u32), u64::from((a as u8).wrapping_pow(b as u32)));
        }
    }
}

#[test]
fn wide_oracle_retains_carries_borrows_and_bits_above_float_precision() {
    let edge = [0, 1, 2, (1 << 24) + 1, (1 << 32) + 1, (1 << 53) + 1, 1 << 63, u64::MAX - 1, u64::MAX];
    for &a in &edge {
        for &b in &edge {
            for (op, expected) in [(Op::Add, a.wrapping_add(b)), (Op::Sub, a.wrapping_sub(b)), (Op::Mul, a.wrapping_mul(b))] {
                assert_eq!(reference(DType::U64, op, a, DType::U64, b), expected);
                assert_eq!(reference(DType::I64, op, a, DType::I64, b), expected);
            }
            if b != 0 {
                assert_eq!(reference(DType::U64, Op::Div, a, DType::U64, b), a / b);
                assert_eq!(reference(DType::U64, Op::FloorRem, a, DType::U64, b), a % b);
                let (x, y) = (a as i64, b as i64);
                if !(x == i64::MIN && y == -1) {
                    let rem = x % y;
                    let floor = if rem != 0 && rem.signum() != y.signum() { rem + y } else { rem };
                    assert_eq!(reference(DType::I64, Op::Div, a, DType::I64, b), (x / y) as u64);
                    assert_eq!(reference(DType::I64, Op::FloorRem, a, DType::I64, b), floor as u64);
                }
            }
        }
        for exponent in [0, 1, 2, 3, 31, 32, 33, 1 << 31, u32::MAX] {
            assert_eq!(power_reference(DType::U64, a, exponent), a.wrapping_pow(exponent));
        }
    }
    for i in 0..257u64 {
        let a = i.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(i as u32 % 64);
        for dtype in DTYPES {
            for count in [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, u64::MAX] {
                let shift = (count & u64::from(integer_arithmetic::width(dtype) - 1)) as u32;
                assert_eq!(reference(dtype, Op::Shl, a, dtype, count), a.wrapping_shl(shift) & mask(dtype));
                let expected = if dtype.is_int() {
                    (decode(dtype, a) >> shift) as u64 & mask(dtype)
                } else {
                    (a & mask(dtype)) >> shift
                };
                assert_eq!(reference(dtype, Op::Shr, a, dtype, count), expected);
            }
        }
    }
}

#[test]
fn exponent_format_and_public_scalar_shortcuts_are_independent_of_base_width() {
    for dtype in DTYPES {
        assert_eq!(
            exponent_bits(dtype, mask(dtype)),
            if dtype.is_int() || integer_arithmetic::width(dtype) >= 32 {
                u32::MAX
            } else {
                mask(dtype) as u32
            }
        );
        assert_eq!(power_reference(dtype, 0, 0), 1);
        assert_eq!(scalar_exponent(dtype, Scalar::Float(-0.75)).unwrap(), 0);
        for (scalar, exponent) in [(0.75, 0), (1.75, 1), (2.75, 2), (3.75, 3)] {
            assert_eq!(scalar_exponent(dtype, Scalar::Float(scalar)).unwrap(), exponent);
        }
        assert!(scalar_exponent(dtype, Scalar::UInt(u64::MAX)).is_err());
    }
    assert_eq!(exponent_bits(DType::I8, 0x80), 0xffff_ff80);
    assert_eq!(exponent_bits(DType::U8, 0x80), 128);
    assert_eq!(exponent_bits(DType::I64, 0xdead_beef_8000_0001), 0x8000_0001);
    assert_eq!(exponent_bits(DType::U64, 0xffff_ffff_0000_0003), 3);
    for base_dtype in DTYPES {
        for rhs_dtype in DTYPES {
            for rhs in [0, 1, 2, 255, 65535, 0x1_0000_0001, u64::MAX] {
                let exponent = exponent_bits(rhs_dtype, rhs);
                assert_eq!(
                    reference(base_dtype, Op::Powi, 3, rhs_dtype, rhs),
                    3u64.wrapping_pow(exponent) & mask(base_dtype)
                );
            }
        }
    }
    assert_eq!(normalize_scalar(DType::I64, Scalar::Int((1 << 53) + 1)).unwrap(), (1 << 53) + 1);
    assert_eq!(normalize_scalar(DType::U64, Scalar::UInt(u64::MAX)).unwrap(), u64::MAX);
    assert_eq!(normalize_scalar(DType::I64, Scalar::Int(i64::MIN)).unwrap(), 1 << 63);
    assert!(normalize_scalar(DType::U8, Scalar::Int(-1)).is_err());
    assert!(normalize_scalar(DType::I8, Scalar::UInt(128)).is_err());
    assert!(normalize_scalar(DType::I32, Scalar::UInt(1 << 31)).is_err());
    assert_eq!(scalar_exponent(DType::U8, Scalar::UInt(200)).unwrap(), 200);
    assert!(scalar_exponent(DType::U8, Scalar::UInt(256)).is_err());
    assert_eq!(normalize_scalar(DType::U8, Scalar::Float(255.75)).unwrap(), 255);
    assert_eq!(normalize_scalar(DType::I8, Scalar::Float(-128.75)).unwrap(), 128);
}

#[test]
fn native_exceptions_are_separate_from_portable_signed_domain() {
    for dtype in DTYPES {
        let sign = 1u64 << (integer_arithmetic::width(dtype) - 1);
        let all = mask(dtype);
        for a in [0, 1, sign, all] {
            assert!(!portable(dtype, Op::Div, a, 0));
            assert!(!portable(dtype, Op::FloorRem, a, 0));
            assert_eq!(reference(dtype, Op::Div, a, dtype, 0), all);
            assert_eq!(reference(dtype, Op::FloorRem, a, dtype, 0), a);
            assert!(!portable(dtype, Op::Shl, a, u64::from(integer_arithmetic::width(dtype))));
        }
        if dtype.is_int() {
            assert!(!portable(dtype, Op::Div, sign, all));
            assert!(!portable(dtype, Op::FloorRem, sign, all));
            assert!(!portable(dtype, Op::Abs, sign, 0));
            assert!(!portable(dtype, Op::Neg, sign, 0));
            assert!(!portable(dtype, Op::Shl, all, 1));
            assert_eq!(reference(dtype, Op::Div, sign, dtype, all), sign);
            assert_eq!(reference(dtype, Op::FloorRem, sign, dtype, all), 0);
            assert_eq!(reference(dtype, Op::Abs, sign, dtype, 0), sign);
            assert_eq!(reference(dtype, Op::Neg, sign, dtype, 0), sign);
            assert_eq!(reference(dtype, Op::Sign, all, dtype, 0), all);
        } else {
            assert!(portable(dtype, Op::Add, all, 1));
            assert_eq!(reference(dtype, Op::Add, all, dtype, 1), 0);
            assert_eq!(reference(dtype, Op::Sign, all, dtype, 0), 1);
        }
    }
}

#[test]
fn view_mapping_and_packing_keep_logical_order_and_every_payload_bit() {
    let source = Values::new(&[3, 7], (0..21).map(|v| 0xfedc_ba98_7654_3200 + v).collect());
    let view = source.slice(&[0, 1], &[1, 2], &[3, 3]).permute(&[1, 0]).flip(&[0, 1]);
    assert_eq!(view.values, [19, 12, 5, 17, 10, 3, 15, 8, 1].map(|v| 0xfedc_ba98_7654_3200 + v));
    assert_eq!(broadcast_shape(&[3, 1], &[1, 7]), [3, 7]);
    assert_eq!(broadcast_shape(&[0, 3], &[1, 3]), [0, 3]);
    assert!(broadcast_shape(&[], &[]).is_empty());
    assert_eq!(Values::new(&[], vec![u64::MAX]).flip(&[]).values, [u64::MAX]);
    assert!(Values::new(&[2, 0, 7], vec![]).permute(&[2, 0, 1]).values.is_empty());
    for dtype in DTYPES {
        for length in [0, 1, 3, 127, 128, 129, 257] {
            let values: Vec<_> = (0..length).map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) & mask(dtype)).collect();
            let bytes = pack(dtype, &values);
            assert_eq!(bytes.len(), length * dtype.size());
            assert_eq!(unpack(dtype, &bytes), values);
            if dtype == DType::U64 && length > 1 {
                assert_eq!(&bytes[8..16], &0x9e37_79b9_7f4a_7c15u64.to_le_bytes());
            }
        }
    }
    assert_eq!(source.values[20], 0xfedc_ba98_7654_3214);
}
