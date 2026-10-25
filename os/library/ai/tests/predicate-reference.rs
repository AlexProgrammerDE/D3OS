#![cfg(feature = "training")]
use burn::backend::DType;
use d3os_ai::validation::predicate::{Classification as Class, Comparison as Cmp, classify_reference, compare_reference};

#[test]
fn integer_order_retains_bits_beyond_binary64_precision() {
    for (dtype, low, high) in [
        (DType::U64, 1 << 53, (1 << 53) + 1),
        (DType::U64, u64::MAX - 1, u64::MAX),
        (DType::U64, 1 << 63, (1 << 63) + 1),
        (DType::I64, i64::MAX as u64 - 1, i64::MAX as u64),
        (DType::I64, i64::MIN as u64, (i64::MIN + 1) as u64),
        (DType::I64, (-(1i64 << 53) - 1) as u64, (-(1i64 << 53)) as u64),
    ] {
        assert!(compare_reference(dtype, low, high, Cmp::Less));
        assert!(compare_reference(dtype, high, low, Cmp::Greater));
        assert!(!compare_reference(dtype, low, high, Cmp::Equal));
        assert!(compare_reference(dtype, high, high, Cmp::GreaterEqual));
        assert!(compare_reference(dtype, low, low, Cmp::LessEqual));
    }
    for dtype in [DType::I8, DType::I16, DType::I32, DType::I64] {
        let sign = 1u64 << (dtype.size() * 8 - 1);
        assert!(compare_reference(dtype, sign, sign - 1, Cmp::Less));
        assert!(compare_reference(dtype, u64::MAX, 0, Cmp::Less));
    }
    for dtype in [DType::U8, DType::U16, DType::U32, DType::U64] {
        let sign = 1u64 << (dtype.size() * 8 - 1);
        assert!(compare_reference(dtype, sign, sign - 1, Cmp::Greater));
        assert!(compare_reference(dtype, u64::MAX, 0, Cmp::Greater));
    }
}

#[test]
fn ieee_zeros_nan_and_subnormals_follow_partial_order() {
    for (dtype, negative_zero, nan, infinity, one) in [
        (DType::F64, 1 << 63, 0x7ff0_0000_0000_0001, 0x7ff0_0000_0000_0000, 1f64.to_bits()),
        (DType::F32, 1 << 31, 0x7f80_0001, 0x7f80_0000, u64::from(1f32.to_bits())),
        (DType::Flex32, 1 << 31, 0x7fc0_1234, 0x7f80_0000, u64::from(1f32.to_bits())),
        (DType::F16, 0x8000, 0x7c01, 0x7c00, 0x3c00),
        (DType::BF16, 0x8000, 0x7f81, 0x7f80, 0x3f80),
    ] {
        assert!(compare_reference(dtype, 0, negative_zero, Cmp::Equal));
        assert!(compare_reference(dtype, 0, negative_zero, Cmp::LessEqual));
        assert!(!compare_reference(dtype, 0, negative_zero, Cmp::Less));
        assert!(compare_reference(dtype, negative_zero | 1, 0, Cmp::Less));
        assert!(compare_reference(dtype, 1, 0, Cmp::Greater));
        assert!(compare_reference(dtype, 1, one, Cmp::Less));
        assert!(compare_reference(dtype, one, infinity, Cmp::Less));
        for rhs in [0, one, infinity, nan] {
            for op in Cmp::ALL {
                let expected = matches!(op, Cmp::NotEqual);
                assert_eq!(compare_reference(dtype, nan, rhs, op), expected);
                assert_eq!(compare_reference(dtype, rhs, nan, op), expected);
            }
        }
        assert!(classify_reference(dtype, nan | negative_zero, Class::Nan));
        assert!(classify_reference(dtype, infinity | negative_zero, Class::Infinite));
        assert!(classify_reference(dtype, 1, Class::Finite));
        assert!(!classify_reference(dtype, nan, Class::Finite));
        assert!(!classify_reference(dtype, infinity, Class::Finite));
    }
}

#[test]
fn exhaustive_half_class_counts_and_neighbor_orders_match_mathematical_anchors() {
    for (dtype, expected_nan, expected_finite, one, min_normal, min_normal_value) in [
        (DType::F16, 2046, 63488, 0x3c00, 0x0400, 2f64.powi(-14)),
        (DType::BF16, 254, 65280, 0x3f80, 0x0080, 2f64.powi(-126)),
    ] {
        let mut counts = [0; 3];
        for bits in 0..=u16::MAX {
            let value = u64::from(bits);
            for (i, class) in Class::ALL.into_iter().enumerate() {
                counts[i] += usize::from(classify_reference(dtype, value, class));
            }
            if bits < 0x7fff && classify_reference(dtype, value + 1, Class::Finite) {
                assert!(compare_reference(dtype, value, value + 1, Cmp::Less));
                assert!(compare_reference(dtype, value | 0x8000, (value + 1) | 0x8000, Cmp::Greater));
            }
        }
        assert_eq!(counts, [expected_nan, 2, expected_finite]);
        assert!(compare_reference(dtype, one - 1, one, Cmp::Less));
        assert!(compare_reference(dtype, one, one + 1, Cmp::Less));
        // Anchor the decoder to an independent mathematical value through F64.
        let host = if dtype == DType::F16 {
            burn::backend::f16::from_bits(min_normal).to_f64()
        } else {
            burn::backend::bf16::from_bits(min_normal).to_f64()
        };
        assert_eq!(host, min_normal_value);
        let tiny = if dtype == DType::F16 { 2f64.powi(-24) } else { 2f64.powi(-133) };
        let decoded = if dtype == DType::F16 {
            burn::backend::f16::from_bits(1).to_f64()
        } else {
            burn::backend::bf16::from_bits(1).to_f64()
        };
        assert_eq!(decoded, tiny);
    }
}
