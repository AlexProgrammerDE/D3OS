#![cfg(feature = "training")]
extern crate alloc;

#[path = "../src/validation/float_arithmetic.rs"]
#[allow(dead_code)]
mod float_arithmetic;
use burn::backend::DType;
use float_arithmetic::{DTYPES, Operation as Op, Values, broadcast_shape, chain_fixtures, expected_bits, is_nan, pack, pool, scalar_fixtures, unpack};

#[test]
fn exact_fixtures_cross_check_host_ieee_arithmetic_at_its_own_precision() {
    // Secondary checks only. Neither host float type computes fixture values.
    // F64 is never used as a universal oracle for F32 or half arithmetic.
    for dtype in [DType::F64, DType::F32, DType::Flex32] {
        for a in pool(dtype) {
            for b in pool(dtype) {
                for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
                    let actual = if dtype == DType::F64 {
                        let (x, y) = (f64::from_bits(a), f64::from_bits(b));
                        match op {
                            Op::Add => x + y,
                            Op::Sub => x - y,
                            Op::Mul => x * y,
                            Op::Div => x / y,
                            _ => unreachable!(),
                        }
                        .to_bits()
                    } else {
                        let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
                        u64::from(
                            match op {
                                Op::Add => x + y,
                                Op::Sub => x - y,
                                Op::Mul => x * y,
                                Op::Div => x / y,
                                _ => unreachable!(),
                            }
                            .to_bits(),
                        )
                    };
                    let expected = expected_bits(dtype, op, a, b).unwrap();
                    assert!(
                        actual == expected || (is_nan(dtype, actual) && is_nan(dtype, expected)),
                        "{dtype:?} {op:?} {a:x} {b:x}: {actual:x} != {expected:x}"
                    );
                }
            }
        }
    }
}

#[test]
fn halfway_cancellation_gradual_underflow_overflow_and_division_are_exact() {
    for dtype in DTYPES {
        let p = pool(dtype);
        let reference = |op, a: usize, b: usize| expected_bits(dtype, op, p[a], p[b]).unwrap();
        assert_eq!(reference(Op::Add, 2, 10), p[2]); // even midpoint at one
        assert_eq!(reference(Op::Add, 8, 10), p[8] + 1); // odd midpoint rounds upward
        assert_eq!(reference(Op::Add, 2, 3), p[0]);
        assert_eq!(reference(Op::Sub, 2, 2), p[0]);
        assert_eq!(reference(Op::Add, 1, 1), p[1]);
        assert_eq!(reference(Op::Sub, 18, 12), p[16]);
        assert_eq!(reference(Op::Add, 16, 12), p[18]);
        assert_eq!(reference(Op::Div, 12, 4), p[0]); // minsub / two is an even tie
        assert_eq!(reference(Op::Div, 14, 4), p[12]);
        assert_eq!(reference(Op::Mul, 20, 4), p[22]);
        assert_eq!(reference(Op::Div, 2, 6), p[30]);
        assert_eq!(reference(Op::Div, 2, 1), p[23]);
        assert_eq!(reference(Op::Div, 3, 22), p[1]);
        assert!(is_nan(dtype, reference(Op::Div, 0, 0)));
        assert!(is_nan(dtype, reference(Op::Mul, 0, 22)));
        assert!(is_nan(dtype, reference(Op::Sub, 22, 22)));
        assert_eq!(reference(Op::Sign, 24, 0), p[0]);
        assert_eq!(reference(Op::Sign, 23, 0), p[3]);
        assert_eq!(reference(Op::Neg, 0, 0), p[1]);
        assert_eq!(reference(Op::Abs, 1, 0), p[0]);
        assert_eq!(expected_bits(dtype, Op::Neg, p[26], 0), Some(p[26] ^ p[1]));
    }
}

#[test]
fn scalar_fixtures_preserve_the_f64_f32_half_schedule_and_variant_distinctions() {
    let f64 = scalar_fixtures(DType::F64);
    assert_eq!(f64[12].bits, (2.0f64).powi(-40).to_bits());
    assert_eq!(f64[13].bits, 0x3ff0_0000_0000_1000);
    assert_eq!(f64[14].bits, 1e100f64.to_bits());
    assert_eq!(f64[15].bits, 1);
    assert_eq!(f64[16].bits, 0x4340_0000_0000_0000); // RN64(2^53 + 1)
    assert_eq!(f64[17].bits, 0x43f0_0000_0000_0000); // RN64(u64::MAX)
    for dtype in [DType::F32, DType::Flex32] {
        let scalars = scalar_fixtures(dtype);
        assert_eq!(scalars[12].bits, 0x5e80_0000); // direct integer RN32 is ...0001
    }
    for (dtype, one) in [(DType::F16, 0x3c00), (DType::BF16, 0x3f80)] {
        let scalars = scalar_fixtures(dtype);
        assert_eq!(scalars[6].bits, one); // early half rounding, before multiplication
        assert_eq!(scalars[12].bits, one); // F64 below the half midpoint
        assert_eq!(scalars[13].bits, one); // F64 above midpoint first rounds to F32 midpoint
        let p = pool(dtype);
        assert_eq!(expected_bits(dtype, Op::Mul, p[6], scalars[6].bits), Some(p[6]));
    }
    for dtype in DTYPES {
        let p = pool(dtype);
        let s = scalar_fixtures(dtype);
        assert_eq!([s[0].bits, s[2].bits, s[4].bits], [p[0]; 3]);
        assert_eq!([s[1].bits, s[3].bits, s[5].bits, s[11].bits], [p[2]; 4]);
        assert_eq!(s[7].bits, p[1]);
        assert_eq!([s[8].bits, s[9].bits], [p[22], p[23]]);
        assert!(is_nan(dtype, s[10].bits));
    }
}

#[test]
fn separately_rounded_chains_cannot_be_replaced_by_fma() {
    for dtype in DTYPES {
        let p = pool(dtype);
        let chains = chain_fixtures(dtype);
        assert_eq!(chains.len(), 3);
        let fma = &chains[0];
        assert_eq!(fma.initial, p[28]);
        assert_eq!(fma.steps[0].expected, p[2]);
        assert_eq!(fma.steps[1].expected, p[0]);
        let fused = match dtype {
            DType::F64 => 0xbc90_0000_0000_0000,
            DType::F32 | DType::Flex32 => 0xb280_0000,
            DType::F16 => 0x8c00,
            DType::BF16 => 0xba80,
            _ => unreachable!(),
        };
        assert_eq!(fma.fused, Some(fused));
        assert_ne!(fma.fused, Some(fma.steps[1].expected));
        let cancellation = &chains[1];
        assert_eq!(cancellation.steps[0].expected, p[2]);
        assert_eq!(cancellation.steps[1].expected, p[0]);
        assert_eq!(cancellation.steps[2].expected, p[1]);
    }
}

#[test]
fn fixture_lookup_and_storage_do_not_narrow_wide_bits_or_hide_empty_shapes() {
    assert_eq!(broadcast_shape(&[0, 3], &[1, 3]), [0, 3]);
    assert_eq!(broadcast_shape(&[], &[]), Vec::<usize>::new());
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
        assert_eq!(expected_bits(dtype, Op::Add, u64::MAX, 0), None);
        for a in &p {
            for s in scalar_fixtures(dtype) {
                for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
                    assert!(expected_bits(dtype, op, *a, s.bits).is_some());
                }
            }
        }
    }
}
