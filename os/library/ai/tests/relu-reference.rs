#![cfg(feature = "training")]
extern crate alloc;

#[path = "../src/validation/relu.rs"]
#[allow(dead_code)]
mod relu;
use burn::backend::{DType, bf16, f16};
use relu::{DTYPES, Options, Values, backward_bits, expected_bits, expected_summary, gradient_pool, is_nan, lower_equal_zero, pack, pool, unpack};

#[test]
fn every_half_encoding_obeys_ordered_mask_and_preserves_unselected_payloads() {
    for dtype in [DType::F16, DType::BF16] {
        let mut nans = 0;
        let mut masked = 0;
        for bits in 0..=u16::MAX {
            // Secondary host classification is independent of the integer oracle.
            let value = if dtype == DType::F16 {
                f16::from_bits(bits).to_f32()
            } else {
                bf16::from_bits(bits).to_f32()
            };
            let bits = u64::from(bits);
            assert_eq!(is_nan(dtype, bits), value.is_nan());
            assert_eq!(lower_equal_zero(dtype, bits), value <= 0.0);
            let expected = if value <= 0.0 { 0 } else { bits };
            assert_eq!(expected_bits(dtype, bits), expected);
            assert_eq!(expected_bits(dtype, expected), expected);
            let gradient = bits ^ 0x8135;
            assert_eq!(backward_bits(dtype, expected, gradient), if value <= 0.0 { 0 } else { gradient });
            nans += usize::from(value.is_nan());
            masked += usize::from(value <= 0.0);
        }
        let expected_nans = if dtype == DType::F16 { 2046 } else { 254 };
        assert_eq!(nans, expected_nans);
        assert_eq!(masked, (65536 - expected_nans) / 2 + 1);
    }
}

#[test]
fn wide_ieee_boundaries_and_gradient_payloads_retain_exact_bits() {
    for dtype in DTYPES {
        let p = pool(dtype);
        assert_eq!(p.len(), relu::POOL_LENGTH);
        assert_eq!(p.iter().copied().collect::<std::collections::BTreeSet<_>>().len(), p.len());
        for (i, &bits) in p.iter().enumerate() {
            let (nan, nonpositive) = match dtype {
                DType::F64 => {
                    let x = f64::from_bits(bits);
                    (x.is_nan(), x <= 0.0)
                }
                DType::F32 | DType::Flex32 => {
                    let x = f32::from_bits(bits as u32);
                    (x.is_nan(), x <= 0.0)
                }
                DType::F16 => {
                    let x = f16::from_bits(bits as u16).to_f32();
                    (x.is_nan(), x <= 0.0)
                }
                DType::BF16 => {
                    let x = bf16::from_bits(bits as u16).to_f32();
                    (x.is_nan(), x <= 0.0)
                }
                _ => unreachable!(),
            };
            assert_eq!(is_nan(dtype, bits), nan);
            assert_eq!(lower_equal_zero(dtype, bits), nonpositive);
            assert_eq!(expected_bits(dtype, bits), if nonpositive { 0 } else { bits });
            let grad = gradient_pool(dtype)[i];
            assert_eq!(backward_bits(dtype, expected_bits(dtype, bits), grad), if nonpositive { 0 } else { grad });
        }
        assert_eq!(p.iter().filter(|&&x| is_nan(dtype, x)).count(), 8);
        assert_eq!(expected_bits(dtype, p[1]), 0);
        assert_eq!(expected_bits(dtype, p[2]), p[2]);
        assert_eq!(expected_bits(dtype, p[3]), 0);
        assert_eq!(unpack(dtype, &pack(dtype, &p)), p);
        assert_eq!(pack(dtype, &p).len(), p.len() * dtype.size());
        assert!(unpack(dtype, &pack(dtype, &[])).is_empty());
        assert_eq!(gradient_pool(dtype).iter().copied().collect::<std::collections::BTreeSet<_>>().len(), p.len());
    }
    // This positive F64 input cannot survive an intermediate F32 narrowing.
    for bits in [1, 0x3ff0_0000_0000_0001, 0x7fef_ffff_ffff_ffff, 0xfff0_0000_0000_0025] {
        assert_eq!(expected_bits(DType::F64, bits), bits);
    }
}

#[test]
fn logical_views_preserve_coordinates_through_signed_and_zero_strides() {
    let root = Values::new(&[3, 7], (0..21).collect());
    assert_eq!(
        root.slice(&[0, 1], &[1, 2], &[3, 3]).permute(&[1, 0]).flip(&[0, 1]).values,
        [19, 12, 5, 17, 10, 3, 15, 8, 1]
    );
    assert_eq!(
        Values::new(&[1, 3], vec![2, 5, 9]).flip(&[1]).expand(&[4, 3]).values,
        [9, 5, 2, 9, 5, 2, 9, 5, 2, 9, 5, 2]
    );
    assert!(Values::new(&[2, 0, 7], vec![]).permute(&[2, 0, 1]).values.is_empty());
    assert_eq!(Values::new(&[], vec![37]).values, [37]);
    let rank8 = Values::new(&[1, 2, 1, 1, 1, 2, 1, 3], (0..12).collect())
        .flip(&[7])
        .expand(&[1, 2, 1, 2, 1, 2, 1, 3]);
    assert_eq!(rank8.values, [2, 1, 0, 5, 4, 3, 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 8, 7, 6, 11, 10, 9]);
    assert_eq!(root.values, (0..21).collect::<Vec<_>>());
}

#[test]
fn corpus_totals_and_filters_describe_actual_bounded_work() {
    let full = expected_summary(&Options::default());
    assert_eq!(full.dtypes, 5);
    assert_eq!(full.forward_cases, 75);
    assert_eq!(full.backward_cases, 75);
    assert_eq!(full.idempotence_cases, 75);
    assert_eq!(full.high_level_cases, 5);
    assert_eq!(full.unique_cases, 5);
    assert_eq!(full.mask_cases, 10);
    assert_eq!(full.cases(), 85);
    assert_eq!(full.chain_steps, 245);
    assert_eq!(full.source_checks, 80);
    assert_eq!(full.alias_checks, 80);
    assert_eq!(full.gradient_checks, 75);
    assert_eq!(full.output_checks, 75);
    assert_eq!(full.poison_checks, 85);
    assert_eq!(full.values, 12200);
    assert_eq!(full.nan_values, 2700);
    assert_eq!(full.bool_values, 320);
    for dtype in DTYPES {
        let single = expected_summary(&Options {
            dtype: Some(dtype),
            ..Options::default()
        });
        assert_eq!(single.dtypes, 1);
        assert_eq!(single.values * 5, full.values);
        assert_eq!(single.nan_values * 5, full.nan_values);
        let no_rank0 = expected_summary(&Options {
            dtype: Some(dtype),
            skip_rank_zero: true,
            ..Options::default()
        });
        assert_eq!(single.forward_cases - no_rank0.forward_cases, 1);
        assert_eq!(single.values - no_rank0.values, 3);
        assert_eq!(single.chain_steps - no_rank0.chain_steps, 3);
    }
    let no_flex = expected_summary(&Options {
        no_flex32: true,
        ..Options::default()
    });
    assert_eq!(no_flex.dtypes, 4);
    assert_eq!(no_flex.values * 5, full.values * 4);
    assert_eq!(
        expected_summary(&Options {
            dtype: Some(DType::Flex32),
            no_flex32: true,
            ..Options::default()
        }),
        relu::Summary::default()
    );
}
