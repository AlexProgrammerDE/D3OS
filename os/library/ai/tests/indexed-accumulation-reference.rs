use d3os_ai::{
    burn::backend::DType,
    validation::indexed_accumulation::{self as indexed, ExecutionScope, Expected, Operation, Options},
};
use std::collections::BTreeSet;

#[test]
fn scope_selection_preserves_fixed_rows_and_separates_model_counts() {
    let native = indexed::population(&Options {
        scope: ExecutionScope::NativeValues,
        dtype: None,
    })
    .unwrap();
    let errors = indexed::population(&Options {
        scope: ExecutionScope::NativeErrors,
        dtype: None,
    })
    .unwrap();
    let common = indexed::population(&Options::default()).unwrap();
    assert_eq!((native.cases().len(), errors.cases().len(), common.cases().len()), (184, 8, 167));
    let all: BTreeSet<_> = native.cases().iter().chain(errors.cases()).map(|c| c.id).collect();
    assert_eq!(all.len(), 192);
    assert!(
        common
            .cases()
            .iter()
            .all(|c| c.dtype != DType::F64 && !matches!(c.expected, Expected::Error(_)))
    );
    assert_eq!(
        (
            native.required_observations().len(),
            errors.required_observations().len(),
            common.required_observations().len()
        ),
        (2556, 122, 2307)
    );
    let models = indexed::run_models(&Options {
        scope: ExecutionScope::ModelReference,
        dtype: None,
    });
    assert!(models.is_ok(), "{:?}", models.failures());
    assert_eq!(
        (
            models.device_counts().unwrap().attempted,
            models.device_counts().unwrap().passed,
            models.model_attempted(),
            models.model_passed()
        ),
        (0, 0, 283, 283)
    );
}

#[test]
fn signed_views_and_aliases_refer_to_original_storage_exactly() {
    for case in indexed::corpus() {
        for source in case.sources {
            let mut coordinates = vec![vec![]];
            for &size in source.shape {
                coordinates = coordinates
                    .into_iter()
                    .flat_map(|prefix| {
                        (0..size).map(move |i| {
                            let mut c = prefix.clone();
                            c.push(i);
                            c
                        })
                    })
                    .collect();
            }
            let offsets: Vec<_> = coordinates
                .iter()
                .map(|c| source.offset_elements as i128 + c.iter().zip(source.strides_elements).map(|(&i, &s)| i as i128 * i128::from(s)).sum::<i128>())
                .collect();
            assert_eq!(offsets, source.logical_to_storage.iter().map(|&i| i as i128).collect::<Vec<_>>());
            assert!(offsets.iter().all(|&i| i >= 0 && i < (source.storage_bits.len() as i128)));
            for alias in case.sources.iter().filter(|s| s.storage_id == source.storage_id) {
                assert_eq!(
                    (alias.dtype, alias.storage_shape, alias.storage_bits),
                    (source.dtype, source.storage_shape, source.storage_bits)
                );
            }
        }
    }
}

#[test]
fn declared_width_arithmetic_keeps_full_signedness_and_per_update_rounding() {
    for dtype in [DType::I8, DType::I16, DType::I32, DType::I64, DType::U8, DType::U16, DType::U32, DType::U64] {
        let width = dtype.size() * 8;
        let mask = u64::MAX >> (64 - width);
        let high = 1u64 << (width - 1);
        assert_eq!(indexed::reference_step(dtype, mask, 1, Operation::Add), 0);
        assert_eq!(indexed::reference_step(dtype, mask, mask, Operation::Mul), 1);
        assert_eq!(
            indexed::reference_step(dtype, high, high - 1, Operation::Min),
            if dtype.is_int() { high } else { high - 1 }
        );
    }
    let first = indexed::reference_step(DType::BF16, 0x3fc0, 0x3f81, Operation::Mul);
    let staged = indexed::reference_step(DType::BF16, first, 0x3f81, Operation::Mul);
    assert_eq!((first, staged), (0x3fc2, 0x3fc4));
    assert_eq!(
        indexed::reference_step(DType::F64, 0x3ff0000000000000, 0x3cb0000000000000, Operation::Add),
        0x3ff0000000000001
    );
}

#[test]
fn model_identity_uses_whole_attachment_and_public_gradients_keep_original_ranks() {
    let mut keys = BTreeSet::new();
    let mut patterns = 0;
    for boundary in indexed::symbolic_corpus() {
        for attachment in boundary.attachments {
            assert!(keys.insert(attachment.key));
            patterns += usize::from(attachment.class_pattern);
        }
    }
    assert_eq!((keys.len(), patterns), (283, 132));
    let gradients: Vec<_> = indexed::corpus().iter().filter(|c| matches!(c.expected, Expected::Gradient(_))).collect();
    assert_eq!(gradients.len(), 20);
    for case in gradients {
        let Expected::Gradient(g) = case.expected else { unreachable!() };
        assert_eq!(g.leaf_gradient.shape.len(), case.design_rank);
        assert_eq!((g.root.dtype, g.seed.dtype, g.leaf_gradient.dtype), (case.dtype, case.dtype, case.dtype));
        assert_eq!(g.root.shape, g.seed.shape);
    }
}
