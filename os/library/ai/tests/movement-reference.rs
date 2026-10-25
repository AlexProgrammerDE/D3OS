#![cfg(feature = "training")]
extern crate alloc;

// Keep host references independent of command and module registration.
#[path = "../src/validation/movement.rs"]
#[allow(dead_code)]
mod movement;
use burn::backend::ops::PadMode;
use movement::{Options, Records, expected_summary, scatter_or_reference};

#[test]
fn views_and_indexing_keep_opaque_multi_byte_records() {
    let bytes: Vec<_> = (0u64..12).flat_map(|i| (0x7ff0_1234_0000_0000 | i).to_le_bytes()).collect();
    let source = Records::new(&[3, 4], 8, &bytes);
    let view = source.slice(&[0, 1], &[1, 2], &[3, 2]).permute(&[1, 0]).flip(&[0, 1]);
    let expected = [11, 7, 3, 9, 5, 1];
    let encode = |indices: &[u64]| indices.iter().flat_map(|&i| (0x7ff0_1234_0000_0000 | i).to_le_bytes()).collect::<Vec<_>>();
    assert_eq!(view.shape, [2, 3]);
    assert_eq!(view.bytes(), encode(&expected));
    assert_eq!(view.select(1, &[2, 0, 2]).bytes(), encode(&[3, 11, 3, 1, 9, 1]));
    assert_eq!(view.gather(0, &[2, 3], &[1, 0, 1, 0, 1, 0]).bytes(), encode(&[9, 7, 1, 11, 5, 3]));
    assert_eq!(view.gather(1, &[2, 3], &[1, 1, 0, 2, 0, 2]).bytes(), encode(&[7, 7, 11, 1, 9, 1]));
    assert_eq!(source.bytes(), bytes);
}

#[test]
fn assignment_snapshots_aliases_and_respects_destination_steps() {
    let source = Records::new(&[8], 1, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let overlap = source.slice(&[0], &[1], &[6]);
    assert_eq!(source.assign(&[1], &[1], &overlap).bytes(), [0, 0, 1, 2, 3, 4, 5, 7]);
    let reversed = source.slice(&[1], &[2], &[3]).flip(&[0]);
    assert_eq!(source.assign(&[0], &[3], &reversed).bytes(), [5, 1, 2, 3, 4, 5, 1, 7]);
    assert_eq!(source.bytes(), [0, 1, 2, 3, 4, 5, 6, 7]);
    let empty = Records::new(&[0], 1, &[]);
    assert_eq!(source.assign(&[8], &[2], &empty), source);
}

#[test]
fn broadcast_masks_and_compaction_preserve_full_integer_records() {
    let values = [0xfedc_ba98_7654_3210u64, (1 << 53) + 1, u64::MAX, 1 << 63, 0, 7];
    let raw: Vec<_> = values.into_iter().flat_map(u64::to_le_bytes).collect();
    let tensor = Records::new(&[2, 3], 8, &raw);
    let replacement = Records::new(&[1, 3], 8, &[101u64, 102, 103].into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>());
    let mask = [false, true, false, true, false, true];
    let decode = |records: Records| {
        records
            .bytes()
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    assert_eq!(decode(tensor.mask_where(&mask, &replacement)), [values[0], 102, values[2], 101, values[4], 103]);
    assert_eq!(decode(tensor.mask_select(&mask)), [values[1], values[3], values[5]]);
    assert_eq!(tensor.mask_select(&[false; 6]).shape, [0]);
    assert_eq!(tensor.mask_select(&[true; 6]).bytes(), raw);
    let column = Records::new(&[2, 1], 1, &[1, 2]);
    assert_eq!(column.broadcast(&[2, 3]).bytes(), [1, 1, 1, 2, 2, 2]);
}

#[test]
fn cat_and_pad_follow_logical_axes_and_boundary_rules() {
    let tensor = Records::new(&[2, 2], 1, &[1, 2, 3, 4]);
    let other = Records::new(&[2, 1], 1, &[5, 6]);
    assert_eq!(tensor.cat(&other, 1).bytes(), [1, 2, 5, 3, 4, 6]);
    assert_eq!(tensor.cat(&tensor.flip(&[0]), 0).bytes(), [1, 2, 3, 4, 3, 4, 1, 2]);
    let padding = [(1, 1), (1, 1)];
    assert_eq!(
        tensor.pad(&padding, PadMode::Constant(9.), &[9]).bytes(),
        [9, 9, 9, 9, 9, 1, 2, 9, 9, 3, 4, 9, 9, 9, 9, 9]
    );
    assert_eq!(
        tensor.pad(&padding, PadMode::Edge, &[0]).bytes(),
        [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
    );
    assert_eq!(
        tensor.pad(&padding, PadMode::Reflect, &[0]).bytes(),
        [4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1]
    );
    let empty = Records::new(&[0, 2], 1, &[]);
    let padded = empty.pad(&padding, PadMode::Constant(9.), &[9]);
    assert_eq!(padded.shape, [2, 4]);
    assert_eq!(padded.bytes(), [9; 8]);
    assert!(empty.cat(&empty, 1).bytes().is_empty());
}

#[test]
fn duplicate_boolean_scatter_uses_or_without_overwriting_true_values() {
    for width in [1, 4] {
        let encode = |values: &[u32]| values.iter().flat_map(|v| v.to_le_bytes()[..width].to_vec()).collect::<Vec<_>>();
        let base = Records::new(&[2, 3], width, &encode(&[0, 1, 0, 0, 0, 1]));
        let value = Records::new(&[2, 3], width, &encode(&[1, 0, 0, 0, 1, 0]));
        let result = scatter_or_reference(&base, 1, &[0, 0, 2, 1, 1, 2], &value);
        assert_eq!(result.bytes(), encode(&[1, 1, 0, 0, 1, 1]));
        assert_eq!(base.bytes(), encode(&[0, 1, 0, 0, 0, 1]));
    }
}

#[test]
fn summary_options_keep_boolean_scopes_when_flex32_is_excluded() {
    let full = expected_summary(&Options::default());
    let linux = expected_summary(&Options {
        include_flex32: false,
        ..Default::default()
    });
    assert_eq!(full.dtypes, 15);
    assert_eq!(linux.dtypes, 14);
    assert_eq!(full.cases(), 13 * (6 * 53 - 1) + 2 * 6 * 56);
    assert_eq!(linux.cases(), full.cases() - (6 * 53 - 1));
    assert_eq!(full.source_checks, 13 * 6 * 33 + 2 * 6 * 35);
    assert_eq!(linux.source_checks, full.source_checks - 6 * 33);
    assert_eq!(full.argwhere, 60);
    assert_eq!(linux.argwhere, full.argwhere);
    assert_eq!(full.bool_scatter_or, 24);
    assert_eq!(linux.bool_scatter_or, full.bool_scatter_or);
    assert_eq!(full.pad - linux.pad, 23);
    assert_eq!(full.values, 1_273_900);
    assert_eq!(linux.values, 1_188_924);
}

#[test]
fn empty_boolean_scatter_diagnostic_keeps_every_other_empty_case() {
    let full = expected_summary(&Options {
        include_flex32: false,
        ..Default::default()
    });
    let diagnostic = expected_summary(&Options {
        include_flex32: false,
        skip_empty_bool_scatter: true,
        ..Default::default()
    });
    assert_eq!(full.cases() - diagnostic.cases(), 4);
    assert_eq!(full.source_checks - diagnostic.source_checks, 4);
    assert_eq!(full.values, diagnostic.values);
    assert_eq!(full.bool_scatter_or - diagnostic.bool_scatter_or, 4);
    assert_eq!(full.gather, diagnostic.gather);
    assert_eq!(full.mask_select, diagnostic.mask_select);
    assert_eq!(full.argwhere, diagnostic.argwhere);
}

#[test]
fn matching_boolean_masks_diagnostic_excludes_only_mixed_where_and_fill() {
    let options = Options {
        include_flex32: false,
        skip_empty_bool_scatter: true,
        ..Default::default()
    };
    let full = expected_summary(&options);
    let matching = expected_summary(&Options {
        matching_bool_masks_only: true,
        ..options
    });
    assert_eq!(full.cases() - matching.cases(), 144);
    assert_eq!(full.mask_where - matching.mask_where, 72);
    assert_eq!(full.mask_fill - matching.mask_fill, 72);
    assert_eq!(full.values - matching.values, 41_088);
    assert_eq!(full.source_checks, matching.source_checks);
    assert_eq!(full.mask_select, matching.mask_select);
    assert_eq!(full.argwhere, matching.argwhere);
    assert_eq!(full.bool_scatter_or, matching.bool_scatter_or);
}
