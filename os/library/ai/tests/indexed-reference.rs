#![cfg(feature = "training")]
extern crate alloc;

// Keep references runnable before command/module registration.
#[path = "../src/validation/indexed.rs"]
#[allow(dead_code)]
mod indexed;
use burn::backend::{DType, IntDType};
use indexed::{Options, Records, argwhere_reference, expected_summary, ordered_reference};

fn records(shape: &[usize], values: &[u64]) -> Records {
    let bytes: Vec<_> = values.iter().flat_map(|v| (0x7ff0_1234_0000_0000 | v).to_le_bytes()).collect();
    Records::new(shape, 8, &bytes)
}
fn labels(data: &Records) -> Vec<u64> {
    data.bytes()
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()) & 0xffff_ffff)
        .collect()
}
fn indices(data: &Records) -> Vec<u64> {
    let count = data.shape.iter().product::<usize>();
    if count == 0 {
        return Vec::new();
    }
    let width = data.bytes().len() / count;
    data.bytes()
        .chunks_exact(width)
        .map(|b| {
            let mut word = [0; 8];
            word[..b.len()].copy_from_slice(b);
            u64::from_le_bytes(word)
        })
        .collect()
}

#[test]
fn strided_transposed_negative_views_index_logical_coordinates() {
    let source = records(&[3, 7], &(0..21).collect::<Vec<_>>());
    let view = source.slice(&[0, 1], &[1, 2], &[3, 3]).permute(&[1, 0]).flip(&[0, 1]);
    assert_eq!(view, records(&[3, 3], &[19, 12, 5, 17, 10, 3, 15, 8, 1]));
    assert_eq!(labels(&view.select(1, &[2, 0, 2])), [5, 19, 5, 3, 17, 3, 1, 15, 1]);
    assert_eq!(labels(&view.gather(0, &[2, 3], &[2, 1, 0, 0, 1, 2])), [15, 10, 5, 19, 10, 1]);
    assert_eq!(labels(&source), (0..21).collect::<Vec<_>>());
}
#[test]
fn nd_tuple_grid_preserves_tuple_axis_and_trailing_slice_order() {
    let source = records(&[2, 3, 2], &(0..12).collect::<Vec<_>>());
    let view = source.permute(&[1, 0, 2]).flip(&[0, 2]);
    assert_eq!(labels(&view), [5, 4, 11, 10, 3, 2, 9, 8, 1, 0, 7, 6]);
    let ids = [2, 1, 0, 0, 1, 1, 2, 0];
    let gathered = view.gather_nd(&[2, 2, 2], &ids);
    assert_eq!(gathered.shape, [2, 2, 2]);
    assert_eq!(labels(&gathered), [7, 6, 5, 4, 9, 8, 1, 0]);
    let single = view.gather_nd(&[2], &[1, 0]);
    assert_eq!(single.shape, [2]);
    assert_eq!(labels(&single), [3, 2]);
    let scalar = view.gather_nd(&[3], &[1, 1, 0]);
    assert!(scalar.shape.is_empty());
    assert_eq!(labels(&scalar), [9]);
    let empty = records(&[2, 0, 3], &[]).gather_nd(&[2, 1], &[1, 0]);
    assert_eq!(empty.shape, [2, 0, 3]);
    assert!(empty.bytes().is_empty());
}
#[test]
fn assignments_snapshot_overlapping_source_views_and_keep_untouched_records() {
    let source = records(&[2, 4], &(0..8).collect::<Vec<_>>());
    let values = source.slice(&[0, 0], &[1, 1], &[2, 3]).flip(&[0, 1]);
    assert_eq!(labels(&values), [6, 5, 4, 2, 1, 0]);
    let selected = source.select_assign(1, &[3, 0, 2], &values);
    assert_eq!(labels(&selected), [5, 1, 4, 6, 1, 5, 0, 2]);
    let scattered = source.scatter(1, &[3, 0, 2, 2, 1, 0], &values);
    assert_eq!(labels(&scattered), [5, 1, 4, 6, 0, 1, 2, 7]);
    let nd = source.scatter_nd(&[2, 1], &[1, 0], &source.flip(&[1]));
    assert_eq!(labels(&nd), [7, 6, 5, 4, 3, 2, 1, 0]);
    assert_eq!(labels(&source), (0..8).collect::<Vec<_>>());
}
#[test]
fn nd_assign_handles_multidimensional_tuple_grids_and_scalar_values() {
    let source = records(&[2, 3, 2], &(0..12).collect::<Vec<_>>());
    let values = records(&[2, 1, 2], &[100, 101, 102, 103]);
    let result = source.scatter_nd(&[2, 1, 2], &[1, 2, 0, 1], &values);
    assert_eq!(labels(&result), [0, 1, 102, 103, 4, 5, 6, 7, 8, 9, 100, 101]);
    let scalar = records(&[], &[0xabcd]);
    let result = source.scatter_nd(&[3], &[1, 0, 1], &scalar);
    assert_eq!(labels(&result), [0, 1, 2, 3, 4, 5, 6, 0xabcd, 8, 9, 10, 11]);
}
#[test]
#[should_panic(expected = "unique destinations")]
fn exact_assignment_reference_rejects_ambiguous_duplicate_destinations() {
    let source = records(&[3], &[0, 1, 2]);
    source.scatter_nd(&[2, 1], &[1, 1], &records(&[2], &[11, 12]));
}
#[test]
fn ordering_maps_each_axis_to_original_records_and_encodes_all_index_widths() {
    let signed = [i32::MIN, 7, i32::MAX, -3, 5, 0];
    let data = Records::new(&[2, 3], 4, &signed.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    for out in indexed::INDEX_DTYPES {
        let (values, ids) = ordered_reference(&data, DType::I32, 1, true, 2, out);
        let decode = |r: &Records| r.bytes().chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        assert_eq!(values.shape, [2, 2]);
        assert_eq!(decode(&values), [i32::MAX, 7, 5, 0]);
        assert_eq!(indices(&ids), [2, 1, 1, 2]);
        let (values, ids) = ordered_reference(&data, DType::I32, 0, false, 1, out);
        assert_eq!(values.shape, [1, 3]);
        assert_eq!(decode(&values), [i32::MIN, 5, 0]);
        assert_eq!(indices(&ids), [0, 1, 1]);
        let (values, ids) = ordered_reference(&data, DType::I32, 1, false, 0, out);
        assert_eq!(values.shape, [2, 0]);
        assert_eq!(ids.shape, [2, 0]);
        assert!(values.bytes().is_empty());
        assert!(ids.bytes().is_empty());
    }
    // Stable ties are ordered by their original logical axis index.
    let floats = [2f32, -1., 2., -1.];
    let data = Records::new(&[4], 4, &floats.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let (values, ids) = ordered_reference(&data, DType::F32, 0, true, 4, IntDType::U64);
    assert_eq!(indices(&ids), [0, 2, 1, 3]);
    assert_eq!(values.bytes(), [2f32, 2., -1., -1.].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
}
#[test]
fn argwhere_uses_logical_coordinates_for_both_stores_and_all_index_widths() {
    for width in [1, 4] {
        let bytes: Vec<_> = [0u32, 1, 0, 1, 1, 0].iter().flat_map(|v| v.to_le_bytes()[..width].to_vec()).collect();
        let view = Records::new(&[2, 3], width, &bytes).permute(&[1, 0]).flip(&[0]);
        for out in indexed::INDEX_DTYPES {
            let result = argwhere_reference(&view, out);
            assert_eq!(result.shape, [3, 2]);
            assert_eq!(indices(&result), [1, 0, 1, 1, 2, 1]);
            let empty = argwhere_reference(&Records::new(&[2, 0, 3], width, &[]), out);
            assert_eq!(empty.shape, [0, 3]);
            assert!(empty.bytes().is_empty());
        }
    }
}
#[test]
fn summaries_scope_only_flex32_and_rank_zero_nd_cases() {
    let full = expected_summary(&Options::default());
    let scoped = expected_summary(&Options {
        include_flex32: false,
        include_scalar_nd: false,
    });
    assert_eq!(full.dtypes, 15);
    assert_eq!(scoped.dtypes, 14);
    assert_eq!(full.index_dtypes, 8);
    assert_eq!(full.unsupported_bool_nd, 4);
    assert_eq!(scoped.unsupported_bool_nd, 4);
    assert_eq!(full.argwhere, 2 * 8 * 3 * 8);
    for (a, b) in [
        (full.argmin, scoped.argmin),
        (full.argmax, scoped.argmax),
        (full.sort_pairs, scoped.sort_pairs),
        (full.topk_pairs, scoped.topk_pairs),
        (full.argsort, scoped.argsort),
        (full.argtopk, scoped.argtopk),
        (full.argwhere, scoped.argwhere),
    ] {
        assert_eq!(a, b);
    }
    assert!(full.gather > scoped.gather);
    assert!(full.gather_nd > scoped.gather_nd);
    assert!(full.scatter_assign > scoped.scatter_assign);
    assert!(full.scatter_nd_assign > scoped.scatter_nd_assign);
    assert!(full.source_checks > scoped.source_checks);
    assert!(full.values > scoped.values);
}
