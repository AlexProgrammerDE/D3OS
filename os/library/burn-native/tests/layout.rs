use burn_backend::{Shape, Slice};
use d3os_burn_native::layout::{Layout, broadcast, matmul_shape};

fn offsets(layout: &Layout) -> Vec<usize> {
    (0..layout.len()).map(|index| layout.element_offset(index).unwrap()).collect()
}

#[test]
fn composed_transpose_reverse_and_stepped_slice_addresses() {
    let original = Layout::contiguous(Shape::new([2, 3, 4])).unwrap();
    let view = original
        .permute(&[2, 0, 1])
        .unwrap()
        .flip(&[0, 2])
        .unwrap()
        .slice(&[Slice::new(1, Some(4), 2), Slice::full(), Slice::new(1, Some(3), -1)])
        .unwrap();
    assert_eq!(view.shape().as_slice(), &[2, 2, 2]);
    assert_eq!(offsets(&view), [2, 6, 14, 18, 0, 4, 12, 16]);
    assert!(!view.is_contiguous());
    assert!(view.reshape(Shape::new([8])).is_err());
    assert_eq!(offsets(&original), (0..24).collect::<Vec<_>>());
}

#[test]
fn all_three_dimensional_permutations_and_reversals() {
    let original = Layout::contiguous(Shape::new([2, 3, 4])).unwrap();
    for axes in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        for mask in 0..8 {
            let flipped: Vec<_> = (0..3).filter(|axis| mask & (1 << axis) != 0).collect();
            let view = original.permute(&axes).unwrap().flip(&flipped).unwrap();
            let mut coordinates = Vec::new();
            for n in 0..2 {
                for c in 0..3 {
                    for x in 0..4 {
                        let coordinate = [n, c, x];
                        let mut order = [0; 3];
                        for (axis, &source) in axes.iter().enumerate() {
                            order[axis] = if mask & (1 << axis) == 0 {
                                coordinate[source]
                            } else {
                                [2, 3, 4][source] - 1 - coordinate[source]
                            };
                        }
                        coordinates.push((order, n * 12 + c * 4 + x));
                    }
                }
            }
            coordinates.sort_by_key(|&(order, _)| order);
            assert_eq!(offsets(&view), coordinates.into_iter().map(|(_, offset)| offset).collect::<Vec<_>>());
        }
    }
}

#[test]
fn expanded_and_overlapping_views_keep_the_original_storage_extent() {
    let original = Layout::contiguous(Shape::new([2, 1, 3])).unwrap();
    let view = original.expand(Shape::new([4, 2, 5, 3])).unwrap();
    assert_eq!(view.storage_elements(), 6);
    assert_eq!(view.len(), 120);
    assert!(!view.is_contiguous());
    let expected: Vec<_> = (0..4)
        .flat_map(|_| (0..2).flat_map(|row| (0..5).flat_map(move |_| (0..3).map(move |col| row * 3 + col))))
        .collect();
    assert_eq!(offsets(&view), expected);
    let windows = Layout::contiguous(Shape::new([2, 6])).unwrap().unfold(1, 3, 2).unwrap();
    assert_eq!(windows.shape().as_slice(), &[2, 2, 3]);
    assert_eq!(offsets(&windows), [0, 1, 2, 2, 3, 4, 6, 7, 8, 8, 9, 10]);
    assert!(!windows.is_contiguous());
}

#[test]
fn contiguous_subviews_reshape_without_losing_their_base_offset() {
    let original = Layout::contiguous(Shape::new([5, 4])).unwrap();
    let view = original.slice(&[Slice::new(1, Some(4), 1)]).unwrap().reshape(Shape::new([2, 1, 6])).unwrap();
    assert!(view.is_contiguous());
    assert_eq!(view.offset(), 4);
    assert_eq!(view.storage_elements(), 20);
    assert_eq!(offsets(&view), (4..16).collect::<Vec<_>>());
    let singleton = Layout::from_parts(Shape::new([2, 3, 1]), vec![3, 1, 9], 0, 6).unwrap();
    assert!(singleton.is_contiguous());
    assert!(singleton.reshape(Shape::new([6])).is_ok());
}

#[test]
fn negative_slice_bounds_and_empty_views() {
    let original = Layout::contiguous(Shape::new([6])).unwrap();
    let view = original.slice(&[Slice::new(-4, None, 2)]).unwrap();
    assert_eq!(offsets(&view), [2, 4]);
    let empty = original.slice(&[Slice::new(4, Some(4), 1)]).unwrap().flip(&[0]).unwrap();
    assert!(empty.is_empty());
    assert!(empty.is_contiguous());
    assert!(empty.element_offset(0).is_err());
    assert!(empty.reshape(Shape::new([3, 0, 2])).is_ok());
    let huge_empty = Layout::contiguous(Shape::new([i32::MAX as usize, i32::MAX as usize, 0])).unwrap();
    assert_eq!(huge_empty.len(), 0);
    assert_eq!(broadcast(&Shape::new([2, 0, 3]), &Shape::new([1, 3])).unwrap().as_slice(), &[2, 0, 3]);
}

#[test]
fn invalid_or_overflowing_views_are_rejected() {
    assert!(Layout::from_parts(Shape::new([3]), vec![-1], 1, 3).is_err());
    assert!(Layout::from_parts(Shape::new([3]), vec![2], 0, 3).is_err());
    assert!(Layout::from_parts(Shape::new([1]), vec![i64::MAX], 0, 1).is_err());
    assert!(Layout::from_parts(Shape::new([2, 3]), vec![1], 0, 6).is_err());
    assert!(Layout::contiguous(Shape::new([i32::MAX as usize, 2])).is_err());
    assert!(Layout::contiguous(Shape::new([1; 9])).is_err());
    let original = Layout::contiguous(Shape::new([2, 3])).unwrap();
    assert!(original.permute(&[1, 1]).is_err());
    assert!(original.permute(&[0, 2]).is_err());
    assert!(original.flip(&[1, 1]).is_err());
    assert!(original.expand(Shape::new([2, 4])).is_err());
    assert!(original.unfold(1, 4, 1).is_err());
    assert!(original.unfold(1, 2, 0).is_err());
    assert!(broadcast(&Shape::new([2, 3]), &Shape::new([4, 3])).is_err());
    let edge = Layout::from_parts(Shape::new([1]), vec![i64::from(i32::MIN)], 0, 1).unwrap();
    assert!(edge.flip(&[0]).is_err());
}

#[test]
fn matrix_outputs_broadcast_batches_without_confusing_matrix_dimensions() {
    for (lhs, rhs, output) in [
        (vec![3, 5], vec![5, 7], vec![3, 7]),
        (vec![2, 3, 5], vec![5, 7], vec![2, 3, 7]),
        (vec![2, 1, 3, 5], vec![4, 5, 7], vec![2, 4, 3, 7]),
        (vec![2, 0, 3, 5], vec![1, 5, 7], vec![2, 0, 3, 7]),
        (vec![3, 0], vec![0, 7], vec![3, 7]),
    ] {
        assert_eq!(matmul_shape(&Shape::from(lhs), &Shape::from(rhs)).unwrap().as_slice(), output);
    }
    for (lhs, rhs) in [(vec![5], vec![5, 7]), (vec![3, 5], vec![4, 7]), (vec![2, 3, 5], vec![3, 5, 7])] {
        assert!(matmul_shape(&Shape::from(lhs), &Shape::from(rhs)).is_err());
    }
}

#[test]
fn reduction_metadata_preserves_rank_signed_strides_and_original_storage() {
    use d3os_nvidia::compute::reduction::Reduction;
    let view = Layout::contiguous(Shape::new([3, 4, 5]))
        .unwrap()
        .permute(&[2, 0, 1])
        .unwrap()
        .flip(&[2])
        .unwrap();
    for (axis, expected) in [(0, [1, 3, 4]), (1, [5, 1, 4]), (2, [5, 3, 1])] {
        assert_eq!(view.reduction_shape(axis).unwrap().as_slice(), expected);
        let p = view.reduction_parameters(axis, Reduction::Sum).unwrap();
        assert_eq!(p[..8], [expected.into_iter().product(), 3, 0, axis, 15, 60, [5, 3, 4][axis], 0]);
        assert_eq!(p[8..11], [5, 3, 4]);
        assert_eq!(p[16..19], [1, 20, (-5i32) as u32 as usize]);
    }
    assert!(view.reduction_shape(3).is_err());
    let empty = Layout::contiguous(Shape::new([3, 0, 4])).unwrap();
    assert_eq!(empty.reduction_shape(1).unwrap().as_slice(), [3, 1, 4]);
    assert!(empty.reduction_parameters(1, Reduction::Sum).is_ok());
    assert!(empty.reduction_parameters(1, Reduction::Max).is_err());
    assert_eq!(empty.reduction_shape(0).unwrap().as_slice(), [1, 0, 4]);
    assert!(empty.reduction_parameters(0, Reduction::Sum).is_err());
}

#[test]
fn typed_words_use_the_same_view_addresses_with_distinct_operand_rules() {
    use d3os_nvidia::compute::words::WordOp;
    let view = Layout::contiguous(Shape::new([3, 257])).unwrap().permute(&[1, 0]).unwrap().flip(&[0]).unwrap();
    let rhs = Layout::contiguous(Shape::new([3])).unwrap().expand(Shape::new([257, 3])).unwrap();
    let parameters = view.word_parameters(&rhs, WordOp::ShiftRight, 0, false).unwrap();
    assert_eq!(parameters[..8], [771, 2, WordOp::ShiftRight as usize, 0, 256, 0, 771, 3]);
    assert_eq!(parameters[16..18], [(-1i32) as u32 as usize, 257]);
    assert_eq!(parameters[24..26], [0, 1]);
    assert!(view.word_parameters(&rhs, WordOp::BoolAnd, 2, true).is_err());
    assert!(view.word_parameters(&rhs, WordOp::FloatToInt, 1, false).is_err());
}
