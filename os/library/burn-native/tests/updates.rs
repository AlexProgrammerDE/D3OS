use burn_backend::{Shape, Slice};
use d3os_burn_native::layout::Layout;
use d3os_nvidia::compute::{Kernel, elementwise::Elementwise};

#[test]
fn permuted_reversed_and_stepped_writes_prove_unique_addresses() {
    let dense = Layout::contiguous(Shape::new([3, 5, 7])).unwrap();
    for axes in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        for flips in 0..8 {
            let flip: Vec<_> = (0..3).filter(|a| flips & (1 << a) != 0).collect();
            let target = dense
                .permute(&axes)
                .unwrap()
                .flip(&flip)
                .unwrap()
                .slice(&[Slice::new(0, None, 2), Slice::full(), Slice::new(1, None, -2)])
                .unwrap();
            let source = Layout::contiguous(target.shape().clone()).unwrap().flip(&[1]).unwrap();
            let p = target.assignment_parameters(&source).unwrap().map(|v| v as u32);
            let work = Kernel::ViewAssign.work(&p).unwrap();
            assert_eq!(work.extents[..2], [105, source.len() as u64]);
            assert_eq!((work.pointers, work.output, work.blocks), (2, 0, (target.len() as u32).div_ceil(128)));
            let mut addresses: Vec<_> = (0..target.len()).map(|i| target.element_offset(i).unwrap()).collect();
            let length = addresses.len();
            addresses.sort_unstable();
            addresses.dedup();
            assert_eq!(addresses.len(), length);
            assert!(work.validate_aliases(&[0x1000, 0x4000]).is_ok());
            assert!(work.validate_aliases(&[0x1000, 0x1004]).is_err());
        }
    }
    assert!(!Kernel::ViewAssign.allocates_output());
}

#[test]
fn overlapping_destinations_fail_while_overlapping_sources_remain_readable() {
    let dense = Layout::contiguous(Shape::new([2, 6])).unwrap();
    let windows = dense.unfold(1, 3, 2).unwrap();
    let target = Layout::contiguous(windows.shape().clone()).unwrap();
    assert!(windows.assignment_parameters(&target).is_err());
    assert!(target.assignment_parameters(&windows).is_ok());
    let scalar = Layout::contiguous(Shape::new([1])).unwrap().expand(Shape::new([2, 3, 4])).unwrap();
    let dense = Layout::contiguous(scalar.shape().clone()).unwrap();
    assert!(scalar.assignment_parameters(&dense).is_err());
    assert!(dense.assignment_parameters(&scalar).is_ok());
    // These distinct addresses interleave and lie outside the conservative
    // stride proof. They must fail rather than bypass the write contract.
    let interleaved = Layout::from_parts(Shape::new([3, 2]), vec![2, 3], 0, 8).unwrap();
    assert!(
        interleaved
            .assignment_parameters(&Layout::contiguous(interleaved.shape().clone()).unwrap())
            .is_err()
    );
}

#[test]
fn masked_selection_and_partial_writes_reject_malformed_ranges_before_dispatch() {
    let a = Layout::contiguous(Shape::new([5, 129])).unwrap().flip(&[0, 1]).unwrap();
    let b = Layout::contiguous(Shape::new([129, 5])).unwrap().permute(&[1, 0]).unwrap();
    let p = a.elementwise_parameters(&b, Elementwise::Copy, 0, false).unwrap().map(|v| v as u32);
    let work = Kernel::MaskWhere.work(&p).unwrap();
    assert_eq!(work.extents[..4], [645; 4]);
    assert_eq!((work.pointers, work.output, work.blocks), (4, 3, 6));
    assert!(Kernel::MaskWhere.allocates_output());
    assert!(work.validate_aliases(&[0x1000, 0x4000, 0x7000, 0xa000]).is_ok());
    for input in [0x1000, 0x4004, 0x7004] {
        assert!(work.validate_aliases(&[0x1000, 0x4000, 0x7000, input]).is_err());
    }
    for (index, value) in [
        (0, 0),
        (0, 644),
        (1, 0),
        (1, 9),
        (2, 1),
        (2, 65536),
        (3, 1),
        (4, 643),
        (6, 644),
        (7, 644),
        (10, 2),
        (18, 1),
        (26, 1),
    ] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::MaskWhere.work(&invalid).is_err());
        assert!(Kernel::ViewAssign.work(&invalid).is_err());
    }
    assert!(Kernel::MaskWhere.work(&p[..31]).is_err());
    assert!(Kernel::ViewAssign.work(&p[..31]).is_err());
}
