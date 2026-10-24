use burn_backend::{Shape, ops::ConvOptions};
use d3os_burn_native::{convolution::Conv2d, layout::Layout};
use d3os_nvidia::compute::{Kernel, convolution as abi};

#[test]
fn grouped_dilated_asymmetric_convolution_checks_both_signed_views() {
    let input = Layout::contiguous(Shape::new([2, 4, 9, 7]))
        .unwrap()
        .permute(&[0, 1, 3, 2])
        .unwrap()
        .flip(&[1, 3])
        .unwrap();
    let weights = Layout::contiguous(Shape::new([2, 6, 3, 2]))
        .unwrap()
        .permute(&[1, 0, 2, 3])
        .unwrap()
        .flip(&[0, 2])
        .unwrap();
    let options = ConvOptions::new_with_padding([2, 3], [(1, 2), (0, 1)], [2, 2], 2);
    let plan = Conv2d::new(input.shape().clone(), weights.shape().clone(), Some(&Shape::new([6])), &options).unwrap();
    assert_eq!(plan.output_shape().as_slice(), [2, 6, 3, 3]);
    assert_eq!(plan.columns_shape().as_slice(), [2, 2, 12, 9]);
    let p = plan.parameters(&input, &weights).unwrap().map(|v| v as u32);
    assert_eq!(p[..18], [2, 4, 7, 9, 3, 2, 3, 3, 2, 3, 1, 0, 2, 2, 2, 2, 1, 6]);
    assert_eq!(p[18..24], [504, 245, 252, (-63i32) as u32, 1, (-7i32) as u32]);
    assert_eq!(p[24..30], [72, 34, (-6i32) as u32, 36, (-2i32) as u32, 1]);
    let work = Kernel::Conv2d.work(&p).unwrap();
    assert_eq!(work.extents[..4], [504, 72, 1, 108]);
    assert_eq!((work.blocks, work.threads, work.pointers, work.output), (2, 128, 4, 3));
    assert!(Kernel::Conv2d.allocates_output());
    let mut biased = p;
    biased[30] = 1;
    assert_eq!(Kernel::Conv2d.work(&biased).unwrap().extents[2], 6);
    for other in 0..3 {
        let mut addresses = [0x1000, 0x3000, 0x4000, 0x5000];
        addresses[3] = addresses[other];
        assert!(work.validate_aliases(&addresses).is_err());
    }
    let window = plan.windows_parameters(&input).unwrap().map(|v| v as u32);
    let work = Kernel::Conv2dWindows.work(&window).unwrap();
    assert_eq!(work.extents[..2], [504, 432]);
    assert_eq!(work.blocks, 4);
    let work = Kernel::Conv2dFold.work(&p[..abi::GEOMETRY]).unwrap();
    assert_eq!(work.extents[..2], [432, 504]);
    assert!(Kernel::Conv2dWindows.allocates_output() && Kernel::Conv2dFold.allocates_output());
    for (index, value) in [
        (0, 0),
        (1, 0),
        (1, 5),
        (4, 0),
        (6, 4),
        (7, 4),
        (8, 0),
        (10, u32::MAX),
        (12, 0),
        (14, 0),
        (14, 3),
        (17, 0),
        (17, 7),
        (18, 503),
        (19, 244),
        (20, 253),
        (24, 71),
        (25, 33),
        (30, 2),
    ] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::Conv2d.work(&invalid).is_err());
    }
    assert!(Kernel::Conv2d.work(&p[..30]).is_err());
    assert!(Kernel::Conv2dWindows.work(&window[..23]).is_err());
    assert!(Kernel::Conv2dFold.work(&p[..17]).is_err());
}

#[test]
fn invalid_shapes_and_options_fail_without_allocation_and_empty_batches_keep_shapes() {
    let input = Shape::new([2, 4, 7, 9]);
    let weights = Shape::new([6, 2, 3, 2]);
    let options = ConvOptions::new_with_padding([2, 3], [(1, 2), (0, 1)], [2, 2], 2);
    for shape in [Shape::new([4, 7, 9]), Shape::new([2, 3, 7, 9]), Shape::new([2, 4, 0, 9])] {
        assert!(Conv2d::new(shape, weights.clone(), None, &options).is_err());
    }
    assert!(Conv2d::new(input.clone(), Shape::new([6, 2, 3]), None, &options).is_err());
    assert!(Conv2d::new(input.clone(), weights.clone(), Some(&Shape::new([1, 6])), &options).is_err());
    for invalid in [
        ConvOptions { groups: 0, ..options.clone() },
        ConvOptions {
            stride: [0, 3],
            ..options.clone()
        },
        ConvOptions {
            dilation: [0, 2],
            ..options.clone()
        },
        ConvOptions {
            padding: [(usize::MAX, 0), (0, 0)],
            ..options.clone()
        },
    ] {
        assert!(Conv2d::new(input.clone(), weights.clone(), None, &invalid).is_err());
    }
    assert!(Conv2d::new(Shape::new([2, 4, 1, 1]), weights.clone(), None, &options).is_err());
    let empty = Layout::contiguous(Shape::new([0, 4, 7, 9])).unwrap();
    let weights = Layout::contiguous(weights).unwrap();
    let plan = Conv2d::new(empty.shape().clone(), weights.shape().clone(), None, &options).unwrap();
    assert_eq!(plan.output_shape().as_slice(), [0, 6, 3, 3]);
    assert_eq!(plan.columns_shape().as_slice(), [0, 2, 12, 9]);
    let p = plan.parameters(&empty, &weights).unwrap().map(|v| v as u32);
    assert_eq!(abi::validate(&p).unwrap().0.outputs, 0);
    assert!(Kernel::Conv2d.work(&p).is_err());
}
