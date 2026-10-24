use burn_backend::{
    Shape,
    ops::{AvgPoolOptions, MaxPoolOptions},
};
use d3os_burn_native::{layout::Layout, pooling::Pooling};
use d3os_nvidia::compute::{
    Kernel,
    pooling::{self as abi, Mode},
};

#[test]
fn ceil_dilation_and_asymmetric_padding_preserve_signed_storage_ranges() {
    let input = Layout::contiguous(Shape::new([2, 3, 9, 7]))
        .unwrap()
        .permute(&[0, 1, 3, 2])
        .unwrap()
        .flip(&[0, 1, 3])
        .unwrap();
    let options = MaxPoolOptions::new([3, 2])
        .with_stride([2, 3])
        .with_padding_pairs([(1, 2), (0, 1)])
        .with_dilation([2, 2])
        .with_ceil_mode(true);
    let plan = Pooling::max(input.shape().clone(), &options).unwrap();
    assert_eq!(plan.output_shape().as_slice(), [2, 3, 4, 3]);
    let p = plan.parameters(&input, false).unwrap().map(|v| v as u32);
    assert_eq!(p[..24], [2, 3, 1, 7, 9, 1, 4, 3, 1, 3, 2, 1, 2, 3, 0, 1, 0, 0, 2, 1, 1, 2, 2, 1]);
    assert_eq!(p[24..31], [378, 371, (-189i32) as u32, (-63i32) as u32, 0, 1, (-7i32) as u32]);
    let work = Kernel::Pool.work(&p).unwrap();
    assert_eq!(work.extents[..2], [378, 144]);
    assert_eq!((work.blocks, work.pointers, work.output), (1, 2, 1));
    assert!(Kernel::Pool.allocates_output());
    assert!(work.validate_aliases(&[0x1000, 0x4000]).is_ok());
    assert!(work.validate_aliases(&[0x1000, 0x1004]).is_err());
    assert!(Kernel::PoolBackward.work(&p).is_err());
    for (index, value) in [
        (0, 0),
        (2, 0),
        (6, 5),
        (7, 4),
        (9, 0),
        (12, 0),
        (15, u32::MAX),
        (21, 0),
        (23, 2),
        (23, 4),
        (24, 377),
        (25, 370),
        (31, 3),
    ] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::Pool.work(&invalid).is_err());
    }
    assert!(Kernel::Pool.work(&p[..31]).is_err());
}

#[test]
fn average_backward_validates_the_gradient_view_instead_of_the_input_view() {
    let shape = Shape::new([2, 3, 7, 9]);
    let options = AvgPoolOptions::new([3, 2])
        .with_stride([2, 3])
        .with_padding_pairs([(1, 2), (0, 1)])
        .with_ceil_mode(true);
    let plan = Pooling::average(shape.clone(), &options).unwrap();
    assert_eq!(plan.output_shape().as_slice(), [2, 3, 4, 3]);
    let gradient = Layout::contiguous(plan.output_shape().clone()).unwrap().flip(&[0, 2]).unwrap();
    let p = plan.parameters(&gradient, true).unwrap().map(|v| v as u32);
    assert_eq!(p[23], 3);
    assert_eq!(p[24..32], [72, 45, (-36i32) as u32, 12, 0, (-3i32) as u32, 1, 1]);
    let work = Kernel::PoolBackward.work(&p).unwrap();
    assert_eq!(work.extents[..2], [72, 378]);
    assert_eq!(work.blocks, 3);
    assert!(Kernel::PoolBackward.allocates_output());
    assert!(plan.parameters(&Layout::contiguous(shape).unwrap(), true).is_err());
    for (index, value) in [(21, 2), (24, 71), (25, 44), (29, (-4i32) as u32), (31, 0)] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::PoolBackward.work(&invalid).is_err());
    }
}

#[test]
fn adaptive_geometry_supports_upsampling_and_empty_leading_dimensions() {
    let input = Layout::contiguous(Shape::new([1, 2, 3, 5, 7])).unwrap();
    let plan = Pooling::adaptive(input.shape().clone(), &[4, 2, 3]).unwrap();
    assert_eq!(plan.output_shape().as_slice(), [1, 2, 4, 2, 3]);
    let p = plan.parameters(&input, false).unwrap().map(|v| v as u32);
    let work = Kernel::Pool.work(&p).unwrap();
    assert_eq!(work.extents[..2], [210, 48]);
    for (index, value) in [(8, 2), (11, 2), (14, 1), (17, 1), (20, 2), (23, 1)] {
        let mut invalid = p;
        invalid[index] = value;
        assert!(Kernel::Pool.work(&invalid).is_err());
    }
    let large = Layout::contiguous(Shape::new([1, 1, 65537])).unwrap();
    let plan = Pooling::adaptive(large.shape().clone(), &[65539]).unwrap();
    assert!(Kernel::Pool.work(&plan.parameters(&large, false).unwrap().map(|v| v as u32)).is_ok());
    for shape in [Shape::new([0, 2, 3, 5, 7]), Shape::new([1, 0, 3, 5, 7])] {
        let input = Layout::contiguous(shape).unwrap();
        let plan = Pooling::adaptive(input.shape().clone(), &[4, 2, 3]).unwrap();
        let p = plan.parameters(&input, false).unwrap().map(|v| v as u32);
        assert_eq!(abi::geometry(&p[..24], Mode::Adaptive).unwrap().outputs, 0);
        assert!(Kernel::Pool.work(&p).is_err());
    }
    assert_eq!(abi::output_extent(3, 5, 3, 0, 0, 1, true).unwrap(), 1);
    assert!(abi::output_extent(3, 7, 3, 0, 0, 1, true).is_err());
    assert!(Pooling::adaptive(Shape::new([1, 2, 3, 5]), &[0, 2]).is_err());
    assert!(Pooling::adaptive(Shape::new([1, 2, 3, 5]), &[2]).is_err());
    assert!(Pooling::max(Shape::new([1, 2, 3, 5]), &MaxPoolOptions::new([2, 2]).with_padding([usize::MAX, 0])).is_err());
}
