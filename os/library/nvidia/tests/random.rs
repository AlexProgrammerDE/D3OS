use d3os_nvidia::compute::{
    Kernel,
    random::{self, Mode},
};

#[test]
fn random_launches_validate_each_distribution_and_exact_counter_extent() {
    let cases = [
        (Mode::Words, [0, 0, 0]),
        (
            Mode::Uniform,
            [(-f32::MAX).to_bits(), f32::MAX.to_bits(), random::predecessor(f32::MAX.to_bits())],
        ),
        (Mode::Uniform, [0, f32::from_bits(1).to_bits(), 0]),
        (Mode::Normal, [(-0.0f32).to_bits(), 0, 0]),
        (Mode::Bernoulli, [u32::MAX, 0, 0]),
        (Mode::Bernoulli, [0, 1, 0]),
        (Mode::IntegerUniform, [i32::MIN as u32, u32::MAX, 1]),
        (Mode::IntegerUniform, [i32::MIN as u32, 0x8000_0001, 0x7fff_ffff]),
    ];
    for (mode, arguments) in cases {
        for count in [1u32, 3, 4, 5, 131, i32::MAX as u32] {
            let start = u64::MAX - u64::from(count).div_ceil(4);
            let p = [
                count,
                u32::MAX,
                17,
                start as u32,
                (start >> 32) as u32,
                mode as u32,
                arguments[0],
                arguments[1],
                arguments[2],
            ];
            let work = Kernel::Random.work(&p).unwrap();
            assert_eq!((work.output, work.pointers, work.blocks), (0, 1, count.div_ceil(128)));
            assert_eq!(work.extents[0], u64::from(count));
            assert_eq!(work.extents[1..], [0; 15]);
            let mut exhausted = p;
            let too_late = start + 1;
            exhausted[3] = too_late as u32;
            exhausted[4] = (too_late >> 32) as u32;
            assert!(Kernel::Random.work(&exhausted).is_err());
            for (index, value) in [(0, 0), (0, u32::MAX), (5, 5)] {
                let mut invalid = p;
                invalid[index] = value;
                assert!(Kernel::Random.work(&invalid).is_err());
            }
            assert!(Kernel::Random.work(&p[..8]).is_err());
        }
    }
}

#[test]
fn malformed_distribution_arguments_fail_before_any_gpu_work() {
    for (mode, args) in [
        (Mode::Words, [0, 0, 1]),
        (Mode::Uniform, [0, 1f32.to_bits(), 1f32.to_bits()]),
        (Mode::Uniform, [1f32.to_bits(), 1f32.to_bits(), random::predecessor(1f32.to_bits())]),
        (Mode::Uniform, [0, f32::INFINITY.to_bits(), f32::MAX.to_bits()]),
        (Mode::Uniform, [f32::NAN.to_bits(), 1f32.to_bits(), random::predecessor(1f32.to_bits())]),
        (Mode::Normal, [0, (-1f32).to_bits(), 0]),
        (Mode::Normal, [f32::INFINITY.to_bits(), 1f32.to_bits(), 0]),
        (Mode::Normal, [0, 1f32.to_bits(), 1]),
        (Mode::Bernoulli, [1, 1, 0]),
        (Mode::Bernoulli, [0, 2, 0]),
        (Mode::Bernoulli, [0, 0, 1]),
        (Mode::IntegerUniform, [0, 0, 0]),
        (Mode::IntegerUniform, [i32::MAX as u32, 1, 0]),
        (Mode::IntegerUniform, [i32::MIN as u32, 0x8000_0001, 0]),
    ] {
        let p = [131, 0, 0, 0, 0, mode as u32, args[0], args[1], args[2]];
        assert!(Kernel::Random.work(&p).is_err());
    }
}
