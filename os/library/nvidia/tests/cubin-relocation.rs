use d3os_nvidia::{
    Error,
    cubin::relocation::{Cuda64Patch, Cuda64Plan, checked_image_section},
};

// Fabricated GPU virtual addresses used only by host tests.
const FAKE_GPU_VIRTUAL_BASE: u64 = 0x1234_0000_0000;

fn patch(slot: usize, symbol: usize) -> Cuda64Patch {
    Cuda64Patch::new(slot, symbol).unwrap()
}

fn assert_atomic_error(plan: &Cuda64Plan, image: &mut [u8], base: u64, error: Error) {
    let before = image.to_vec();
    assert_eq!(plan.apply(image, base), Err(error));
    assert_eq!(image, before);
}

#[test]
fn section_layout_preserves_minimum_alignment_and_padding_order() {
    for alignment in [0, 1, 2, 4, 8, 16, 32, 64, 128] {
        assert_eq!(checked_image_section(129, 9, alignment), Ok((256, 265)));
        assert_eq!(checked_image_section(128, 0, alignment), Ok((128, 128)));
        assert_eq!(checked_image_section(129, 0, alignment), Ok((256, 256)));
    }
    assert_eq!(checked_image_section(129, 9, 256), Ok((256, 265)));
    assert_eq!(checked_image_section(257, 9, 4096), Ok((4096, 4105)));
    // The helper does not impose the adapter's allocation-base alignment policy.
    assert_eq!(checked_image_section(1, 1, 8192), Ok((8192, 8193)));
    let mut image = vec![0xab; 129];
    let (start, end) = checked_image_section(image.len(), 3, 16).unwrap();
    image.resize(start, 0);
    image.extend_from_slice(&[4, 5, 6]);
    assert_eq!(image.len(), end);
    assert_eq!(&image[..129], &[0xab; 129]);
    assert!(image[129..256].iter().all(|&byte| byte == 0));
    assert_eq!(&image[256..], &[4, 5, 6]);
}

#[test]
fn layout_matches_wide_arithmetic_across_alignment_and_range_boundaries() {
    let cursors = [0, 1, 127, 128, 129, 4095, usize::MAX - 127, usize::MAX];
    for cursor in cursors {
        for size in [0, 1, 8, 127, usize::MAX] {
            for exponent in 0..64 {
                let alignment = 1_u64 << exponent;
                let effective = u128::from(alignment.max(128));
                let start = (cursor as u128).div_ceil(effective) * effective;
                let end = start + size as u128;
                let expected = if alignment as u128 > usize::MAX as u128 || end > usize::MAX as u128 {
                    Err(Error::InvalidField)
                } else {
                    Ok((start as usize, end as usize))
                };
                assert_eq!(checked_image_section(cursor, size, alignment), expected);
            }
        }
    }
}

#[test]
fn layout_rejects_invalid_alignment_rounding_and_size_overflow() {
    for alignment in [3, 7, 24, 127, 129, u64::MAX] {
        assert_eq!(checked_image_section(0, 0, alignment), Err(Error::InvalidField));
    }
    assert_eq!(checked_image_section(usize::MAX, 0, 128), Err(Error::InvalidField));
    assert_eq!(checked_image_section(1, usize::MAX, 128), Err(Error::InvalidField));
    assert_eq!(checked_image_section(usize::MAX - 127, 127, 128), Ok((usize::MAX - 127, usize::MAX)));
    assert_eq!(checked_image_section(usize::MAX - 127, 128, 128), Err(Error::InvalidField));
}

#[test]
fn patch_constructor_rejects_partial_slot_alignment_and_overflow() {
    for offset in 1..8 {
        assert_eq!(Cuda64Patch::new(offset, 0), Err(Error::InvalidField));
    }
    assert_eq!(Cuda64Patch::new(usize::MAX - 7, 0), Err(Error::InvalidField));
    assert_eq!(patch(8, 3).slot_offset(), 8);
    assert_eq!(patch(8, 3).symbol_image_offset(), 3);
}

#[test]
fn plan_requires_complete_slots_and_interior_symbol_starts() {
    for (image_bytes, slot, symbol) in [(7, 0, 0), (16, 16, 0), (16, 0, 16), (0, 0, 0)] {
        assert!(matches!(Cuda64Plan::new(image_bytes, vec![patch(slot, symbol)]), Err(Error::InvalidField)));
    }
    assert!(matches!(Cuda64Plan::new(16, vec![patch(0, usize::MAX)]), Err(Error::InvalidField)));
    let plan = Cuda64Plan::new(16, vec![patch(8, 15)]).unwrap();
    let mut image = [0; 16];
    plan.apply(&mut image, FAKE_GPU_VIRTUAL_BASE).unwrap();
    assert_eq!(&image[8..], &(FAKE_GPU_VIRTUAL_BASE + 15).to_le_bytes());
}

#[test]
fn plan_sorts_deterministically_accepts_adjacent_slots_and_rejects_duplicates() {
    let original = vec![patch(16, 2), patch(0, 1), patch(8, 0)];
    let plan = Cuda64Plan::new(24, original.clone()).unwrap();
    assert_eq!(plan.image_bytes(), 24);
    assert_eq!(plan.patches(), &[patch(0, 1), patch(8, 0), patch(16, 2)]);
    assert_eq!(original, vec![patch(16, 2), patch(0, 1), patch(8, 0)]);
    for patches in [vec![patch(0, 1), patch(0, 2)], vec![patch(8, 1), patch(0, 2), patch(8, 3)]] {
        assert!(matches!(Cuda64Plan::new(24, patches), Err(Error::InvalidField)));
    }
    // With eight-byte alignment, every possible partial overlap is rejected
    // at construction, before a patch can enter a plan.
    for offset in 9..16 {
        assert_eq!(Cuda64Patch::new(offset, 0), Err(Error::InvalidField));
    }
}

#[test]
fn writes_are_little_endian_and_preserve_every_nonpatch_byte() {
    let mut image: Vec<u8> = (0..64).collect();
    image[8..16].fill(0);
    image[40..48].fill(0);
    let before = image.clone();
    let plan = Cuda64Plan::new(64, vec![patch(40, 63), patch(8, 17)]).unwrap();
    plan.apply(&mut image, FAKE_GPU_VIRTUAL_BASE).unwrap();
    assert_eq!(&image[8..16], &(FAKE_GPU_VIRTUAL_BASE + 17).to_le_bytes());
    assert_eq!(&image[40..48], &(FAKE_GPU_VIRTUAL_BASE + 63).to_le_bytes());
    image[8..16].copy_from_slice(&before[8..16]);
    image[40..48].copy_from_slice(&before[40..48]);
    assert_eq!(image, before);
}

#[test]
fn unqualified_addends_including_signed_looking_values_fail_before_any_write() {
    let plan = Cuda64Plan::new(32, vec![patch(0, 16), patch(8, 24)]).unwrap();
    for addend in [1_u64, (1 << 63) - 1, 1 << 63, u64::MAX] {
        for slot in [0, 8] {
            let mut image = [0; 32];
            image[slot..slot + 8].copy_from_slice(&addend.to_le_bytes());
            assert_atomic_error(&plan, &mut image, FAKE_GPU_VIRTUAL_BASE, Error::Unsupported);
        }
    }
}

#[test]
fn late_nonzero_byte_at_every_position_is_atomic() {
    let plan = Cuda64Plan::new(40, vec![patch(0, 24), patch(8, 32), patch(16, 39)]).unwrap();
    for byte in 16..24 {
        let mut image = [0; 40];
        image[byte] = 0x80;
        assert_atomic_error(&plan, &mut image, FAKE_GPU_VIRTUAL_BASE, Error::Unsupported);
    }
}

#[test]
fn exact_image_length_is_checked_before_patching() {
    let plan = Cuda64Plan::new(32, vec![patch(0, 24)]).unwrap();
    for length in [0, 7, 8, 24, 31, 33, 64] {
        let mut image = vec![0; length];
        let error = if length < 32 { Error::Truncated } else { Error::InvalidField };
        assert_atomic_error(&plan, &mut image, FAKE_GPU_VIRTUAL_BASE, error);
    }
}

#[test]
fn same_plan_rebases_fresh_images_and_rejects_reapplication_atomically() {
    let plan = Cuda64Plan::new(32, vec![patch(0, 16), patch(8, 24)]).unwrap();
    let mut first = [0; 32];
    let mut second = first;
    plan.apply(&mut first, FAKE_GPU_VIRTUAL_BASE).unwrap();
    plan.apply(&mut second, FAKE_GPU_VIRTUAL_BASE + 0x10_0000).unwrap();
    for slot in [0, 8] {
        let a = u64::from_le_bytes(first[slot..slot + 8].try_into().unwrap());
        let b = u64::from_le_bytes(second[slot..slot + 8].try_into().unwrap());
        assert_eq!(b - a, 0x10_0000);
    }
    assert_atomic_error(&plan, &mut first, FAKE_GPU_VIRTUAL_BASE, Error::Unsupported);
    assert_atomic_error(&plan, &mut second, FAKE_GPU_VIRTUAL_BASE + 0x20_0000, Error::Unsupported);
}

#[test]
fn u64_virtual_extent_checks_do_not_impose_architecture_domain_limits() {
    let plan = Cuda64Plan::new(16, vec![patch(0, 15)]).unwrap();
    for base in [0, 1, (1 << 48) + 1, (1 << 63) + 3, u64::MAX - 16] {
        let mut image = [0; 16];
        plan.apply(&mut image, base).unwrap();
        assert_eq!(&image[..8], &(base + 15).to_le_bytes());
    }
    for base in [u64::MAX - 15, u64::MAX - 1, u64::MAX] {
        assert_atomic_error(&plan, &mut [0; 16], base, Error::InvalidField);
    }
    // Even a low-offset pointer cannot hide overflow of the full image extent.
    let low_symbol = Cuda64Plan::new(16, vec![patch(0, 0)]).unwrap();
    assert_atomic_error(&low_symbol, &mut [0; 16], u64::MAX - 8, Error::InvalidField);
}

#[test]
fn null_pointer_rejection_keeps_successful_images_detectably_relocated() {
    let plan = Cuda64Plan::new(16, vec![patch(0, 8), patch(8, 0)]).unwrap();
    assert_atomic_error(&plan, &mut [0; 16], 0, Error::InvalidField);
}

#[test]
fn zero_patch_images_are_identical_but_still_require_valid_length_and_extent() {
    let plan = Cuda64Plan::new(32, vec![]).unwrap();
    let mut image = [0xa5; 32];
    plan.apply(&mut image, FAKE_GPU_VIRTUAL_BASE).unwrap();
    plan.apply(&mut image, FAKE_GPU_VIRTUAL_BASE + 0x1000).unwrap();
    assert_eq!(image, [0xa5; 32]);
    assert_atomic_error(&plan, &mut image, u64::MAX - 31, Error::InvalidField);
    assert_atomic_error(&plan, &mut [0; 31], FAKE_GPU_VIRTUAL_BASE, Error::Truncated);
    assert_atomic_error(&plan, &mut [0; 33], FAKE_GPU_VIRTUAL_BASE, Error::InvalidField);
    Cuda64Plan::new(0, vec![]).unwrap().apply(&mut [], u64::MAX).unwrap();
}

// Captured .nv.global.init payload shared by the F64 Sin/Cos, dynamic trig,
// and combined modules. Only these 272 bytes and their 24-byte CB4 are retained.
// Payload SHA256: 78b4bac7d222df155a17884f3f5312a6709b15b0d5b994f4b9b1f9a4f95f6502.
// Words preserve exact bits, without floating-point conversion or an ELF parser.
const CAPTURED_COEFFICIENTS: [u64; 16] = [
    0xbe5ae5f12cb0d246,
    0x3ec71de369ace392,
    0xbf2a01a019db62a1,
    0x3f81111111110818,
    0xbfc5555555555554,
    0x0000000000000000,
    0x0000000000000000,
    0xbda8ff8320fd8164,
    0x3e21eea7c1ef8528,
    0xbe927e4f8e06e6d9,
    0x3efa01a019ddbce9,
    0xbf56c16c16c15d47,
    0x3fa5555555555551,
    0xbfe0000000000000,
    0x3ff0000000000000,
    0x0000000000000000,
];
const CAPTURED_REDUCTION: [u64; 18] = [
    0x6bfb5fb11f8d5d08,
    0x3d0739f78a5292ea,
    0x7527bac7ebe5f17b,
    0x4f463f669e5fea2d,
    0x6d367ecf27cb09b7,
    0xef2f118b5a0a6d1f,
    0x1ff897ffde05980f,
    0x9c845f8bbdf9283b,
    0x3991d639835339f4,
    0xe99c7026b45f7e41,
    0xe88235f52ebb4484,
    0xfe1deb1cb129a73e,
    0x06492eea09d1921c,
    0xb7246e3a424dd2e0,
    0xfe5163abdebbc561,
    0xdb6295993c439041,
    0xfc2757d1f534ddc0,
    0xa2f9836e4e441529,
];

#[test]
fn captured_trig_pointers_resolve_complete_tables_and_preserve_legacy_slot() {
    let coefficients: Vec<_> = CAPTURED_COEFFICIENTS.into_iter().flat_map(u64::to_le_bytes).collect();
    let reduction: Vec<_> = CAPTURED_REDUCTION.into_iter().flat_map(u64::to_le_bytes).collect();
    // Captured image offsets and model-patched CB4 words at FAKE_GPU_VIRTUAL_BASE.
    let cases = [
        (4352, [0, 0x0000_1234_0000_1180_u64, 0x0000_1234_0000_1100]), // F64 Sin
        (4480, [0, 0x0000_1234_0000_1200, 0x0000_1234_0000_1180]),     // F64 Cos
        (34432, [0, 0x0000_1234_0000_8700, 0x0000_1234_0000_8680]),    // Dynamic trig
        (93824, [0, 0x0000_1234_0001_6f00, 0x0000_1234_0001_6e80]),    // Combined
    ];
    for (table_offset, expected_cb4) in cases {
        // Intervening code and constants are synthetic sentinels. This tests
        // captured data-pointer patching, not reconstruction of complete cubins.
        let mut image = vec![0xa5; table_offset];
        image[..24].fill(0); // Actual pristine CB4, including the legacy slot.
        image.extend_from_slice(&coefficients);
        image.extend_from_slice(&reduction);
        let pristine = image.clone();
        let plan = Cuda64Plan::new(image.len(), vec![patch(16, table_offset), patch(8, table_offset + 128)]).unwrap();
        plan.apply(&mut image, FAKE_GPU_VIRTUAL_BASE).unwrap();
        let expected_cb4: Vec<_> = expected_cb4.into_iter().flat_map(u64::to_le_bytes).collect();
        assert_eq!(&image[..24], expected_cb4);
        for (slot, payload) in [(8, &reduction), (16, &coefficients)] {
            let pointer = u64::from_le_bytes(image[slot..slot + 8].try_into().unwrap());
            let relative = usize::try_from(pointer - FAKE_GPU_VIRTUAL_BASE).unwrap();
            assert_eq!(&image[relative..relative + payload.len()], payload);
        }
        image[8..24].copy_from_slice(&pristine[8..24]);
        assert_eq!(image, pristine);
    }
}
