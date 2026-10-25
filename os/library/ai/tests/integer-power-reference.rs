#![cfg(feature = "training")]
use d3os_ai::validation::integer::power_reference;

#[test]
fn modular_power_reference_matches_primitive_wrapping_and_group_identities() {
    for i in 0u32..4099 {
        let base = i.wrapping_mul(0x9e37_79b9).rotate_left(i % 32) as i32;
        for exponent in [i, 0, 1, 2, 31, 32, 33, u32::MAX, 1 << 31] {
            assert_eq!(power_reference(base, exponent as i32), base.wrapping_pow(exponent));
        }
    }
    assert_eq!(power_reference(16_777_217, 1), 16_777_217);
    assert_eq!(power_reference(16_777_217, 3), 50_331_649);
    assert_eq!(power_reference(3, -1) as u32, 0xaaaa_aaab);
    assert_eq!(power_reference(0, 0), 1);
    for base in [-65537, -3, -1, 1, 3, 65537, i32::MAX] {
        assert_eq!(power_reference(base, 1 << 30), 1);
        assert_eq!(base.wrapping_mul(power_reference(base, -1)), 1);
    }
    for base in [i32::MIN, -2, 0, 2, i32::MAX - 1] {
        assert_eq!(power_reference(base, -1), 0);
    }
}
