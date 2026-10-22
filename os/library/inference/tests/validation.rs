use d3os_compute::{Backend, cpu::Cpu};
use d3os_inference::{Error, archive::Archive, self_test, top_k, verify};

fn archive(records: &[(&str, &[u64], &[f32])]) -> Vec<u8> {
    let mut data = b"D3TENSR\0".to_vec();
    data.extend(1u32.to_le_bytes());
    data.extend((records.len() as u32).to_le_bytes());
    for &(name, shape, values) in records {
        data.extend((name.len() as u16).to_le_bytes());
        data.extend((shape.len() as u16).to_le_bytes());
        data.extend((values.len() as u64).to_le_bytes());
        data.extend(name.bytes());
        for dim in shape {
            data.extend(dim.to_le_bytes());
        }
        for value in values {
            data.extend(value.to_le_bytes());
        }
    }
    data
}

#[test]
fn native_self_test_exercises_the_complete_small_network() {
    self_test(&Cpu).unwrap();
}

#[test]
fn archive_decodes_endianness_and_rejects_every_truncated_prefix() {
    let data = archive(&[("input", &[1, 3], &[1.25, -2.5, 0.125])]);
    for length in 0..data.len() {
        assert!(Archive::parse(&data[..length]).is_err());
    }
    let mut parsed = Archive::parse(&data).unwrap();
    let input = parsed.take("input").unwrap().load(&Cpu).unwrap();
    parsed.finish().unwrap();
    assert_eq!(input.shape(), [1, 3]);
    assert_eq!(Cpu.download(&input).unwrap(), [1.25, -2.5, 0.125]);
}

#[test]
fn archive_rejects_duplicate_names_trailing_bytes_and_mismatched_shapes() {
    assert!(Archive::parse(&archive(&[("x", &[1], &[1.]), ("x", &[1], &[2.])])).is_err());
    assert!(Archive::parse(&archive(&[("x", &[2], &[1.])])).is_err());
    assert!(Archive::parse(&archive(&[("x", &[u64::MAX, 2], &[1.])])).is_err());
    let mut data = archive(&[("x", &[1], &[1.])]);
    data.push(0);
    assert!(Archive::parse(&data).is_err());
    let data = archive(&[("x", &[1], &[f32::NAN])]);
    assert!(Archive::parse(&data).unwrap().take("x").unwrap().load(&Cpu).is_err());
}

#[test]
fn verification_rejects_nonfinite_values_wrong_lengths_and_excess_error() {
    assert_eq!(verify(&[f32::NAN], &[0.], 1e-3, 1e-3), Err(Error::VerificationFailed));
    assert!(verify(&[1.], &[1., 2.], 0., 0.).is_err());
    assert!(verify(&[1.01], &[1.], 0.001, 0.001).is_err());
    assert!(verify(&[1.0001, 0.00001], &[1., 0.], 0.0001, 0.0001).is_ok());
    assert!(verify(&[1.], &[1.], f32::NAN, 0.).is_err());
}

#[test]
fn class_ranking_is_stable_and_limits_output() {
    assert_eq!(top_k(&[0.2, 0.4, 0.4], 2).unwrap(), [(1, 0.4), (2, 0.4)]);
    assert!(top_k(&[f32::NAN], 1).is_err());
}
