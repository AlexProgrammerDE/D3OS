use d3os_nvidia::{Bits, Error, Record};

#[test]
fn hardware_fields_cross_bytes_and_preserve_every_neighbor() {
    for low in 0..64 {
        for width in 1..=64 {
            let field = Bits::new(low + width - 1, low);
            let mut record = Record::<16>::from_bytes(&[0xa5; 16]).unwrap();
            let before = record.as_bytes().to_vec();
            let value = 0xfedcba9876543210 & (u64::MAX >> (64 - width));
            record.set_bits(field, value).unwrap();
            assert_eq!(record.get_bits(field), Ok(value));
            for bit in 0..128 {
                if !(low..low + width).contains(&bit) {
                    assert_eq!((record.as_bytes()[bit / 8] >> (bit % 8)) & 1, (before[bit / 8] >> (bit % 8)) & 1);
                }
            }
        }
    }
}

#[test]
fn invalid_fields_do_not_partially_modify_records() {
    let mut record = Record::<16>::from_bytes(&[0xa5; 16]).unwrap();
    for field in [Bits::new(4, 5), Bits::new(128, 127), Bits::new(64, 0), Bits::new(usize::MAX, 0)] {
        assert_eq!(record.set_bits(field, 0), Err(Error::InvalidField));
        assert_eq!(record.get_bits(field), Err(Error::InvalidField));
        assert_eq!(record.as_bytes(), &[0xa5; 16]);
    }
    assert_eq!(record.set_bits(Bits::new(67, 4), u64::MAX), Ok(()));
    let before = record.as_bytes().to_vec();
    assert_eq!(record.set_bits(Bits::new(8, 2), 128), Err(Error::InvalidField));
    assert_eq!(record.as_bytes().as_slice(), before);
}
