use d3os_nvidia::{Error, Field, Record, context::*};

fn template() -> Record<HEADER_BYTES> {
    let mut header = Record::<HEADER_BYTES>::default();
    header.set(Field::<u32>::new(4), 0x14b800).unwrap();
    header.set(Field::<u32>::new(0xc), 0xcafe0020).unwrap();
    header.set(Field::<u32>::new(0xfc), 0x600dc0de).unwrap();
    header.set(Field::<u64>::new(0xa4), 0x1042910000).unwrap();
    header
}

#[test]
fn relocation_changes_only_the_context_self_pointer() {
    let header = template();
    let relocated = relocate_header(header.as_bytes(), 0x18c000, 0x104a331000).unwrap();
    let decoded = Record::<HEADER_BYTES>::from_bytes(&relocated).unwrap();
    assert_eq!(decoded.get(Field::<u32>::new(0x78)), Ok(0x10));
    assert_eq!(decoded.get(Field::<u32>::new(0x7c)), Ok(0x4a331000));
    for (i, (&actual, &original)) in relocated.iter().zip(header.as_bytes()).enumerate() {
        if !(0x78..0x80).contains(&i) {
            assert_eq!(actual, original);
        }
    }
}

#[test]
fn invalid_images_and_addresses_are_rejected() {
    let header = template();
    assert!(relocate_header(&header.as_bytes()[..255], 0x18c000, 0x1000000000).is_err());
    for (allocation, address) in [(0x1000, 0x1000000000), (0x18c000, 0), (0x18c000, 0x1000000001), (0x18c000, (1 << 49) - 4096)] {
        assert_eq!(relocate_header(header.as_bytes(), allocation, address), Err(Error::InvalidField));
    }
    for (offset, value) in [(0xfc, 0), (0xc, 0x0020), (4, 0)] {
        let mut invalid = template();
        invalid.set(Field::<u32>::new(offset), value).unwrap();
        assert!(relocate_header(invalid.as_bytes(), 0x18c000, 0x1000000000).is_err());
    }
}

#[test]
fn virtual_binding_preserves_the_page_directories_and_fifo_state() {
    let mut initial = [0xa5; INSTANCE_BYTES];
    initial[0xae] &= !1;
    let mut instance = Record::<INSTANCE_BYTES>::from_bytes(&initial).unwrap();
    for address in [0, 4097, 1 << 40] {
        assert_eq!(bind_virtual_image(&mut instance, address), Err(Error::InvalidField));
        assert_eq!(instance.as_bytes(), &initial);
    }
    bind_virtual_image(&mut instance, 0x104a331000).unwrap();
    assert_eq!(instance.get(Field::<u32>::new(0x210)), Ok(0x4a331004));
    assert_eq!(instance.get(Field::<u32>::new(0x214)), Ok(0x10));
    assert_eq!(instance.get(Field::<u32>::new(0x218)), Ok(0xa5a5a580));
    assert_eq!(instance.get(Field::<u32>::new(0xac)), Ok(0xa5a5a5a5));
    for (i, (&actual, &original)) in instance.as_bytes().iter().zip(&initial).enumerate() {
        if i != 0xae && !(0x210..0x219).contains(&i) {
            assert_eq!(actual, original);
        }
    }
}
