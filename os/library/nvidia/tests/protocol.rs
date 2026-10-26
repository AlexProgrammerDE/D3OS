use d3os_nvidia::{Error, Field, Record, rpc};

#[test]
fn unaligned_wire_fields_preserve_neighbors_and_reject_overflow() {
    let mut record = Record::<12>::default();
    record.set(Field::<u8>::new(0), 0xaa).unwrap();
    record.set(Field::<u64>::new(1), 0x1122334455667788).unwrap();
    record.set(Field::<u16>::new(10), 0xbbcc).unwrap();
    assert_eq!(record.as_bytes(), &[0xaa, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0, 0xcc, 0xbb]);
    assert_eq!(record.get(Field::<u64>::new(1)), Ok(0x1122334455667788));
    assert_eq!(record.get(Field::<u32>::new(10)), Err(Error::Truncated));
    assert_eq!(record.set(Field::<u64>::new(usize::MAX), 1), Err(Error::Truncated));
    assert!(matches!(Record::<12>::from_bytes(&[0; 11]), Err(Error::Truncated)));
}

#[test]
fn rpc_payloads_cross_page_boundaries_without_exposing_padding() {
    for length in [0, 1, 7, 8, 15, rpc::PAGE - rpc::PREFIX, rpc::PAGE - rpc::PREFIX + 1, 16384, rpc::MAX_PAYLOAD] {
        let payload: Vec<_> = (0..length).map(|i| i as u8).collect();
        let bytes = rpc::encode(0x1001, u32::MAX, &payload).unwrap();
        let response = rpc::decode(&bytes).unwrap();
        assert_eq!(response.function, 0x1001);
        assert_eq!(response.sequence, u32::MAX);
        assert_eq!(response.payload, payload);
        assert_eq!(response.pages * rpc::PAGE, bytes.len());
        assert!(bytes[rpc::PREFIX + length..].iter().all(|byte| *byte == 0));
    }
    assert!(matches!(rpc::encode(1, 0, &vec![0; rpc::MAX_PAYLOAD + 1]), Err(Error::InvalidQueue)));
}

#[test]
fn rpc_rejects_corruption_and_truncation() {
    let original = rpc::encode(2, 3, &[0xaa; 19]).unwrap();
    for length in 0..rpc::PREFIX + 19 {
        assert!(rpc::decode(&original[..length]).is_err());
    }
    let mut corrupt = original.clone();
    corrupt[rpc::PREFIX + 5] ^= 0x80;
    assert!(matches!(rpc::decode(&corrupt), Err(Error::InvalidChecksum)));
    let mut corrupt = original.clone();
    corrupt[40..44].copy_from_slice(&17u32.to_le_bytes());
    assert!(matches!(rpc::decode(&corrupt), Err(Error::InvalidQueue)));
    // The firmware can leave old bytes after the declared message.
    let mut padding = original;
    padding[rpc::PREFIX + 19] = 0xff;
    assert!(rpc::decode(&padding).is_ok());
}

#[test]
fn ring_transfers_wrap_and_keep_one_empty_slot() {
    for count in 2..32 {
        for start in 0..count {
            for pages in 1..count {
                let (first, second) = rpc::spans(start, pages, count).unwrap();
                assert_eq!(first + second, pages);
                let slots: Vec<_> = (start..start + first).chain(0..second).collect();
                assert!(slots.iter().all(|slot| *slot < count));
                assert_eq!(slots, (0..pages).map(|i| (start + i) % count).collect::<Vec<_>>());
            }
            assert_eq!(rpc::available(start, start, count), Ok(count - 1));
            assert_eq!(rpc::available((start + count - 1) % count, start, count), Ok(0));
        }
    }
    assert_eq!(rpc::spans(63, 1, 63), Err(Error::InvalidQueue));
    assert_eq!(rpc::available(0, 0, 1), Err(Error::InvalidQueue));
}
