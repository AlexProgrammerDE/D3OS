use d3os_ai::qwen::{
    Error,
    gguf::{
        ReadAt,
        shards::{Index, Sharded},
    },
};

fn index(length: u64, part_bytes: u32, parts: u32) -> Vec<u8> {
    let mut bytes = b"D3MODEL\0".to_vec();
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(part_bytes.to_le_bytes());
    bytes.extend(length.to_le_bytes());
    bytes.extend(parts.to_le_bytes());
    bytes.extend([37; 32]);
    bytes
}

#[test]
fn sharded_reads_preserve_order_across_boundaries_and_reject_truncated_parts() {
    const PART: usize = 1 << 20;
    let bytes: Vec<_> = (0..2 * PART + 17).map(|i| (i as u8).wrapping_mul(37).wrapping_add((i >> 20) as u8)).collect();
    let metadata = Index::parse(&index(bytes.len() as u64, PART as u32, 3)).unwrap();
    let mut source = Sharded::new(metadata.clone(), bytes.chunks(PART).collect()).unwrap();
    assert_eq!(source.len(), bytes.len() as u64);
    assert_eq!(source.index().source_sha256(), &[37; 32]);
    assert_eq!(source.index().part_len(3), None);
    for (offset, length) in [(0, 31), (PART - 5, PART + 13), (2 * PART - 7, 24), (bytes.len(), 0), (PART + 19, 227)] {
        let mut actual = vec![0; length];
        source.read_at(offset as u64, &mut actual).unwrap();
        assert_eq!(actual, bytes[offset..offset + length]);
    }
    assert_eq!(source.read_at(bytes.len() as u64, &mut [0]), Err(Error::SourceRead));
    assert_eq!(source.read_at(u64::MAX, &mut [0]), Err(Error::SourceRead));
    assert!(Sharded::new(metadata.clone(), vec![&bytes[..PART], &bytes[PART..2 * PART]]).is_err());
    assert!(Sharded::new(metadata, vec![&bytes[..PART], &bytes[PART..2 * PART], &bytes[2 * PART..bytes.len() - 1]]).is_err());
}

#[test]
fn shard_index_rejects_invalid_extents_versions_and_counts() {
    for bytes in [
        index(0, 1 << 20, 1),
        index(1, 1 << 20, 0),
        index(1, 1 << 20, 65),
        index(1, 1 << 19, 1),
        index(1, 3 << 20, 1),
        index((1 << 20) + 1, 1 << 20, 1),
        index(u64::MAX, 1 << 30, 64),
    ] {
        assert!(Index::parse(&bytes).is_err());
    }
    let bytes = index(7, 1 << 20, 1);
    for length in 0..bytes.len() {
        assert!(Index::parse(&bytes[..length]).is_err());
    }
    for offset in [0, 8] {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 1;
        assert!(Index::parse(&corrupt).is_err());
    }
}
