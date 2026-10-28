//! Synthetic wire fixtures exercise parsing, never establish qualification.

use d3os_nvidia::cubin::qualification::{InputKind, ModuleQualification, RuntimeInventory};

fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn receipt(kind: u32) -> Vec<u8> {
    let mut bytes = b"D3QMOD01".to_vec();
    for value in [1_u32, 1, 89, 3, kind, 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for digest in 1..=5 {
        bytes.extend_from_slice(&[digest; 32]);
    }
    bytes
}

fn inventory(records: &[(&str, u64, [u8; 32])]) -> Vec<u8> {
    let mut bytes = b"D3QRT001".to_vec();
    bytes.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for &(path, length, digest) in records {
        bytes.extend_from_slice(&(path.len() as u32).to_le_bytes());
        bytes.extend_from_slice(path.as_bytes());
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(&digest);
    }
    bytes
}

#[test]
fn module_decodes_both_inputs_and_all_digest_positions() {
    for (wire, kind) in [(1, InputKind::TileIr), (2, InputKind::Ptx)] {
        let parsed = ModuleQualification::parse(&receipt(wire)).unwrap();
        assert_eq!(parsed.input_kind(), kind);
        assert_eq!(parsed.source_sha256(), &[1; 32]);
        assert_eq!(parsed.cubin_sha256(), &[2; 32]);
        assert_eq!(parsed.sass_sha256(), &[3; 32]);
        assert_eq!(parsed.compiler_sha256(), &[4; 32]);
        assert_eq!(parsed.runtime_sha256(), &[5; 32]);
    }
}

#[test]
fn module_requires_exact_framing_and_supported_header_fields() {
    let valid = receipt(1);
    for end in 0..valid.len() {
        assert!(ModuleQualification::parse(&valid[..end]).is_err(), "prefix length {end}");
    }
    let mut extra = valid.clone();
    extra.push(0);
    assert!(ModuleQualification::parse(&extra).is_err());

    let mut magic = valid.clone();
    magic[7] ^= 1;
    assert!(ModuleQualification::parse(&magic).is_err());
    for (offset, unsupported) in [(8, 0), (8, 2), (12, 0), (12, 2), (16, 90), (20, 2), (24, 0), (24, 3), (28, 1)] {
        let mut bytes = valid.clone();
        put32(&mut bytes, offset, unsupported);
        assert!(ModuleQualification::parse(&bytes).is_err(), "field at {offset} = {unsupported}");
    }
    for offset in (8..32).step_by(4) {
        let mut bytes = valid.clone();
        put32(&mut bytes, offset, u32::MAX);
        assert!(ModuleQualification::parse(&bytes).is_err());
    }
}

#[test]
fn module_rejects_each_missing_digest_but_allows_sparse_nonzero_digests() {
    let valid = receipt(2);
    for offset in (32..192).step_by(32) {
        let mut bytes = valid.clone();
        bytes[offset..offset + 32].fill(0);
        assert!(ModuleQualification::parse(&bytes).is_err(), "digest at {offset}");
        bytes[offset + 31] = 1;
        assert!(ModuleQualification::parse(&bytes).is_ok());
    }
}

#[test]
fn inventory_decodes_sorted_assets_and_preserves_canonical_bytes() {
    let records = [
        ("lib64/libgcc_s.so.1", 1, [1; 32]),
        ("usr/cuda/bin/tileiras", u64::MAX, [2; 32]),
        ("usr/cuda/nvvm/libdevice/libdevice.10.bc", 4096, [3; 32]),
        ("usr/share/nvidia/family.ptx", 42, [4; 32]),
        ("usr/share/nvidia/family.tilebc", 84, [5; 32]),
    ];
    let bytes = inventory(&records);
    let parsed = RuntimeInventory::parse(&bytes).unwrap();
    assert_eq!(parsed.canonical_bytes(), &bytes[12..]);
    assert_eq!(parsed.canonical_bytes().as_ptr(), bytes[12..].as_ptr());
    assert_eq!(parsed.records().len(), records.len());
    for (actual, &(path, length, digest)) in parsed.records().iter().zip(&records) {
        assert_eq!(actual.path(), path);
        assert_eq!(actual.bytes(), length);
        assert_eq!(actual.sha256(), &digest);
    }
}

#[test]
fn inventory_admits_only_the_three_registered_cached_module_paths() {
    for family in ["vector-add", "tensor-matmul", "inference"] {
        let path = format!("usr/share/nvidia/{family}.cubin");
        let bytes = inventory(&[(&path, 4096, [7; 32])]);
        let parsed = RuntimeInventory::parse(&bytes).unwrap();
        assert_eq!(parsed.records()[0].path(), path);
        assert_eq!(parsed.records()[0].bytes(), 4096);
        assert_eq!(parsed.records()[0].sha256(), &[7; 32]);
    }
    for path in [
        "usr/share/nvidia/other.cubin",
        "usr/share/nvidia/.cubin",
        "usr/share/nvidia/inference.cubin.extra",
        "usr/share/nvidia/sub/inference.cubin",
    ] {
        assert!(RuntimeInventory::parse(&inventory(&[(path, 4096, [7; 32])])).is_err());
    }
}

#[test]
fn inventory_rejects_every_truncation_trailing_data_and_count_mismatch() {
    let valid = inventory(&[("usr/cuda/bin/tileiras", 123, [7; 32])]);
    for end in 0..valid.len() {
        assert!(RuntimeInventory::parse(&valid[..end]).is_err(), "prefix length {end}");
    }
    let mut extra = valid.clone();
    extra.push(0);
    assert!(RuntimeInventory::parse(&extra).is_err());
    let mut magic = valid.clone();
    magic[0] ^= 1;
    assert!(RuntimeInventory::parse(&magic).is_err());
    for count in [0, 2, 129, u32::MAX] {
        let mut bytes = valid.clone();
        put32(&mut bytes, 8, count);
        assert!(RuntimeInventory::parse(&bytes).is_err(), "record count {count}");
    }
    let path_length = "usr/cuda/bin/tileiras".len() as u32;
    for length in [0, 1, path_length - 1, path_length + 1, 513, u32::MAX] {
        let mut bytes = valid.clone();
        put32(&mut bytes, 12, length);
        assert!(RuntimeInventory::parse(&bytes).is_err(), "path length {length}");
    }
}

#[test]
fn inventory_requires_strict_path_order_without_duplicates() {
    for paths in [
        ["usr/cuda/a", "usr/cuda/a"],
        ["usr/cuda/b", "usr/cuda/a"],
        ["usr/share/nvidia/a.ptx", "lib64/libgcc_s.so.1"],
    ] {
        let bytes = inventory(&[(paths[0], 1, [1; 32]), (paths[1], 2, [2; 32])]);
        assert!(RuntimeInventory::parse(&bytes).is_err());
    }
    let bytes = inventory(&[("usr/cuda/A", 1, [1; 32]), ("usr/cuda/a", 2, [2; 32])]);
    assert!(RuntimeInventory::parse(&bytes).is_ok());
}

#[test]
fn inventory_rejects_path_aliases_controls_and_unstaged_locations() {
    for path in [
        "",
        "/usr/cuda/a",
        "//usr/cuda/a",
        "./usr/cuda/a",
        "usr/cuda/./a",
        "usr/cuda/../a",
        "usr/cuda//a",
        "usr/cuda/a/",
        "usr/cuda/a/..",
        "usr/cuda/a/.",
        "usr/cuda/a\0b",
        "usr/cuda/a\nb",
        "usr/cuda/a\u{7f}b",
        "usr/cuda/a\u{85}b",
        "usr/cuda/a\\b",
        "usr/cuda",
        "usr/cuda-other/a",
        "usr/share/nvidia/compiler-manifest.json",
        "usr/share/nvidia/runtime.qualification",
        "usr/share/nvidia/.tilebc",
        "usr/share/nvidia/.ptx",
        "usr/share/nvidia/sub/a.ptx",
        "usr/share/nvidia/a.ptx.extra",
        "lib64",
        "lib/a.so",
        "etc/a",
    ] {
        assert!(RuntimeInventory::parse(&inventory(&[(path, 1, [1; 32])])).is_err(), "path {path:?}");
    }
    let mut invalid_utf8 = inventory(&[("usr/cuda/a", 1, [1; 32])]);
    invalid_utf8[16 + "usr/cuda/".len()] = 0xff;
    assert!(RuntimeInventory::parse(&invalid_utf8).is_err());
}

#[test]
fn inventory_enforces_path_and_record_limits_without_overflow() {
    let longest = format!("usr/cuda/{}", "a".repeat(512 - "usr/cuda/".len()));
    assert!(RuntimeInventory::parse(&inventory(&[(&longest, 1, [1; 32])])).is_ok());
    let too_long = format!("{longest}a");
    assert!(RuntimeInventory::parse(&inventory(&[(&too_long, 1, [1; 32])])).is_err());

    let paths: Vec<_> = (0..128).map(|n| format!("usr/cuda/{n:03}")).collect();
    let mut records: Vec<_> = paths.iter().map(|path| (path.as_str(), 1, [1; 32])).collect();
    let bytes = inventory(&records);
    let parsed = RuntimeInventory::parse(&bytes).unwrap();
    assert_eq!(parsed.records().len(), 128);
    records.push(("usr/cuda/128", 1, [1; 32]));
    assert!(RuntimeInventory::parse(&inventory(&records)).is_err());

    let empty = inventory(&[]);
    let parsed = RuntimeInventory::parse(&empty).unwrap();
    assert!(parsed.records().is_empty());
    assert!(parsed.canonical_bytes().is_empty());
}

#[test]
fn inventory_requires_nonzero_asset_sizes_and_digests() {
    assert!(RuntimeInventory::parse(&inventory(&[("usr/cuda/a", 0, [1; 32])])).is_err());
    assert!(RuntimeInventory::parse(&inventory(&[("usr/cuda/a", 1, [0; 32])])).is_err());
    let mut sparse = [0; 32];
    sparse[31] = 1;
    assert!(RuntimeInventory::parse(&inventory(&[("usr/cuda/a", 1, sparse)])).is_ok());
}
