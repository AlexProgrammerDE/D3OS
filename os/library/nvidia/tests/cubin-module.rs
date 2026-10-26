//! Exercise the kernel ELF consumer with the real SDK, without kernel runtime.
extern crate alloc;

#[path = "../../../kernel/src/device/nvidia/cubin.rs"]
pub mod cubin;

use cubin::{LegacyPolicy, LegacyQualification, prepare_module};
use goblin::elf::{Elf, section_header::*};

const FAKE_GPU_VA: u64 = 0x1234_0000_0000;

fn put16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// A synthetic ELF container, never executable GPU code or runtime admission.
fn module(legacy: bool) -> Vec<u8> {
    let names = [
        "",
        ".shstrtab",
        ".strtab",
        ".symtab",
        ".nv.constant4",
        ".nv.global.init",
        ".text.probe",
        ".nv.info.probe",
        ".rel.nv.constant4",
        ".nv.global",
    ];
    let mut strings = Vec::new();
    let mut offsets = Vec::new();
    for name in names {
        offsets.push(strings.len() as u32);
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);
    }
    let symbols = b"\0first\0second\0___NV_TILE_LAUNCH_META_DATA___\0";
    let mut table = vec![0; if legacy { 96 } else { 72 }];
    for (index, name, section, value, size) in [(1, 1, 5, 0, 8), (2, 7, 5, 8, 8), (3, 14, 9, 0, 64)] {
        if index == 3 && !legacy {
            continue;
        }
        let offset = index * 24;
        put32(&mut table, offset, name);
        table[offset + 4] = 0x11; // GLOBAL OBJECT
        put16(&mut table, offset + 6, section);
        put64(&mut table, offset + 8, value);
        put64(&mut table, offset + 16, size);
    }
    let mut relocs = Vec::new();
    for (offset, symbol) in if legacy { vec![(0, 3), (8, 1), (16, 2)] } else { vec![(0, 1), (8, 2)] } {
        relocs.extend_from_slice(&(offset as u64).to_le_bytes());
        relocs.extend_from_slice(&((symbol as u64) << 32 | 2).to_le_bytes());
    }
    let mut info = vec![4, 0x10, 12, 0];
    for dimension in [128_u32, 1, 1] {
        info.extend_from_slice(&dimension.to_le_bytes());
    }
    let payloads = [
        Vec::new(),
        strings,
        symbols.to_vec(),
        table,
        vec![0; if legacy { 24 } else { 16 }],
        (1..=16).collect(),
        vec![0x42; 128],
        info,
        relocs,
        Vec::new(),
    ];
    let kinds = [
        SHT_NULL,
        SHT_STRTAB,
        SHT_STRTAB,
        SHT_SYMTAB,
        SHT_PROGBITS,
        SHT_PROGBITS,
        SHT_PROGBITS,
        SHT_PROGBITS,
        SHT_REL,
        SHT_NOBITS,
    ];
    let flags = [0, 0, 0, 0, 2, 3, 6, 0, 0, 3];
    let alignments = [0, 1, 1, 8, 8, 8, 128, 1, 8, 8];
    let mut bytes = vec![0; 64];
    let mut headers = vec![[0_u8; 64]; names.len()];
    for i in 1..names.len() {
        let alignment = alignments[i] as usize;
        bytes.resize(bytes.len().div_ceil(alignment) * alignment, 0);
        let h = &mut headers[i];
        put32(h, 0, offsets[i]);
        put32(h, 4, kinds[i]);
        put64(h, 8, flags[i]);
        put64(h, 24, bytes.len() as u64);
        put64(h, 32, if i == 9 { 64 } else { payloads[i].len() as u64 });
        put64(h, 48, alignments[i]);
        if i == 3 {
            put32(h, 40, 2);
            put32(h, 44, 1);
            put64(h, 56, 24);
        }
        if i == 6 {
            put32(h, 44, 8 << 24);
        }
        if i == 8 {
            put32(h, 40, 3);
            put32(h, 44, 4);
            put64(h, 56, 16);
        }
        bytes.extend_from_slice(&payloads[i]);
    }
    bytes.resize(bytes.len().div_ceil(8) * 8, 0);
    let section_offset = bytes.len();
    for h in headers {
        bytes.extend_from_slice(&h);
    }
    bytes[..9].copy_from_slice(b"\x7fELF\x02\x01\x01\x41\x08");
    put16(&mut bytes, 16, 2);
    put16(&mut bytes, 18, 190);
    put32(&mut bytes, 20, 1);
    put64(&mut bytes, 40, section_offset as u64);
    put32(&mut bytes, 48, 0x0600_5904);
    put16(&mut bytes, 52, 64);
    put16(&mut bytes, 58, 64);
    put16(&mut bytes, 60, names.len() as u16);
    put16(&mut bytes, 62, 1);
    bytes
}

fn qualified(bytes: &[u8]) -> LegacyPolicy<'_> {
    LegacyPolicy::QualifiedUnused(
        LegacyQualification::from_trusted_admission(bytes, [1; 32], "synthetic parser test", "no runtime or SASS qualification").unwrap(),
    )
}
fn header(bytes: &[u8], section: usize) -> usize {
    u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize + section * 64
}
fn payload(bytes: &[u8], section: usize) -> usize {
    let h = header(bytes, section);
    u64::from_le_bytes(bytes[h + 24..h + 32].try_into().unwrap()) as usize
}

#[test]
fn complete_module_preserves_layout_and_patches_only_gpu_virtual_pointers() {
    let bytes = module(false);
    let prepared = prepare_module(&bytes, LegacyPolicy::Reject).unwrap();
    let entry = prepared.entry("probe").unwrap();
    assert_eq!((entry.image_offset, entry.registers, entry.required_threads), (256, 8, Some([128, 1, 1])));
    assert_eq!(entry.constants[4].unwrap().offset, 0);
    assert_eq!(prepared.plan().patches().len(), 2);
    let pristine = prepared.pristine_bytes().to_vec();
    let patched = prepared.apply(FAKE_GPU_VA).unwrap();
    for (slot, offset) in [(0, 128), (8, 136)] {
        assert_eq!(u64::from_le_bytes(patched[slot..slot + 8].try_into().unwrap()), FAKE_GPU_VA + offset);
    }
    assert_eq!(&patched[16..], &pristine[16..]);
    assert_eq!(&patched[128..144], &(1_u8..=16).collect::<Vec<_>>());
    assert_eq!(bytes, module(false));
    let rebased = prepare_module(&bytes, LegacyPolicy::Reject).unwrap().apply(FAKE_GPU_VA + 4096).unwrap();
    assert_eq!(u64::from_le_bytes(rebased[..8].try_into().unwrap()), FAKE_GPU_VA + 4096 + 128);
}

#[test]
fn legacy_exception_requires_complete_module_qualification_and_exact_shape() {
    let bytes = module(true);
    assert!(prepare_module(&bytes, LegacyPolicy::default()).is_err());
    let prepared = prepare_module(&bytes, qualified(&bytes)).unwrap();
    assert_eq!(prepared.legacy_slots(), &[0]);
    assert_eq!(prepared.plan().patches().len(), 2);
    let image = prepared.apply(FAKE_GPU_VA).unwrap();
    assert_eq!(&image[..8], &[0; 8]);
    let other = module(false);
    assert!(prepare_module(&bytes, qualified(&other)).is_err());
    let mut changed = bytes.clone();
    let symbol = payload(&changed, 3) + 3 * 24;
    put64(&mut changed, symbol + 16, 63);
    assert!(prepare_module(&changed, qualified(&changed)).is_err());
}

#[test]
fn unsupported_live_forms_and_partial_symbols_fail_before_upload() {
    for (field, value) in [(8, 1_u64 << 32 | 99), (8, 9_u64 << 32 | 2), (0, 1), (16, 0)] {
        let mut bytes = module(false);
        let offset = payload(&bytes, 8);
        put64(&mut bytes, offset + field, value);
        assert!(prepare_module(&bytes, LegacyPolicy::Reject).is_err());
    }
    let mut bytes = module(false);
    let slot = payload(&bytes, 4);
    bytes[slot] = 1;
    assert!(prepare_module(&bytes, LegacyPolicy::Reject).is_err());
    let mut bytes = module(false);
    let symbol = payload(&bytes, 3) + 24;
    put64(&mut bytes, symbol + 16, 17);
    assert!(prepare_module(&bytes, LegacyPolicy::Reject).is_err());
    put64(&mut bytes, symbol + 16, 8);
    put16(&mut bytes, symbol + 6, 0);
    assert!(prepare_module(&bytes, LegacyPolicy::Reject).is_err());
}

#[test]
fn debug_targets_skip_runtime_resolution_but_keep_structural_validation() {
    let mut bytes = module(false);
    let target = header(&bytes, 4);
    put64(&mut bytes, target + 8, 0);
    let record = payload(&bytes, 8);
    put64(&mut bytes, record, u64::MAX);
    put64(&mut bytes, record + 8, u64::MAX);
    let prepared = prepare_module(&bytes, LegacyPolicy::Reject).unwrap();
    assert_eq!(prepared.ignored_debug_records(), 2);
    assert!(prepared.plan().patches().is_empty());
    let container = header(&bytes, 8);
    put32(&mut bytes, container + 40, 99);
    assert!(prepare_module(&bytes, LegacyPolicy::Reject).is_err());
}

#[test]
fn malformed_ranges_strides_alignments_and_headers_reject_without_panics() {
    let bytes = module(false);
    for end in 0..bytes.len() {
        assert!(prepare_module(&bytes[..end], LegacyPolicy::Reject).is_err());
    }
    for (section, field, value) in [(3, 56, 23), (8, 56, 24), (5, 48, 3), (5, 16, 1), (5, 24, u64::MAX), (5, 32, u64::MAX)] {
        let mut changed = bytes.clone();
        let h = header(&changed, section);
        put64(&mut changed, h + field, value);
        assert!(prepare_module(&changed, LegacyPolicy::Reject).is_err());
    }
    let mut changed = bytes;
    put32(&mut changed, 48, 0);
    assert!(prepare_module(&changed, LegacyPolicy::Reject).is_err());
}

#[test]
fn reservation_alignment_domain_and_missing_entry_are_checked() {
    let bytes = module(false);
    let prepared = prepare_module(&bytes, LegacyPolicy::Reject).unwrap();
    assert!(prepared.entry("missing").is_err());
    for base in [0, 1, FAKE_GPU_VA + 1, (1 << 48) - 4096, u64::MAX] {
        assert!(prepared.validate_gpu_base(base).is_err());
    }
    assert!(prepared.validate_gpu_base(FAKE_GPU_VA).is_ok());
    assert!(prepared.allocation_bytes() >= prepared.pristine_bytes().len() as u64 + 4096);
    assert!(Elf::parse(&bytes).is_ok());
}
