//! Prepare the observed SM89 CUDA ELF profile before device allocation.
//!
//! This is a qualified zero-address, zero-addend profile, not a CUDA psABI.
//! Only the trusted admission caller can authorize the legacy unused slot.
//! No allocator, upload, cache, device, or hardware recovery belongs here.

use alloc::vec::Vec;
use core::{fmt, ops::Range};
use d3os_nvidia::cubin::relocation::{Cuda64Patch, Cuda64Plan, checked_image_section};
use d3os_nvidia::{Error, cubin};
use goblin::container::{Container, Ctx, Endian};
use goblin::elf::{
    Elf,
    header::Header,
    program_header,
    section_header::*,
    sym::{self, Sym, Symtab},
};

const CTX: Ctx = Ctx {
    container: Container::Big,
    le: Endian::Little,
};
const LEGACY_NAME: &str = "___NV_TILE_LAUNCH_META_DATA___";
const GPU_VA_END: u64 = 1 << 48;
const BASE_ALIGNMENT: u64 = 4096;
// ELF RELR and Android packed relocation containers, absent from Goblin 0.9.3 constants.
const SHT_RELR: u32 = 19;
const SHT_ANDROID_REL: u32 = 0x6000_0001;
const SHT_ANDROID_RELA: u32 = 0x6000_0002;
const SHT_ANDROID_RELR: u32 = 0x6fff_ff00;
// Bounded staging memory for this first observed-profile adapter.
const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;

/// A recoverable input rejection with ELF indices when available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterError {
    pub code: Error,
    pub reason: &'static str,
    pub section: Option<usize>,
    pub record: Option<usize>,
}

pub type Result<T> = core::result::Result<T, AdapterError>;

impl AdapterError {
    fn at(mut self, section: usize, record: Option<usize>) -> Self {
        self.section = Some(section);
        self.record = record;
        self
    }
}

fn error(code: Error, reason: &'static str) -> AdapterError {
    AdapterError {
        code,
        reason,
        section: None,
        record: None,
    }
}
fn invalid(reason: &'static str) -> AdapterError {
    error(Error::InvalidField, reason)
}
fn unsupported(reason: &'static str) -> AdapterError {
    error(Error::Unsupported, reason)
}
fn index(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("value does not fit usize"))
}
fn sum(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| invalid("size overflow"))
}
fn range(offset: u64, size: u64, limit: usize) -> Result<Range<usize>> {
    let end = offset.checked_add(size).ok_or_else(|| invalid("range overflow"))?;
    let r = index(offset)?..index(end)?;
    if r.end > limit {
        return Err(error(Error::Truncated, "range exceeds input"));
    }
    Ok(r)
}
fn payload<'a>(bytes: &'a [u8], section: &SectionHeader) -> Result<&'a [u8]> {
    if section.sh_type == SHT_NOBITS {
        return Err(invalid("NOBITS has no payload"));
    }
    Ok(&bytes[range(section.sh_offset, section.sh_size, bytes.len())?])
}
fn string(bytes: &[u8], offset: usize) -> Result<&str> {
    let tail = bytes.get(offset..).ok_or_else(|| invalid("string index"))?;
    let end = tail.iter().position(|&b| b == 0).ok_or_else(|| invalid("unterminated string"))?;
    core::str::from_utf8(&tail[..end]).map_err(|_| invalid("invalid UTF-8 string"))
}

/// Provenance asserted by a trusted complete-module admission path.
///
/// The caller verifies SHA256, compiler/profile, and reviewed metadata-use evidence
/// before construction. This adapter does not compute SHA256 or analyze SASS.
/// The borrowed immutable bytes bind the assertion to the exact prepared module.
#[derive(Clone, Copy)]
pub struct LegacyQualification<'a> {
    module: &'a [u8],
    sha256: [u8; 32],
    compiler: &'a str,
    evidence: &'a str,
}
impl fmt::Debug for LegacyQualification<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LegacyQualification")
            .field("module_bytes", &self.module.len())
            .field("sha256", &self.sha256)
            .field("compiler", &self.compiler)
            .field("evidence", &self.evidence)
            .finish()
    }
}
impl<'a> LegacyQualification<'a> {
    pub fn from_trusted_admission(module: &'a [u8], sha256: [u8; 32], compiler: &'a str, evidence: &'a str) -> Result<Self> {
        if module.is_empty() || sha256 == [0; 32] || compiler.trim().is_empty() || evidence.trim().is_empty() {
            return Err(invalid("missing module qualification provenance"));
        }
        Ok(Self {
            module,
            sha256,
            compiler,
            evidence,
        })
    }
    pub const fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }
    pub const fn compiler(&self) -> &'a str {
        self.compiler
    }
    pub const fn evidence(&self) -> &'a str {
        self.evidence
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum LegacyPolicy<'a> {
    #[default]
    Reject,
    QualifiedUnused(LegacyQualification<'a>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSection {
    pub offset: usize,
    pub bytes: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstantBinding {
    pub offset: usize,
    pub bytes: u32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryMetadata<'a> {
    pub name: &'a str,
    pub section_index: usize,
    pub image_offset: usize,
    pub code_bytes: u64,
    pub registers: u32,
    pub local_bytes: u64,
    pub shared_bytes: u32,
    pub required_threads: Option<[u32; 3]>,
    pub constants: [Option<ConstantBinding>; 8],
}

/// Private pristine bytes and immutable validated plan. Prepare once per module.
/// Entry selection and every metadata check occur before allocation.
pub struct PreparedImage<'a> {
    image: Vec<u8>,
    plan: Cuda64Plan,
    sections: Vec<Option<ImageSection>>,
    entries: Vec<EntryMetadata<'a>>,
    legacy_slots: Vec<usize>,
    debug_records: usize,
    allocation_bytes: u64,
    qualification: Option<LegacyQualification<'a>>,
}
impl fmt::Debug for PreparedImage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedImage")
            .field("image_bytes", &self.image.len())
            .field("patches", &self.plan.patches().len())
            .field("entries", &self.entries.len())
            .field("legacy_slots", &self.legacy_slots)
            .field("debug_records", &self.debug_records)
            .field("allocation_bytes", &self.allocation_bytes)
            .finish()
    }
}
impl<'a> PreparedImage<'a> {
    pub fn pristine_bytes(&self) -> &[u8] {
        &self.image
    }
    pub const fn plan(&self) -> &Cuda64Plan {
        &self.plan
    }
    pub fn sections(&self) -> &[Option<ImageSection>] {
        &self.sections
    }
    pub fn entries(&self) -> &[EntryMetadata<'a>] {
        &self.entries
    }
    pub fn entry(&self, name: &str) -> Result<&EntryMetadata<'a>> {
        self.entries.iter().find(|entry| entry.name == name).ok_or_else(|| invalid("entry not found"))
    }
    pub fn legacy_slots(&self) -> &[usize] {
        &self.legacy_slots
    }
    pub const fn ignored_debug_records(&self) -> usize {
        self.debug_records
    }
    pub const fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }
    pub const fn qualification(&self) -> Option<&LegacyQualification<'a>> {
        self.qualification.as_ref()
    }

    /// Consume pristine staging bytes and patch with the actual GPU virtual base.
    /// The caller reserves `allocation_bytes()` at 4096-byte alignment first.
    /// No physical address enters this method. Returned bytes are ready for upload.
    pub fn apply(mut self, gpu_virtual_base: u64) -> Result<Vec<u8>> {
        self.validate_gpu_base(gpu_virtual_base)?;
        self.plan
            .apply(&mut self.image, gpu_virtual_base)
            .map_err(|code| error(code, "pointer patch validation"))?;
        Ok(self.image)
    }

    /// Check the full reservation, including the native extra 4096 bytes.
    pub fn validate_gpu_base(&self, base: u64) -> Result<()> {
        if base == 0 || !base.is_multiple_of(BASE_ALIGNMENT) {
            return Err(invalid("GPU virtual base alignment"));
        }
        let end = base.checked_add(self.allocation_bytes).ok_or_else(|| invalid("GPU virtual extent overflow"))?;
        if end > GPU_VA_END {
            return Err(invalid("GPU virtual extent exceeds 48 bits"));
        }
        Ok(())
    }
}

/// Parse, pack, and validate all allocated relocations before GPU allocation.
/// Allocated PROGBITS retain native order and max(128, sh_addralign) packing.
/// Nonallocated targets skip only runtime arithmetic, after container validation.
pub fn prepare_module<'a>(bytes: &'a [u8], legacy: LegacyPolicy<'a>) -> Result<PreparedImage<'a>> {
    if let LegacyPolicy::QualifiedUnused(q) = legacy
        && q.module != bytes
    {
        return Err(unsupported("qualification binds a different complete module"));
    }
    let headers = preflight(bytes)?;
    let elf = Elf::parse(bytes).map_err(|_| invalid("goblin ELF parse"))?;
    if elf.section_headers != headers {
        return Err(invalid("inconsistent section parsing"));
    }
    let names_section = &headers[usize::from(elf.header.e_shstrndx)];
    let names = payload(bytes, names_section)?;
    let section_name = |i: usize| string(names, headers[i].sh_name);

    let mut sections = alloc::vec![None; headers.len()];
    let mut file_ranges: Vec<Range<usize>> = Vec::new();
    let mut cursor = 0;
    for (i, s) in headers.iter().enumerate() {
        if s.sh_type != SHT_PROGBITS || s.sh_flags & u64::from(SHF_ALLOC) == 0 {
            continue;
        }
        let validate = || -> Result<(usize, usize, Range<usize>)> {
            if s.sh_addr != 0 {
                return Err(unsupported("nonzero allocated section address"));
            }
            if !matches!(s.sh_flags, 2 | 3 | 6 | 0x42) {
                return Err(unsupported("allocated PROGBITS flags"));
            }
            if s.sh_addralign > BASE_ALIGNMENT {
                return Err(unsupported("section exceeds base alignment guarantee"));
            }
            let source = range(s.sh_offset, s.sh_size, bytes.len())?;
            if !s.sh_offset.is_multiple_of(s.sh_addralign.max(1)) {
                return Err(invalid("misaligned section payload"));
            }
            let (start, end) = checked_image_section(cursor, index(s.sh_size)?, s.sh_addralign).map_err(|code| error(code, "section layout"))?;
            if end > MAX_IMAGE_BYTES {
                return Err(unsupported("staging image exceeds 64 MiB profile limit"));
            }
            Ok((start, end, source))
        };
        let (start, end, source) = validate().map_err(|e| e.at(i, None))?;
        if !source.is_empty() {
            file_ranges.push(source);
        }
        sections[i] = Some(ImageSection {
            offset: start,
            bytes: index(s.sh_size)?,
        });
        cursor = end;
    }
    file_ranges.sort_unstable_by_key(|r| r.start);
    if file_ranges.windows(2).any(|w| w[0].end > w[1].start) {
        return Err(invalid("overlapping allocated payloads"));
    }
    // Preserve native allocation allowance and four-byte upload padding.
    let allocation_bytes = u64::try_from(sum(cursor, 4096)?).map_err(|_| invalid("allocation size"))?;
    let image_bytes = sum(cursor, 3)? & !3;
    let mut image = Vec::new();
    image.try_reserve_exact(image_bytes).map_err(|_| invalid("staging allocation failed"))?;
    image.resize(image_bytes, 0);
    for (i, placement) in sections.iter().enumerate() {
        if let Some(p) = placement {
            image[p.offset..sum(p.offset, p.bytes)?].copy_from_slice(payload(bytes, &headers[i])?);
        }
    }
    let entries = entry_metadata(bytes, &headers, &sections, names)?;
    let mut tables: Vec<Option<SymbolTable<'_>>> = (0..headers.len()).map(|_| None).collect();
    for (i, s) in headers.iter().enumerate() {
        if s.sh_type == SHT_SYMTAB {
            let table = Symtab::parse(bytes, index(s.sh_offset)?, index(s.sh_size / 24)?, CTX).map_err(|_| invalid("goblin symbol table").at(i, None))?;
            let strings = payload(bytes, &headers[index(u64::from(s.sh_link))?])?;
            for (record, symbol) in table.iter().enumerate() {
                string(strings, symbol.st_name).map_err(|e| e.at(i, Some(record)))?;
            }
            tables[i] = Some(SymbolTable { symbols: table, strings });
        }
    }
    let mut patches = Vec::new();
    let mut occupied = Vec::new();
    let mut legacy_slots = Vec::new();
    let mut debug_records = 0;
    for (rel_index, relocs) in &elf.shdr_relocs {
        // Goblin gives the relocation CONTAINER index, never its target index.
        let container = &headers[*rel_index];
        let target_index = index(u64::from(container.sh_info))?;
        let target = &headers[target_index];
        let table = tables[index(u64::from(container.sh_link))?]
            .as_ref()
            .ok_or_else(|| invalid("relocation symbol table").at(*rel_index, None))?;
        if target.sh_flags & u64::from(SHF_ALLOC) == 0 {
            debug_records = sum(debug_records, relocs.len())?;
            continue;
        }
        for (record_index, relocation) in relocs.iter().enumerate() {
            let validate = || -> Result<(usize, Option<Cuda64Patch>)> {
                if container.sh_type != SHT_REL || relocation.r_addend.is_some() || relocation.r_type != 2 {
                    return Err(unsupported("live relocation form or type"));
                }
                let placement = sections[target_index].ok_or_else(|| unsupported("live target is not mapped PROGBITS"))?;
                if target.sh_flags != 2 {
                    return Err(unsupported("live pointer target flags"));
                }
                let loc = relative_extent(relocation.r_offset, 8, target)?;
                if !loc.is_multiple_of(8) {
                    return Err(invalid("unaligned pointer slot"));
                }
                let slot = sum(placement.offset, loc)?;
                if image[slot..sum(slot, 8)?].iter().any(|&b| b != 0) {
                    return Err(unsupported("nonzero implicit REL addend"));
                }
                let symbol = table.symbols.get(relocation.r_sym).ok_or_else(|| invalid("live symbol index"))?;
                if symbol.st_shndx == 0 || symbol.st_shndx >= SHN_LORESERVE as usize {
                    return Err(unsupported("undefined or reserved live symbol section"));
                }
                let source = headers.get(symbol.st_shndx).ok_or_else(|| invalid("live symbol section index"))?;
                let symbol_offset = relative_extent(symbol.st_value, symbol.st_size, source)?;
                let symbol_name = string(table.strings, symbol.st_name)?;
                if symbol_name == LEGACY_NAME {
                    if !legacy_shape(target, section_name(target_index)?, loc, &symbol, source, section_name(symbol.st_shndx)?) {
                        return Err(unsupported("legacy metadata differs from qualified shape"));
                    }
                    if !matches!(legacy, LegacyPolicy::QualifiedUnused(_)) {
                        return Err(unsupported("unqualified legacy metadata"));
                    }
                    return Ok((slot, None));
                }
                let source_placement = sections[symbol.st_shndx].ok_or_else(|| unsupported("live symbol is not mapped PROGBITS"))?;
                if source.sh_flags != 3
                    || !matches!(symbol.st_bind(), sym::STB_LOCAL | sym::STB_GLOBAL)
                    || symbol.st_type() != sym::STT_OBJECT
                    || symbol.st_other != 0
                {
                    return Err(unsupported("live symbol flags, binding, type, or visibility"));
                }
                if symbol.st_size == 0 || symbol_offset >= source_placement.bytes {
                    return Err(invalid("empty or exterior data object"));
                }
                let relative = sum(source_placement.offset, symbol_offset)?;
                let patch = Cuda64Patch::new(slot, relative).map_err(|code| error(code, "pointer patch"))?;
                Ok((slot, Some(patch)))
            };
            let (slot, patch) = validate().map_err(|e| e.at(*rel_index, Some(record_index)))?;
            occupied.push(slot);
            if let Some(patch) = patch {
                patches.push(patch);
            } else {
                legacy_slots.push(slot);
            }
        }
    }
    // Include skipped legacy slots. Duplicates across containers also fail.
    occupied.sort_unstable();
    if occupied.windows(2).any(|w| w[0].checked_add(8).is_none_or(|end| end > w[1])) {
        return Err(invalid("overlapping live or skipped legacy slots"));
    }
    let plan = Cuda64Plan::new(image_bytes, patches).map_err(|code| error(code, "module relocation plan"))?;
    Ok(PreparedImage {
        image,
        plan,
        sections,
        entries,
        legacy_slots,
        debug_records,
        allocation_bytes,
        qualification: match legacy {
            LegacyPolicy::Reject => None,
            LegacyPolicy::QualifiedUnused(q) => Some(q),
        },
    })
}

struct SymbolTable<'a> {
    symbols: Symtab<'a>,
    strings: &'a [u8],
}

fn relative_extent(value: u64, size: u64, section: &SectionHeader) -> Result<usize> {
    // ET_EXEC values are full section addresses. Never assume st_value is an offset.
    let offset = value.checked_sub(section.sh_addr).ok_or_else(|| invalid("address precedes section"))?;
    let end = offset.checked_add(size).ok_or_else(|| invalid("object extent overflow"))?;
    index(size)?;
    if end > section.sh_size {
        return Err(invalid("extent exceeds section"));
    }
    // No nonzero-address convention is qualified by the captured modules.
    if section.sh_addr != 0 {
        return Err(unsupported("nonzero referenced section address"));
    }
    index(offset)
}

fn legacy_shape(target: &SectionHeader, target_name: &str, offset: usize, symbol: &Sym, source: &SectionHeader, source_name: &str) -> bool {
    target_name == ".nv.constant4"
        && target.sh_type == SHT_PROGBITS
        && target.sh_flags == 2
        && matches!(target.sh_size, 8 | 24)
        && offset == 0
        && symbol.st_bind() == sym::STB_GLOBAL
        && symbol.st_type() == sym::STT_OBJECT
        && symbol.st_value == 0
        && symbol.st_size == 64
        && symbol.st_other == 0
        && source_name == ".nv.global"
        && source.sh_type == SHT_NOBITS
        && source.sh_flags == 3
        && source.sh_addr == 0
        && source.sh_size == 64
        && source.sh_addralign == 8
}

/// Validate raw table ranges before Goblin's full parser can follow any links.
fn preflight(bytes: &[u8]) -> Result<Vec<SectionHeader>> {
    if bytes.len() < 64 {
        return Err(error(Error::Truncated, "ELF64 header"));
    }
    if &bytes[..4] != b"\x7fELF" {
        return Err(invalid("ELF magic"));
    }
    if bytes[4..9] != [2, 1, 1, 0x41, 8] {
        return Err(unsupported("ELF class, endian, version, OSABI or ABI version"));
    }
    let h: Header = Elf::parse_header(bytes).map_err(|_| invalid("goblin ELF header"))?;
    if h.e_type != 2 || h.e_machine != 190 || h.e_version != 1 || h.e_flags != 0x06005904 || h.e_entry != 0 {
        return Err(unsupported("unqualified observed SM89 CUDA header profile"));
    }
    if h.e_ehsize != 64
        || h.e_shentsize != 64
        || h.e_shnum == 0
        || h.e_shoff < 64
        || h.e_shstrndx == 0
        || h.e_shstrndx >= h.e_shnum
        || h.e_shnum >= SHN_LORESERVE as u16
    {
        return Err(invalid("section table shape or unsupported extended indices"));
    }
    range(h.e_shoff, u64::from(h.e_shnum) * 64, bytes.len())?;
    if h.e_phnum != 0 {
        if h.e_phnum == 0xffff || h.e_phentsize != 56 || h.e_phoff < 64 {
            return Err(invalid("program table shape"));
        }
        range(h.e_phoff, u64::from(h.e_phnum) * 56, bytes.len())?;
        let programs =
            program_header::ProgramHeader::parse(bytes, index(h.e_phoff)?, usize::from(h.e_phnum), CTX).map_err(|_| invalid("goblin program headers"))?;
        for p in programs {
            range(p.p_offset, p.p_filesz, bytes.len())?;
            p.p_vaddr.checked_add(p.p_memsz).ok_or_else(|| invalid("program address overflow"))?;
            if p.p_type == program_header::PT_DYNAMIC {
                return Err(unsupported("dynamic ELF loading"));
            }
        }
    }
    let sections = SectionHeader::parse(bytes, index(h.e_shoff)?, usize::from(h.e_shnum), CTX).map_err(|_| invalid("goblin section headers"))?;
    if sections[0] != SectionHeader::default() {
        return Err(invalid("nonempty null section or extended indices"));
    }
    for (i, s) in sections.iter().enumerate() {
        let validate = || -> Result<()> {
            index(s.sh_size)?;
            index(s.sh_offset)?;
            index(s.sh_addr)?;
            index(s.sh_addralign)?;
            index(s.sh_entsize)?;
            s.sh_addr.checked_add(s.sh_size).ok_or_else(|| invalid("section address extent overflow"))?;
            if s.sh_type != SHT_NOBITS {
                range(s.sh_offset, s.sh_size, bytes.len())?;
            }
            if matches!(
                s.sh_type,
                SHT_DYNAMIC
                    | SHT_DYNSYM
                    | SHT_SYMTAB_SHNDX
                    | SHT_RELR
                    | SHT_ANDROID_REL
                    | SHT_ANDROID_RELA
                    | SHT_ANDROID_RELR
                    | SHT_GNU_VERSYM
                    | SHT_GNU_VERDEF
                    | SHT_GNU_VERNEED
            ) {
                return Err(unsupported("dynamic, packed, or extended-index section"));
            }
            if s.sh_flags & u64::from(SHF_ALLOC) != 0 && !matches!(s.sh_type, SHT_PROGBITS | SHT_NOBITS | SHT_REL | SHT_RELA) {
                return Err(unsupported("unqualified allocated section kind"));
            }
            if s.sh_type == SHT_SYMTAB {
                if s.sh_entsize != 24 || !s.sh_size.is_multiple_of(24) {
                    return Err(invalid("symbol table stride"));
                }
                let linked = sections.get(index(u64::from(s.sh_link))?).ok_or_else(|| invalid("symbol string table index"))?;
                if s.sh_link == 0 || linked.sh_type != SHT_STRTAB {
                    return Err(invalid("symbol string table link"));
                }
                if u64::from(s.sh_info) > s.sh_size / 24 {
                    return Err(invalid("symbol local/global boundary"));
                }
            }
            if matches!(s.sh_type, SHT_REL | SHT_RELA) {
                let stride = if s.sh_type == SHT_REL { 16 } else { 24 };
                if s.sh_entsize != stride || !s.sh_size.is_multiple_of(stride) {
                    return Err(invalid("relocation stride"));
                }
                let linked = sections
                    .get(index(u64::from(s.sh_link))?)
                    .ok_or_else(|| invalid("relocation symbol table index"))?;
                if s.sh_link == 0 || linked.sh_type != SHT_SYMTAB {
                    return Err(invalid("relocation symbol table link"));
                }
                if s.sh_info == 0 || index(u64::from(s.sh_info))? >= sections.len() {
                    return Err(invalid("relocation target index"));
                }
            }
            Ok(())
        };
        validate().map_err(|e| e.at(i, None))?;
    }
    let names = &sections[usize::from(h.e_shstrndx)];
    if names.sh_type != SHT_STRTAB {
        return Err(invalid("section names table type"));
    }
    let names = payload(bytes, names)?;
    for (i, s) in sections.iter().enumerate() {
        string(names, s.sh_name).map_err(|e| e.at(i, None))?;
    }
    Ok(sections)
}

fn entry_metadata<'a>(bytes: &'a [u8], headers: &[SectionHeader], placements: &[Option<ImageSection>], names: &'a [u8]) -> Result<Vec<EntryMetadata<'a>>> {
    let mut stack = 0;
    for (i, s) in headers.iter().enumerate() {
        if string(names, s.sh_name)? == ".nv.info" {
            stack = stack.max(cubin::stack_bytes(payload(bytes, s)?).map_err(|code| error(code, "module stack metadata").at(i, None))?);
        }
    }
    let local_bytes = u64::from(stack).checked_add(0x240).ok_or_else(|| invalid("local reservation overflow"))?;
    let mut entries: Vec<EntryMetadata<'a>> = Vec::new();
    for (i, s) in headers.iter().enumerate() {
        let Some(name) = string(names, s.sh_name)?.strip_prefix(".text.") else {
            continue;
        };
        let placement = placements[i].ok_or_else(|| invalid("entry is not mapped PROGBITS").at(i, None))?;
        if name.is_empty() || s.sh_flags != 6 || s.sh_size == 0 || entries.iter().any(|e| e.name == name) {
            return Err(invalid("entry name, flags, extent, or uniqueness").at(i, None));
        }
        let mut entry = EntryMetadata {
            name,
            section_index: i,
            image_offset: placement.offset,
            code_bytes: s.sh_size,
            registers: s.sh_info >> 24,
            local_bytes,
            shared_bytes: 0x400,
            required_threads: None,
            constants: [None; 8],
        };
        for (j, metadata) in headers.iter().enumerate() {
            let section_name = string(names, metadata.sh_name)?;
            if section_name.strip_prefix(".nv.info.") == Some(name) {
                entry.required_threads = cubin::required_threads(payload(bytes, metadata)?).map_err(|code| error(code, "entry thread metadata").at(j, None))?;
            }
            if section_name.strip_prefix(".nv.shared.") == Some(name) {
                if metadata.sh_type != SHT_NOBITS {
                    return Err(unsupported("shared metadata must be NOBITS").at(j, None));
                }
                entry.shared_bytes = cubin::shared_bytes(metadata.sh_size).map_err(|code| error(code, "entry shared metadata").at(j, None))?;
            }
            let Some(p) = placements[j] else {
                continue;
            };
            if let Some(constant) = section_name.strip_prefix(".nv.constant") {
                let (bank, scope) = constant.split_once('.').map_or((constant, None), |(bank, scope)| (bank, Some(scope)));
                if let Ok(bank) = bank.parse::<usize>()
                    && (1..8).contains(&bank)
                    && scope.is_none_or(|scope| scope == name)
                {
                    let size = u32::try_from(metadata.sh_size).map_err(|_| invalid("constant bank size").at(j, None))?;
                    if size > 65536 {
                        return Err(invalid("constant bank exceeds 64 KiB").at(j, None));
                    }
                    entry.constants[bank] = Some(ConstantBinding { offset: p.offset, bytes: size });
                }
            }
        }
        // Use the actual SDK launch constraint helpers before allocation.
        cubin::shared_partition(entry.shared_bytes).map_err(|code| error(code, "shared partition").at(i, None))?;
        if let Some(dimensions) = entry.required_threads {
            let threads = dimensions
                .into_iter()
                .try_fold(1u32, |n, d| n.checked_mul(d))
                .ok_or_else(|| invalid("thread product"))?;
            cubin::validate_ada_registers(u64::from(entry.registers), threads).map_err(|code| error(code, "entry register budget").at(i, None))?;
        }
        entries.push(entry);
    }
    if entries.is_empty() {
        return Err(invalid("module has no entry"));
    }
    Ok(entries)
}
