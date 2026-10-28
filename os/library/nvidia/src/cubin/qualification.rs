//! Checked receipts for native module admission. Callers verify asset digests
//! and producer identity separately before admitting a parsed receipt.

use alloc::vec::Vec;

use crate::{Error, Result};

pub const COMPILER_VERSION: &str = "13.4.92";
pub const PROFILE_LABEL: &str = "sm89-direct-constant-v1";
pub const MODULE_MAGIC: &[u8; 8] = b"D3QMOD01";
pub const MODULE_RECEIPT_BYTES: usize = 192;
pub const SCHEMA_VERSION: u32 = 1;
pub const PROFILE_ID: u32 = 1;
pub const SM: u32 = 89;
pub const OPT_LEVEL: u32 = 3;
pub const RUNTIME_MAGIC: &[u8; 8] = b"D3QRT001";
pub const MAX_RUNTIME_RECORDS: usize = 128;
pub const MAX_RUNTIME_PATH_BYTES: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    TileIr,
    Ptx,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleQualification {
    input_kind: InputKind,
    source_sha256: [u8; 32],
    cubin_sha256: [u8; 32],
    sass_sha256: [u8; 32],
    compiler_sha256: [u8; 32],
    runtime_sha256: [u8; 32],
}

impl ModuleQualification {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != MODULE_RECEIPT_BYTES {
            return Err(if bytes.len() < MODULE_RECEIPT_BYTES {
                Error::Truncated
            } else {
                Error::InvalidField
            });
        }
        let mut reader = Reader(bytes);
        if reader.take(8)? != MODULE_MAGIC
            || reader.u32()? != SCHEMA_VERSION
            || reader.u32()? != PROFILE_ID
            || reader.u32()? != SM
            || reader.u32()? != OPT_LEVEL
        {
            return Err(Error::InvalidField);
        }
        let input_kind = match reader.u32()? {
            1 => InputKind::TileIr,
            2 => InputKind::Ptx,
            _ => return Err(Error::InvalidField),
        };
        if reader.u32()? != 0 {
            return Err(Error::InvalidField);
        }
        Ok(Self {
            input_kind,
            source_sha256: reader.digest()?,
            cubin_sha256: reader.digest()?,
            sass_sha256: reader.digest()?,
            compiler_sha256: reader.digest()?,
            runtime_sha256: reader.digest()?,
        })
    }

    pub const fn input_kind(&self) -> InputKind {
        self.input_kind
    }

    pub const fn source_sha256(&self) -> &[u8; 32] {
        &self.source_sha256
    }

    pub const fn cubin_sha256(&self) -> &[u8; 32] {
        &self.cubin_sha256
    }

    pub const fn sass_sha256(&self) -> &[u8; 32] {
        &self.sass_sha256
    }

    pub const fn compiler_sha256(&self) -> &[u8; 32] {
        &self.compiler_sha256
    }

    pub const fn runtime_sha256(&self) -> &[u8; 32] {
        &self.runtime_sha256
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAsset<'a> {
    path: &'a str,
    bytes: u64,
    sha256: [u8; 32],
}

impl<'a> RuntimeAsset<'a> {
    pub const fn path(&self) -> &'a str {
        self.path
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }
}

#[derive(Debug)]
pub struct RuntimeInventory<'a> {
    canonical_bytes: &'a [u8],
    records: Vec<RuntimeAsset<'a>>,
}

impl<'a> RuntimeInventory<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let mut reader = Reader(bytes);
        if reader.take(8)? != RUNTIME_MAGIC {
            return Err(Error::InvalidField);
        }
        let count = usize::try_from(reader.u32()?).map_err(|_| Error::InvalidField)?;
        let canonical_bytes = reader.0;
        if count > MAX_RUNTIME_RECORDS {
            return Err(Error::InvalidField);
        }
        let mut records: Vec<RuntimeAsset<'a>> = Vec::with_capacity(count);
        for _ in 0..count {
            let length = usize::try_from(reader.u32()?).map_err(|_| Error::InvalidField)?;
            if !(1..=MAX_RUNTIME_PATH_BYTES).contains(&length) {
                return Err(Error::InvalidField);
            }
            let path = core::str::from_utf8(reader.take(length)?).map_err(|_| Error::InvalidField)?;
            validate_path(path)?;
            if records.last().is_some_and(|previous| previous.path >= path) {
                return Err(Error::InvalidField);
            }
            let bytes = u64::from_le_bytes(reader.array()?);
            if bytes == 0 {
                return Err(Error::InvalidField);
            }
            records.push(RuntimeAsset {
                path,
                bytes,
                sha256: reader.digest()?,
            });
        }
        if !reader.0.is_empty() {
            return Err(Error::InvalidField);
        }
        Ok(Self { canonical_bytes, records })
    }

    pub fn records(&self) -> &[RuntimeAsset<'a>] {
        &self.records
    }

    /// The exact encoded records, excluding the magic and record count.
    /// Qualification receipts hash this sequence on both host and guest.
    pub const fn canonical_bytes(&self) -> &'a [u8] {
        self.canonical_bytes
    }
}

fn validate_path(path: &str) -> Result<()> {
    if path.chars().any(|c| c.is_control() || c == '\\') || path.split('/').any(|part| matches!(part, "" | "." | "..")) {
        return Err(Error::InvalidField);
    }
    let allowed = path.starts_with("usr/cuda/")
        || path.starts_with("lib64/")
        || path.strip_prefix("usr/share/nvidia/").is_some_and(|name| {
            !name.contains('/')
                && ([".tilebc", ".ptx"]
                    .iter()
                    .any(|suffix| name.strip_suffix(suffix).is_some_and(|family| !family.is_empty()))
                    || matches!(name, "vector-add.cubin" | "tensor-matmul.cubin" | "inference.cubin"))
        });
    if !allowed {
        return Err(Error::InvalidField);
    }
    Ok(())
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let (value, rest) = self.0.split_at_checked(length).ok_or(Error::Truncated)?;
        self.0 = rest;
        Ok(value)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.take(N)?.try_into().map_err(|_| Error::Truncated)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn digest(&mut self) -> Result<[u8; 32]> {
        let digest = self.array()?;
        if digest == [0; 32] {
            return Err(Error::InvalidField);
        }
        Ok(digest)
    }
}
