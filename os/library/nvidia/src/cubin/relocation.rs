//! Checked image layout and absolute CUDA data pointers for a qualified profile.
//!
//! The caller admits only ELF64 little-endian CUDA REL/type-2 zero-addend data
//! relocations. It resolves symbols, validates their full nonempty object extents
//! inside mapped sections, and handles the qualified legacy metadata exception.
//! It also reserves skipped legacy slots against all live relocation ranges.
//! This module does not parse ELF or validate those adapter policies.

use alloc::vec::Vec;

use crate::{Error, Result};

const POINTER_BYTES: usize = 8;
const MIN_SECTION_ALIGNMENT: usize = 128;

/// Return the section's aligned start and exclusive end without allocating.
///
/// Accept an ELF alignment of zero, one, or a power of two. Preserve the native
/// minimum alignment of 128 bytes. Invalid alignment, conversion, or arithmetic
/// returns `Error::InvalidField`.
///
/// The caller first validates the source range and this result. It then extends
/// the image with zero padding up to `start`, appends the section bytes, and uses
/// `end` as the next cursor. A zero-size section can still require padding.
/// Validate every section before allocation to avoid partial image construction.
/// The caller checks allocation-base alignment and architecture size limits.
pub fn checked_image_section(cursor: usize, size: usize, alignment: u64) -> Result<(usize, usize)> {
    if alignment != 0 && !alignment.is_power_of_two() {
        return Err(Error::InvalidField);
    }
    let alignment = usize::try_from(alignment).map_err(|_| Error::InvalidField)?.max(MIN_SECTION_ALIGNMENT);
    let mask = alignment - 1;
    let start = cursor.checked_add(mask).ok_or(Error::InvalidField)? & !mask;
    let end = start.checked_add(size).ok_or(Error::InvalidField)?;
    Ok((start, end))
}

/// One eight-byte little-endian pointer to an object in the same upload image.
///
/// Private fields preserve constructor checks. Image-relative offsets never
/// represent physical addresses or cubin file positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cuda64Patch {
    slot_offset: usize,
    symbol_image_offset: usize,
}

impl Cuda64Patch {
    /// Validate slot alignment, slot arithmetic, and the symbol's u64 conversion.
    /// The plan constructor additionally checks both offsets against the image.
    pub fn new(slot_offset: usize, symbol_image_offset: usize) -> Result<Self> {
        if !slot_offset.is_multiple_of(POINTER_BYTES) {
            return Err(Error::InvalidField);
        }
        slot_offset.checked_add(POINTER_BYTES).ok_or(Error::InvalidField)?;
        u64::try_from(symbol_image_offset).map_err(|_| Error::InvalidField)?;
        Ok(Self {
            slot_offset,
            symbol_image_offset,
        })
    }

    pub const fn slot_offset(&self) -> usize {
        self.slot_offset
    }

    pub const fn symbol_image_offset(&self) -> usize {
        self.symbol_image_offset
    }
}

/// An immutable plan for one exact image length, sorted by destination offset.
#[derive(Debug)]
pub struct Cuda64Plan {
    image_bytes: usize,
    patches: Vec<Cuda64Patch>,
}

impl Cuda64Plan {
    /// Validate complete slots, interior symbol starts, and disjoint patch ranges.
    ///
    /// Reject duplicates and partial overlaps. The caller must validate each
    /// symbol's full extent in its source section before constructing a patch.
    /// Empty images are valid only with an empty patch list.
    pub fn new(image_bytes: usize, mut patches: Vec<Cuda64Patch>) -> Result<Self> {
        u64::try_from(image_bytes).map_err(|_| Error::InvalidField)?;
        patches.sort_unstable_by_key(Cuda64Patch::slot_offset);
        let mut previous_end = 0;
        for patch in &patches {
            let end = patch.slot_offset.checked_add(POINTER_BYTES).ok_or(Error::InvalidField)?;
            let symbol_end = patch.symbol_image_offset.checked_add(1).ok_or(Error::InvalidField)?;
            if patch.slot_offset < previous_end || end > image_bytes || symbol_end > image_bytes {
                return Err(Error::InvalidField);
            }
            previous_end = end;
        }
        Ok(Self { image_bytes, patches })
    }

    pub const fn image_bytes(&self) -> usize {
        self.image_bytes
    }

    pub fn patches(&self) -> &[Cuda64Patch] {
        &self.patches
    }

    /// Preflight the complete image and every patch, then write all pointers.
    ///
    /// Require exactly the planned length. A shorter image returns
    /// `Error::Truncated`, and a longer image returns `Error::InvalidField`.
    /// Reject nonzero original slots as `Error::Unsupported`, including reuse of
    /// a relocated image. Use a pristine image for each new GPU virtual base.
    /// Reject a null resolved pointer so every successful patch becomes nonzero.
    /// Address overflow and null pointers return `Error::InvalidField`.
    ///
    /// Only checked u64 virtual-address arithmetic belongs here. The kernel
    /// validates architecture address limits, base alignment, and mapping extent.
    /// No physical address enters this API. An empty plan preserves all bytes,
    /// but still validates the length and exclusive virtual end of the image.
    ///
    /// Every returned error leaves the image unchanged. This guarantee covers
    /// validation and byte patching only, not allocation, upload, or hardware.
    pub fn apply(&self, image: &mut [u8], gpu_virtual_base: u64) -> Result<()> {
        if image.len() < self.image_bytes {
            return Err(Error::Truncated);
        }
        if image.len() != self.image_bytes {
            return Err(Error::InvalidField);
        }
        let image_bytes = u64::try_from(self.image_bytes).map_err(|_| Error::InvalidField)?;
        gpu_virtual_base.checked_add(image_bytes).ok_or(Error::InvalidField)?;
        for patch in &self.patches {
            let symbol_offset = u64::try_from(patch.symbol_image_offset).map_err(|_| Error::InvalidField)?;
            gpu_virtual_base
                .checked_add(symbol_offset)
                .filter(|&value| value != 0)
                .ok_or(Error::InvalidField)?;
            if image[patch.slot_offset..patch.slot_offset + POINTER_BYTES].iter().any(|&byte| byte != 0) {
                return Err(Error::Unsupported);
            }
        }
        // Private, validated ranges and the first pass make these writes infallible.
        for patch in &self.patches {
            let value = gpu_virtual_base + patch.symbol_image_offset as u64;
            image[patch.slot_offset..patch.slot_offset + POINTER_BYTES].copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}
