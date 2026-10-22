//! Ordered column/row command bins for indexed GPU rendering.
use crate::{Error, Result};
use alloc::vec::Vec;

/// Build absolute CSR offsets followed by command IDs. The first `width` lists
/// contain columns; the next `height` contain rows. Each list preserves draw
/// order, so a pixel can merge its two lists without scanning unrelated draws.
pub fn command_bins(commands: &[u32], width: usize, height: usize) -> Result<Vec<u32>> {
    if width == 0 || height == 0 || !commands.len().is_multiple_of(12) {
        return Err(Error::InvalidParameter);
    }
    let lists = width.checked_add(height).ok_or(Error::SizeOverflow)?;
    let header = lists.checked_add(1).ok_or(Error::SizeOverflow)?;
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(header).map_err(|_| Error::AllocationFailed)?;
    offsets.resize(header, 0usize);
    let visit = |command: &[u32], callback: &mut dyn FnMut(usize)| -> Result<()> {
        let x = command[1] as usize;
        let y = command[2] as usize;
        if x >= width || y >= height || command[3] < if command[0] == 0 { command[2] } else { command[1] } {
            return Ok(());
        }
        match command[0] {
            0 => {
                callback(x);
                if command[11] & 1 != 0 && x + 1 < width {
                    callback(x + 1);
                }
            }
            1 => callback(width + y),
            _ => return Err(Error::InvalidParameter),
        }
        Ok(())
    };
    for command in commands.chunks_exact(12) {
        visit(command, &mut |list| offsets[list + 1] += 1)?;
    }
    offsets[0] = header;
    for i in 1..=lists {
        offsets[i] = offsets[i].checked_add(offsets[i - 1]).ok_or(Error::SizeOverflow)?;
    }
    let length = offsets[lists];
    u32::try_from(length).map_err(|_| Error::SizeOverflow)?;
    u32::try_from(commands.len() / 12).map_err(|_| Error::SizeOverflow)?;
    let mut result = Vec::new();
    result.try_reserve_exact(length).map_err(|_| Error::AllocationFailed)?;
    result.extend(offsets.iter().map(|&offset| offset as u32));
    result.resize(length, 0);
    for (index, command) in commands.chunks_exact(12).enumerate() {
        visit(command, &mut |list| {
            result[offsets[list]] = index as u32;
            offsets[list] += 1;
        })?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_preserve_order_and_clip_double_columns() {
        let mut commands = [[0u32; 12]; 5];
        commands[0][..4].copy_from_slice(&[0, 2, 0, 4]);
        commands[0][11] = 1;
        commands[1][..4].copy_from_slice(&[1, 0, 1, 3]);
        commands[2][..4].copy_from_slice(&[0, 3, 0, 4]);
        commands[2][11] = 1;
        commands[3][..4].copy_from_slice(&[0, 1, 3, 2]); // Empty draw.
        commands[4][..4].copy_from_slice(&[0, 9, 0, 4]); // Outside framebuffer.
        let bins = command_bins(commands.as_flattened(), 4, 5).unwrap();
        let list = |i: usize| &bins[bins[i] as usize..bins[i + 1] as usize];
        assert_eq!(list(2), &[0]);
        assert_eq!(list(3), &[0, 2]);
        assert_eq!(list(5), &[1]);
        for i in [0, 1, 4, 6, 7, 8] {
            assert!(list(i).is_empty());
        }
    }
}
