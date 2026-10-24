use alloc::vec;
use alloc::vec::Vec;

#[derive(Debug)]
pub struct InitialStack {
    pub bytes: Vec<u8>,
    pub pointer: u64,
}

/// Construct the AMD64 process stack. Pointers refer to the final guest mapping.
pub fn build(base: u64, size: usize, args: &[&str], env: &[&str], aux: &[(u64, u64)], random: [u8; 16]) -> Option<InitialStack> {
    let mut bytes = vec![0; size];
    base.checked_add(size as u64)?;
    let mut cursor = size;
    let mut put = |data: &[u8]| -> Option<u64> {
        cursor = cursor.checked_sub(data.len())?;
        bytes.get_mut(cursor..cursor + data.len())?.copy_from_slice(data);
        Some(base + cursor as u64)
    };
    let random_address = put(&random)?;
    let mut argv = Vec::new();
    let mut envp = Vec::new();
    for (strings, pointers) in [(args, &mut argv), (env, &mut envp)] {
        for s in strings {
            if s.as_bytes().contains(&0) {
                return None;
            }
            put(&[0])?;
            pointers.push(put(s.as_bytes())?);
        }
    }
    let mut words = vec![args.len() as u64];
    words.extend(argv.iter());
    words.push(0);
    words.extend(envp);
    words.push(0);
    for &(key, value) in aux {
        if matches!(key, 0 | 25 | 31) {
            return None;
        }
        words.extend([key, value]);
    }
    words.extend([25, random_address, 31, *argv.first()?, 0, 0]);
    cursor = cursor.checked_sub(words.len() * 8)? & !15;
    for (offset, word) in words.into_iter().enumerate() {
        bytes
            .get_mut(cursor + offset * 8..cursor + offset * 8 + 8)?
            .copy_from_slice(&word.to_le_bytes());
    }
    Some(InitialStack {
        pointer: base + cursor as u64,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_has_aligned_pointer_and_guest_addresses() {
        let stack = build(0x10000, 4096, &["/bin/ptxas", "--version"], &["LANG=C"], &[(6, 4096)], [7; 16]).unwrap();
        assert_eq!(stack.pointer % 16, 0);
        let offset = (stack.pointer - 0x10000) as usize;
        let word = |i: usize| u64::from_le_bytes(stack.bytes[offset + i * 8..offset + i * 8 + 8].try_into().unwrap());
        assert_eq!(word(0), 2);
        let argv0 = (word(1) - 0x10000) as usize;
        assert_eq!(&stack.bytes[argv0..argv0 + 11], b"/bin/ptxas\0");
        assert_eq!(word(3), 0);
        assert_eq!(word(5), 0);
    }

    #[test]
    fn rejects_small_stacks_and_embedded_nuls() {
        assert!(build(0, 16, &["ptxas"], &[], &[], [0; 16]).is_none());
        assert!(build(0, 4096, &["bad\0arg"], &[], &[], [0; 16]).is_none());
    }
}
