use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Truncated,
    Unsupported,
    Invalid,
    Overflow,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub address: u64,
    pub offset: usize,
    pub file_size: usize,
    pub memory_size: u64,
    pub flags: u32,
}

#[derive(Debug)]
pub struct Elf<'a> {
    pub entry: u64,
    pub phdr: u64,
    pub phnum: u16,
    pub interpreter: Option<&'a str>,
    pub segments: Vec<Segment>,
    pub end: u64,
}

fn word<const N: usize>(data: &[u8], at: usize) -> Result<[u8; N], Error> {
    data.get(at..at.checked_add(N).ok_or(Error::Overflow)?)
        .ok_or(Error::Truncated)?
        .try_into()
        .map_err(|_| Error::Truncated)
}

impl<'a> Elf<'a> {
    pub fn parse(data: &'a [u8], bias: u64) -> Result<Self, Error> {
        if data.get(..7) != Some(b"\x7fELF\x02\x01\x01") {
            return Err(Error::Unsupported);
        }
        let kind = u16::from_le_bytes(word(data, 16)?);
        if !matches!(kind, 2 | 3) || u16::from_le_bytes(word(data, 18)?) != 62 {
            return Err(Error::Unsupported);
        }
        if kind == 2 && bias != 0 {
            return Err(Error::Unsupported);
        }
        let entry = u64::from_le_bytes(word(data, 24)?).checked_add(bias).ok_or(Error::Overflow)?;
        let phoff = usize::try_from(u64::from_le_bytes(word(data, 32)?)).map_err(|_| Error::Overflow)?;
        let phentsize = u16::from_le_bytes(word(data, 54)?);
        let phnum = u16::from_le_bytes(word(data, 56)?);
        if phentsize != 56 || phnum == 0 || phnum > 1024 {
            return Err(Error::Invalid);
        }
        let table_end = phoff.checked_add(phnum as usize * 56).ok_or(Error::Overflow)?;
        if table_end > data.len() {
            return Err(Error::Truncated);
        }
        let mut result = Self {
            entry,
            phdr: 0,
            phnum,
            interpreter: None,
            segments: Vec::new(),
            end: 0,
        };
        for i in 0..phnum as usize {
            let p = phoff + i * 56;
            let typ = u32::from_le_bytes(word(data, p)?);
            let flags = u32::from_le_bytes(word(data, p + 4)?);
            let offset = usize::try_from(u64::from_le_bytes(word(data, p + 8)?)).map_err(|_| Error::Overflow)?;
            let address = u64::from_le_bytes(word(data, p + 16)?).checked_add(bias).ok_or(Error::Overflow)?;
            let file_size = usize::try_from(u64::from_le_bytes(word(data, p + 32)?)).map_err(|_| Error::Overflow)?;
            let memory_size = u64::from_le_bytes(word(data, p + 40)?);
            let file_end = offset.checked_add(file_size).ok_or(Error::Overflow)?;
            if file_end > data.len() {
                return Err(Error::Truncated);
            }
            match typ {
                1 => {
                    if file_size as u64 > memory_size || offset as u64 % 4096 != address % 4096 {
                        return Err(Error::Invalid);
                    }
                    let end = address.checked_add(memory_size).ok_or(Error::Overflow)?;
                    if end >= 0x8000_0000_0000 {
                        return Err(Error::Invalid);
                    }
                    if offset <= phoff && file_end >= table_end {
                        result.phdr = address + (phoff - offset) as u64;
                    }
                    result.end = result.end.max(end);
                    if memory_size > 0 {
                        result.segments.push(Segment {
                            address,
                            offset,
                            file_size,
                            memory_size,
                            flags,
                        });
                    }
                }
                3 => {
                    let bytes = data.get(offset..file_end).ok_or(Error::Truncated)?;
                    let path = bytes.strip_suffix(&[0]).ok_or(Error::Invalid)?;
                    if path.contains(&0) || result.interpreter.is_some() {
                        return Err(Error::Invalid);
                    }
                    result.interpreter = Some(core::str::from_utf8(path).map_err(|_| Error::Invalid)?);
                }
                _ => {}
            }
        }
        if result.segments.is_empty()
            || result.phdr == 0
            || !result
                .segments
                .iter()
                .any(|s| s.flags & 1 != 0 && entry >= s.address && entry < s.address + s.memory_size)
        {
            return Err(Error::Invalid);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn image() -> Vec<u8> {
        let mut data = vec![0; 4096];
        data[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        data[16..18].copy_from_slice(&2u16.to_le_bytes());
        data[18..20].copy_from_slice(&62u16.to_le_bytes());
        data[24..32].copy_from_slice(&0x400100u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());
        data[54..56].copy_from_slice(&56u16.to_le_bytes());
        data[56..58].copy_from_slice(&1u16.to_le_bytes());
        data[64..68].copy_from_slice(&1u32.to_le_bytes());
        data[68..72].copy_from_slice(&5u32.to_le_bytes());
        data[80..88].copy_from_slice(&0x400000u64.to_le_bytes());
        data[96..104].copy_from_slice(&4096u64.to_le_bytes());
        data[104..112].copy_from_slice(&8192u64.to_le_bytes());
        data
    }

    #[test]
    fn derives_headers_and_zero_fill_extent() {
        let data = image();
        let elf = Elf::parse(&data, 0).unwrap();
        assert_eq!(elf.phdr, 0x400040);
        assert_eq!(elf.end, 0x402000);
        assert_eq!(elf.segments[0].file_size, 4096);
    }

    #[test]
    fn rejects_truncated_or_noncongruent_segments() {
        let mut data = image();
        assert_eq!(Elf::parse(&data[..100], 0).unwrap_err(), Error::Truncated);
        data[80..88].copy_from_slice(&0x400001u64.to_le_bytes());
        assert_eq!(Elf::parse(&data, 0).unwrap_err(), Error::Invalid);
    }

    #[test]
    fn rejects_relocating_fixed_executables() {
        assert_eq!(Elf::parse(&image(), 0x10000000000).unwrap_err(), Error::Unsupported);
    }
}
