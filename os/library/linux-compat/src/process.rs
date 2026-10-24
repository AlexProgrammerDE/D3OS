use crate::{Host, Result, elf::Elf, linux_errno, page_up, stack};
use alloc::{collections::BTreeMap, string::String, vec::Vec};
use naming::shared_types::{OpenOptions, SeekOrigin};
use syscall::return_vals::Errno;

#[derive(Debug, Clone, Copy)]
pub struct StartContext {
    pub instruction: u64,
    pub stack: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyscallOutcome {
    Return(isize),
    Exit(u8),
}

#[repr(C)]
pub struct SyscallFrame {
    pub number: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rsp: u64,
}
const _: () = assert!(core::mem::size_of::<SyscallFrame>() == 80);

impl SyscallFrame {
    pub fn arguments(&self) -> [u64; 6] {
        [self.rdi, self.rsi, self.rdx, self.r10, self.r8, self.r9]
    }
}

const PAGE: u64 = 4096;
const HIGH: u64 = 0x100_0000_0000;
const STACK: u64 = 0x300_0000_0000;
const STACK_SIZE: usize = 2 * 1024 * 1024;
const MMAP: u64 = 0x400_0000_0000;
const HEAP: u64 = 0x200_0000_0000;

pub struct LinuxProcess<H: Host> {
    host: H,
    id: u32,
    cwd: String,
    pages: BTreeMap<u64, u32>,
    files: BTreeMap<u64, File>,
    next_map: u64,
    brk: u64,
    signal_mask: u64,
    actions: [[u8; 32]; 64],
    clear_tid: u64,
    robust_list: u64,
}

struct File {
    handle: usize,
    path: String,
    flags: u64,
}

impl<H: Host> LinuxProcess<H> {
    pub fn new(host: H, id: u32, cwd: &str) -> Self {
        Self {
            host,
            id,
            cwd: cwd.into(),
            pages: BTreeMap::new(),
            files: BTreeMap::new(),
            next_map: MMAP,
            brk: HEAP,
            signal_mask: 0,
            actions: [[0; 32]; 64],
            clear_tid: 0,
            robust_list: 0,
        }
    }

    pub fn host(&self) -> &H {
        &self.host
    }

    /// Execute a Linux syscall in the process's active address space.
    ///
    /// # Safety
    /// The caller must activate this process's mappings, provide a valid kernel
    /// stack, and prevent concurrent access to the process's user memory.
    pub unsafe fn dispatch(&mut self, number: u64, args: [u64; 6]) -> SyscallOutcome {
        if matches!(number, 60 | 231) {
            if self.clear_tid != 0 {
                let _ = self.output(self.clear_tid, &0u32.to_le_bytes());
            }
            self.close_files();
            SyscallOutcome::Exit(args[0] as u8)
        } else {
            SyscallOutcome::Return(self.call(number, args).unwrap_or_else(|e| e))
        }
    }

    pub fn start(&mut self, executable: &[u8], interpreter: Option<&[u8]>, args: &[&str]) -> Result<StartContext> {
        let elf = Elf::parse(executable, 0).map_err(|_| -8isize)?;
        let (entry, phdr, phnum) = self.image(executable, 0)?;
        let (instruction, interpreter_base) = if elf.interpreter.is_some() {
            let (start, _, _) = self.image(interpreter.ok_or(-2isize)?, HIGH)?;
            (start, HIGH)
        } else {
            (entry, 0)
        };
        self.map(STACK, STACK_SIZE as u64, 3, false)?;
        let mut random = [0u8; 16];
        random[..8].copy_from_slice(&self.host.random_word()?.to_le_bytes());
        random[8..].copy_from_slice(&self.host.random_word()?.to_le_bytes());
        let stack = stack::build(
            STACK,
            STACK_SIZE,
            args,
            &["LANG=C", "LC_ALL=C", "LD_LIBRARY_PATH=/lib64"],
            &[
                (3, phdr),
                (4, 56),
                (5, phnum as u64),
                (6, PAGE),
                (7, interpreter_base),
                (8, 0),
                (9, entry),
                (11, 0),
                (12, 0),
                (13, 0),
                (14, 0),
                (16, 0),
                (17, 100),
                (23, 0),
                (26, 0),
            ],
            random,
        )
        .ok_or(-7isize)?;
        self.host.copy_into(STACK, &stack.bytes)?;
        self.protect(STACK, STACK_SIZE as u64, 3)?;
        Ok(StartContext {
            instruction,
            stack: stack.pointer,
        })
    }

    fn close_files(&mut self) {
        for file in self.files.values() {
            let _ = self.host.close(file.handle);
        }
        self.files.clear();
    }

    fn check(&self, address: u64, length: usize, write: bool) -> Result<()> {
        if length == 0 {
            return Ok(());
        }
        let end = address.checked_add(length as u64).ok_or(-14isize)?;
        if end > 0x8000_0000_0000 {
            return Err(-14);
        }
        for page in (address & !(PAGE - 1)..end).step_by(PAGE as usize) {
            let protection = self.pages.get(&page).ok_or(-14isize)?;
            if protection & if write { 2 } else { 1 } == 0 {
                return Err(-14);
            }
        }
        Ok(())
    }

    fn bytes(&self, address: u64, length: usize) -> Result<&[u8]> {
        self.check(address, length, false)?;
        if length == 0 {
            return Ok(&[]);
        }
        Ok(unsafe { core::slice::from_raw_parts(address as *const u8, length) })
    }

    fn output(&self, address: u64, bytes: &[u8]) -> Result<()> {
        self.check(address, bytes.len(), true)?;
        if !bytes.is_empty() {
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len());
            }
        }
        Ok(())
    }

    fn path(&self, address: u64) -> Result<String> {
        let mut bytes = Vec::new();
        for i in 0..4096 {
            let b = *self.bytes(address.checked_add(i).ok_or(-14isize)?, 1)?.first().unwrap();
            if b == 0 {
                let s = core::str::from_utf8(&bytes).map_err(|_| -2isize)?;
                return normalize(s, &self.cwd);
            }
            bytes.push(b);
        }
        Err(-36)
    }

    fn map(&mut self, address: u64, length: u64, protection: u32, replace: bool) -> Result<()> {
        let end = address.checked_add(length).and_then(page_up).ok_or(-12isize)?;
        if !address.is_multiple_of(PAGE)
            || length == 0
            || protection & !7 != 0
            || !(address >= HIGH && end < 0x8000_0000_0000 || address >= 0x400000 && end <= 0x4000000)
        {
            return Err(-22);
        }
        for p in (address..end).step_by(PAGE as usize) {
            if self.pages.contains_key(&p) {
                if !replace {
                    return Err(-17);
                }
                self.host.unmap_page(p)?;
                self.pages.remove(&p);
            }
            self.host.allocate_page(p)?;
            self.pages.insert(p, protection);
        }
        Ok(())
    }

    fn protect(&self, address: u64, length: u64, protection: u32) -> Result<()> {
        let end = address.checked_add(length).and_then(page_up).ok_or(-22isize)?;
        for p in (address..end).step_by(PAGE as usize) {
            if !self.pages.contains_key(&p) {
                return Err(-12);
            }
            self.host.protect_page(p, protection)?;
        }
        Ok(())
    }

    fn image(&mut self, data: &[u8], bias: u64) -> Result<(u64, u64, u16)> {
        let elf = Elf::parse(data, bias).map_err(|_| -8isize)?;
        for segment in &elf.segments {
            let start = segment.address & !(PAGE - 1);
            let end = page_up(segment.address + segment.memory_size).ok_or(-8isize)?;
            for p in (start..end).step_by(PAGE as usize) {
                if !self.pages.contains_key(&p) {
                    self.map(p, PAGE, 7, false)?;
                }
            }
            self.host
                .copy_into(segment.address, &data[segment.offset..segment.offset + segment.file_size])?;
        }
        for segment in &elf.segments {
            let start = segment.address & !(PAGE - 1);
            let end = page_up(segment.address + segment.memory_size).ok_or(-8isize)?;
            let protection = (segment.flags & 4) >> 2 | segment.flags & 2 | (segment.flags & 1) << 2;
            for p in (start..end).step_by(PAGE as usize) {
                self.pages.insert(p, protection);
            }
            self.protect(start, end - start, protection)?;
        }
        Ok((elf.entry, elf.phdr, elf.phnum))
    }

    fn file(&self, fd: u64) -> Result<&File> {
        self.files.get(&fd).ok_or(-9)
    }

    fn open(&mut self, path: String, flags: u64) -> Result<isize> {
        let allowed = 3 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x8000 | 0x10000 | 0x80000;
        if flags & !allowed != 0 || flags & 3 == 3 {
            return Err(-22);
        }
        if flags & 0x400 != 0 {
            return Err(-95);
        }
        let exists = self.host.metadata(&path).is_ok();
        if flags & 0x40 != 0 && flags & 0x80 != 0 && exists {
            return Err(-17);
        }
        let mut options = match flags & 3 {
            0 => OpenOptions::READONLY,
            1 => OpenOptions::WRITEONLY,
            _ => OpenOptions::READWRITE,
        };
        if flags & 0x40 != 0 && !exists {
            options |= OpenOptions::CREATE;
        }
        if flags & 0x10000 != 0 {
            options |= OpenOptions::DIRECTORY;
        }
        if flags & 0x200 != 0 && exists {
            self.host.truncate(&path).map_err(errno)?;
        }
        let fd = (3..4096).find(|fd| !self.files.contains_key(fd)).ok_or(-24isize)?;
        let handle = self.host.open(&path, options).map_err(errno)?;
        self.files.insert(fd, File { handle, path, flags });
        Ok(fd as isize)
    }

    fn stat(&self, path: Option<&str>, out: u64) -> Result<isize> {
        let mut bytes = [0u8; 144];
        let (mode, size) = if let Some(path) = path {
            let metadata = self.host.metadata(path).map_err(errno)?;
            if metadata.directory {
                (0o40755u32, metadata.size as u64)
            } else {
                (0o100644u32, metadata.size as u64)
            }
        } else {
            (0o20666u32, 0u64)
        };
        let inode = path
            .map(|p| p.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3)))
            .unwrap_or(1);
        bytes[8..16].copy_from_slice(&inode.to_le_bytes());
        bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
        bytes[24..28].copy_from_slice(&mode.to_le_bytes());
        bytes[48..56].copy_from_slice(&size.to_le_bytes());
        bytes[56..64].copy_from_slice(&4096u64.to_le_bytes());
        bytes[64..72].copy_from_slice(&size.div_ceil(512).to_le_bytes());
        self.output(out, &bytes)?;
        Ok(0)
    }

    fn mmap(&mut self, a: [u64; 6]) -> Result<isize> {
        let [requested, length, protection, flags, fd, offset] = a;
        if flags & !0x1833 != 0 || flags & 3 != 2 || offset % PAGE != 0 || length == 0 || protection & !7 != 0 {
            return Err(-22);
        }
        let size = page_up(length).ok_or(-12isize)?;
        let fixed = flags & 0x10 != 0;
        let address = if fixed { requested } else { self.next_map };
        if !fixed {
            self.next_map = address.checked_add(size).and_then(|n| n.checked_add(PAGE)).ok_or(-12isize)?;
        }
        let file = if flags & 0x20 == 0 { Some(self.file(fd)?.handle) } else { None };
        self.map(address, size, protection as u32, fixed)?;
        if let Some(handle) = file {
            let old = self.host.seek(handle, 0, SeekOrigin::Current).map_err(errno)?;
            self.host
                .seek(handle, offset.try_into().map_err(|_| -22isize)?, SeekOrigin::Start)
                .map_err(errno)?;
            let mut buffer = [0u8; 4096];
            let mut done = 0;
            while done < length {
                let count = (length - done).min(PAGE) as usize;
                let read = self.host.read(handle, &mut buffer[..count]).map_err(errno)?;
                if read == 0 {
                    break;
                }
                self.host.copy_into(address + done, &buffer[..read])?;
                done += read as u64;
            }
            self.host.seek(handle, old as isize, SeekOrigin::Start).map_err(errno)?;
        }
        self.protect(address, size, protection as u32)?;
        Ok(address as isize)
    }

    fn call(&mut self, number: u64, a: [u64; 6]) -> Result<isize> {
        let [x, y, z, w, _, _] = a;
        match number {
            0 | 17 => {
                if z == 0 {
                    self.file(x)?;
                    return Ok(0);
                }
                self.check(y, z.try_into().map_err(|_| -22isize)?, true)?;
                let file = self.file(x)?;
                if file.flags & 3 == 1 {
                    return Err(-9);
                }
                let handle = file.handle;
                let old = if number == 17 {
                    let old = self.host.seek(handle, 0, SeekOrigin::Current).map_err(errno)?;
                    self.host.seek(handle, w.try_into().map_err(|_| -22isize)?, SeekOrigin::Start).map_err(errno)?;
                    Some(old)
                } else {
                    None
                };
                let result = self
                    .host
                    .read(handle, unsafe { core::slice::from_raw_parts_mut(y as *mut u8, z as usize) })
                    .map_err(errno);
                if let Some(old) = old {
                    self.host.seek(handle, old as isize, SeekOrigin::Start).map_err(errno)?;
                }
                result.map(|n| n as isize)
            }
            1 => {
                let bytes = self.bytes(y, z.try_into().map_err(|_| -22isize)?)?;
                if matches!(x, 1 | 2) {
                    self.host.console_write(x, bytes).map(|n| n as isize)
                } else {
                    let file = self.file(x)?;
                    if file.flags & 3 == 0 {
                        return Err(-9);
                    }
                    self.host.write(file.handle, bytes).map(|n| n as isize).map_err(errno)
                }
            }
            20 => {
                if z > 1024 {
                    return Err(-22);
                }
                let entries = self.bytes(y, (z as usize).checked_mul(16).ok_or(-22isize)?)?.to_vec();
                let mut total = 0;
                for entry in entries.chunks_exact(16) {
                    let address = u64::from_le_bytes(entry[..8].try_into().unwrap());
                    let length = u64::from_le_bytes(entry[8..].try_into().unwrap());
                    match self.call(1, [x, address, length, 0, 0, 0]) {
                        Ok(written) => {
                            total += written;
                            if written as u64 != length {
                                break;
                            }
                        }
                        Err(error) if total == 0 => return Err(error),
                        Err(_) => break,
                    }
                }
                Ok(total)
            }
            2 => self.open(self.path(x)?, y),
            3 => {
                let file = self.files.remove(&x).ok_or(-9isize)?;
                self.host.close(file.handle).map_err(errno)?;
                Ok(0)
            }
            4 => self.stat(Some(&self.path(x)?), y),
            5 => self.stat(if x <= 2 { None } else { Some(&self.file(x)?.path) }, y),
            8 => {
                let file = self.file(x)?;
                let base = match z {
                    0 => 0,
                    1 => self.host.seek(file.handle, 0, SeekOrigin::Current).map_err(errno)? as i128,
                    2 => self.host.metadata(&file.path).map_err(errno)?.size as i128,
                    _ => return Err(-22),
                };
                let position = base + (y as i64) as i128;
                if position < 0 {
                    return Err(-22);
                }
                if position > i64::MAX as i128 {
                    return Err(-75);
                }
                self.host
                    .seek(file.handle, position as isize, SeekOrigin::Start)
                    .map(|n| n as isize)
                    .map_err(errno)
            }
            9 => self.mmap(a),
            10 => {
                if x % PAGE != 0 || z & !7 != 0 {
                    return Err(-22);
                }
                self.protect(x, y, z as u32)?;
                let end = page_up(x.checked_add(y).ok_or(-22isize)?).ok_or(-22isize)?;
                for p in (x..end).step_by(PAGE as usize) {
                    self.pages.insert(p, z as u32);
                }
                Ok(0)
            }
            11 => {
                if x % PAGE != 0 || y == 0 {
                    return Err(-22);
                }
                let end = page_up(x.checked_add(y).ok_or(-22isize)?).ok_or(-22isize)?;
                for p in (x..end).step_by(PAGE as usize) {
                    if self.pages.remove(&p).is_some() {
                        self.host.unmap_page(p)?;
                    }
                }
                Ok(0)
            }
            12 => {
                if x == 0 {
                    return Ok(self.brk as isize);
                }
                if !(HEAP..HEAP + 512 * 1024 * 1024).contains(&x) {
                    return Ok(self.brk as isize);
                }
                let old_end = page_up(self.brk).ok_or(-12isize)?;
                let end = page_up(x).ok_or(-12isize)?;
                if end > old_end {
                    self.map(old_end, end - old_end, 3, false)?;
                }
                if end < old_end {
                    for p in (end..old_end).step_by(PAGE as usize) {
                        if self.pages.remove(&p).is_some() {
                            self.host.unmap_page(p)?;
                        }
                    }
                }
                self.brk = x;
                Ok(x as isize)
            }
            13 => {
                if x == 0 || x > 64 || x == 9 || x == 19 || w != 8 {
                    return Err(-22);
                }
                let old = self.actions[(x - 1) as usize];
                let new = if y != 0 { Some(self.bytes(y, 32)?.try_into().unwrap()) } else { None };
                if z != 0 {
                    self.output(z, &old)?;
                }
                if let Some(new) = new {
                    self.actions[(x - 1) as usize] = new;
                }
                Ok(0)
            }
            14 => {
                if w != 8 {
                    return Err(-22);
                }
                let old = self.signal_mask;
                let new = if y != 0 {
                    Some(u64::from_le_bytes(self.bytes(y, 8)?.try_into().unwrap()))
                } else {
                    None
                };
                if z != 0 {
                    self.output(z, &old.to_le_bytes())?;
                }
                if let Some(mask) = new {
                    self.signal_mask = match x {
                        0 => old | mask,
                        1 => old & !mask,
                        2 => mask,
                        _ => return Err(-22),
                    } & !(1 << 8 | 1 << 18);
                }
                Ok(0)
            }
            16 => {
                if x > 2 {
                    self.file(x)?;
                }
                Err(-25)
            }
            21 => {
                if y & !7 != 0 {
                    return Err(-22);
                }
                self.host.metadata(&self.path(x)?).map_err(errno)?;
                Ok(0)
            }
            24 => {
                self.host.yield_now();
                Ok(0)
            }
            39 | 186 => Ok(self.id as isize),
            96 => {
                if x != 0 {
                    let ms = self.host.realtime_ms()?;
                    self.output(x, &(ms / 1000).to_le_bytes())?;
                    self.output(x + 8, &((ms % 1000) * 1000).to_le_bytes())?;
                }
                if y != 0 {
                    self.output(y, &[0; 8])?;
                }
                Ok(0)
            }
            102 | 104 | 107 | 108 => Ok(0),
            158 => match x {
                0x1002 => {
                    if y >= 0x8000_0000_0000 {
                        return Err(-1);
                    }
                    self.host.set_thread_pointer(y);
                    Ok(0)
                }
                0x1003 => {
                    let value = self.host.thread_pointer();
                    self.output(y, &value.to_le_bytes())?;
                    Ok(0)
                }
                _ => Err(-22),
            },
            146 | 147 => match x {
                0 | 3 | 5 => Ok(0),
                1 | 2 => Ok(if number == 146 { 99 } else { 1 }),
                _ => Err(-22),
            },
            202 => {
                self.check(x, 4, false)?;
                let op = y & !0x80;
                if op == 1 {
                    Ok(0)
                } else if op == 0 && u32::from_le_bytes(self.bytes(x, 4)?.try_into().unwrap()) != z as u32 {
                    Err(-11)
                } else {
                    Err(-38)
                }
            }
            218 => {
                self.clear_tid = x;
                Ok(self.id as isize)
            }
            228 => {
                let ms = match x {
                    0 => self.host.realtime_ms()?,
                    1 | 4 | 6 | 7 => self.host.monotonic_ms(),
                    2 | 3 => self.host.cpu_time_ms(),
                    _ => return Err(-22),
                };
                self.output(y, &(ms / 1000).to_le_bytes())?;
                self.output(y + 8, &((ms % 1000) * 1_000_000).to_le_bytes())?;
                Ok(0)
            }
            257 => {
                let path = self.path(y)?;
                if x as i64 != -100 && self.bytes(y, 1)? != b"/" {
                    return Err(-9);
                }
                self.open(path, z)
            }
            262 => {
                if w & !0x1100 != 0 {
                    return Err(-22);
                }
                if self.bytes(y, 1)? == [0] && w & 0x1000 != 0 {
                    return self.stat(if x <= 2 { None } else { Some(&self.file(x)?.path) }, z);
                }
                if x as i64 != -100 && self.bytes(y, 1)? != b"/" {
                    return Err(-9);
                }
                self.stat(Some(&self.path(y)?), z)
            }
            273 => {
                if y != 24 {
                    return Err(-22);
                }
                self.check(x, 24, false)?;
                self.robust_list = x;
                Ok(0)
            }
            302 => {
                if x > 1 {
                    return Err(-3);
                }
                if z != 0 {
                    return Err(-1);
                }
                let limit = match y {
                    3 => [STACK_SIZE as u64, STACK_SIZE as u64],
                    7 => [4096, 4096],
                    _ => return Err(-22),
                };
                if w != 0 {
                    let mut bytes = [0; 16];
                    bytes[..8].copy_from_slice(&limit[0].to_le_bytes());
                    bytes[8..].copy_from_slice(&limit[1].to_le_bytes());
                    self.output(w, &bytes)?;
                }
                Ok(0)
            }
            318 => {
                if z & !7 != 0 {
                    return Err(-22);
                }
                self.check(x, y.try_into().map_err(|_| -22isize)?, true)?;
                let mut done = 0;
                while done < y {
                    let random = self.host.random_word()?;
                    let count = (y - done).min(8) as usize;
                    self.output(x + done, &random.to_le_bytes()[..count])?;
                    done += count as u64;
                }
                Ok(y as isize)
            }
            334 => Err(-38), // glibc falls back when restartable sequences are unavailable.
            _ => {
                log::warn!("linux: unsupported syscall {number} args={a:x?}");
                Err(-38)
            }
        }
    }
}

fn errno(e: Errno) -> isize {
    linux_errno(e as isize)
}

pub fn normalize(path: &str, cwd: &str) -> Result<String> {
    if path.is_empty() {
        return Err(-2);
    }
    let joined: String = if path.starts_with('/') { path.into() } else { alloc::format!("{cwd}/{path}") };
    let mut parts = Vec::new();
    for part in joined.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            _ => parts.push(part),
        }
    }
    Ok(alloc::format!("/{}", parts.join("/")))
}

impl<H: Host> Drop for LinuxProcess<H> {
    fn drop(&mut self) {
        self.close_files();
    }
}
