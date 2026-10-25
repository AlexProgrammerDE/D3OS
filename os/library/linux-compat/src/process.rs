use crate::{Host, LOW_EXECUTABLE_END, LOW_EXECUTABLE_START, Result, elf::Elf, linux_errno, page_up, stack};
use alloc::{collections::BTreeMap, string::String, vec::Vec};
use naming::shared_types::{OpenOptions, SeekOrigin};
use syscall::return_vals::Errno;

#[derive(Debug, Clone, Copy)]
pub struct StartContext {
    pub instruction: u64,
    pub stack: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyscallOutcome {
    Return(isize),
    Exit(u8),
    CloneThread(CloneThread),
    Vfork {
        stack: u64,
    },
    Exec {
        path: String,
        arguments: Vec<String>,
        environment: Vec<String>,
    },
    Wait(WaitRequest),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloneThread {
    pub stack: u64,
    pub tls: Option<u64>,
    pub parent_tid: Option<u64>,
    pub child_tid: Option<u64>,
    pub clear_tid: Option<u64>,
}

impl CloneThread {
    fn new(flags: u64, stack: u64, parent_tid: u64, child_tid: u64, tls: u64) -> Result<Self> {
        const REQUIRED: u64 = 0x100 | 0x200 | 0x400 | 0x800 | 0x10000 | 0x40000;
        const SET_TLS: u64 = 0x80000;
        const PARENT_TID: u64 = 0x100000;
        const CLEAR_TID: u64 = 0x200000;
        const CHILD_TID: u64 = 0x1000000;
        if flags & REQUIRED != REQUIRED || flags & !(REQUIRED | SET_TLS | PARENT_TID | CLEAR_TID | CHILD_TID) != 0 {
            return Err(-38);
        }
        stack.checked_sub(16).ok_or(-22isize)?;
        Ok(Self {
            stack,
            tls: (flags & SET_TLS != 0).then_some(tls),
            parent_tid: (flags & PARENT_TID != 0).then_some(parent_tid),
            child_tid: (flags & CHILD_TID != 0).then_some(child_tid),
            clear_tid: (flags & CLEAR_TID != 0).then_some(child_tid),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaitRequest {
    pub pid: i64,
    pub output: u64,
    pub usage: u64,
    pub nonblocking: bool,
    pub keep: bool,
    pub information: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SyscallFrame {
    pub number: u64,
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub rbp: u64,
    pub rbx: u64,
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
const _: () = assert!(core::mem::size_of::<SyscallFrame>() == 128);

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

#[derive(Clone, Copy)]
struct PageMapping {
    protection: u32,
    backed: bool,
}

pub struct LinuxProcess<H: Host> {
    host: H,
    id: u32,
    cwd: String,
    executable: String,
    pages: BTreeMap<u64, PageMapping>,
    files: BTreeMap<u64, File>,
    consoles: BTreeMap<u64, u64>,
    next_map: u64,
    brk: u64,
    signal_mask: u64,
    actions: [[u8; 32]; 64],
    clear_tid: u64,
    robust_list: u64,
    alternate_stack: [u8; 24],
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
            executable: String::new(),
            pages: BTreeMap::new(),
            files: BTreeMap::new(),
            consoles: [(0, 0), (1, 1), (2, 2)].into(),
            next_map: MMAP,
            brk: HEAP,
            signal_mask: 0,
            actions: [[0; 32]; 64],
            clear_tid: 0,
            robust_list: 0,
            alternate_stack: {
                let mut stack = [0; 24];
                stack[8] = 2;
                stack
            },
        }
    }

    pub fn host(&self) -> &H {
        &self.host
    }

    pub fn id(&self) -> u32 {
        self.id
    }
    pub fn cwd(&self) -> &str {
        &self.cwd
    }
    pub fn set_executable_path(&mut self, path: &str) {
        self.executable = path.into();
    }

    /// Copy process metadata and descriptors for a serialized vfork child.
    /// The host still refers to the shared address space until exec or exit.
    pub fn vfork(&self, host: H, id: u32) -> Result<Self> {
        let mut child = Self::new(host, id, &self.cwd);
        child.pages = self.pages.clone();
        child.next_map = self.next_map;
        child.brk = self.brk;
        child.executable = self.executable.clone();
        child.signal_mask = self.signal_mask;
        child.alternate_stack = self.alternate_stack;
        child.inherit_descriptors(self, false)?;
        Ok(child)
    }

    pub fn inherit_descriptors(&mut self, parent: &Self, exec: bool) -> Result<()> {
        self.close_files();
        self.consoles = parent.consoles.clone();
        for (&fd, file) in &parent.files {
            if exec && file.flags & 0x80000 != 0 {
                continue;
            }
            let handle = self.host.duplicate(file.handle).map_err(errno)?;
            self.files.insert(
                fd,
                File {
                    handle,
                    path: file.path.clone(),
                    flags: file.flags,
                },
            );
        }
        Ok(())
    }

    /// Write a syscall result into validated guest memory.
    ///
    /// # Safety
    /// The caller must activate this process's mappings and exclude concurrent
    /// access to its memory, as required by `dispatch`.
    pub unsafe fn write_memory(&self, address: u64, data: &[u8]) -> Result<()> {
        self.output(address, data)
    }

    /// Validate a user-memory range before a host operation accesses it.
    pub fn check_memory(&self, address: u64, length: usize, write: bool) -> Result<()> {
        self.check(address, length, write)
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
        } else if matches!(number, 56 | 59 | 61 | 247 | 435) {
            self.process_call(number, args).unwrap_or_else(SyscallOutcome::Return)
        } else {
            SyscallOutcome::Return(self.call(number, args).unwrap_or_else(|e| e))
        }
    }

    pub fn start(&mut self, executable: &[u8], interpreter: Option<&[u8]>, args: &[&str]) -> Result<StartContext> {
        self.start_with_environment(executable, interpreter, args, &["LANG=C", "LC_ALL=C", "LD_LIBRARY_PATH=/lib64"])
    }

    pub fn start_with_environment(&mut self, executable: &[u8], interpreter: Option<&[u8]>, args: &[&str], environment: &[&str]) -> Result<StartContext> {
        let elf = Elf::parse(executable, 0).map_err(|_| -8isize)?;
        self.executable = (*args.first().ok_or(-22isize)?).into();
        let bias = if elf.position_independent { HIGH + 0x8000_0000 } else { 0 };
        let (entry, phdr, phnum) = self.image(executable, bias)?;
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
            environment,
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

    fn process_call(&self, number: u64, args: [u64; 6]) -> Result<SyscallOutcome> {
        match number {
            56 => self.clone_thread(args[0], args[1], args[2], args[3], args[4]),
            435 => {
                if args[1] < 64 || args[1] > 88 {
                    return Err(-22);
                }
                let data = self.bytes(args[0], args[1] as usize)?;
                let word = |offset| u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                if word(0) & 0x10000 != 0 {
                    if word(32) != 0 || data[64..].iter().any(|&byte| byte != 0) {
                        return Err(-38);
                    }
                    let stack = word(40).checked_add(word(48)).ok_or(-22isize)?;
                    if word(48) < 16 {
                        return Err(-22);
                    }
                    return self.clone_thread(word(0), stack, word(24), word(16), word(56));
                }
                // Serialized, shared-address-space spawn with cleared signal handlers.
                if word(0) != 0x100004100 || word(32) != 17 || [8, 16, 24, 56].iter().any(|&at| word(at) != 0) || data[64..].iter().any(|&byte| byte != 0) {
                    return Err(-38);
                }
                let stack = word(40).checked_add(word(48)).ok_or(-22isize)?;
                if word(48) < 16 {
                    return Err(-22);
                }
                self.check(word(40), word(48).try_into().map_err(|_| -22isize)?, true)?;
                Ok(SyscallOutcome::Vfork { stack })
            }
            59 => Ok(SyscallOutcome::Exec {
                path: self.path(args[0])?,
                arguments: self.strings(args[1])?,
                environment: self.strings(args[2])?,
            }),
            61 => {
                if args[2] & !1 != 0 {
                    return Err(-22);
                }
                if args[1] != 0 {
                    self.check(args[1], 4, true)?;
                }
                if args[3] != 0 {
                    self.check(args[3], 144, true)?;
                }
                Ok(SyscallOutcome::Wait(WaitRequest {
                    pid: args[0] as i64,
                    output: args[1],
                    usage: args[3],
                    nonblocking: args[2] & 1 != 0,
                    keep: false,
                    information: false,
                }))
            }
            247 => {
                if args[0] > 1 || args[3] & !0x1000005 != 0 || args[3] & 4 == 0 {
                    return Err(-22);
                }
                if args[2] != 0 {
                    self.check(args[2], 128, true)?;
                }
                if args[4] != 0 {
                    self.check(args[4], 144, true)?;
                }
                Ok(SyscallOutcome::Wait(WaitRequest {
                    pid: if args[0] == 0 { -1 } else { args[1] as i64 },
                    output: args[2],
                    usage: args[4],
                    nonblocking: args[3] & 1 != 0,
                    keep: args[3] & 0x1000000 != 0,
                    information: true,
                }))
            }
            _ => unreachable!(),
        }
    }

    fn clone_thread(&self, flags: u64, stack: u64, parent_tid: u64, child_tid: u64, tls: u64) -> Result<SyscallOutcome> {
        let request = CloneThread::new(flags, stack, parent_tid, child_tid, tls)?;
        self.check(stack - 16, 16, true)?;
        if request.tls.is_some() {
            self.check(tls, 8, true)?;
        }
        if request.parent_tid.is_some() {
            self.check(parent_tid, 4, true)?;
        }
        if request.clear_tid.is_some() || request.child_tid.is_some() {
            self.check(child_tid, 4, true)?;
        }
        Ok(SyscallOutcome::CloneThread(request))
    }

    fn strings(&self, address: u64) -> Result<Vec<String>> {
        let mut strings = Vec::new();
        if address == 0 {
            return Ok(strings);
        }
        let mut total = 0;
        for i in 0..4096u64 {
            let pointer = address.checked_add(i * 8).ok_or(-14isize)?;
            let pointer = u64::from_le_bytes(self.bytes(pointer, 8)?.try_into().unwrap());
            if pointer == 0 {
                return Ok(strings);
            }
            let mut bytes = Vec::new();
            for offset in 0..4096 {
                let byte = self.bytes(pointer.checked_add(offset).ok_or(-14isize)?, 1)?[0];
                total += 1;
                if total > 1024 * 1024 {
                    return Err(-7);
                }
                if byte == 0 {
                    break;
                }
                bytes.push(byte);
            }
            if bytes.len() == 4096 {
                return Err(-7);
            }
            strings.push(String::from_utf8(bytes).map_err(|_| -22isize)?);
        }
        Err(-7)
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
            let mapping = self.pages.get(&page).ok_or(-14isize)?;
            if mapping.protection & if write { 2 } else { 1 } == 0 {
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
            || !(address >= HIGH && end < 0x8000_0000_0000 || address >= LOW_EXECUTABLE_START && end <= LOW_EXECUTABLE_END)
        {
            return Err(-22);
        }
        for p in (address..end).step_by(PAGE as usize) {
            if self.pages.contains_key(&p) {
                if !replace {
                    return Err(-17);
                }
                self.unmap(p)?;
            }
            let backed = protection != 0;
            if backed {
                self.host.allocate_page(p)?;
            }
            self.pages.insert(p, PageMapping { protection, backed });
        }
        Ok(())
    }

    fn unmap(&mut self, address: u64) -> Result<()> {
        if let Some(mapping) = self.pages.get(&address) {
            if mapping.backed {
                self.host.unmap_page(address)?;
            }
            self.pages.remove(&address);
        }
        Ok(())
    }

    fn protect(&mut self, address: u64, length: u64, protection: u32) -> Result<()> {
        let end = address.checked_add(length).and_then(page_up).ok_or(-22isize)?;
        for p in (address..end).step_by(PAGE as usize) {
            let mapping = self.pages.get_mut(&p).ok_or(-12isize)?;
            // Anonymous PROT_NONE mappings reserve addresses only. An already
            // backed page retains its contents when access is removed.
            if !mapping.backed && protection != 0 {
                self.host.allocate_page(p)?;
                mapping.backed = true;
            }
            if mapping.backed {
                self.host.protect_page(p, protection)?;
            }
            mapping.protection = protection;
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
        let fd = (0..4096)
            .find(|fd| !self.files.contains_key(fd) && !self.consoles.contains_key(fd))
            .ok_or(-24isize)?;
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
        if flags & !0x21833 != 0 || flags & 3 != 2 || offset % PAGE != 0 || length == 0 || protection & !7 != 0 {
            return Err(-22);
        }
        let size = page_up(length).ok_or(-12isize)?;
        let fixed = flags & 0x10 != 0;
        let address = if fixed { requested } else { self.next_map };
        if !fixed {
            self.next_map = address.checked_add(size).and_then(|n| n.checked_add(PAGE)).ok_or(-12isize)?;
        }
        let file = if flags & 0x20 == 0 { Some(self.file(fd)?.handle) } else { None };
        // File mappings need backing even for PROT_NONE so their initial
        // contents survive a later permission change without reading again.
        self.map(address, size, if file.is_some() { 3 } else { protection as u32 }, fixed)?;
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
                if let Some(&console) = self.consoles.get(&x) {
                    if console == 0 {
                        return Err(-9);
                    }
                    self.host.console_write(console, bytes).map(|n| n as isize)
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
                if self.consoles.remove(&x).is_some() {
                    return Ok(0);
                }
                let file = self.files.remove(&x).ok_or(-9isize)?;
                self.host.close(file.handle).map_err(errno)?;
                Ok(0)
            }
            4 | 6 => self.stat(Some(&self.path(x)?), y),
            5 => self.stat(if self.consoles.contains_key(&x) { None } else { Some(&self.file(x)?.path) }, y),
            32 | 33 => {
                if !self.files.contains_key(&x) && !self.consoles.contains_key(&x) {
                    return Err(-9);
                }
                let target = if number == 33 {
                    y
                } else {
                    (0..4096)
                        .find(|fd| !self.files.contains_key(fd) && !self.consoles.contains_key(fd))
                        .ok_or(-24isize)?
                };
                if target >= 4096 {
                    return Err(-9);
                }
                if target == x {
                    return Ok(target as isize);
                }
                let file = self
                    .files
                    .get(&x)
                    .map(|file| -> Result<File> {
                        Ok(File {
                            handle: self.host.duplicate(file.handle).map_err(errno)?,
                            path: file.path.clone(),
                            flags: file.flags & !0x80000,
                        })
                    })
                    .transpose()?;
                let console = self.consoles.get(&x).copied();
                if let Some(old) = self.files.remove(&target) {
                    let _ = self.host.close(old.handle);
                }
                self.consoles.remove(&target);
                if let Some(file) = file {
                    self.files.insert(target, file);
                }
                if let Some(console) = console {
                    self.consoles.insert(target, console);
                }
                Ok(target as isize)
            }
            87 => {
                self.host.unlink(&self.path(x)?).map_err(errno)?;
                Ok(0)
            }
            72 => {
                if let Some(file) = self.files.get_mut(&x) {
                    match y {
                        1 => Ok(isize::from(file.flags & 0x80000 != 0)),
                        2 => {
                            if z & !1 != 0 {
                                return Err(-22);
                            }
                            file.flags = (file.flags & !0x80000) | ((z & 1) << 19);
                            Ok(0)
                        }
                        3 => Ok((file.flags & !0x80000) as isize),
                        _ => Err(-22),
                    }
                } else if self.consoles.contains_key(&x) {
                    match y {
                        1 => Ok(0),
                        3 => Ok(if x == 0 { 0 } else { 1 }),
                        _ => Err(-95),
                    }
                } else {
                    Err(-9)
                }
            }
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
                Ok(0)
            }
            11 => {
                if x % PAGE != 0 || y == 0 {
                    return Err(-22);
                }
                let end = page_up(x.checked_add(y).ok_or(-22isize)?).ok_or(-22isize)?;
                for p in (x..end).step_by(PAGE as usize) {
                    self.unmap(p)?;
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
                        self.unmap(p)?;
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
                let path = self.path(x)?;
                if path != "/proc/self/exe" {
                    self.host.metadata(&path).map_err(errno)?;
                }
                Ok(0)
            }
            89 | 267 => {
                let (path, out, size) = if number == 89 {
                    (x, y, z)
                } else {
                    if x as i32 != -100 {
                        return Err(-9);
                    }
                    (y, z, w)
                };
                if size == 0 {
                    return Err(-22);
                }
                let path = self.path(path)?;
                if path != "/proc/self/exe" {
                    self.host.metadata(&path).map_err(errno)?;
                    return Err(-22);
                }
                let size = (size as usize).min(self.executable.len());
                self.output(out, &self.executable.as_bytes()[..size])?;
                Ok(size as isize)
            }
            97 => {
                let limit = match x {
                    3 => [STACK_SIZE as u64; 2],
                    7 => [4096; 2],
                    _ => return Err(-22),
                };
                self.output(y, &limit[0].to_le_bytes())?;
                self.output(y + 8, &limit[1].to_le_bytes())?;
                Ok(0)
            }
            98 => {
                if !matches!(x as i64, -1..=1) {
                    return Err(-22);
                }
                let mut usage = [0u8; 144];
                let ms = self.host.cpu_time_ms();
                usage[..8].copy_from_slice(&(ms / 1000).to_le_bytes());
                usage[8..16].copy_from_slice(&((ms % 1000) * 1000).to_le_bytes());
                self.output(y, &usage)?;
                Ok(0)
            }
            131 => {
                if x != 0 {
                    let new = self.bytes(x, 24)?.to_vec();
                    let flags = u32::from_le_bytes(new[8..12].try_into().unwrap());
                    if flags & !0x80000002 != 0 {
                        return Err(-22);
                    }
                    if flags & 2 == 0 {
                        let size = u64::from_le_bytes(new[16..24].try_into().unwrap());
                        if size < 2048 {
                            return Err(-12);
                        }
                    }
                    if y != 0 {
                        self.output(y, &self.alternate_stack)?;
                    }
                    self.alternate_stack = new.try_into().unwrap();
                } else if y != 0 {
                    self.output(y, &self.alternate_stack)?;
                }
                Ok(0)
            }
            204 => {
                if x != 0 && x != self.id as u64 {
                    return Err(-3);
                }
                if y < 8 {
                    return Err(-22);
                }
                let cpus = self.host.cpu_count().clamp(1, 64);
                let mask = u64::MAX >> (64 - cpus);
                self.output(z, &mask.to_le_bytes())?;
                Ok(8)
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
                if x as i32 != -100 && self.bytes(y, 1)? != b"/" {
                    return Err(-9);
                }
                self.open(path, z)
            }
            262 => {
                if w & !0x1100 != 0 {
                    return Err(-22);
                }
                if self.bytes(y, 1)? == [0] && w & 0x1000 != 0 {
                    return self.stat(if self.consoles.contains_key(&x) { None } else { Some(&self.file(x)?.path) }, z);
                }
                if x as i32 != -100 && self.bytes(y, 1)? != b"/" {
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

#[cfg(test)]
mod tests {
    use super::CloneThread;

    #[test]
    fn thread_requests_select_only_flagged_tid_and_tls_actions() {
        let request = CloneThread::new(0x3d0f00, 0x8000, 0x1000, 0x2000, 0x3000).unwrap();
        assert_eq!(request.stack, 0x8000);
        assert_eq!(request.parent_tid, Some(0x1000));
        assert_eq!(request.clear_tid, Some(0x2000));
        assert_eq!(request.child_tid, None);
        assert_eq!(request.tls, Some(0x3000));
        let inherited = CloneThread::new(0x50f00, 0x8000, 1, 2, 3).unwrap();
        assert_eq!(
            (inherited.parent_tid, inherited.child_tid, inherited.clear_tid, inherited.tls),
            (None, None, None, None)
        );
    }

    #[test]
    fn thread_requests_reject_unshared_state_and_unsupported_clone_modes() {
        for required in [0x100, 0x200, 0x400, 0x800, 0x10000, 0x40000] {
            assert_eq!(CloneThread::new(0x3d0f00 & !required, 0x8000, 0, 0, 0), Err(-38));
        }
        for unsupported in [17, 0x4000, 0x1000, 0x200000000] {
            assert_eq!(CloneThread::new(0x3d0f00 | unsupported, 0x8000, 0, 0, 0), Err(-38));
        }
        assert_eq!(CloneThread::new(0x3d0f00, 15, 0, 0, 0), Err(-22));
    }
}
