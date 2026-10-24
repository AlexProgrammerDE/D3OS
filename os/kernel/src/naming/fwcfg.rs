//! Read-only firmware files, streamed through a bounded DMA buffer.
use alloc::{sync::Arc, vec::Vec};
use core::fmt;
use naming::shared_types::{DirEntry, FileType, OpenOptions};
use spin::Mutex;
use syscall::return_vals::Errno;

use super::{
    stat::{MODE_DIR, MODE_FILE, Mode, Stat},
    traits::{DirectoryObject, FileObject, FileSystem, NamedObject},
};
use crate::device::qemu_cfg::{Files, FirmwareFile};

pub struct FirmwareFs {
    root: Arc<Directory>,
}

impl FirmwareFs {
    pub fn new() -> Option<Self> {
        let files = match Files::probe() {
            Ok(Some(files)) => files,
            Ok(None) => return None,
            Err(error) => {
                log::warn!("Cannot initialize firmware files: {error:?}");
                return None;
            }
        };
        let entries = files.entries.clone();
        let device = Arc::new(Mutex::new(files));
        let files = entries.into_iter().map(|info| Arc::new(File { info, device: device.clone() })).collect();
        Some(Self {
            root: Arc::new(Directory { files }),
        })
    }
}

impl FileSystem for FirmwareFs {
    fn root_dir(&self) -> Arc<dyn DirectoryObject> {
        self.root.clone()
    }
}

#[derive(Debug)]
struct Directory {
    files: Vec<Arc<File>>,
}

impl DirectoryObject for Directory {
    fn lookup(&self, name: &str) -> Result<NamedObject, Errno> {
        self.files
            .iter()
            .find(|file| file.info.name == name)
            .map(|file| NamedObject::FileObject(file.clone()))
            .ok_or(Errno::ENOENT)
    }
    fn create_file(&self, _: &str, _: Mode) -> Result<NamedObject, Errno> {
        Err(Errno::ERDONLY)
    }
    fn create_dir(&self, _: &str, _: Mode) -> Result<NamedObject, Errno> {
        Err(Errno::ERDONLY)
    }
    fn create_pipe(&self, _: &str, _: Mode) -> Result<NamedObject, Errno> {
        Err(Errno::ERDONLY)
    }
    fn stat(&self) -> Result<Stat, Errno> {
        Ok(Stat::new(Mode::new(MODE_DIR), 0))
    }
    fn readdir(&self, index: usize) -> Result<Option<DirEntry>, Errno> {
        Ok(self.files.get(index).map(|file| DirEntry {
            file_type: FileType::Regular,
            name: file.info.name.clone(),
        }))
    }
}

struct File {
    info: FirmwareFile,
    device: Arc<Mutex<Files>>,
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FirmwareFile").field("info", &self.info).finish()
    }
}

impl FileObject for File {
    fn stat(&self) -> Result<Stat, Errno> {
        Ok(Stat::new(Mode::new(MODE_FILE), self.info.size))
    }
    fn read(&self, buffer: &mut [u8], offset: usize, _: OpenOptions) -> Result<usize, Errno> {
        self.device.lock().read(&self.info, offset, buffer).map_err(|error| {
            log::warn!("Firmware file {} read failed at {offset}: {error:?}", self.info.name);
            Errno::EUNKN
        })
    }
    fn write(&self, _: &[u8], _: usize, _: OpenOptions) -> Result<usize, Errno> {
        Err(Errno::ERDONLY)
    }
}
