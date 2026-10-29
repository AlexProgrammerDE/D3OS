use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_int, c_uint, c_void},
    path::Path,
    ptr,
    sync::{Arc, Mutex},
};

use d3os_compute::{Buffer, Error, Result};
use d3os_nvidia::compute::{KERNEL_COUNT, Kernel, ProfileEntry, Work};
use libloading::Library;

use crate::gpu::Device;

type Handle = *mut c_void;
type DevicePointer = u64;

macro_rules! cuda_api {
    ($($field:ident: $symbol:literal => $signature:ty),* $(,)?) => {
        struct Api {
            _library: Library,
            $($field: $signature,)*
        }
        impl Api {
            fn load() -> Result<Self> {
                // Keep the library alive for every copied function pointer.
                let library = unsafe { Library::new("libcuda.so.1") }.map_err(|_| Error::DeviceUnavailable)?;
                Ok(Self {
                    $($field: unsafe { *library.get::<$signature>(concat!($symbol, "\0").as_bytes()).map_err(|_| Error::DeviceUnavailable)? },)*
                    _library: library,
                })
            }
        }
    };
}

cuda_api! {
    init: "cuInit" => unsafe extern "C" fn(c_uint) -> c_int,
    device_get: "cuDeviceGet" => unsafe extern "C" fn(*mut c_int, c_int) -> c_int,
    device_name: "cuDeviceGetName" => unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    driver_version: "cuDriverGetVersion" => unsafe extern "C" fn(*mut c_int) -> c_int,
    capability: "cuDeviceComputeCapability" => unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> c_int,
    context_create: "cuCtxCreate_v2" => unsafe extern "C" fn(*mut Handle, c_uint, c_int) -> c_int,
    context_destroy: "cuCtxDestroy_v2" => unsafe extern "C" fn(Handle) -> c_int,
    context_push: "cuCtxPushCurrent_v2" => unsafe extern "C" fn(Handle) -> c_int,
    context_pop: "cuCtxPopCurrent_v2" => unsafe extern "C" fn(*mut Handle) -> c_int,
    context_get: "cuCtxGetCurrent" => unsafe extern "C" fn(*mut Handle) -> c_int,
    synchronize: "cuCtxSynchronize" => unsafe extern "C" fn() -> c_int,
    module_load: "cuModuleLoad" => unsafe extern "C" fn(*mut Handle, *const c_char) -> c_int,
    module_unload: "cuModuleUnload" => unsafe extern "C" fn(Handle) -> c_int,
    function: "cuModuleGetFunction" => unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> c_int,
    allocate: "cuMemAlloc_v2" => unsafe extern "C" fn(*mut DevicePointer, usize) -> c_int,
    free: "cuMemFree_v2" => unsafe extern "C" fn(DevicePointer) -> c_int,
    upload: "cuMemcpyHtoD_v2" => unsafe extern "C" fn(DevicePointer, *const c_void, usize) -> c_int,
    download: "cuMemcpyDtoH_v2" => unsafe extern "C" fn(*mut c_void, DevicePointer, usize) -> c_int,
    launch: "cuLaunchKernel" => unsafe extern "C" fn(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void) -> c_int,
}

fn check(status: c_int, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        eprintln!("LINUX_DOOM_CUDA_FAIL operation={operation} status={status}");
        Err(Error::DeviceFault)
    }
}

struct Context {
    api: Api,
    context: Handle,
    module: Handle,
    functions: BTreeMap<&'static str, Handle>,
    profile: [ProfileEntry; KERNEL_COUNT],
}

// CUDA contexts may move between threads. Every operation holds the mutex and
// restores the caller's context if it had to switch. The benchmark thread keeps
// its context current, as a normal single-threaded CUDA client would.
unsafe impl Send for Context {}

impl Context {
    fn current<T>(&mut self, work: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let mut current = ptr::null_mut();
        check(unsafe { (self.api.context_get)(&mut current) }, "context_get")?;
        if current == self.context {
            return work(self);
        }
        check(unsafe { (self.api.context_push)(self.context) }, "context_push")?;
        let result = work(self);
        let mut popped = ptr::null_mut();
        let restored = check(unsafe { (self.api.context_pop)(&mut popped) }, "context_pop");
        match result {
            Err(error) => Err(error),
            Ok(value) => {
                restored?;
                if popped != self.context {
                    return Err(Error::DeviceFault);
                }
                Ok(value)
            }
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let _ = self.current(|state| check(unsafe { (state.api.module_unload)(state.module) }, "module_unload"));
        let _ = check(unsafe { (self.api.context_destroy)(self.context) }, "context_destroy");
    }
}

pub(crate) struct CudaDevice(Arc<Mutex<Context>>);

pub(crate) struct CudaWords {
    owner: Arc<Mutex<Context>>,
    pointer: DevicePointer,
    words: usize,
}

impl Buffer for CudaWords {
    fn len(&self) -> usize {
        self.words
    }
}

impl Drop for CudaWords {
    fn drop(&mut self) {
        if let Ok(mut state) = self.owner.lock() {
            let _ = state.current(|state| check(unsafe { (state.api.free)(self.pointer) }, "free"));
        }
    }
}

impl CudaDevice {
    pub(crate) fn new(cubin: &Path) -> Result<Self> {
        let api = Api::load()?;
        check(unsafe { (api.init)(0) }, "init")?;
        let mut device = 0;
        check(unsafe { (api.device_get)(&mut device, 0) }, "device_get")?;
        let (mut major, mut minor, mut driver) = (0, 0, 0);
        check(unsafe { (api.capability)(&mut major, &mut minor, device) }, "capability")?;
        check(unsafe { (api.driver_version)(&mut driver) }, "driver_version")?;
        if (major, minor) != (8, 9) {
            return Err(Error::DeviceUnavailable);
        }
        let mut name = [0i8; 256];
        check(unsafe { (api.device_name)(name.as_mut_ptr(), name.len() as c_int, device) }, "device_name")?;
        println!(
            "LINUX_DOOM_DEVICE sm={major}{minor} driver={driver} name={}",
            unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy()
        );
        let path = CString::new(cubin.as_os_str().as_encoded_bytes()).map_err(|_| Error::InvalidParameter)?;
        let mut context = ptr::null_mut();
        check(unsafe { (api.context_create)(&mut context, 0, device) }, "context_create")?;
        let mut module = ptr::null_mut();
        let loaded = check(unsafe { (api.module_load)(&mut module, path.as_ptr()) }, "module_load");
        if loaded.is_err() {
            if !module.is_null() {
                let _ = unsafe { (api.module_unload)(module) };
            }
            let mut popped = ptr::null_mut();
            let _ = unsafe { (api.context_pop)(&mut popped) };
            let _ = unsafe { (api.context_destroy)(context) };
            return Err(Error::DeviceFault);
        }
        Ok(Self(Arc::new(Mutex::new(Context {
            api,
            context,
            module,
            functions: BTreeMap::new(),
            profile: [ProfileEntry::default(); KERNEL_COUNT],
        }))))
    }

    fn context<T>(&self, work: impl FnOnce(&mut Context) -> Result<T>) -> Result<T> {
        self.0.lock().map_err(|_| Error::DeviceFault)?.current(work)
    }

    fn address(&self, buffer: &CudaWords, offset: usize, words: usize) -> Result<DevicePointer> {
        if !Arc::ptr_eq(&self.0, &buffer.owner) {
            return Err(Error::InvalidParameter);
        }
        checked_address(buffer.pointer, buffer.words, offset, words)
    }
}

fn checked_address(pointer: DevicePointer, capacity: usize, offset: usize, words: usize) -> Result<DevicePointer> {
    if offset.checked_add(words).is_none_or(|end| end > capacity) {
        return Err(Error::ShapeMismatch);
    }
    pointer
        .checked_add(
            u64::try_from(offset)
                .map_err(|_| Error::SizeOverflow)?
                .checked_mul(4)
                .ok_or(Error::SizeOverflow)?,
        )
        .ok_or(Error::SizeOverflow)
}

fn packet(kernel: Kernel, parameters: &[usize], buffers: &[(DevicePointer, usize)]) -> Result<(Vec<u32>, Work)> {
    let parameters: Vec<_> = parameters
        .iter()
        .copied()
        .map(u32::try_from)
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| Error::SizeOverflow)?;
    let work = kernel.work(&parameters).map_err(|_| Error::InvalidParameter)?;
    if buffers.len() != work.pointers {
        return Err(Error::InvalidParameter);
    }
    for ((_, available), &extent) in buffers.iter().zip(&work.extents[..work.pointers]) {
        if u64::try_from(*available).map_err(|_| Error::SizeOverflow)? < extent {
            return Err(Error::ShapeMismatch);
        }
    }
    work.validate_aliases(&buffers.iter().map(|&(pointer, _)| pointer).collect::<Vec<_>>())
        .map_err(|_| Error::InvalidParameter)?;
    Ok((parameters, work))
}

impl Device for CudaDevice {
    type Words = CudaWords;

    fn allocate_words(&self, elements: usize) -> Result<CudaWords> {
        d3os_compute::element_count(&[elements])?;
        u32::try_from(elements).map_err(|_| Error::SizeOverflow)?;
        let mut pointer = 0;
        self.context(|state| {
            check(
                unsafe { (state.api.allocate)(&mut pointer, elements.checked_mul(4).ok_or(Error::SizeOverflow)?) },
                "allocate",
            )
        })?;
        let buffer = CudaWords {
            owner: self.0.clone(),
            pointer,
            words: elements,
        };
        self.dispatch_words(Kernel::Zero, &[(&buffer, 0)], &[elements])?;
        Ok(buffer)
    }

    fn write_words(&self, buffer: &CudaWords, offset: usize, words: &[u32]) -> Result<()> {
        let address = self.address(buffer, offset, words.len())?;
        if words.is_empty() {
            return Ok(());
        }
        self.context(|state| check(unsafe { (state.api.upload)(address, words.as_ptr().cast(), size_of_val(words)) }, "upload"))
    }

    fn write_bytes(&self, buffer: &CudaWords, offset: usize, bytes: &[u8]) -> Result<()> {
        let mut words = vec![0u32; bytes.len().div_ceil(4)];
        // Both buffers own initialized, disjoint storage; the last word stays
        // zero padded and writes outside this word range are rejected.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast::<u8>(), bytes.len()) };
        self.write_words(buffer, offset, &words)
    }

    fn read_words_into(&self, buffer: &CudaWords, offset: usize, words: &mut [u32]) -> Result<()> {
        let address = self.address(buffer, offset, words.len())?;
        if words.is_empty() {
            return Ok(());
        }
        self.context(|state| {
            check(
                unsafe { (state.api.download)(words.as_mut_ptr().cast(), address, size_of_val(words)) },
                "download",
            )
        })
    }

    fn dispatch_words(&self, kernel: Kernel, buffers: &[(&CudaWords, usize)], parameters: &[usize]) -> Result<()> {
        let ranges: Vec<_> = buffers
            .iter()
            .map(|&(buffer, offset)| Ok((self.address(buffer, offset, 0)?, buffer.words - offset)))
            .collect::<Result<_>>()?;
        let (mut parameters, work) = packet(kernel, parameters, &ranges)?;
        let mut addresses: Vec<_> = ranges.iter().map(|&(address, _)| address).collect();
        self.context(|state| {
            let start = crate::gpu::cycles();
            let name = kernel.entry(&parameters);
            let function = if let Some(&function) = state.functions.get(name) {
                function
            } else {
                let symbol = CString::new(name).map_err(|_| Error::InvalidParameter)?;
                let mut function = ptr::null_mut();
                check(unsafe { (state.api.function)(&mut function, state.module, symbol.as_ptr()) }, "function")?;
                state.functions.insert(name, function);
                function
            };
            let mut arguments: Vec<*mut c_void> = addresses
                .iter_mut()
                .map(|value| (value as *mut u64).cast())
                .chain(parameters.iter_mut().map(|value| (value as *mut u32).cast()))
                .collect();
            check(
                unsafe {
                    (state.api.launch)(
                        function,
                        work.blocks,
                        1,
                        1,
                        work.threads,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        arguments.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "launch",
            )?;
            let prepared = crate::gpu::cycles();
            check(unsafe { (state.api.synchronize)() }, "synchronize")?;
            let entry = &mut state.profile[kernel as usize];
            entry.calls += 1;
            entry.cycles += prepared.wrapping_sub(start);
            entry.wait_cycles += crate::gpu::cycles().wrapping_sub(prepared);
            Ok(())
        })
    }

    fn profile(&self) -> Result<[ProfileEntry; KERNEL_COUNT]> {
        self.context(|state| Ok(state.profile))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_ranges_check_extent_offsets_and_pointer_overflow() {
        assert_eq!(checked_address(4096, 5, 3, 2), Ok(4108));
        assert_eq!(checked_address(4096, 5, 5, 0), Ok(4116));
        assert_eq!(checked_address(4096, 5, 4, 2), Err(Error::ShapeMismatch));
        assert_eq!(checked_address(4096, 5, usize::MAX, 2), Err(Error::ShapeMismatch));
        assert_eq!(checked_address(u64::MAX - 3, 5, 1, 1), Err(Error::SizeOverflow));
    }

    #[test]
    fn launch_packets_use_real_sdk_extents_and_reject_aliasing() {
        let parameters = [320, 200, 1, 4096, 524];
        let buffers = [(0x100000, 16000), (0x200000, 12), (0x300000, 1024), (0x400000, 16000), (0x500000, 524)];
        let (_, work) = packet(Kernel::RasterBinnedBytes, &parameters, &buffers).unwrap();
        assert_eq!(work.pointers, 5);
        let mut short = buffers;
        short[3].1 -= 1;
        assert!(matches!(packet(Kernel::RasterBinnedBytes, &parameters, &short), Err(Error::ShapeMismatch)));
        let mut alias = buffers;
        alias[3].0 = alias[0].0;
        assert!(matches!(packet(Kernel::RasterBinnedBytes, &parameters, &alias), Err(Error::InvalidParameter)));
        assert!(matches!(
            packet(Kernel::RasterBinnedBytes, &[usize::MAX, 200, 1, 4096, 524], &buffers),
            Err(Error::SizeOverflow)
        ));
    }
}
