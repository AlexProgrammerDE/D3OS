//! User pointers stay on the CPU side; GPU commands receive only owned handles.
use crate::{
    memory::{MemorySpace, vma::VmaType},
    process_manager,
};
use d3os_nvidia::compute::{self as abi, Request};
use syscall::return_vals::{Errno, convert_syscall_result_to_ret_code};

fn valid_range(process: &crate::process::process::Process, pointer: u64, bytes: u64, writable: bool) -> bool {
    if pointer == 0 || bytes == 0 || x86_64::VirtAddr::try_new(pointer).is_err() {
        return false;
    }
    let Some(end) = pointer.checked_add(bytes) else {
        return false;
    };
    if x86_64::VirtAddr::try_new(end - 1).is_err() {
        return false;
    }
    let mut cursor = pointer;
    while cursor < end {
        let Some(vma) = process.virtual_address_space.vma_for_address(cursor) else {
            return false;
        };
        if vma.space != MemorySpace::User
            || (writable && !matches!(vma.typ, VmaType::Heap | VmaType::Anonymous | VmaType::UserStack | VmaType::SharedMemory { .. }))
        {
            return false;
        }
        cursor = vma.end().as_u64().min(end);
    }
    true
}

pub extern "sysv64" fn sys_gpu_compute(pointer: *const Request) -> isize {
    let process = process_manager().read().current_process();
    if !valid_range(&process, pointer as u64, size_of::<Request>() as u64, false) {
        return Errno::EINVAL as isize;
    }
    let request = unsafe { pointer.read_unaligned() };
    if request.operation == abi::PROFILE {
        if request.elements != abi::KERNEL_COUNT as u64
            || !valid_range(&process, request.pointer, (size_of::<abi::ProfileEntry>() * abi::KERNEL_COUNT) as u64, true)
        {
            return Errno::EINVAL as isize;
        }
    }
    if request.operation == abi::SERVICE_PROFILE
        && (request.elements != abi::SERVICE_OPERATIONS as u64
            || !valid_range(
                &process,
                request.pointer,
                (size_of::<abi::ServiceEntry>() * abi::SERVICE_OPERATIONS) as u64,
                true,
            ))
    {
        return Errno::EINVAL as isize;
    }
    if request.operation == abi::QUEUE_PROFILE
        && (request.elements != 1 || !valid_range(&process, request.pointer, size_of::<abi::QueueProfile>() as u64, true))
    {
        return Errno::EINVAL as isize;
    }
    if matches!(request.operation, abi::UPLOAD | abi::DOWNLOAD) {
        let Some(bytes) = request.elements.checked_mul(4) else {
            return Errno::EINVAL as isize;
        };
        if request.pointer & 3 != 0 || !valid_range(&process, request.pointer, bytes, request.operation == abi::DOWNLOAD) {
            return Errno::EINVAL as isize;
        }
    }
    convert_syscall_result_to_ret_code(crate::device::nvidia::compute_request(process.id(), &request))
}
