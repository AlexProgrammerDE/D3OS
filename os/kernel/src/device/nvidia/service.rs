//! Process-owned compute buffers and explicit command scopes.
use super::{DEVICE, Device, mmu::Allocation};
use alloc::{collections::BTreeMap, vec::Vec};
use d3os_nvidia::compute::{self as abi, Kernel, Request};
use syscall::return_vals::{Errno, SyscallResult};
use uuid::Uuid;

#[derive(Clone, Copy)]
struct Buffer {
    owner: Uuid,
    allocation: Allocation,
    elements: u64,
}

#[derive(Default)]
pub(super) struct Buffers {
    next: u64,
    live: BTreeMap<u64, Buffer>,
    free: Vec<Allocation>,
    retired: Vec<(Allocation, u64)>,
}

impl Buffers {
    pub(super) fn reclaim_retired(&mut self, completed: u64) {
        let mut index = 0;
        while index < self.retired.len() {
            if d3os_nvidia::fence::reached(completed, self.retired[index].1) {
                let (allocation, _) = self.retired.swap_remove(index);
                self.recycle(allocation);
            } else {
                index += 1;
            }
        }
    }

    fn take_free(&mut self, bytes: u64) -> Option<Allocation> {
        let (index, _) = self.free.iter().enumerate().filter(|(_, b)| b.bytes >= bytes).min_by_key(|(_, b)| b.bytes)?;
        let mut allocation = self.free.swap_remove(index);
        if allocation.bytes > bytes {
            self.free.push(Allocation {
                physical: allocation.physical + bytes,
                virtual_address: allocation.virtual_address + bytes,
                bytes: allocation.bytes - bytes,
            });
            allocation.bytes = bytes;
        }
        Some(allocation)
    }

    fn get(&self, handle: u64, owner: Uuid) -> Result<Buffer, Errno> {
        self.live.get(&handle).filter(|buffer| buffer.owner == owner).copied().ok_or(Errno::EINVALH)
    }

    fn recycle(&mut self, mut allocation: Allocation) {
        // Merge only ranges contiguous in both physical and GPU virtual memory.
        let mut i = 0;
        while i < self.free.len() {
            let other = self.free[i];
            if allocation.physical + allocation.bytes == other.physical && allocation.virtual_address + allocation.bytes == other.virtual_address {
                allocation.bytes += other.bytes;
                self.free.swap_remove(i);
                i = 0;
            } else if other.physical + other.bytes == allocation.physical && other.virtual_address + other.bytes == allocation.virtual_address {
                allocation.physical = other.physical;
                allocation.virtual_address = other.virtual_address;
                allocation.bytes += other.bytes;
                self.free.swap_remove(i);
                i = 0;
            } else {
                i += 1;
            }
        }
        self.free.push(allocation);
    }
}

pub(crate) fn release_process(owner: Uuid) {
    let Some(device) = DEVICE.get() else {
        return;
    };
    let _preempt = crate::process::core_local_storage::PreemptGuard::new();
    let mut device = device.lock();
    let Some(mut compute) = device.compute.take() else {
        return;
    };
    // A faulted GPU may still reference both live and retired buffers.
    if compute.healthy && compute.flush_pending(&device).is_ok() {
        compute.command_scopes.remove(&owner);
        let handles: Vec<_> = compute.buffers.live.iter().filter(|(_, b)| b.owner == owner).map(|(&h, _)| h).collect();
        for handle in handles {
            if let Some(buffer) = compute.buffers.live.remove(&handle) {
                compute.buffers.recycle(buffer.allocation);
            }
        }
    }
    device.compute = Some(compute);
}

pub(crate) fn request(owner: Uuid, request: &Request) -> SyscallResult {
    if request.version != abi::VERSION {
        return Err(Errno::ENOTSUP);
    }
    // In-guest compilation belongs to device load time, before any service
    // timing, preemption guard, or GPU spinlock acquisition.
    super::compiler::wait()?;
    // Keep the spinlock owner runnable until its synchronous GPU transaction
    // completes. Interrupts remain enabled for timers and cross-core work.
    let start = unsafe { core::arch::x86_64::_rdtsc() };
    let _preempt = crate::process::core_local_storage::PreemptGuard::new();
    let mut device = DEVICE.get().ok_or(Errno::ENOTSUP)?.lock();
    let mut compute = device.compute.take().ok_or(Errno::ENOTSUP)?;
    let queue_before = compute.queue_profile.cycles;
    compute.defer_launch = compute.command_scopes.contains_key(&owner) && !compute.device_timing;
    let result = if compute.healthy {
        serve(&device, &mut compute, owner, request)
    } else {
        Err(Errno::EUNKN)
    };
    let queue_cycles = compute.queue_profile.cycles.wrapping_sub(queue_before);
    if let Some(entry) = compute.service_profile.get_mut(request.operation as usize) {
        entry.calls = entry.calls.wrapping_add(1);
        entry.cycles = entry.cycles.wrapping_add(unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start));
        entry.queue_cycles = entry.queue_cycles.wrapping_add(queue_cycles);
    }
    device.compute = Some(compute);
    result
}

fn serve(device: &Device, compute: &mut super::rm::Compute, owner: Uuid, request: &Request) -> SyscallResult {
    match request.operation {
        abi::SERVICE_PROFILE => {
            if request.elements != abi::SERVICE_OPERATIONS as u64 {
                return Err(Errno::EINVAL);
            }
            gpu_result(compute.flush_pending(device))?;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    compute.service_profile.as_ptr().cast::<u8>(),
                    request.pointer as *mut u8,
                    size_of_val(&compute.service_profile),
                );
            }
            Ok(0)
        }
        abi::BEGIN_COMMANDS => {
            let depth = compute.command_scopes.entry(owner).or_default();
            *depth = depth.checked_add(1).ok_or(Errno::EINVAL)?;
            Ok(0)
        }
        abi::END_COMMANDS => {
            if request.elements & !abi::DEFER_NESTED_COMPLETION != 0 {
                return Err(Errno::EINVAL);
            }
            let depth = compute.command_scopes.get(&owner).copied().ok_or(Errno::EINVAL)?;
            if depth == 1 || request.elements == 0 {
                gpu_result(compute.flush_pending(device))?;
            }
            if depth == 1 {
                compute.command_scopes.remove(&owner);
            } else {
                compute.command_scopes.insert(owner, depth - 1);
            }
            Ok(0)
        }
        abi::MEMORY_PROFILE => {
            if request.elements != 1 {
                return Err(Errno::EINVAL);
            }
            gpu_result(compute.flush_pending(device))?;
            let profile = abi::MemoryProfile {
                process_live_bytes: compute.buffers.live.values().filter(|b| b.owner == owner).map(|b| b.allocation.bytes).sum(),
                device_cached_bytes: compute.buffers.free.iter().map(|a| a.bytes).sum(),
                device_retired_bytes: compute.buffers.retired.iter().map(|(a, _)| a.bytes).sum(),
            };
            unsafe {
                (request.pointer as *mut abi::MemoryProfile).write_unaligned(profile);
            }
            Ok(0)
        }
        abi::QUEUE_PROFILE => {
            if request.elements != 1 {
                return Err(Errno::EINVAL);
            }
            gpu_result(compute.flush_pending(device))?;
            unsafe {
                (request.pointer as *mut abi::QueueProfile).write_unaligned(compute.queue_profile);
            }
            Ok(0)
        }
        abi::DEVICE_TIMING => {
            if request.elements > 1 {
                return Err(Errno::EINVAL);
            }
            gpu_result(compute.flush_pending(device))?;
            compute.device_timing = request.elements != 0;
            Ok(0)
        }
        abi::PROFILE => {
            if request.elements != abi::KERNEL_COUNT as u64 {
                return Err(Errno::EINVAL);
            }
            gpu_result(compute.flush_pending(device))?;
            // sys_gpu_compute validated the complete writable user range.
            unsafe {
                core::ptr::copy_nonoverlapping(compute.profile.as_ptr().cast::<u8>(), request.pointer as *mut u8, size_of_val(&compute.profile));
            }
            Ok(0)
        }
        abi::INFO => {
            // The complete inference module must exist; a register probe alone
            // is not an operational userspace backend.
            inference_module(compute)?;
            Ok(abi::VERSION as usize)
        }
        abi::ALLOCATE => {
            let bytes = request
                .elements
                .checked_mul(4)
                .filter(|&n| n != 0)
                .and_then(|n| n.checked_add(4095))
                .ok_or(Errno::EINVAL)?
                & !4095;
            if request.elements > u32::MAX as u64 {
                return Err(Errno::EINVAL);
            }
            let allocation = allocate(device, compute, bytes)?;
            inference_module(compute)?;
            // Clear fresh and recycled storage before exposing a handle. The
            // queue completes before another process can reuse this allocation.
            gpu_result(compute.dispatch(device, Kernel::Zero, &[allocation.virtual_address], &[(allocation.bytes / 4) as u32]))?;
            compute.buffers.next = compute
                .buffers
                .next
                .checked_add(1)
                .filter(|&n| n <= isize::MAX as u64)
                .ok_or(Errno::ENOHANDLES)?;
            let handle = compute.buffers.next;
            compute.buffers.live.insert(
                handle,
                Buffer {
                    owner,
                    allocation,
                    elements: request.elements,
                },
            );
            Ok(handle as usize)
        }
        abi::FREE => {
            let buffer = compute.buffers.get(request.handle, owner)?;
            let retire = compute.defer_launch && compute.pending_commands();
            if !retire {
                gpu_result(compute.flush_pending(device))?;
            }
            compute.buffers.live.remove(&request.handle);
            if retire {
                let sequence = compute.retirement_fence().ok_or(Errno::EUNKN)?;
                compute.buffers.retired.push((buffer.allocation, sequence));
            } else {
                compute.buffers.recycle(buffer.allocation);
            }
            Ok(0)
        }
        abi::UPLOAD | abi::DOWNLOAD => {
            let buffer = compute.buffers.get(request.handle, owner)?;
            if request.offset.checked_add(request.elements).is_none_or(|end| end > buffer.elements) {
                return Err(Errno::EINVAL);
            }
            // The syscall boundary validated the CPU range; handle ownership
            // and the full GPU extent were checked before any staging access.
            gpu_result(unsafe {
                compute.transfer(
                    device,
                    buffer.allocation.virtual_address + request.offset * 4,
                    request.pointer as *mut u8,
                    request.elements as usize,
                    request.operation == abi::UPLOAD,
                )
            })?;
            Ok(0)
        }
        abi::LAUNCH | abi::LAUNCH_ALLOCATE => {
            let kernel = Kernel::from_raw(request.kernel).map_err(|_| Errno::ENOTSUP)?;
            let parameters = request.parameters.get(..request.parameter_count as usize).ok_or(Errno::EINVAL)?;
            let work = kernel.work(parameters).map_err(|_| Errno::EINVAL)?;
            if request.buffer_count as usize != work.pointers {
                return Err(Errno::EINVAL);
            }
            let produce = request.operation == abi::LAUNCH_ALLOCATE;
            if produce
                && (!kernel.allocates_output()
                    || request.elements != work.extents[work.output]
                    || request.buffers[work.output] != 0
                    || request.offsets[work.output] != 0)
            {
                return Err(Errno::EINVAL);
            }
            let mut addresses = [0u64; 16];
            for (i, address) in addresses.iter_mut().enumerate().take(work.pointers) {
                if produce && i == work.output {
                    continue;
                }
                let buffer = compute.buffers.get(request.buffers[i], owner)?;
                if request.offsets[i].checked_add(work.extents[i]).is_none_or(|end| end > buffer.elements) {
                    return Err(Errno::EINVAL);
                }
                *address = buffer.allocation.virtual_address + request.offsets[i] * 4;
            }
            inference_module(compute)?;
            let output = if produce {
                // All inputs were checked before allocation. Do not publish the
                // new handle until a proven full writer is accepted. Scoped
                // readback and scope completion wait for that writer to finish.
                let bytes = request.elements.checked_mul(4).and_then(|n| n.checked_add(4095)).ok_or(Errno::EINVAL)? & !4095;
                let handle = compute
                    .buffers
                    .next
                    .checked_add(1)
                    .filter(|&n| n <= isize::MAX as u64)
                    .ok_or(Errno::ENOHANDLES)?;
                let allocation = allocate(device, compute, bytes)?;
                addresses[work.output] = allocation.virtual_address;
                Some((handle, allocation))
            } else {
                None
            };
            if work.validate_aliases(&addresses[..work.pointers]).is_err() {
                if let Some((_, allocation)) = output {
                    compute.buffers.recycle(allocation);
                }
                return Err(Errno::EINVAL);
            }
            gpu_result(compute.dispatch(device, kernel, &addresses[..work.pointers], parameters))?;
            if let Some((handle, allocation)) = output {
                compute.buffers.next = handle;
                compute.buffers.live.insert(
                    handle,
                    Buffer {
                        owner,
                        allocation,
                        elements: request.elements,
                    },
                );
                Ok(handle as usize)
            } else {
                Ok(0)
            }
        }
        _ => Err(Errno::EINVAL),
    }
}

fn allocate(device: &Device, compute: &mut super::rm::Compute, bytes: u64) -> Result<Allocation, Errno> {
    if let Some(allocation) = compute.buffers.take_free(bytes) {
        return Ok(allocation);
    }
    // Keep page-table changes outside live GPU work. Completion can also make
    // a retired allocation reusable without expanding the address space.
    gpu_result(compute.flush_pending(device))?;
    if let Some(allocation) = compute.buffers.take_free(bytes) {
        return Ok(allocation);
    }
    compute.memory.allocate_buffer(device, bytes).map_err(|_| Errno::ENOMEM)
}

fn gpu_result(result: super::Result<()>) -> Result<(), Errno> {
    result.map_err(|error| {
        log::warn!("NVIDIA compute operation failed: {error}");
        error.errno()
    })
}

fn inference_module(compute: &mut super::rm::Compute) -> Result<&'static [u8], Errno> {
    if let Some(module) = compute.inference_module {
        return Ok(module);
    }
    let module = super::compiler::inference().ok_or(Errno::ENOTSUP)?;
    compute.inference_module = Some(module);
    Ok(module)
}
