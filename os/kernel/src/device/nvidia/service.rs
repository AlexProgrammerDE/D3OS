//! Synchronous, process-owned compute buffers and built-in GPU dispatch.
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
}

impl Buffers {
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
    let mut device = device.lock();
    let Some(compute) = device.compute.as_mut() else {
        return;
    };
    if !compute.healthy {
        return;
    } // A faulted GPU may still reference its buffers.
    let handles: Vec<_> = compute.buffers.live.iter().filter(|(_, b)| b.owner == owner).map(|(&h, _)| h).collect();
    for handle in handles {
        if let Some(buffer) = compute.buffers.live.remove(&handle) {
            compute.buffers.recycle(buffer.allocation);
        }
    }
}

pub(crate) fn request(owner: Uuid, request: &Request) -> SyscallResult {
    if request.version != abi::VERSION {
        return Err(Errno::ENOTSUP);
    }
    let mut device = DEVICE.get().ok_or(Errno::ENOTSUP)?.lock();
    let mut compute = device.compute.take().ok_or(Errno::ENOTSUP)?;
    let result = if compute.healthy {
        serve(&device, &mut compute, owner, request)
    } else {
        Err(Errno::EUNKN)
    };
    device.compute = Some(compute);
    result
}

fn serve(device: &Device, compute: &mut super::rm::Compute, owner: Uuid, request: &Request) -> SyscallResult {
    let vram = device.vram.as_ref().ok_or(Errno::ENOTSUP)?;
    match request.operation {
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
            let allocation = if let Some((index, _)) = compute
                .buffers
                .free
                .iter()
                .enumerate()
                .filter(|(_, b)| b.bytes >= bytes)
                .min_by_key(|(_, b)| b.bytes)
            {
                let mut allocation = compute.buffers.free.swap_remove(index);
                if allocation.bytes > bytes {
                    compute.buffers.free.push(Allocation {
                        physical: allocation.physical + bytes,
                        virtual_address: allocation.virtual_address + bytes,
                        bytes: allocation.bytes - bytes,
                    });
                    allocation.bytes = bytes;
                }
                // Reused storage must not expose another process's weights or data.
                for offset in (0..allocation.bytes).step_by(4) {
                    vram.write((allocation.physical + offset) as usize, 0).map_err(|_| Errno::EUNKN)?;
                }
                allocation
            } else {
                // Memory::allocate clears new VRAM before installing its PTEs.
                compute.memory.allocate(device, bytes, 4096).map_err(|_| Errno::ENOMEM)?
            };
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
            compute.buffers.live.remove(&request.handle);
            compute.buffers.recycle(buffer.allocation);
            Ok(0)
        }
        abi::UPLOAD | abi::DOWNLOAD => {
            let buffer = compute.buffers.get(request.handle, owner)?;
            if request.offset.checked_add(request.elements).is_none_or(|end| end > buffer.elements) {
                return Err(Errno::EINVAL);
            }
            // The syscall boundary validates the userspace range before this call.
            for i in 0..request.elements {
                let physical = (buffer.allocation.physical + (request.offset + i) * 4) as usize;
                let pointer = (request.pointer as *mut u32).wrapping_add(i as usize);
                if request.operation == abi::UPLOAD {
                    vram.write(physical, unsafe { pointer.read_volatile() }).map_err(|_| Errno::EUNKN)?;
                } else {
                    unsafe {
                        pointer.write_volatile(vram.read(physical).ok_or(Errno::EUNKN)?);
                    }
                }
            }
            Ok(0)
        }
        abi::LAUNCH => {
            let kernel = Kernel::from_raw(request.kernel).map_err(|_| Errno::ENOTSUP)?;
            let parameters = request.parameters.get(..request.parameter_count as usize).ok_or(Errno::EINVAL)?;
            let work = kernel.work(parameters).map_err(|_| Errno::EINVAL)?;
            if request.buffer_count as usize != work.pointers {
                return Err(Errno::EINVAL);
            }
            let mut addresses = [0u64; 16];
            for (i, address) in addresses.iter_mut().enumerate().take(work.pointers) {
                let buffer = compute.buffers.get(request.buffers[i], owner)?;
                if request.offsets[i].checked_add(work.extents[i]).is_none_or(|end| end > buffer.elements) {
                    return Err(Errno::EINVAL);
                }
                *address = buffer.allocation.virtual_address + request.offsets[i] * 4;
            }
            work.validate_aliases(&addresses[..work.pointers]).map_err(|_| Errno::EINVAL)?;
            let mut arguments = Vec::with_capacity(work.pointers * 8 + parameters.len() * 4);
            for address in addresses.iter().take(work.pointers) {
                arguments.extend_from_slice(&address.to_le_bytes());
            }
            for parameter in parameters {
                arguments.extend_from_slice(&parameter.to_le_bytes());
            }
            let cubin = inference_module(compute)?;
            if let Err(error) = compute.launch(device, cubin, kernel.symbol(), &arguments, [work.threads, work.blocks]) {
                compute.healthy = false;
                log::warn!("NVIDIA inference dispatch failed: {error}");
                return Err(Errno::EUNKN);
            }
            Ok(0)
        }
        _ => Err(Errno::EINVAL),
    }
}

fn inference_module(compute: &mut super::rm::Compute) -> Result<&'static [u8], Errno> {
    if let Some(module) = compute.inference_module {
        return Ok(module);
    }
    let module = super::gsp::asset("usr/share/nvidia/inference.cubin").map_err(|_| Errno::ENOTSUP)?;
    compute.inference_module = Some(module);
    Ok(module)
}
