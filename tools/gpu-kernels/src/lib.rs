//! GPU device code. Compile one kernel family at a time.
#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]
#![cfg_attr(any(feature = "tensor-matmul", feature = "inference"), feature(asm_experimental_arch))]

use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
use core::panic::PanicInfo;

const _: () = assert!(
    cfg!(feature = "vector-add") as usize
        + cfg!(feature = "relu") as usize
        + cfg!(feature = "matmul") as usize
        + cfg!(feature = "tensor-matmul") as usize
        + cfg!(feature = "inference") as usize
        == 1,
    "select exactly one GPU kernel feature"
);

#[cfg(feature = "tensor-matmul")]
mod tensor;

#[cfg(feature = "inference")]
mod inference;
#[cfg(feature = "inference")]
mod tf32;
#[cfg(feature = "inference")]
mod transformer;
#[cfg(feature = "inference")]
mod raster;

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[inline(always)]
fn index() -> u32 {
    unsafe { _block_idx_x() * _block_dim_x() + _thread_idx_x() }
}

/// # Safety
/// Addresses must refer to at least `count` f32 values. Output cannot alias input.
#[cfg(feature = "vector-add")]
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vector_add(a: u64, b: u64, out: u64, count: u32) {
    let (a, b, out) = (a as *const f32, b as *const f32, out as *mut f32);
    let i = index();
    if i < count {
        unsafe { out.add(i as usize).write(a.add(i as usize).read() + b.add(i as usize).read()) };
    }
}

/// # Safety
/// Input and output must address `count` f32 values and cannot overlap.
#[cfg(feature = "relu")]
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn relu(input: u64, out: u64, count: u32) {
    let (input, out) = (input as *const f32, out as *mut f32);
    let i = index();
    if i < count {
        let value = unsafe { input.add(i as usize).read() };
        unsafe { out.add(i as usize).write(if value < 0.0 { 0.0 } else { value }) };
    }
}

/// Row-major matrix multiplication, one output element per GPU thread.
/// # Safety
/// A, B, and output must address rows*inner, inner*cols, and rows*cols f32 values.
/// Output cannot alias either input. Products must fit u32.
#[cfg(feature = "matmul")]
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn matmul(a: u64, b: u64, out: u64, rows: u32, inner: u32, cols: u32) {
    let (a, b, out) = (a as *const f32, b as *const f32, out as *mut f32);
    if rows == 0 || inner == 0 || cols == 0 {
        return;
    }
    let i = index();
    if i < rows * cols {
        let row = i / cols;
        let col = i % cols;
        let mut sum = 0.0;
        for k in 0..inner {
            sum += unsafe { a.add((row * inner + k) as usize).read() * b.add((k * cols + col) as usize).read() };
        }
        unsafe { out.add(i as usize).write(sum) };
    }
}
