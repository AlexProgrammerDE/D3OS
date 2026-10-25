/* ╔═════════════════════════════════════════════════════════════════════════╗
   ║ Module: kheap                                                           ║
   ╟─────────────────────────────────────────────────────────────────────────╢
   ║ Allocator for the kernel heap.                                          ║
   ╟─────────────────────────────────────────────────────────────────────────╢
   ║ Author: Fabian Ruhland, Univ. Duesseldorf, 02.03.2025                   ║
   ╚═════════════════════════════════════════════════════════════════════════╝
*/
use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ptr::NonNull;
use talc::{DefaultBinning, base::Talc, source::Manual};
use x86_64::structures::paging::frame::PhysFrameRange;
use crate::memory::PAGE_SIZE;
use core::sync::atomic::{AtomicBool, AtomicUsize};
use crate::sync::irqsave_spinlock::IrqSaveSpinlock;

pub struct KernelAllocator {
    heap: IrqSaveSpinlock<Talc<Manual, DefaultBinning>>,
    initialized: AtomicBool,
}

// Heap capacity minus live requested allocations, excluding allocator overhead.
static FREE_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn get_free_bytes() -> usize {
    FREE_BYTES.load(core::sync::atomic::Ordering::SeqCst)
}   

impl KernelAllocator {
    pub const fn new() -> Self {
        Self { heap: IrqSaveSpinlock::new(Talc::new(Manual)), initialized: AtomicBool::new(false) }
    }

    pub unsafe fn init(&self, frames: &PhysFrameRange) {
        let mut heap = self.heap.lock();
        unsafe {
            heap.claim(frames.start.start_address().as_u64() as *mut u8, (frames.end - frames.start) as usize * PAGE_SIZE)
                .expect("kernel heap region cannot hold allocator metadata");
        }
        FREE_BYTES.store((frames.end - frames.start) as usize * PAGE_SIZE, core::sync::atomic::Ordering::SeqCst);
        self.initialized.store(true, core::sync::atomic::Ordering::SeqCst);
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(core::sync::atomic::Ordering::SeqCst)
    }

    pub fn is_locked(&self) -> bool {
        self.heap.is_locked()
    }
}

unsafe impl Allocator for KernelAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }

        match unsafe { self.heap.lock().allocate(layout) } {
            Some(ptr) => {
                FREE_BYTES.fetch_sub(layout.size(), core::sync::atomic::Ordering::SeqCst);
                Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
            },
            None => Err(AllocError),
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            let mut heap = self.heap.lock();
            unsafe { heap.deallocate(ptr.as_ptr(), layout); }
            FREE_BYTES.fetch_add(layout.size(), core::sync::atomic::Ordering::SeqCst);
        }
    }
}

unsafe impl GlobalAlloc for KernelAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Allocator::allocate(self, layout)
            .map_or(core::ptr::null_mut(), |allocation| allocation.as_ptr().cast())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { Allocator::deallocate(self, NonNull::new_unchecked(ptr), layout); }
    }
}

