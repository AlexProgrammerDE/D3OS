#[cfg(not(feature = "std"))]
use spin::{Mutex as MutexImported, MutexGuard, Once as OnceImported, RwLock as RwLockImported};
#[cfg(feature = "std")]
use std::sync::{
    Mutex as MutexImported, MutexGuard, OnceLock as OnceImported, RwLock as RwLockImported,
};

#[cfg(not(feature = "std"))]
pub use spin::{LazyLock, RwLockReadGuard, RwLockWriteGuard};
#[cfg(feature = "std")]
pub use std::sync::{LazyLock, RwLockReadGuard, RwLockWriteGuard};

/// A spin-based one-time initialization cell, identical on every target.
///
/// Prefer [`SyncOnceCell`] for plain lazy initialization; use this when the
/// fallible [`spin::Once::try_call_once`] API is needed.
pub use spin::Once;

#[cfg(target_has_atomic = "ptr")]
pub use alloc::sync::Arc;
#[cfg(not(target_has_atomic = "ptr"))]
pub use portable_atomic_util::Arc;

#[cfg(target_has_atomic = "ptr")]
pub use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering};
#[cfg(not(target_has_atomic = "ptr"))]
pub use portable_atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering};

// No `AtomicU64`: `target_has_atomic = "ptr"` says nothing about 64-bit
// atomics, and a 32-bit target can have pointer-width CAS without them
// (thumbv7m). Its one user is behind `stream_local`, which is `std`, and every
// std target has them — so it takes `core`'s directly rather than making this
// shim carry a width it cannot provide on both arms.

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will block threads waiting for the lock to become available. The
/// mutex can also be statically initialized or created via a [`Mutex::new`]
///
/// [Mutex] wrapper to make `spin::Mutex` API compatible with `std::sync::Mutex` to swap
#[derive(Debug, Default)]
pub struct Mutex<T> {
    inner: MutexImported<T>,
}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: MutexImported::new(value),
        }
    }

    /// Locks the mutex blocking the current thread until it is able to do so.
    ///
    /// Locking cannot fail, so no `Result` is returned. A poisoned lock is
    /// recovered rather than reported: a panic under a guard must not turn
    /// every later lock into a panic, matching the unwind behavior of the spin
    /// implementation used off-std.
    #[inline(always)]
    pub fn lock(&self) -> MutexGuard<'_, T> {
        #[cfg(not(feature = "std"))]
        {
            self.inner.lock()
        }

        #[cfg(feature = "std")]
        {
            self.inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }

    /// Consumes the mutex, returning the value it protected. A poisoned lock
    /// is recovered, as in [`lock`](Self::lock).
    #[inline(always)]
    pub fn into_inner(self) -> T {
        #[cfg(not(feature = "std"))]
        {
            self.inner.into_inner()
        }

        #[cfg(feature = "std")]
        {
            self.inner
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }
}

/// A reader-writer lock which is exclusively locked for writing or shared for reading.
/// This reader-writer lock will block threads waiting for the lock to become available.
/// The lock can also be statically initialized or created via a [`RwLock::new`]
/// [`RwLock`] wrapper to make `spin::RwLock` API compatible with `std::sync::RwLock` to swap
#[derive(Debug)]
pub struct RwLock<T> {
    inner: RwLockImported<T>,
}

impl<T> RwLock<T> {
    /// Creates a new reader-writer lock in an unlocked state ready for use.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: RwLockImported::new(value),
        }
    }

    /// Locks this rwlock with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// Poisoning is recovered, never reported; see [`Mutex::lock`].
    #[inline(always)]
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        #[cfg(not(feature = "std"))]
        {
            self.inner.read()
        }
        #[cfg(feature = "std")]
        {
            self.inner
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }

    /// Locks this rwlock with exclusive write access, blocking the current thread
    /// until it can be acquired.
    ///
    /// Poisoning is recovered, never reported; see [`Mutex::lock`].
    #[inline(always)]
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        #[cfg(not(feature = "std"))]
        {
            self.inner.write()
        }

        #[cfg(feature = "std")]
        {
            self.inner
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }
}

/// An opaque thread identifier supplied by the host platform.
///
/// This type and its provider are available with and without `std`.
/// They do not convert or replace `std::thread::ThreadId`.
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct ThreadId(core::num::NonZeroU64);

/// A thread-id provider is already installed for this process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlreadyInstalled;

impl core::fmt::Display for AlreadyInstalled {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a thread-id provider is already installed")
    }
}

impl core::error::Error for AlreadyInstalled {}

static THREAD_ID_PROVIDER: Mutex<Option<fn() -> core::num::NonZeroU64>> = Mutex::new(None);

/// Install the process-wide provider for [`current_thread_id`].
///
/// Install the provider before worker threads use thread-keyed state.
/// Exactly one registration succeeds. Every later registration returns
/// [`AlreadyInstalled`], including a registration of the same function.
///
/// The provider must return one stable identity for each OS thread.
/// Identities must remain distinct for the process lifetime, including after
/// thread exit, because shared state can retain keys from terminated threads.
/// CPU migration, device selection, and asynchronous task changes must not
/// change the identity of an OS thread.
///
/// The provider must fail if the platform cannot identify the current thread.
/// It must not call [`current_thread_id`] or thread-keyed module operations.
/// No standard-library, single-thread, or stream identity fallback is supplied.
pub fn register_thread_id_provider(
    provider: fn() -> core::num::NonZeroU64,
) -> Result<(), AlreadyInstalled> {
    let mut installed = THREAD_ID_PROVIDER.lock();
    if installed.is_some() {
        return Err(AlreadyInstalled);
    }
    *installed = Some(provider);
    Ok(())
}

/// Resolve the current OS thread through the installed platform provider.
///
/// The registration lock is released before the provider is called.
/// This function also requires registration when `std` is enabled.
/// A lookup that precedes registration does not wait for an installer.
///
/// # Panics
///
/// Panics if no provider is installed or if the provider fails.
pub fn current_thread_id() -> ThreadId {
    let provider = { *THREAD_ID_PROVIDER.lock() };
    let provider = provider.expect(
        "Current thread identity is unsupported: no thread-id provider is installed",
    );
    ThreadId(provider())
}

/// A cell that provides lazy one-time initialization that implements [Sync] and [Send].
///
/// This module is a stub when no std is available to swap with [`std::sync::OnceLock`].
pub struct SyncOnceCell<T>(OnceImported<T>);

impl<T> Default for SyncOnceCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SyncOnceCell<T> {
    /// Create a new once.
    #[inline(always)]
    pub fn new() -> Self {
        Self(OnceImported::new())
    }

    /// Initialize the cell with a value.
    #[inline(always)]
    pub fn initialized(value: T) -> Self {
        #[cfg(not(feature = "std"))]
        {
            let cell = OnceImported::initialized(value);
            Self(cell)
        }

        #[cfg(feature = "std")]
        {
            let cell = OnceImported::new();
            // Infallible: the cell was just created, so it is empty. Ignoring
            // the `Err` is what keeps `T: Debug` off this whole impl.
            let _ = cell.set(value);

            Self(cell)
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell
    /// was empty.
    #[inline(always)]
    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        #[cfg(not(feature = "std"))]
        {
            self.0.call_once(f)
        }

        #[cfg(feature = "std")]
        {
            self.0.get_or_init(f)
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Regression: a panic under a guard must not poison the lock — one
    /// failed autotune must not take down every later kernel launch.
    #[test]
    fn poisoned_mutex_recovers() {
        let mutex = Mutex::new(0u32);

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = mutex.lock();
            panic!("poison the lock");
        }))
        .unwrap_err();

        *mutex.lock() += 1;
        assert_eq!(*mutex.lock(), 1);
    }

    #[test]
    fn poisoned_rwlock_recovers() {
        let lock = RwLock::new(0u32);

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = lock.write();
            panic!("poison the lock");
        }))
        .unwrap_err();

        assert_eq!(*lock.read(), 0);
    }
}
