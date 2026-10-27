//! D3OS process services for portable Burn modules.
#![no_std]

#[cfg(target_os = "none")]
use core::num::NonZeroU64;
#[cfg(target_os = "none")]
use cubecl_environment::sync::{AlreadyInstalled, register_thread_id_provider};

#[cfg(target_os = "none")]
static INITIALIZED: spin::Once<Result<(), AlreadyInstalled>> = spin::Once::new();

/// Install the D3OS thread identity provider before Burn module execution.
///
/// The Burn-enabled application runtime calls this after heap and thread setup.
/// Native devices also call it when opened. Repeated calls return the original
/// result. A provider installed by another component is never replaced.
#[cfg(target_os = "none")]
pub fn initialize() -> Result<(), AlreadyInstalled> {
    *INITIALIZED.call_once(|| register_thread_id_provider(current_thread_id))
}

#[cfg(target_os = "none")]
fn current_thread_id() -> NonZeroU64 {
    let thread = concurrent::thread::current().expect("D3OS cannot identify the current thread for Burn module state");
    let id = u64::try_from(thread.id()).expect("D3OS thread ID exceeds the Burn identity width");
    NonZeroU64::new(id).expect("D3OS returned a zero thread ID for Burn module state")
}
