//! Burn-based model APIs shared by D3OS applications and host validation.
#![no_std]

extern crate alloc;

pub use burn;

pub mod mnist;
#[cfg(target_os = "none")]
mod platform;
#[cfg(any(test, target_os = "none"))]
mod timer_queue;
#[cfg(feature = "training")]
pub mod validation;
