#![cfg(all(feature = "nvidia", not(target_os = "none")))]

use d3os_compute::{Error, nvidia::Nvidia};

#[test]
fn host_build_never_issues_d3os_syscalls_or_falls_back_to_cpu() {
    assert!(matches!(Nvidia::new(), Err(Error::DeviceUnavailable)));
}
