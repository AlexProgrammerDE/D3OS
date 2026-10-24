#![cfg(feature = "native")]

use burn_dispatch::{Dispatch, DispatchDevice, DispatchDeviceId};
use d3os_ai::{
    burn::backend::{Device, DeviceId},
    native::NativeDevice,
};

#[test]
fn native_device_round_trips_through_the_public_dispatch_registry() {
    let device = DispatchDevice::from(NativeDevice);
    let id = device.to_id();
    assert_eq!(id, DeviceId::new(12 << 8, 0));
    assert_eq!(DispatchDevice::from_id(id), device);
    assert_eq!(Dispatch::enumerate(DispatchDeviceId::Native), vec![device]);
}
