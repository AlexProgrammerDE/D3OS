use burn_backend::{BoolStore, DType, Device, DeviceId, DeviceOps, DeviceSettings};
use d3os_compute::{Result, nvidia::Nvidia};

/// The process-owned GPU exposed by D3OS. No host CUDA runtime is involved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeDevice;

impl NativeDevice {
    pub fn seed(&self, seed: u64) {
        crate::random::DEVICE_RANDOM.seed(seed);
    }

    pub fn random_state(&self) -> crate::random::RandomState {
        crate::random::DEVICE_RANDOM.state()
    }

    pub fn restore_random_state(&self, state: crate::random::RandomState) {
        crate::random::DEVICE_RANDOM.restore(state);
    }

    pub fn open() -> Result<Self> {
        Nvidia::new()?;
        Ok(Self)
    }
}

impl Device for NativeDevice {
    fn from_id(id: DeviceId) -> Self {
        assert_eq!(id, DeviceId::new(0, 0), "D3OS exposes one native GPU");
        Self
    }

    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, 0)
    }
}

impl DeviceOps for NativeDevice {
    fn defaults(&self) -> DeviceSettings {
        DeviceSettings::with_dtypes(DType::F32, DType::I32, BoolStore::U32)
    }
}
