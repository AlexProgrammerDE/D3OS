use burn_backend::{BoolStore, DType, Device, DeviceId, DeviceOps, DeviceSettings};
use d3os_compute::{Backend as _, Result, nvidia::Nvidia};

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

    /// Queue native Burn operations and wait for completion. The scope reports
    /// deferred tensor errors; callback errors still close the transaction.
    pub fn with_commands<T, E>(&self, work: impl FnOnce() -> core::result::Result<T, E>) -> core::result::Result<T, E>
    where
        E: From<crate::NativeError> + From<burn_backend::ExecutionError>,
    {
        let value = Nvidia.with_commands(work)?;
        crate::backend::check_errors().map_err(E::from)?;
        Ok(value)
    }

    pub fn open() -> Result<Self> {
        #[cfg(target_os = "none")]
        d3os_burn_platform::initialize().map_err(|_| d3os_compute::Error::DeviceUnavailable)?;
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
