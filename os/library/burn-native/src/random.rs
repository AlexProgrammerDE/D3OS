//! Seeded GPU initialization and portable random-state checkpoints.
use crate::{NativeTensor, layout::Layout};
use alloc::sync::Arc;
use burn_backend::{BoolStore, DType, Distribution, Shape};
use d3os_compute::{Backend, Error, Result, nvidia::Nvidia};
use d3os_nvidia::compute::{
    Kernel,
    random::{self as abi, Mode},
};
use spin::Mutex;

const STATE_TAG: [u8; 8] = *b"D3RNG001";

/// Philox4x32-10 key and the next four-word block in its 64-bit stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RandomState {
    pub seed: u64,
    pub next_block: u64,
}

impl RandomState {
    pub fn to_bytes(self) -> [u8; 24] {
        let mut bytes = [0; 24];
        bytes[..8].copy_from_slice(&STATE_TAG);
        bytes[8..16].copy_from_slice(&self.seed.to_le_bytes());
        bytes[16..].copy_from_slice(&self.next_block.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: [u8; 24]) -> Result<Self> {
        if bytes[..8] != STATE_TAG {
            return Err(Error::InvalidParameter);
        }
        let mut seed = [0; 8];
        let mut next = [0; 8];
        seed.copy_from_slice(&bytes[8..16]);
        next.copy_from_slice(&bytes[16..]);
        Ok(Self {
            seed: u64::from_le_bytes(seed),
            next_block: u64::from_le_bytes(next),
        })
    }
}

/// A serialized stream. Distinct generators can retain independent states.
#[derive(Debug)]
pub struct RandomGenerator {
    state: Mutex<RandomState>,
}

pub(crate) static DEVICE_RANDOM: RandomGenerator = RandomGenerator::new(0);

impl RandomGenerator {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: Mutex::new(RandomState { seed, next_block: 0 }),
        }
    }

    pub fn state(&self) -> RandomState {
        *self.state.lock()
    }

    /// Restore at a training/checkpoint boundary, after all users of the stream
    /// have stopped submitting operations.
    pub fn restore(&self, state: RandomState) {
        *self.state.lock() = state;
    }

    pub fn seed(&self, seed: u64) {
        self.restore(RandomState { seed, next_block: 0 });
    }

    fn reserve<T>(&self, blocks: u64, produce: impl FnOnce(RandomState) -> Result<T>) -> Result<T> {
        let mut state = self.state.lock();
        let next = state.next_block.checked_add(blocks).ok_or(Error::SizeOverflow)?;
        let result = produce(*state)?;
        state.next_block = next;
        Ok(result)
    }

    /// Generate values entirely on the GPU, using Burn's distribution types.
    /// Invalid requests and failed submissions leave the stream state intact.
    pub fn random(&self, shape: Shape, dtype: DType, distribution: Distribution) -> Result<NativeTensor> {
        let plan = Plan::new(shape, dtype, distribution)?;
        if plan.layout.is_empty() {
            return NativeTensor::empty(plan.layout.shape().clone(), dtype);
        }
        self.reserve((plan.layout.len() as u64).div_ceil(4), |state| {
            let parameters = plan.parameters(state)?;
            Nvidia.with_commands(|| {
                let buffer = Nvidia.produce_words(plan.layout.len(), Kernel::Random, &[], &parameters)?;
                let output = NativeTensor {
                    buffer: Arc::new(buffer),
                    layout: plan.layout,
                    dtype: plan.storage_dtype,
                };
                if plan.storage_dtype == dtype { Ok(output) } else { output.cast(dtype) }
            })
        })
    }
}

impl NativeTensor {
    /// Use the device stream controlled by `NativeDevice::seed` and its checkpoint.
    pub fn random(shape: Shape, dtype: DType, distribution: Distribution) -> Result<Self> {
        DEVICE_RANDOM.random(shape, dtype, distribution)
    }
}

struct Plan {
    layout: Layout,
    mode: Mode,
    arguments: [u32; 3],
    storage_dtype: DType,
}

impl Plan {
    fn new(shape: Shape, dtype: DType, distribution: Distribution) -> Result<Self> {
        let layout = Layout::contiguous(shape)?;
        if !matches!(dtype, DType::F32 | DType::I32 | DType::Bool(BoolStore::U32)) {
            return Err(Error::InvalidParameter);
        }
        let mut storage_dtype = dtype;
        let (mode, arguments) = match distribution {
            Distribution::Default if dtype == DType::I32 => (Mode::Words, [0; 3]),
            Distribution::Default if dtype == DType::Bool(BoolStore::U32) => (Mode::Bernoulli, [0x8000_0000, 0, 0]),
            Distribution::Default => (Mode::Uniform, [0, 1f32.to_bits(), abi::predecessor(1f32.to_bits())]),
            Distribution::Uniform(a, b) if dtype == DType::I32 => {
                if !a.is_finite() || !b.is_finite() || a < i32::MIN as f64 || b > i32::MAX as f64 {
                    return Err(Error::InvalidParameter);
                }
                let (a, b) = (a as i32, b as i32);
                if a >= b {
                    return Err(Error::InvalidParameter);
                }
                let width = (i64::from(b) - i64::from(a)) as u32;
                (Mode::IntegerUniform, [a as u32, width, width.wrapping_neg() % width])
            }
            Distribution::Uniform(a, b) if dtype == DType::F32 => {
                if !a.is_finite() || !b.is_finite() {
                    return Err(Error::InvalidParameter);
                }
                let (a, b) = (a as f32, b as f32);
                (Mode::Uniform, [a.to_bits(), b.to_bits(), abi::predecessor(b.to_bits())])
            }
            Distribution::Normal(mean, deviation) if dtype != DType::Bool(BoolStore::U32) => {
                if !mean.is_finite() || !deviation.is_finite() || deviation < 0.0 {
                    return Err(Error::InvalidParameter);
                }
                storage_dtype = DType::F32;
                (Mode::Normal, [(mean as f32).to_bits(), (deviation as f32).to_bits(), 0])
            }
            Distribution::Bernoulli(probability) => {
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(Error::InvalidParameter);
                }
                let threshold = (probability * 4294967296.0) as u64;
                if dtype == DType::F32 {
                    storage_dtype = DType::I32;
                }
                (Mode::Bernoulli, [threshold as u32, (threshold >> 32) as u32, 0])
            }
            _ => return Err(Error::InvalidParameter),
        };
        let result = Self {
            layout,
            mode,
            arguments,
            storage_dtype,
        };
        // Empty outputs still validate their dtype and distribution, but do not
        // reserve counters or ask the driver to launch zero element work.
        let mut parameters = result.parameters_unchecked(RandomState::default());
        parameters[0] = parameters[0].max(1);
        abi::validate(&parameters.map(|v| v as u32)).map_err(|_| Error::InvalidParameter)?;
        Ok(result)
    }

    fn parameters_unchecked(&self, state: RandomState) -> [usize; 9] {
        [
            self.layout.len(),
            state.seed as u32 as usize,
            (state.seed >> 32) as usize,
            state.next_block as u32 as usize,
            (state.next_block >> 32) as usize,
            self.mode as usize,
            self.arguments[0] as usize,
            self.arguments[1] as usize,
            self.arguments[2] as usize,
        ]
    }

    fn parameters(&self, state: RandomState) -> Result<[usize; 9]> {
        let parameters = self.parameters_unchecked(state);
        abi::validate(&parameters.map(|v| v as u32)).map_err(|_| Error::InvalidParameter)?;
        Ok(parameters)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::{thread, vec::Vec};

    #[test]
    fn checkpoints_and_failed_reservations_preserve_the_exact_stream() {
        let generator = RandomGenerator::new(0x0123_4567_89ab_cdef);
        let first = generator.reserve(7, Ok).unwrap();
        assert_eq!(
            first,
            RandomState {
                seed: 0x0123_4567_89ab_cdef,
                next_block: 0
            }
        );
        let checkpoint = generator.state();
        assert_eq!(checkpoint.next_block, 7);
        assert_eq!(RandomState::from_bytes(checkpoint.to_bytes()).unwrap(), checkpoint);
        let mut corrupt = checkpoint.to_bytes();
        corrupt[3] ^= 1;
        assert_eq!(RandomState::from_bytes(corrupt), Err(Error::InvalidParameter));
        assert_eq!(generator.reserve(19, |_| Err::<(), _>(Error::AllocationFailed)), Err(Error::AllocationFailed));
        assert_eq!(generator.state(), checkpoint);
        let next = generator.reserve(31, Ok).unwrap();
        generator.restore(checkpoint);
        assert_eq!(generator.reserve(31, Ok).unwrap(), next);
        generator.restore(RandomState {
            seed: u64::MAX,
            next_block: u64::MAX - 1,
        });
        assert_eq!(generator.reserve(2, |_| panic!()), Err::<(), _>(Error::SizeOverflow));
        assert_eq!(generator.state().next_block, u64::MAX - 1);
        generator.reserve(1, |_| Ok(())).unwrap();
        assert_eq!(generator.reserve(1, |_| panic!()), Err::<(), _>(Error::SizeOverflow));
        generator.seed(42);
        assert_eq!(generator.state(), RandomState { seed: 42, next_block: 0 });
    }

    #[test]
    fn concurrent_reservations_never_overlap_or_lose_counter_updates() {
        let generator = Arc::new(RandomGenerator::new(93));
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let generator = generator.clone();
                let ranges = ranges.clone();
                thread::spawn(move || {
                    for n in 0..127 {
                        let count = (n % 4 + 1) as u64;
                        generator
                            .reserve(count, |state| {
                                assert_eq!(state.seed, 93);
                                ranges.lock().push((state.next_block, count));
                                Ok(())
                            })
                            .unwrap();
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        let mut ranges = ranges.lock();
        ranges.sort_unstable();
        let mut next = 0;
        for &(start, length) in ranges.iter() {
            assert_eq!(start, next);
            next += length;
        }
        assert_eq!(ranges.len(), 8 * 127);
        assert_eq!(generator.state().next_block, next);
    }

    #[test]
    fn distribution_plans_reject_invalid_ranges_and_validate_counter_carries() {
        for (dtype, distribution) in [
            (DType::F32, Distribution::Uniform(-f32::MAX as f64, f32::MAX as f64)),
            (DType::F32, Distribution::Uniform(0.0, f32::from_bits(1) as f64)),
            (DType::F32, Distribution::Normal(-0.0, 0.0)),
            (DType::F32, Distribution::Normal(7.0, 3.0)),
            (DType::F32, Distribution::Bernoulli(0.0)),
            (DType::F32, Distribution::Bernoulli(1.0)),
            (DType::I32, Distribution::Default),
            (DType::I32, Distribution::Uniform(i32::MIN as f64, i32::MAX as f64)),
            (DType::I32, Distribution::Uniform(-2147483648.0, 1.0)),
            (DType::I32, Distribution::Normal(0.0, 1.0)),
            (DType::Bool(BoolStore::U32), Distribution::Default),
        ] {
            let plan = Plan::new(Shape::new([5, 129]), dtype, distribution).unwrap();
            let p = plan
                .parameters(RandomState {
                    seed: u64::MAX,
                    next_block: u32::MAX as u64 - 1,
                })
                .unwrap()
                .map(|v| v as u32);
            let work = Kernel::Random.work(&p).unwrap();
            assert_eq!((work.pointers, work.output, work.blocks), (1, 0, 6));
            assert_eq!(work.extents[0], 645);
            assert_eq!(p[1..5], [u32::MAX, u32::MAX, u32::MAX - 1, 0]);
            assert!(
                plan.parameters(RandomState {
                    seed: 0,
                    next_block: u64::MAX - 160
                })
                .is_err()
            );
            let mut malformed = p;
            malformed[5] = 5;
            assert!(Kernel::Random.work(&malformed).is_err());
            assert!(Kernel::Random.work(&p[..8]).is_err());
        }
        for (dtype, distribution) in [
            (DType::F32, Distribution::Uniform(1.0, 1.0)),
            (DType::F32, Distribution::Uniform(2.0, 1.0)),
            (DType::F32, Distribution::Uniform(f64::NAN, 1.0)),
            (DType::F32, Distribution::Uniform(0.0, f64::MAX)),
            (DType::F32, Distribution::Normal(0.0, -0.1)),
            (DType::F32, Distribution::Normal(f64::INFINITY, 1.0)),
            (DType::I32, Distribution::Uniform(1.1, 1.9)),
            (DType::I32, Distribution::Uniform(i32::MIN as f64 - 1.0, 3.0)),
            (DType::I32, Distribution::Uniform(0.0, i32::MAX as f64 + 1.0)),
            (DType::F32, Distribution::Bernoulli(-0.1)),
            (DType::F32, Distribution::Bernoulli(1.1)),
            (DType::F32, Distribution::Bernoulli(f64::NAN)),
            (DType::Bool(BoolStore::U32), Distribution::Normal(0.0, 1.0)),
            (DType::F64, Distribution::Default),
        ] {
            assert!(Plan::new(Shape::new([1]), dtype, distribution).is_err());
            assert!(Plan::new(Shape::new([0]), dtype, distribution).is_err());
        }
        assert!(Kernel::Random.allocates_output());
    }
}
