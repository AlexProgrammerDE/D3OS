//! Boot check for relocated F64 coefficient and large-argument reduction tables.
use super::{Device, Error, Result, rm::Compute};
use d3os_nvidia::{Field, Record};

// MPFR independent reference bits for the relocation probe only.
const INPUT_BITS: [u64; 12] = [
    0x0000000000000000,
    0x8000000000000000,
    0x3fe0000000000000,
    0xbfe0000000000000,
    0x3ff0000000000000,
    0xc000000000000000,
    0x4045200000000000,
    0x4202a05f20000000,
    0x54b249ad2594c37d,
    0x7e37e43c8800759c,
    0x400921fb54442d18,
    0x3ff921fb54442d18,
];
const EXPECTED_BITS: [u64; 24] = [
    0x0000000000000000,
    0x3ff0000000000000,
    0x8000000000000000,
    0x3ff0000000000000,
    0x3fdeaee8744b05f0,
    0x3fec1528065b7d50,
    0xbfdeaee8744b05f0,
    0x3fec1528065b7d50,
    0x3feaed548f090cee,
    0x3fe14a280fb5068c,
    0xbfed18f6ead1b446,
    0xbfdaa22657537205,
    0xbfef9565e4b49d24,
    0xbfc495157c78ea06,
    0xbfdf334c7896a4e3,
    0x3febf098901c931a,
    0xbfd85c5e5b929359,
    0x3fed9757496841f5,
    0xbfea2c16b010e385,
    0xbfe2699022adc4c1,
    0x3ca1a62633145c07,
    0xbff0000000000000,
    0x3ff0000000000000,
    0x3c91a62633145c07,
];

fn ordered(bits: u64) -> u64 {
    if bits >> 63 != 0 { !bits } else { bits | (1 << 63) }
}

impl Compute {
    pub(super) fn relocation_test(&mut self, device: &Device, cubin: &'static [u8]) -> Result<()> {
        const GUARD: u64 = 0x5a17_8ec3_216d_490b;
        const POISON: u64 = 0x7ff8_a617_39b4_8001;
        const OUTPUT: usize = 264;
        let allocation = self.memory.allocate(device, 4096, 4096)?;
        let vram = device.vram.as_ref().ok_or(Error::Firmware)?;
        let write = |offset: usize, bits: u64| -> Result<()> {
            vram.write(allocation.physical as usize + offset, bits as u32)?;
            vram.write(allocation.physical as usize + offset + 4, (bits >> 32) as u32)
        };
        let read = |offset: usize| -> Result<u64> {
            let low = vram.read(allocation.physical as usize + offset).ok_or(Error::Firmware)?;
            let high = vram.read(allocation.physical as usize + offset + 4).ok_or(Error::Firmware)?;
            Ok(u64::from(low) | (u64::from(high) << 32))
        };
        for offset in [0, 8 + INPUT_BITS.len() * 8, OUTPUT - 8, OUTPUT + EXPECTED_BITS.len() * 8] {
            write(offset, GUARD)?;
        }
        for (i, bits) in INPUT_BITS.into_iter().enumerate() {
            write(8 + i * 8, bits)?;
        }
        let images = self.program_images.len();
        let mut max_ulp = 0;
        let mut comparisons = 0;
        for count in [0, 1, 11, 12] {
            for i in 0..EXPECTED_BITS.len() {
                write(OUTPUT + i * 8, POISON)?;
            }
            let mut arguments = Record::<20>::default();
            arguments.set(Field::<u64>::new(0), allocation.virtual_address + 8)?;
            arguments.set(Field::<u64>::new(8), allocation.virtual_address + OUTPUT as u64)?;
            arguments.set(Field::<u32>::new(16), count)?;
            self.launch(device, cubin, "relocation_probe", arguments.as_bytes(), [128, 1])?;
            for (i, expected) in EXPECTED_BITS.into_iter().enumerate() {
                let actual = read(OUTPUT + i * 8)?;
                if i >= count as usize * 2 {
                    if actual != POISON {
                        return Err(Error::Firmware);
                    }
                    continue;
                }
                let distance = ordered(actual).abs_diff(ordered(expected));
                if actual & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 || distance > 1 || (expected & 0x7fff_ffff_ffff_ffff == 0 && actual != expected) {
                    log::warn!(
                        "NVIDIA_RELOCATION_FAIL input={} output={i} actual={actual:#018x} expected={expected:#018x} ulp={distance}",
                        i / 2
                    );
                    return Err(Error::Firmware);
                }
                max_ulp = max_ulp.max(distance);
                comparisons += 1;
            }
            for offset in [0, 8 + INPUT_BITS.len() * 8, OUTPUT - 8, OUTPUT + EXPECTED_BITS.len() * 8] {
                if read(offset)? != GUARD {
                    return Err(Error::Firmware);
                }
            }
            for (i, bits) in INPUT_BITS.into_iter().enumerate() {
                if read(8 + i * 8)? != bits {
                    return Err(Error::Firmware);
                }
            }
            // Timestamp and local-stack checks already uploaded this module.
            // A new entry and repeated invocations reuse that patched image.
            if self.program_images.len() != images {
                return Err(Error::Firmware);
            }
        }
        log::info!(
            "NVIDIA_RELOCATION_OK inputs=12 cases=4 comparisons={comparisons} max_ulp={max_ulp} table_addresses=gpu_virtual guards=ok tails=preserved source=preserved image_cache=reused backend=nvidia"
        );
        Ok(())
    }
}
