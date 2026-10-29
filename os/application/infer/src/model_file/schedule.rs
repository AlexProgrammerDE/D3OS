#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Variant {
    Control,
    Candidate,
}

impl Variant {
    pub(super) fn index(self) -> usize {
        match self {
            Self::Control => 0,
            Self::Candidate => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pass {
    pub variant: Variant,
    pub sample: Option<usize>,
    pub instrumented: bool,
}

impl Pass {
    pub(super) fn warm(self) -> bool {
        self.sample.is_some() || self.instrumented
    }
}

pub(super) fn passes(paired: bool, device_timing: bool) -> impl Iterator<Item = Pass> {
    let warmup = [Variant::Control, Variant::Candidate]
        .into_iter()
        .take(if paired { 2 } else { 1 })
        .map(|variant| Pass {
            variant,
            sample: None,
            instrumented: false,
        });
    let measured = [
        Variant::Control,
        Variant::Candidate,
        Variant::Candidate,
        Variant::Control,
        Variant::Candidate,
        Variant::Control,
        Variant::Control,
        Variant::Candidate,
    ]
    .into_iter()
    .take(if paired { 8 } else { 2 })
    .enumerate()
    .map(move |(sample, variant)| Pass {
        variant: if paired { variant } else { Variant::Control },
        sample: Some(sample),
        instrumented: false,
    });
    let instrumented = [Variant::Control, Variant::Candidate]
        .into_iter()
        .take(if !device_timing {
            0
        } else if paired {
            2
        } else {
            1
        })
        .map(|variant| Pass {
            variant,
            sample: None,
            instrumented: true,
        });
    warmup.chain(measured).chain(instrumented)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_samples_balance_variant_order_and_exclude_other_passes() {
        for device_timing in [false, true] {
            let all: Vec<_> = passes(true, device_timing).collect();
            let measured: Vec<_> = all.iter().filter(|pass| pass.sample.is_some()).collect();
            assert_eq!(measured.len(), 8);
            let mut population = [0; 2];
            for pass in &measured {
                population[pass.variant.index()] += 1;
            }
            assert_eq!(population, [4, 4]);
            for variant in [Variant::Control, Variant::Candidate] {
                let positions: Vec<_> = measured
                    .iter()
                    .filter(|pass| pass.variant == variant)
                    .map(|pass| pass.sample.unwrap())
                    .collect();
                assert_eq!(positions.len(), 4);
                assert_eq!(positions.iter().sum::<usize>(), 14);
                assert_eq!(all.iter().filter(|pass| pass.variant == variant && !pass.warm()).count(), 1);
                assert_eq!(
                    all.iter().filter(|pass| pass.variant == variant && pass.instrumented).count(),
                    usize::from(device_timing)
                );
            }
            for block in measured.chunks_exact(4) {
                assert_eq!(block[0].variant, block[3].variant);
                assert_eq!(block[1].variant, block[2].variant);
                assert_ne!(block[0].variant, block[1].variant);
            }
            assert!(all.iter().filter(|pass| pass.instrumented).all(|pass| pass.sample.is_none()));
            assert_eq!(all.len(), 10 + 2 * usize::from(device_timing));
        }
    }

    #[test]
    fn ordinary_benchmark_keeps_its_existing_passes() {
        for device_timing in [false, true] {
            let all: Vec<_> = passes(false, device_timing).collect();
            assert_eq!(all.len(), 3 + usize::from(device_timing));
            assert!(all.iter().all(|pass| pass.variant == Variant::Control));
            assert!(!all[0].warm());
            assert_eq!(all[1].sample, Some(0));
            assert_eq!(all[2].sample, Some(1));
            assert!(all[1..].iter().all(|pass| pass.warm()));
            assert_eq!(all.iter().filter(|pass| pass.instrumented).count(), usize::from(device_timing));
        }
    }
}
