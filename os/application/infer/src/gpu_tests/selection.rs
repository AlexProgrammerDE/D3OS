use alloc::{format, string::String, vec, vec::Vec};
use d3os_compute::{Backend, Error, Tensor, cpu::Cpu, nvidia::Nvidia};
use d3os_nvidia::compute::Kernel;

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let check = |values: Vec<f32>| -> Result<(), String> {
        let cpu = Tensor::from_vec(&Cpu, &[values.len()], values.clone()).map_err(|e| format!("{e}"))?;
        let gpu = Tensor::from_vec(backend, &[values.len()], values).map_err(|e| format!("{e}"))?;
        let expected = Cpu.argmax(&cpu).map_err(|e| format!("{e}"))?;
        let actual = backend.argmax(&gpu).map_err(|e| format!("{e}"))?;
        if actual != expected {
            return Err(format!("argmax: expected index {expected}, got {actual}"));
        }
        Ok(())
    };
    for width in [1, 7, 33, 1023, 1024, 1025, 248320] {
        let mut values: Vec<_> = (0..width).map(|i| -((i * 73 % 257) as f32) - 1.).collect();
        values[width - 1] = 0.;
        check(values.clone())?;
        // Ties cross warp, CTA, and grid-stride boundaries.
        values[width / 3] = 0.;
        check(values)?;
    }
    check(vec![-0., 0., -0.])?;
    check(vec![f32::MIN; 65])?;
    check(vec![0., f32::from_bits(1), f32::from_bits(2)])?;

    // Raw word uploads permit IEEE special values without weakening the typed
    // tensor upload contract. A non-finite loser must still invalidate selection.
    for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let input = backend
            .upload_words(&[1f32.to_bits(), invalid.to_bits(), 2f32.to_bits()])
            .map_err(|e| format!("{e}"))?;
        let output = backend.allocate_words(64).map_err(|e| format!("{e}"))?;
        backend
            .dispatch_words(Kernel::ArgMax, &[(&input, 0), (&output, 0)], &[3])
            .map_err(|e| format!("{e}"))?;
        let mut partial = [0; 2];
        backend.read_words_into(&output, 0, &mut partial).map_err(|e| format!("{e}"))?;
        if partial[0] != 0 || !f32::from_bits(partial[1]).is_nan() {
            return Err("argmax ignored a non-finite input".into());
        }
    }
    let input = Tensor::from_vec(backend, &[1, 1], vec![f32::MAX]).map_err(|e| format!("{e}"))?;
    let output = backend.matmul(&input, &input).map_err(|e| format!("{e}"))?;
    if !matches!(backend.argmax(&output), Err(Error::NonFinite)) {
        return Err("argmax accepted an overflowed tensor".into());
    }
    super::super::report(format_args!(
        "NVIDIA_SELECTION_OK shapes=7 ties=stable guards=finite transfer_bytes=256 backend=nvidia"
    ));
    Ok(())
}
