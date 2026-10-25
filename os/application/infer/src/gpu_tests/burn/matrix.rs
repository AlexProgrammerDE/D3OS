//! Split reductions retain FP32 accuracy, offsets, and output bounds.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{burn::backend::TensorData, native::NativeTensor};
use d3os_compute::{Backend as _, nvidia::Nvidia};
use d3os_nvidia::compute::Kernel;

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    for (rows, inner, cols, splits) in [(17, 4099, 33, 32), (3, 7, 5, 32), (16, 36865, 25, 128)] {
        let left: Vec<_> = (0..rows * inner).map(|i| ((i * 17 % 101) as f32 - 50.) / 37.).collect();
        let right: Vec<_> = (0..inner * cols).map(|i| ((i * 37 % 127) as f32 - 63.) / 71.).collect();
        let expected = super::matrix_reference(&left, &right, rows, inner, cols);
        let guard = 0x7fc0dead;
        let guarded = |values: &[f32], offset: usize| {
            let mut words = vec![guard; offset];
            words.extend(values.iter().map(|value| value.to_bits()));
            words.push(guard);
            words
        };
        let left_words = guarded(&left, 3);
        let right_words = guarded(&right, 5);
        let a = backend.upload_words(&left_words).map_err(|e| format!("{e}"))?;
        let b = backend.upload_words(&right_words).map_err(|e| format!("{e}"))?;
        let partial_count = rows * cols * splits;
        let partials = backend.upload_words(&vec![guard; partial_count + 2]).map_err(|e| format!("{e}"))?;
        let output = backend.upload_words(&vec![guard; rows * cols + 2]).map_err(|e| format!("{e}"))?;
        backend
            .with_commands(|| {
                backend.dispatch_words(Kernel::Matmul, &[(&a, 3), (&b, 5), (&partials, 1)], &[rows, inner, cols, splits])?;
                backend.dispatch_words(Kernel::SumProjectionPartials, &[(&partials, 1), (&output, 1)], &[rows, cols, splits])
            })
            .map_err(|e| format!("{e}"))?;
        let mut values = vec![0; rows * cols + 2];
        backend.read_words_into(&output, 0, &mut values).map_err(|e| format!("{e}"))?;
        if values[0] != guard || values[rows * cols + 1] != guard {
            return Err(String::from("split matrix output guard changed"));
        }
        for (index, (&word, &reference)) in values[1..rows * cols + 1].iter().zip(&expected).enumerate() {
            let actual = f32::from_bits(word);
            if !actual.is_finite() || (actual - reference).abs() > 2e-5 * (1. + reference.abs()) {
                return Err(format!(
                    "split matrix [{rows},{inner},{cols}] splits={splits} index={index}: {actual} != {reference}"
                ));
            }
        }
        let mut first = [0];
        let mut last = [0];
        backend.read_words_into(&partials, 0, &mut first).map_err(|e| format!("{e}"))?;
        backend.read_words_into(&partials, partial_count + 1, &mut last).map_err(|e| format!("{e}"))?;
        if first != [guard] || last != [guard] {
            return Err(String::from("split matrix partial guard changed"));
        }
        for (buffer, expected) in [(&a, &left_words), (&b, &right_words)] {
            let mut actual = vec![0; expected.len()];
            backend.read_words_into(buffer, 0, &mut actual).map_err(|e| format!("{e}"))?;
            if &actual != expected {
                return Err(String::from("split matrix source changed"));
            }
        }
        // The public path must materialize signed views before splitting and
        // preserve broadcast batches. Reversing both K axes keeps the product.
        let left = NativeTensor::from_data(TensorData::new(left, [1, rows, inner]))
            .and_then(|tensor| tensor.flip(&[2]))
            .and_then(|tensor| tensor.expand(d3os_ai::burn::backend::Shape::new([2, rows, inner])))
            .map_err(|e| format!("{e}"))?;
        let right = NativeTensor::from_data(TensorData::new(right, [inner, cols]))
            .and_then(|tensor| tensor.flip(&[0]))
            .map_err(|e| format!("{e}"))?;
        let output = backend.with_commands(|| left.matmul(&right)).map_err(|e| format!("{e}"))?;
        let mut batches = expected.clone();
        batches.extend(expected);
        let actual = output.read().map_err(|e| format!("split matrix broadcast read: {e}"))?;
        let actual = actual.as_slice::<f32>().map_err(|e| format!("split matrix broadcast dtype: {e:?}"))?;
        if actual.len() != batches.len() {
            return Err(format!("split matrix broadcast length: {} != {}", actual.len(), batches.len()));
        }
        let mut max_scaled_error = 0f32;
        for (index, (&actual, &reference)) in actual.iter().zip(&batches).enumerate() {
            let scaled_error = (actual - reference).abs() / (1. + reference.abs());
            max_scaled_error = max_scaled_error.max(scaled_error);
            if !actual.is_finite() || scaled_error > 2e-5 {
                return Err(format!(
                    "split matrix broadcast [{rows},{inner},{cols}] index={index}: {actual} != {reference}, scaled_error={scaled_error}"
                ));
            }
        }
        crate::report(format_args!(
            "NVIDIA_BURN_SPLIT_MATMUL_REFERENCE rows={rows} inner={inner} cols={cols} batches=2 max_scaled_error={max_scaled_error} tolerance=0.00002"
        ));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_SPLIT_MATMUL_OK cases=3 precision=fp32 reference=f64 partial_tiles=ok empty_splits=ok guards=ok offsets=ok"
    ));
    Ok(())
}
