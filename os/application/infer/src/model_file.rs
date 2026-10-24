//! Seek model firmware parts without retaining the complete model in guest RAM.
use super::{File, read_file};
use alloc::{format, string::String, vec::Vec};
use core::fmt::Write;
use d3os_ai::{
    burn::tensor::Device,
    qwen::{
        Error, Qwen,
        gguf::{
            Gguf, ReadAt,
            shards::{Index, Sharded},
        },
    },
};
use naming::shared_types::{OpenOptions, SeekOrigin};

type Result<T> = core::result::Result<T, Error>;
struct ModelPart {
    file: File,
    length: u64,
}
impl ReadAt for ModelPart {
    fn len(&self) -> u64 {
        self.length
    }
    fn read_at(&mut self, offset: u64, output: &mut [u8]) -> Result<()> {
        if offset.checked_add(output.len() as u64).is_none_or(|end| end > self.length) {
            return Err(Error::SourceRead);
        }
        let position = isize::try_from(offset).map_err(|_| Error::SourceRead)?;
        naming::seek(self.file.0, position, SeekOrigin::Start).map_err(|_| Error::SourceRead)?;
        let mut read = 0;
        while read < output.len() {
            let count = naming::read(self.file.0, &mut output[read..]).map_err(|_| Error::SourceRead)?;
            if count == 0 {
                return Err(Error::SourceRead);
            }
            read += count;
        }
        Ok(())
    }
}

pub(super) fn load(device: &Device) -> core::result::Result<(Qwen, [u8; 32]), String> {
    let load = || -> Result<(Qwen, [u8; 32])> {
        let bytes = read_file("/fwcfg/model.index").map_err(|_| Error::SourceRead)?;
        let index = Index::parse(&bytes)?;
        let mut files = Vec::with_capacity(index.parts());
        for part in 0..index.parts() {
            let file = File(naming::open(&format!("/fwcfg/model.{part:03}"), OpenOptions::READONLY).map_err(|_| Error::SourceRead)?);
            let length = naming::seek(file.0, 0, SeekOrigin::End).map_err(|_| Error::SourceRead)? as u64;
            files.push(ModelPart { file, length });
        }
        let source = Sharded::new(index, files)?;
        let identity = *source.index().source_sha256();
        let mut checksum = String::new();
        for byte in source.index().source_sha256() {
            write!(checksum, "{byte:02x}").map_err(|_| Error::InvalidArchive)?;
        }
        let bytes = source.len();
        let parsed = Gguf::parse(source)?;
        super::report(format_args!(
            "GGUF_MODEL_START source_sha256={checksum} bytes={bytes} config={:?}",
            parsed.config()
        ));
        Ok((parsed.load(device)?, identity))
    };
    load().map_err(|error| format!("GGUF model: {error}"))
}

pub(super) fn verify(device: &Device) -> core::result::Result<(), String> {
    let (model, _) = load(device)?;
    let bytes = read_file("/fwcfg/model.fixture")?;
    let positions = d3os_ai::qwen::verify_fixture(&model, &bytes).map_err(|e| format!("GGUF layer reference: {e}"))?;
    super::report(format_args!(
        "INFERENCE_OK qwen-gguf-verify backend=nvidia framework=burn os=d3os positions={positions} layers={} cache_comparison=ok",
        model.config().layers
    ));
    Ok(())
}

pub(super) fn benchmark(device: &Device, backend: &d3os_compute::nvidia::Nvidia, device_timing: bool) -> core::result::Result<(), String> {
    let bytes = read_file("/fwcfg/model.reference")?;
    if bytes.len() < 64 || &bytes[..8] != b"D3QLOG\0\0" {
        return Err(String::from("invalid GGUF logit reference"));
    }
    let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let vocab = word(12) as usize;
    let prompt_count = word(16) as usize;
    let decode_count = word(20) as usize;
    // Match the Linux benchmark's exact work, cache precision, and batching.
    if word(8) != 1 || vocab != 248320 || prompt_count == 0 || prompt_count > 256 || decode_count != 128 || word(24) != 2048 || word(28) != 32 {
        return Err(String::from("GGUF reference has unsupported dimensions"));
    }
    let ids_end = 64 + 4 * (prompt_count + decode_count);
    if bytes.len() != ids_end + 4 * (decode_count + 1) * vocab {
        return Err(String::from("GGUF reference logit extent differs"));
    }
    let ids: Vec<_> = bytes[64..ids_end]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    if ids.iter().any(|&id| id as usize >= vocab) {
        return Err(String::from("GGUF reference token is outside the vocabulary"));
    }
    let (prompt, reference) = ids.split_at(prompt_count);
    let start = time::systime().num_milliseconds();
    let (model, identity) = load(device)?;
    if identity.as_slice() != &bytes[32..64] {
        return Err(String::from("GGUF reference targets a different model"));
    }
    let load_ms = time::systime().num_milliseconds() - start;
    let memory = backend.memory_profile().map_err(|e| format!("{e}"))?;
    super::report(format_args!(
        "NATIVE_LARGE_QWEN_MODEL os=d3os framework=burn backend=nvidia layers={} hidden={} load_ms={load_ms} live_bytes={} cached_bytes={} retired_bytes={}",
        model.config().layers,
        model.config().hidden,
        memory.process_live_bytes,
        memory.device_cached_bytes,
        memory.device_retired_bytes
    ));
    // Preserve all three ordinary samples. An optional fourth pass isolates
    // serialized device timings without presenting them as warm throughput.
    for run in 0..3 + usize::from(device_timing) {
        let instrumented = run == 3;
        let timing = if instrumented {
            Some(super::benchmark::DeviceTiming::start(backend).map_err(|e| format!("{e}"))?)
        } else {
            None
        };
        let state_start = time::systime().num_milliseconds();
        let mut session = model.session(2048).map_err(|e| format!("{e}"))?;
        let state_ms = time::systime().num_milliseconds() - state_start;
        let prefill_before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let profile = run == 1 || instrumented;
        let prefill_profile = if profile {
            Some((
                backend.profile().map_err(|e| format!("{e}"))?,
                backend.queue_profile().map_err(|e| format!("{e}"))?,
            ))
        } else {
            None
        };
        let start = time::systime().num_milliseconds();
        let mut logits = session.prefill(prompt).map_err(|e| format!("{e}"))?;
        let prefill_ms = time::systime().num_milliseconds() - start;
        let decode_before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let decode_profile = if profile {
            Some((
                backend.profile().map_err(|e| format!("{e}"))?,
                backend.queue_profile().map_err(|e| format!("{e}"))?,
            ))
        } else {
            None
        };
        let start = time::systime().num_milliseconds();
        let mut generated = Vec::with_capacity(decode_count);
        let mut checksum = 0u64;
        for &input in reference {
            let token = d3os_ai::qwen::greedy(&logits).map_err(|e| format!("{e}"))?;
            generated.push(token);
            checksum = checksum.wrapping_mul(31).wrapping_add(u64::from(token));
            logits = session.step(input).map_err(|e| format!("{e}"))?;
        }
        let decode_ms = time::systime().num_milliseconds() - start;
        let decode_after = backend.service_profile().map_err(|e| format!("{e}"))?;
        if let Some((kernels, queue)) = decode_profile {
            let after = backend.profile().map_err(|e| format!("{e}"))?;
            let queue_after = backend.queue_profile().map_err(|e| format!("{e}"))?;
            let prefill_phase = if instrumented { "large_prefill_device" } else { "large_prefill" };
            let decode_phase = if instrumented { "large_decode_device" } else { "large_decode" };
            let (prefill_kernels, prefill_queue) = prefill_profile.expect("profile snapshots are paired");
            super::benchmark::profile(&prefill_kernels, &kernels, prefill_phase);
            super::benchmark::queue_profile(prefill_queue, queue, prefill_phase);
            super::benchmark::service_profile(&prefill_before, &decode_before, prefill_phase);
            super::benchmark::profile(&kernels, &after, decode_phase);
            super::benchmark::queue_profile(queue, queue_after, decode_phase);
            super::benchmark::service_profile(&decode_before, &decode_after, decode_phase);
        }
        if let Some(timing) = timing {
            timing.finish().map_err(|e| format!("{e}"))?;
        }
        let upload = d3os_nvidia::compute::UPLOAD as usize;
        let download = d3os_nvidia::compute::DOWNLOAD as usize;
        let prompt_uploads = decode_before[upload].calls.wrapping_sub(prefill_before[upload].calls);
        let prompt_downloads = decode_before[download].calls.wrapping_sub(prefill_before[download].calls);
        let decode_uploads = decode_after[upload].calls.wrapping_sub(decode_before[upload].calls);
        let decode_downloads = decode_after[download].calls.wrapping_sub(decode_before[download].calls);
        if prompt_uploads != prompt_count.div_ceil(32) as u64
            || prompt_downloads != 0
            || decode_uploads != decode_count as u64
            || decode_downloads != decode_count as u64
        {
            return Err(String::from("GGUF benchmark transferred resident activation tensors"));
        }
        super::report(format_args!(
            "NATIVE_LARGE_QWEN_RESIDENT_OK os=d3os framework=burn backend=nvidia run={run} prompt_id_uploads={prompt_uploads} decode_id_uploads={decode_uploads} selection_read_calls={decode_downloads} selection_bytes_per_read=256 selection_candidates_per_read=32 activation_uploads=0 activation_downloads=0"
        ));
        let matching_prefix = generated.iter().zip(reference).take_while(|(actual, expected)| actual == expected).count();
        super::report(format_args!(
            "NATIVE_LARGE_QWEN_BENCH os=d3os framework=burn backend=nvidia run={run} warm={} instrumented={instrumented} state_ms={state_ms} prompt_tokens={prompt_count} prefill_ms={prefill_ms} decode_tokens={decode_count} decode_ms={decode_ms} decode_tps={:.3} checksum={checksum} cache=f32 n_ctx=2048 reference_prefix={matching_prefix} inputs=linux_reference eos=continue_for_benchmark",
            run > 0,
            decode_count as f64 * 1000. / decode_ms.max(1) as f64
        ));
        super::report(format_args!("NATIVE_LARGE_QWEN_TOKENS run={run} ids={generated:?}"));
    }
    let mut session = model.session(2048).map_err(|e| format!("{e}"))?;
    let mut logits = session.prefill(prompt).map_err(|e| format!("{e}"))?;
    let mut failures = 0usize;
    let mut max_absolute = 0f32;
    let mut max_scaled_error = 0f32;
    let mut selection_mismatches = 0usize;
    for row in 0..=decode_count {
        let actual = logits.clone().try_into_vec_as::<f32>().map_err(|e| format!("GGUF validation read: {e:?}"))?;
        if actual.len() != vocab || actual.iter().any(|v| !v.is_finite()) {
            return Err(String::from("nonfinite or invalid native GGUF logits"));
        }
        let expected: Vec<_> = bytes[ids_end + row * vocab * 4..ids_end + (row + 1) * vocab * 4]
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        for (actual, expected) in actual.iter().zip(&expected) {
            if !expected.is_finite() {
                return Err(String::from("nonfinite GGUF reference logits"));
            }
            let error = (actual - expected).abs();
            // Retain the existing pretrained Qwen reference tolerances.
            let scaled = error / (2e-3 + 5e-4 * expected.abs());
            max_absolute = max_absolute.max(error);
            max_scaled_error = max_scaled_error.max(scaled);
            failures += usize::from(scaled > 1.);
        }
        if row < decode_count {
            let selected = d3os_ai::qwen::greedy(&logits).map_err(|e| format!("{e}"))?;
            let expected_id = reference[row] as usize;
            if selected as usize != expected_id {
                selection_mismatches += 1;
                super::report(format_args!(
                    "NATIVE_LARGE_QWEN_SELECTION row={row} actual={selected} expected={expected_id} actual_margin={} reference_margin={}",
                    actual[selected as usize] - actual[expected_id],
                    expected[expected_id] - expected[selected as usize]
                ));
            }
            logits = session.step(reference[row]).map_err(|e| format!("{e}"))?;
        }
    }
    super::report(format_args!(
        "NATIVE_LARGE_QWEN_REFERENCE os=d3os framework=burn backend=nvidia rows={} vocab={vocab} values={} failures={failures} max_absolute={max_absolute} max_scaled_error={max_scaled_error} selection_mismatches={selection_mismatches} absolute_tolerance=0.002 relative_tolerance=0.0005 inputs=linux_reference",
        decode_count + 1,
        (decode_count + 1) * vocab
    ));
    if failures != 0 {
        return Err(String::from("native GGUF logits differ from the staged numerical reference"));
    }
    super::report(format_args!(
        "INFERENCE_BENCH_OK qwen-gguf backend=nvidia framework=burn os=d3os reference_tokens={decode_count}"
    ));
    Ok(())
}
