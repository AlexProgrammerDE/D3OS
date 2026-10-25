//! Check Burn tensor ownership and views on real native GPU allocations.
mod activation;
mod compaction;
mod conversions;
mod convolution;
mod cross;
mod dispatch;
mod indexing;
mod integers;
mod inverse;
mod linear;
mod math;
mod matrix;
mod memory;
mod nd_indexing;
mod optimizer;
mod pooling;
mod quantization;
mod qwen;
mod random;
mod remainder;
mod scans;
mod scatter;
mod scatter_updates;
mod selection;
mod sort;
mod storage;
mod updates;
mod words;
use abi::elementwise::Elementwise;
use abi::reduction::Reduction;
use alloc::{format, string::String, vec::Vec};
use d3os_ai::{
    burn::backend::{BoolStore, DType, Shape, Slice, TensorData, TensorMetadata},
    native::{NativeDevice, NativeTensor},
};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute as abi;

fn verify(backend: &Nvidia) -> Result<(), Error> {
    NativeDevice::open()?;
    let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 0.5).collect();
    let original = NativeTensor::from_data(TensorData::new(values.clone(), [2, 3, 4]))?;
    if !original.can_mut() {
        return Err(Error::DeviceFault);
    }
    let before = backend.service_profile()?;
    let view = original
        .clone()
        .permute(&[2, 0, 1])?
        .flip(&[0, 2])?
        .slice(&[Slice::new(1, Some(4), 2), Slice::full(), Slice::new(1, Some(3), -1)])?;
    let after = backend.service_profile()?;
    if original.can_mut()
        || view.can_mut()
        || [abi::ALLOCATE, abi::UPLOAD, abi::DOWNLOAD]
            .iter()
            .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(Error::DeviceFault);
    }
    let expected: Vec<_> = [2, 6, 14, 18, 0, 4, 12, 16].iter().map(|&i| values[i]).collect();
    if view.read()?.as_slice::<f32>().map_err(|_| Error::DeviceFault)? != expected {
        return Err(Error::DeviceFault);
    }
    drop(view);
    if !original.can_mut() || original.read()?.as_slice::<f32>().map_err(|_| Error::DeviceFault)? != values {
        return Err(Error::DeviceFault);
    }
    let survivor = original.clone().slice(&[Slice::new(1, Some(2), 1)])?;
    drop(original);
    if !survivor.can_mut() || survivor.read()?.as_slice::<f32>().map_err(|_| Error::DeviceFault)? != &values[12..] {
        return Err(Error::DeviceFault);
    }
    let windows = NativeTensor::from_data(TensorData::new((0..12).collect::<Vec<i32>>(), [2, 6]))?.unfold(1, 3, 2)?;
    if windows.can_mut() || windows.read()?.as_slice::<i32>().map_err(|_| Error::DeviceFault)? != [0, 1, 2, 2, 3, 4, 6, 7, 8, 8, 9, 10] {
        return Err(Error::DeviceFault);
    }
    let bits = [0u32, 0x8000_0000, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000, 0x0000_0001];
    let special = NativeTensor::from_data(TensorData::new(bits.iter().copied().map(f32::from_bits).collect(), [6]))?;
    let roundtrip: Vec<_> = special
        .read()?
        .as_slice::<f32>()
        .map_err(|_| Error::DeviceFault)?
        .iter()
        .map(|v| v.to_bits())
        .collect();
    if roundtrip != bits {
        return Err(Error::DeviceFault);
    }
    let booleans = NativeTensor::from_data(TensorData::new(alloc::vec![true, false, true], [3]))?;
    if booleans.dtype() != DType::Bool(BoolStore::U32) || booleans.read()?.as_bytes() != [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] {
        return Err(Error::DeviceFault);
    }
    let empty = NativeTensor::from_data(TensorData::new(Vec::<f32>::new(), [2, 0, 3]))?;
    if !empty.layout().is_empty() || !empty.read()?.as_bytes().is_empty() {
        return Err(Error::DeviceFault);
    }
    if survivor.with_layout(d3os_ai::native::layout::Layout::contiguous(Shape::new([1]))?).is_ok() {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let device = d3os_ai::native_device().map_err(|e| format!("capability device: {e}"))?;
    let capabilities = d3os_ai::validation::capabilities::declared(&device);
    for capability in &capabilities {
        crate::report(format_args!(
            "BURN_DTYPE_CAPABILITY os=d3os backend=native name={} storage={} arithmetic={} accelerated={}",
            capability.name, capability.storage, capability.arithmetic, capability.accelerated
        ));
    }
    crate::report(format_args!(
        "BURN_CAPABILITIES_OK os=d3os backend=native rows={} flags_only=true",
        capabilities.len()
    ));
    memory::run(backend)?;
    verify(backend).map_err(|e| format!("Burn native tensor storage: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_STORAGE_OK views=ok aliasing=ok lifetime=ok ieee_bits=ok bool=ok empty=ok backend=nvidia"
    ));
    verify_elementwise(backend).map_err(|e| format!("Burn native elementwise: {e}"))?;
    verify_guarded_elementwise(backend).map_err(|e| format!("Burn native elementwise guards: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_ELEMENTWISE_OK operations=11 views=strided_broadcast scalar=ok guards=ok resident=ok backend=nvidia"
    ));
    verify_matrices(backend).map_err(|e| format!("Burn native matrices: {e}"))?;
    matrix::run(backend).map_err(|e| format!("Burn native split matrices: {e}"))?;
    optimizer::run(backend).map_err(|e| format!("Burn native Adam: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_MATRIX_OK cases=5 batched=ok subviews=ok empty=ok tensor_cores=true backend=nvidia"
    ));
    verify_reductions(backend).map_err(|e| format!("Burn native reductions: {e}"))?;
    verify_reduction_edges(backend).map_err(|e| format!("Burn native reduction edges: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_REDUCTION_OK operations=7 views=strided_broadcast nan=first ties=first empty=ok guards=ok resident=ok backend=nvidia"
    ));
    math::run(backend).map_err(|e| format!("Burn native math: {e}"))?;
    d3os_ai::validation::erf::run(&device, d3os_ai::validation::erf::GeluGradient::BurnDefault)?;
    let profile = d3os_ai::validation::erf::profile(&device)?;
    crate::report(format_args!(
        "BURN_ERF_PROFILE os=d3os backend=native cases={} strict_mismatches={} special_mismatches={} max_abs_error={} max_ulp_error={}",
        profile.input_bits.len(),
        profile.strict_mismatches,
        profile.special_mismatches,
        profile.max_absolute_error,
        profile.max_ulp_error
    ));
    crate::report(format_args!(
        "BURN_ERF_CONFORMANCE_OK os=d3os backend=native erf_ulp=8 erf=forward,backward gelu=forward,backward shared_views=verified"
    ));
    inverse::run(backend)?;
    for (domain, name) in [
        (d3os_ai::validation::inverse::Domain::Ieee, "ieee"),
        (d3os_ai::validation::inverse::Domain::Finite, "finite"),
    ] {
        let r = d3os_ai::validation::inverse::run(&device, domain)?;
        crate::report(format_args!(
            "BURN_INVERSE_OK os=d3os backend=native domain={} unary_ops=6 unary_inputs={} atan2_pairs={} ulp=8 gradients={} signed_views=true broadcast=true empty=true",
            name, r.unary_inputs, r.atan2_pairs, r.gradients
        ));
    }
    remainder::run(backend)?;
    for (domain, name) in [
        (d3os_ai::validation::remainder::Domain::Ieee, "ieee"),
        (d3os_ai::validation::remainder::Domain::Finite, "finite"),
    ] {
        let report = d3os_ai::validation::remainder::run(&device, domain)?;
        crate::report(format_args!(
            "BURN_REMAINDER_OK os=d3os backend=native domain={} pairs={} scalar_cases={} gradients={} reference=libm_fmod_floor finite_bits=exact zero_sign=unspecified signed_views=true broadcast=true empty=true",
            name, report.pairs, report.scalar_cases, report.gradients
        ));
    }
    integers::run(backend).map_err(|e| format!("Burn native integer reductions: {e}"))?;
    words::run(backend).map_err(|e| format!("Burn native words: {e}"))?;
    scans::run(backend)?;
    sort::run(backend)?;
    compaction::run(backend)?;
    d3os_ai::burn::tensor::try_read_sync(d3os_ai::validation::compaction::run(&device, true, true))
        .ok_or_else(|| String::from("native compaction future did not complete"))??;
    crate::report(format_args!(
        "BURN_COMPACTION_CONFORMANCE_OK os=d3os backend=native dtypes=f32,i32,bool,q8 argwhere=true nonzero=true mask_select=true gradients=4 ranks=1,3,8 quantized_domain=compact_tails empty=true shared_views=true"
    ));
    cross::run(backend)?;
    d3os_ai::validation::cross::run(&device, true)?;
    crate::report(format_args!(
        "BURN_CROSS_CONFORMANCE_OK os=d3os backend=native dtype=f32 axes=all signed_views=true broadcasting=true gradients=6 empty=true"
    ));
    for domain in [d3os_ai::validation::sort::FloatDomain::Ieee, d3os_ai::validation::sort::FloatDomain::Finite] {
        d3os_ai::validation::sort::run(&device, true, d3os_ai::validation::sort::TopKOrdering::NanFirst, domain)?;
    }
    crate::report(format_args!(
        "BURN_SORT_CONFORMANCE_OK os=d3os backend=native sorting=gpu topk_order=nan_first float_domains=ieee,finite dtypes=f32,i32 sort=true argsort=true topk=true gradients=6 empty=true shared_views=true"
    ));
    d3os_ai::validation::scan::run(&device, true)?;
    crate::report(format_args!(
        "BURN_SCAN_CONFORMANCE_OK os=d3os backend=native operations=8 reference=i128,f64 lengths=1,127,128,129,257,531 gradients=4 empty=true shared_views=true"
    ));
    d3os_ai::validation::integer::run(&device)?;
    crate::report(format_args!(
        "BURN_INTEGER_CONFORMANCE_OK os=d3os backend=native operations=div,remainder tensor=true scalar=true shared_views=true"
    ));
    d3os_ai::validation::integer_matrix::run(&device, true)?;
    crate::report(format_args!(
        "BURN_INTEGER_MATMUL_CONFORMANCE_OK os=d3os backend=native cases=8 domain=all reference=i128 wrapping=exact shared_views=true empty=true"
    ));
    indexing::run(backend).map_err(|e| format!("Burn native indexing: {e}"))?;
    scatter::run(backend).map_err(|e| format!("Burn native scatter: {e}"))?;
    scatter_updates::run(backend)?;
    d3os_ai::validation::scatter_updates::run(&device, true, true)?;
    crate::report(format_args!(
        "BURN_SCATTER_UPDATES_OK os=d3os backend=native dtypes=f32,i32 modes=4 axes=all unique_indices=true gradient_cases=12 ieee=true empty=true shared_views=true"
    ));
    d3os_ai::validation::scatter_updates::run_select(&device, true, true)?;
    crate::report(format_args!(
        "BURN_SELECT_UPDATES_OK os=d3os backend=native dtypes=f32,i32 modes=5 axes=all unique_gradient_indices=true gradient_cases=15 duplicates=add,mul,min,max ranks=3,8 ieee=true empty=true shared_views=true"
    ));
    nd_indexing::run(backend)?;
    d3os_ai::validation::nd_indexing::run(&device, true, true, true)?;
    crate::report(format_args!(
        "BURN_ND_INDEXING_OK os=d3os backend=native dtypes=f32,i32 scatter_modes=5 widths=1,2,3 index_ranks=1,2,3 scatter_gradient_cases=15 gather_gradient_cases=3 scalar=true exact_i32=true duplicates=true empty_slices=true shared_views=true"
    ));
    storage::run(backend)?;
    d3os_ai::validation::storage::run(&device, true)?;
    d3os_ai::validation::storage::flex32_direct(&device)?;
    crate::report(format_args!("BURN_STORAGE_FLEX32_DIRECT_OK os=d3os backend=native"));
    crate::report(format_args!(
        "BURN_STORAGE_CONFORMANCE_OK os=d3os backend=native dtypes=15 flex32=direct bytes=1,2,4,8 views=signed,broadcast,overlap fill=true bool_storage=explicit_backend reload=true half_bits=exhaustive empty=true scope=storage"
    ));
    conversions::run(backend)?;
    let pairs = d3os_ai::validation::conversions::run(&device, true)?;
    let gradient_cases = 8;
    crate::report(format_args!(
        "BURN_CONVERSIONS_OK os=d3os backend=native pairs={pairs} flex32=true float_to_int=finite_in_range rounding=nearest_even integer_narrowing=low_bits half_float_codes=exhaustive bf16_midpoints=all_intervals gradients={gradient_cases} signed_views=true empty=true"
    ));
    convolution::run(backend).map_err(|e| format!("Burn native convolution: {e}"))?;
    linear::run(backend).map_err(|e| format!("Burn native linear: {e}"))?;
    activation::run(backend).map_err(|e| format!("Burn native ReLU: {e}"))?;
    pooling::run(backend).map_err(|e| format!("Burn native pooling: {e}"))?;
    updates::run(backend).map_err(|e| format!("Burn native updates: {e}"))?;
    random::run(backend).map_err(|e| format!("Burn native random: {e}"))?;
    qwen::run(backend).map_err(|e| format!("Burn Qwen extensions: {e}"))?;
    selection::run(backend)?;
    quantization::run(backend)?;
    dispatch::run(backend).map_err(|e| format!("Burn public dispatch: {e}"))?;
    for (name, case) in [
        ("direct", d3os_ai::validation::quantized_empty::Case::Direct),
        ("select", d3os_ai::validation::quantized_empty::Case::Select),
        ("mask", d3os_ai::validation::quantized_empty::Case::Mask),
    ] {
        let result = d3os_ai::burn::tensor::try_read_sync(d3os_ai::validation::quantized_empty::run(&device, case))
            .ok_or_else(|| String::from("native quantized-empty future did not complete"))?;
        if name == "direct" && !result.as_ref().is_ok_and(|bytes| bytes == &[0, 0, 128, 63]) {
            return Err(format!("empty static quantized record: {result:?}"));
        }
        match result {
            Ok(bytes) => crate::report(format_args!(
                "BURN_QUANTIZED_EMPTY_RESULT os=d3os backend=native case={name} status=ok bytes={bytes:?}"
            )),
            Err(error) => crate::report(format_args!(
                "BURN_QUANTIZED_EMPTY_RESULT os=d3os backend=native case={name} status=error error={error}"
            )),
        }
        let mut acknowledged = false;
        for _ in 0..16 {
            match device.sync() {
                Ok(()) => {
                    acknowledged = true;
                    break;
                }
                Err(error) if !error.is_device_poisoned() => crate::report(format_args!("BURN_QUANTIZED_EMPTY_ACK case={name} error={error:?}")),
                Err(error) => return Err(format!("quantized-empty poisoned the device: {error:?}")),
            }
        }
        if !acknowledged {
            return Err(String::from("quantized-empty left unacknowledged execution failures"));
        }
    }
    Ok(())
}

fn reduction_reference(values: &[f32], dimensions: &[usize], axis: usize, op: Reduction) -> Vec<u32> {
    let width = dimensions[axis];
    let suffix: usize = dimensions[axis + 1..].iter().product();
    let rows: usize = dimensions.iter().enumerate().filter(|&(a, _)| a != axis).map(|(_, &d)| d).product();
    (0..rows)
        .map(|row| {
            let base = (row / suffix) * width * suffix + row % suffix;
            let value = |k: usize| values[base + k * suffix];
            match op {
                Reduction::Sum | Reduction::Mean => {
                    let sum: f64 = (0..width).map(|k| f64::from(value(k))).sum();
                    (if op == Reduction::Mean { sum / width as f64 } else { sum } as f32).to_bits()
                }
                Reduction::Product => ((0..width).map(|k| f64::from(value(k))).product::<f64>() as f32).to_bits(),
                _ => {
                    let minimum = matches!(op, Reduction::Min | Reduction::ArgMin);
                    let mut winner = 0;
                    for k in 0..width {
                        if value(k).is_nan() {
                            winner = k;
                            break;
                        }
                        if (minimum && value(k) < value(winner)) || (!minimum && value(k) > value(winner)) {
                            winner = k;
                        }
                    }
                    if op.index() { winner as u32 } else { value(winner).to_bits() }
                }
            }
        })
        .collect()
}

fn check_reduction(tensor: &NativeTensor, expected: &[u32], op: Reduction) -> Result<(), Error> {
    let data = tensor.read()?;
    let actual: Vec<u32> = data.as_bytes().chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(&a, &b)| {
            if op.index() || !op.allows_empty() {
                a != b
            } else {
                let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                !(a == b || (a.is_nan() && b.is_nan()) || (a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-5 * (1.0 + b.abs())))
            }
        })
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn verify_reductions(backend: &Nvidia) -> Result<(), Error> {
    let values: Vec<f32> = (0..3 * 257 * 5).map(|i| 1.0 + ((i % 17) as f32 - 8.0) / 1024.0).collect();
    let original = NativeTensor::from_data(TensorData::new(values.clone(), [3, 257, 5]))?;
    let input = original.clone().permute(&[2, 0, 1])?.flip(&[0, 2])?;
    let ordered: Vec<f32> = (0..5)
        .flat_map(|x| (0..3).flat_map(move |n| (0..257).map(move |c| (n * 257 + 256 - c) * 5 + 4 - x)))
        .map(|index| values[index])
        .collect();
    let before = backend.service_profile()?;
    let mut outputs = Vec::new();
    for axis in 0..3 {
        for op in Reduction::ALL {
            outputs.push((axis, op, input.reduce(axis, op)?));
        }
    }
    let after = backend.service_profile()?;
    if [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 21
    {
        return Err(Error::DeviceFault);
    }
    for (axis, op, output) in outputs {
        let mut shape = [5, 3, 257];
        shape[axis] = 1;
        if output.shape().as_slice() != shape || output.dtype() != if op.index() { DType::I32 } else { DType::F32 } {
            return Err(Error::DeviceFault);
        }
        check_reduction(&output, &reduction_reference(&ordered, &[5, 3, 257], axis, op), op)?;
    }
    check_values(&original, &values)?;
    let input = NativeTensor::from_data(TensorData::new(alloc::vec![1.25f32, 0.75], [2, 1]))?.expand(Shape::new([2, 257]))?;
    for op in Reduction::ALL {
        let values: Vec<_> = alloc::vec![1.25f32; 257].into_iter().chain(alloc::vec![0.75f32; 257]).collect();
        check_reduction(&input.reduce(1, op)?, &reduction_reference(&values, &[2, 257], 1, op), op)?;
    }
    Ok(())
}

fn verify_reduction_edges(backend: &Nvidia) -> Result<(), Error> {
    let mut values = alloc::vec![1.0f32; 6*257];
    values[..257].fill(f32::INFINITY);
    values[257..514].fill(f32::NEG_INFINITY);
    values[514..771].fill(0.0);
    values[514] = -0.0;
    values[771 + 3] = f32::from_bits(0x7fc0_1234);
    values[771 + 131] = f32::from_bits(0x7fc0_5678);
    values[1028 + 1] = -7.0;
    values[1028 + 129] = -7.0;
    values[1028 + 2] = 9.0;
    values[1028 + 130] = 9.0;
    values[1285..].fill(f32::from_bits(0x7fc0_9999));
    let input = NativeTensor::from_data(TensorData::new(values.clone(), [6, 257]))?;
    for reversed in [false, true] {
        let view = if reversed { input.clone().flip(&[1])? } else { input.clone() };
        let ordered: Vec<_> = values
            .chunks_exact(257)
            .flat_map(|row| (0..257).map(move |k| row[if reversed { 256 - k } else { k }]))
            .collect();
        for op in Reduction::ALL {
            check_reduction(&view.reduce(1, op)?, &reduction_reference(&ordered, &[6, 257], 1, op), op)?;
        }
    }
    let empty = NativeTensor::from_data(TensorData::new(Vec::<f32>::new(), [3, 0, 4]))?;
    for op in [Reduction::Sum, Reduction::Mean, Reduction::Product] {
        check_reduction(&empty.reduce(1, op)?, &reduction_reference(&[], &[3, 0, 4], 1, op), op)?;
    }
    let before = backend.queue_profile()?;
    for op in [Reduction::Min, Reduction::Max, Reduction::ArgMin, Reduction::ArgMax] {
        if !matches!(empty.reduce(1, op), Err(Error::InvalidParameter)) {
            return Err(Error::DeviceFault);
        }
    }
    let result = empty.reduce(0, Reduction::Sum)?;
    if !result.layout().is_empty() || !result.read()?.as_bytes().is_empty() || before.launches != backend.queue_profile()?.launches {
        return Err(Error::DeviceFault);
    }
    use d3os_ai::native::layout::Layout;
    let mut guarded = alloc::vec![0x6bad_beefu32];
    guarded.extend((0..3 * 257).map(|i| (i as f32 / 8.0).to_bits()));
    guarded.push(0x6bad_beef);
    let input = backend.upload_words(&guarded)?;
    let output = backend.upload_words(&[0x6bad_beef; 5])?;
    let layout = Layout::contiguous(Shape::new([3, 257]))?.flip(&[1])?;
    let parameters = layout.reduction_parameters(1, Reduction::Max)?;
    backend.dispatch_words(abi::Kernel::TensorReduction, &[(&input, 1), (&output, 1)], &parameters)?;
    let mut actual = [0; 5];
    backend.read_words_into(&output, 0, &mut actual)?;
    let expected = [
        0x6bad_beef,
        (256.0f32 / 8.0).to_bits(),
        (513.0f32 / 8.0).to_bits(),
        (770.0f32 / 8.0).to_bits(),
        0x6bad_beef,
    ];
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    let mut malformed = parameters;
    malformed[4] += 257;
    if backend.dispatch_words(abi::Kernel::TensorReduction, &[(&input, 1), (&output, 1)], &malformed) != Err(Error::InvalidParameter) {
        return Err(Error::DeviceFault);
    }
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    let mut unchanged = alloc::vec![0; guarded.len()];
    backend.read_words_into(&input, 0, &mut unchanged)?;
    if unchanged != guarded {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn matrix_reference(a: &[f32], b: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            (0..inner)
                .map(|k| f64::from(a[(i / cols) * inner + k]) * f64::from(b[k * cols + i % cols]))
                .sum::<f64>() as f32
        })
        .collect()
}

fn verify_matrices(backend: &Nvidia) -> Result<(), Error> {
    let a: Vec<f32> = (0..30).map(|i| (i % 11) as f32 / 8.0 - 0.75).collect();
    let b: Vec<f32> = (0..35).map(|i| (i % 13) as f32 / 16.0 - 0.25).collect();
    let left = NativeTensor::from_data(TensorData::new(a.clone(), [2, 3, 5]))?.flip(&[1])?;
    let right = NativeTensor::from_data(TensorData::new(b.clone(), [1, 7, 5]))?.permute(&[0, 2, 1])?;
    let before = backend.service_profile()?;
    let output = left.matmul(&right)?;
    let after = backend.service_profile()?;
    if [abi::UPLOAD, abi::DOWNLOAD]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
    {
        return Err(Error::DeviceFault);
    }
    let mut expected = Vec::new();
    for batch in 0..2 {
        let reversed: Vec<_> = (0..15).map(|i| a[batch * 15 + (2 - i / 5) * 5 + i % 5]).collect();
        let transposed: Vec<_> = (0..35).map(|i| b[(i % 7) * 5 + i / 7]).collect();
        expected.extend(matrix_reference(&reversed, &transposed, 3, 5, 7));
    }
    check_values(&output, &expected)?;
    if output.shape().as_slice() != [2, 3, 7] {
        return Err(Error::DeviceFault);
    }

    let a: Vec<f32> = (0..20).map(|i| i as f32 / 16.0 - 0.5).collect();
    let b: Vec<f32> = (0..42).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();
    let left = NativeTensor::from_data(TensorData::new(a.clone(), [4, 5]))?.slice(&[Slice::new(1, Some(4), 1)])?;
    let right = NativeTensor::from_data(TensorData::new(b.clone(), [6, 7]))?.slice(&[Slice::new(1, Some(6), 1)])?;
    check_values(&left.matmul(&right)?, &matrix_reference(&a[5..], &b[7..], 3, 5, 7))?;

    let a: Vec<f32> = (0..30).map(|i| (i % 11) as f32 / 8.0 - 0.5).collect();
    let b: Vec<f32> = (0..140).map(|i| (i % 13) as f32 / 16.0 - 0.25).collect();
    let left = NativeTensor::from_data(TensorData::new(a.clone(), [2, 1, 3, 5]))?;
    let right = NativeTensor::from_data(TensorData::new(b.clone(), [1, 4, 5, 7]))?;
    let output = left.matmul(&right)?;
    let expected: Vec<_> = (0..2)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .flat_map(|(i, j)| matrix_reference(&a[i * 15..(i + 1) * 15], &b[j * 35..(j + 1) * 35], 3, 5, 7))
        .collect();
    check_values(&output, &expected)?;
    if output.shape().as_slice() != [2, 4, 3, 7] {
        return Err(Error::DeviceFault);
    }

    let empty = NativeTensor::from_data(TensorData::new(Vec::<f32>::new(), [2, 0, 5]))?;
    let rhs = NativeTensor::from_data(TensorData::new(alloc::vec![1.0f32;35], [5, 7]))?;
    let zero_inner_a = NativeTensor::from_data(TensorData::new(Vec::<f32>::new(), [3, 0]))?;
    let zero_inner_b = NativeTensor::from_data(TensorData::new(Vec::<f32>::new(), [0, 2]))?;
    let before = backend.profile()?;
    let empty = empty.matmul(&rhs)?;
    let zeros = zero_inner_a.matmul(&zero_inner_b)?;
    let after = backend.profile()?;
    if empty.shape().as_slice() != [2, 0, 7]
        || !empty.read()?.as_bytes().is_empty()
        || before[abi::Kernel::Matmul as usize].calls != after[abi::Kernel::Matmul as usize].calls
    {
        return Err(Error::DeviceFault);
    }
    check_values(&zeros, &[0.0; 6])?;
    Ok(())
}

fn check_values(tensor: &NativeTensor, expected: &[f32]) -> Result<(), Error> {
    let actual = tensor.read()?;
    let actual = actual.as_slice::<f32>().map_err(|_| Error::DeviceFault)?;
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(&a, &b)| !a.is_finite() || (a - b).abs() > 1e-6 + 1e-6 * b.abs())
    {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn verify_elementwise(backend: &Nvidia) -> Result<(), Error> {
    let values: Vec<f32> = (0..2 * 3 * 131).map(|i| (i % 37) as f32 / 8.0 - 2.25).collect();
    let original = NativeTensor::from_data(TensorData::new(values.clone(), [2, 3, 131]))?;
    let input = original.clone().permute(&[2, 0, 1])?.flip(&[0])?;
    let rhs = NativeTensor::from_data(TensorData::new(alloc::vec![2.0f32, -4.0], [1, 2, 1]))?;
    let expected: Vec<f32> = (0..131)
        .flat_map(|x| (0..2).flat_map(move |row| (0..3).map(move |col| (row * 3 + col) * 131 + (130 - x))))
        .map(|i| values[i])
        .collect();
    let factors: Vec<f32> = (0..131).flat_map(|_| [2.0f32, 2.0, 2.0, -4.0, -4.0, -4.0]).collect();
    check_values(&input.copy()?, &expected)?;
    for op in [Elementwise::Add, Elementwise::Subtract, Elementwise::Multiply, Elementwise::Divide] {
        let reference: Vec<_> = expected
            .iter()
            .zip(&factors)
            .map(|(&a, &b)| match op {
                Elementwise::Add => a + b,
                Elementwise::Subtract => a - b,
                Elementwise::Multiply => a * b,
                _ => a / b,
            })
            .collect();
        check_values(&input.binary(&rhs, op)?, &reference)?;
        let reference: Vec<_> = expected
            .iter()
            .map(|&a| match op {
                Elementwise::Add => a + 0.5,
                Elementwise::Subtract => a - 0.5,
                Elementwise::Multiply => a * 0.5,
                _ => a / 0.5,
            })
            .collect();
        check_values(&input.scalar(op, 0.5)?, &reference)?;
    }
    for op in [
        Elementwise::Equal,
        Elementwise::Greater,
        Elementwise::GreaterEqual,
        Elementwise::Less,
        Elementwise::LessEqual,
    ] {
        for scalar in [false, true] {
            let actual = if scalar { input.scalar(op, 0.5)? } else { input.binary(&rhs, op)? };
            if actual.dtype() != DType::Bool(BoolStore::U32) {
                return Err(Error::DeviceFault);
            }
            let reference: Vec<u8> = expected
                .iter()
                .zip(&factors)
                .flat_map(|(&a, &b)| {
                    let b = if scalar { 0.5 } else { b };
                    let result = match op {
                        Elementwise::Equal => a == b,
                        Elementwise::Greater => a > b,
                        Elementwise::GreaterEqual => a >= b,
                        Elementwise::Less => a < b,
                        _ => a <= b,
                    };
                    u32::from(result).to_le_bytes()
                })
                .collect();
            if actual.read()?.as_bytes() != reference {
                return Err(Error::DeviceFault);
            }
        }
    }
    let before = backend.service_profile()?;
    let shifted = input.binary(&rhs, Elementwise::Add)?;
    let mask = shifted.scalar(Elementwise::Less, 0.0)?;
    let masked = shifted.mask_fill(&mask, -123.5)?;
    let after = backend.service_profile()?;
    if [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 3
    {
        return Err(Error::DeviceFault);
    }
    let reference: Vec<_> = expected.iter().zip(&factors).map(|(&a, &b)| if a + b < 0.0 { -123.5 } else { a + b }).collect();
    check_values(&masked, &reference)?;
    let small_mask = NativeTensor::from_data(TensorData::new(alloc::vec![false, true, false], [3]))?;
    let reference: Vec<_> = expected.iter().enumerate().map(|(i, &v)| if i % 3 == 1 { 9.0 } else { v }).collect();
    check_values(&input.mask_fill(&small_mask, 9.0)?, &reference)?;
    check_values(&original, &values)?;
    let special_bits = [0x8000_0000u32, 0x7fc0_1234, 0x7f80_0000];
    let special = NativeTensor::from_data(TensorData::new(special_bits.map(f32::from_bits).to_vec(), [3]))?
        .flip(&[0])?
        .copy()?;
    let special = special.read()?;
    let actual: Vec<_> = special.as_slice::<f32>().map_err(|_| Error::DeviceFault)?.iter().map(|v| v.to_bits()).collect();
    if actual != special_bits.into_iter().rev().collect::<Vec<_>>() {
        return Err(Error::DeviceFault);
    }
    let integers = NativeTensor::from_data(TensorData::new(alloc::vec![i32::MIN, -1, 0, i32::MAX], [4]))?
        .flip(&[0])?
        .copy()?;
    if integers.read()?.as_slice::<i32>().map_err(|_| Error::DeviceFault)? != [i32::MAX, 0, -1, i32::MIN] {
        return Err(Error::DeviceFault);
    }
    let empty = NativeTensor::from_data(TensorData::new(Vec::<f32>::new(), [2, 0, 3]))?;
    let before = backend.queue_profile()?;
    let empty = empty.scalar(Elementwise::Equal, 0.5)?;
    let after = backend.queue_profile()?;
    if !empty.layout().is_empty() || !empty.read()?.as_bytes().is_empty() || before.launches != after.launches {
        return Err(Error::DeviceFault);
    }
    Ok(())
}

fn verify_guarded_elementwise(backend: &Nvidia) -> Result<(), Error> {
    use d3os_ai::native::layout::Layout;
    let count = 257;
    let left: Vec<u32> = (0..count).map(|i| (i as f32 / 16.0 - 4.0).to_bits()).collect();
    let mut guarded = alloc::vec![0x6bad_beefu32];
    guarded.extend(&left);
    guarded.push(0x6bad_beef);
    let a = backend.upload_words(&guarded)?;
    let right = [0x6bad_beef, 0.25f32.to_bits(), 0x6bad_beef];
    let b = backend.upload_words(&right)?;
    let output = backend.upload_words(&alloc::vec![0x6bad_beef; count+2])?;
    let lhs = Layout::contiguous(Shape::new([count]))?.flip(&[0])?;
    let rhs = Layout::contiguous(Shape::new([1]))?.expand(Shape::new([count]))?;
    let parameters = lhs.elementwise_parameters(&rhs, Elementwise::Add, 0, false)?;
    backend.dispatch_words(abi::Kernel::TensorElementwise, &[(&a, 1), (&b, 1), (&output, 1)], &parameters)?;
    let mut actual = alloc::vec![0; count+2];
    backend.read_words_into(&output, 0, &mut actual)?;
    let mut expected = alloc::vec![0x6bad_beef];
    expected.extend(left.iter().rev().map(|&v| (f32::from_bits(v) + 0.25).to_bits()));
    expected.push(0x6bad_beef);
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    let mut malformed = parameters;
    malformed[4] += count;
    if backend.dispatch_words(abi::Kernel::TensorElementwise, &[(&a, 1), (&b, 1), (&output, 1)], &malformed) != Err(Error::InvalidParameter) {
        return Err(Error::DeviceFault);
    }
    backend.read_words_into(&output, 0, &mut actual)?;
    if actual != expected {
        return Err(Error::DeviceFault);
    }
    backend.read_words_into(&a, 0, &mut actual)?;
    if actual != guarded {
        return Err(Error::DeviceFault);
    }
    let mut actual = [0u32; 3];
    backend.read_words_into(&b, 0, &mut actual)?;
    if actual != right {
        return Err(Error::DeviceFault);
    }
    Ok(())
}
