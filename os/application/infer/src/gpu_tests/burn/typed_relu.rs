//! Guarded typed ReLU composition, direct backward, and resident lifetimes.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{
            Backend as _, BoolDType, BoolStore, DType, Dispatch, Shape, Slice, TensorData, TensorMetadata,
            ops::{ActivationOps, BoolTensorOps, FloatTensorOps, IntTensorOps},
            tensor::FloatTensor,
        },
        tensor::{Tensor, TensorCreationOptions, activation::relu, try_read_sync},
    },
    native::{Native, NativeDevice, layout::Layout},
    validation::relu as oracle,
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{
    self as abi, Kernel,
    cast::Format,
    elementwise::Elementwise,
    packed_updates,
    predicate::{self, Predicate},
    storage::Width,
    words::WordOp,
};

type Result<T> = core::result::Result<T, String>;
const GUARD: u32 = 0x6bad_beef;
const POISON: u32 = 0xa53c_d791;
const FORMATS: [(DType, Format); 4] = [
    (DType::F64, Format::F64),
    (DType::Flex32, Format::Flex32),
    (DType::F16, Format::F16),
    (DType::BF16, Format::BF16),
];
#[derive(Default, Debug, PartialEq, Eq)]
struct Counts {
    raw: usize,
    native: usize,
    backward: usize,
    masks: usize,
    public: usize,
    rejected: usize,
    empty: usize,
    queued: usize,
    exhaustive: usize,
    exhaustive_values: usize,
}
fn gpu<T>(value: d3os_compute::Result<T>) -> Result<T> {
    value.map_err(|e| format!("typed ReLU: {e}"))
}
fn data(dtype: DType, shape: &[usize], values: &[u64]) -> TensorData {
    TensorData::from_bytes_vec(oracle::pack(dtype, values), Shape::from(shape), dtype)
}
fn packed(dtype: DType, values: &[u64]) -> Vec<u32> {
    oracle::pack(dtype, values)
        .chunks(4)
        .map(|chunk| {
            let mut bytes = [0; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect()
}
fn check_data(actual: &TensorData, dtype: DType, shape: &[usize], expected: &[u64]) -> Result<()> {
    if actual.dtype() != dtype || actual.shape().as_slice() != shape || actual.as_bytes() != oracle::pack(dtype, expected) {
        let bits = if actual.as_bytes().len().is_multiple_of(dtype.size()) {
            oracle::unpack(dtype, actual.as_bytes())
        } else {
            vec![]
        };
        let mismatch = bits.iter().zip(expected).position(|(a, b)| a != b);
        return Err(format!(
            "typed ReLU bits/metadata dtype={dtype:?} shape={shape:?} mismatch={mismatch:?} actual_bits={:x?} expected_bits={:x?} actual_dtype={:?} actual_shape={:?} bytes={}",
            mismatch.and_then(|i| bits.get(i)),
            mismatch.and_then(|i| expected.get(i)),
            actual.dtype(),
            actual.shape(),
            actual.as_bytes().len(),
        ));
    }
    Ok(())
}
fn read(tensor: FloatTensor<Native>, dtype: DType, shape: &[usize], expected: &[u64]) -> Result<()> {
    let value = try_read_sync(Native::float_into_data(tensor))
        .ok_or("typed ReLU read pending")?
        .map_err(|e| format!("typed ReLU read: {e:?}"))?;
    check_data(&value, dtype, shape, expected)
}
fn unchanged(backend: &Nvidia, buffer: &d3os_compute::nvidia::GpuBuffer<u32>, expected: &[u32]) -> Result<()> {
    let mut actual = vec![0; expected.len()];
    gpu(backend.read_words_into(buffer, 0, &mut actual))?;
    if actual != expected {
        return Err(String::from("typed ReLU source/guards/poison changed"));
    }
    Ok(())
}
fn words(backend: &Nvidia, buffer: &d3os_compute::nvidia::GpuBuffer<u32>, dtype: DType, expected: &[u64]) -> Result<()> {
    let mut want = vec![GUARD];
    want.extend(packed(dtype, expected));
    want.push(GUARD);
    // Equality includes every output word and the zeroed high half of odd tails.
    unchanged(backend, buffer, &want)
}
fn resident(
    backend: &Nvidia, services: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT], raw: bool,
) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE] {
        if after[service as usize].calls != services[service as usize].calls {
            return Err(format!("typed ReLU resident service={service}"));
        }
    }
    for (service, expected) in [(abi::LAUNCH, if raw { 3 } else { 0 }), (abi::LAUNCH_ALLOCATE, if raw { 0 } else { 2 })] {
        if after[service as usize].calls - services[service as usize].calls != expected {
            return Err(format!("typed ReLU service={service} expected={expected}"));
        }
    }
    let after = gpu(backend.profile())?;
    for (index, (old, new)) in profile.iter().zip(after).enumerate() {
        let expected = if index == Kernel::TensorPredicate as usize {
            1
        } else if index == Kernel::PackedMaskWhere as usize {
            if raw { 2 } else { 1 }
        } else {
            0
        };
        if new.calls - old.calls != expected {
            return Err(format!("typed ReLU kernel={index} expected={expected}"));
        }
    }
    Ok(())
}
fn no_work(backend: &Nvidia, services: &[abi::ServiceEntry; abi::SERVICE_OPERATIONS], profile: &[abi::ProfileEntry; abi::KERNEL_COUNT]) -> Result<()> {
    let after = gpu(backend.service_profile())?;
    for service in [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE, abi::LAUNCH, abi::LAUNCH_ALLOCATE] {
        if after[service as usize].calls != services[service as usize].calls {
            return Err(format!("typed ReLU no-work service={service}"));
        }
    }
    if profile.iter().zip(gpu(backend.profile())?).any(|(a, b)| a.calls != b.calls) {
        return Err(String::from("typed ReLU empty/invalid launched work"));
    }
    Ok(())
}

#[derive(Clone)]
enum View {
    Shape,
    Slice { step: usize, reverse: bool },
    Permuted,
    Broadcast,
}
struct Input {
    dtype: DType,
    format: Format,
    storage: Vec<u64>,
    logical: Vec<u64>,
    layout: Layout,
    view: View,
}
impl Input {
    fn flat(dtype: DType, format: Format, values: Vec<u64>) -> Result<Self> {
        Ok(Self {
            dtype,
            format,
            layout: gpu(Layout::contiguous(Shape::new([values.len()])))?,
            logical: values.clone(),
            storage: values,
            view: View::Shape,
        })
    }
    fn primitive(&self) -> FloatTensor<Native> {
        let tensor = Native::float_from_data(data(self.dtype, &[self.storage.len()], &self.storage), &NativeDevice);
        match self.view {
            View::Shape => Native::float_reshape(tensor, self.layout.shape().clone()),
            View::Slice { step, reverse } => {
                let value = Native::float_slice(tensor, &[Slice::new(1, Some((1 + step * self.logical.len()) as isize), step as isize)]);
                if reverse { Native::float_flip(value, &[0]) } else { value }
            }
            View::Permuted => Native::float_flip(Native::float_permute(Native::float_reshape(tensor, Shape::new([2, 3, 4])), &[2, 0, 1]), &[0, 2]),
            View::Broadcast => Native::float_expand(
                Native::float_reshape(Native::float_slice(tensor, &[Slice::new(1, Some(2), 1)]), Shape::new([1, 1])),
                Shape::new([3, 5]),
            ),
        }
    }
    fn expected(&self) -> Vec<u64> {
        self.logical.iter().map(|&x| oracle::expected_bits(self.dtype, x)).collect()
    }
    fn gradients(&self) -> Vec<u64> {
        let pool = oracle::gradient_pool(self.dtype);
        (0..self.logical.len()).map(|i| pool[i % pool.len()]).collect()
    }
    fn gradient_view(&self) -> Result<Self> {
        let pool = oracle::gradient_pool(self.dtype);
        let storage: Vec<_> = (0..self.storage.len()).map(|i| pool[i % pool.len()]).collect();
        let logical = (0..self.logical.len())
            .map(|i| gpu(self.layout.element_offset(i)).map(|offset| storage[offset]))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dtype: self.dtype,
            format: self.format,
            storage,
            logical,
            layout: self.layout.clone(),
            view: self.view.clone(),
        })
    }
}

fn predicate_parameters(input: &Input, narrow: bool) -> Result<[usize; 32]> {
    let mut p = gpu(input.layout.word_parameters(&input.layout, WordOp::Equal, 0, false))?;
    p[2] = Predicate::LessEqual as usize
        | (input.format as usize) << 8
        | predicate::SCALAR_RHS as usize
        | if narrow { predicate::PACKED_BOOL as usize } else { 0 };
    p[3] = 0;
    p[5] = 0;
    p[7] = 0;
    p[24..32].fill(0);
    Ok(p)
}
fn mask_parameters(dtype: DType, layout: &Layout, narrow: bool) -> Result<[usize; 32]> {
    let width = match dtype.size() {
        2 => Width::Two,
        4 => Width::Four,
        8 => Width::Eight,
        _ => unreachable!(),
    };
    let mut p = gpu(layout.elementwise_parameters(layout, Elementwise::Copy, 0, false))?;
    p[2] = width as usize | packed_updates::SCALAR as usize | if narrow { packed_updates::MASK_U8 as usize } else { 0 };
    p[3] = 0;
    p[5] = 0;
    p[7] = 0;
    p[24..32].fill(0);
    Ok(p)
}

/// Independent destinations are poisoned, with guards around source, predicate,
/// forward and backward buffers. Both bool stores use the exact same data view.
fn raw(backend: &Nvidia, input: &Input, bool_dtype: BoolDType, counts: &mut Counts) -> Result<()> {
    let narrow = bool_dtype == BoolDType::U8;
    let mut source_words = vec![GUARD];
    source_words.extend(packed(input.dtype, &input.storage));
    if input.dtype.size() == 2 && input.storage.len() % 2 == 1 {
        *source_words.last_mut().unwrap() |= 0xffff_0000;
    }
    source_words.push(GUARD);
    let source = gpu(backend.upload_words(&source_words))?;
    let gradients = input.gradients();
    let mut grad_words = vec![GUARD];
    grad_words.extend(packed(input.dtype, &gradients));
    grad_words.push(GUARD);
    let grad = gpu(backend.upload_words(&grad_words))?;
    let mask_words = if narrow { input.logical.len().div_ceil(4) } else { input.logical.len() };
    let mut mask_poison = vec![POISON; mask_words + 2];
    mask_poison[0] = GUARD;
    mask_poison[mask_words + 1] = GUARD;
    let mask = gpu(backend.upload_words(&mask_poison))?;
    let n = packed(input.dtype, &gradients).len();
    let mut output_poison = vec![POISON; n + 2];
    output_poison[0] = GUARD;
    output_poison[n + 1] = GUARD;
    let output = gpu(backend.upload_words(&output_poison))?;
    let backward = gpu(backend.upload_words(&output_poison))?;
    let unused = gpu(backend.upload_words(&[GUARD, POISON, GUARD]))?;
    let p = predicate_parameters(input, narrow)?;
    let forward_p = mask_parameters(input.dtype, &input.layout, narrow)?;
    let backward_p = mask_parameters(input.dtype, &gpu(Layout::contiguous(input.layout.shape().clone()))?, narrow)?;
    let services = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    gpu(backend.with_commands(|| {
        backend.dispatch_words(Kernel::TensorPredicate, &[(&source, 1), (&unused, 1), (&mask, 1)], &p)?;
        backend.dispatch_words(Kernel::PackedMaskWhere, &[(&source, 1), (&unused, 1), (&mask, 1), (&output, 1)], &forward_p)?;
        // <=0(input) equals <=0(ReLU(input)), including all NaN encodings.
        backend.dispatch_words(Kernel::PackedMaskWhere, &[(&grad, 1), (&unused, 1), (&mask, 1), (&backward, 1)], &backward_p)
    }))?;
    resident(backend, &services, &profile, true)?;
    let expected = input.expected();
    words(backend, &output, input.dtype, &expected)?;
    let dx: Vec<_> = expected
        .iter()
        .zip(&gradients)
        .map(|(&x, &g)| oracle::backward_bits(input.dtype, x, g))
        .collect();
    words(backend, &backward, input.dtype, &dx)?;
    let mut mask_expected = vec![GUARD];
    let bytes: Vec<_> = input
        .logical
        .iter()
        .flat_map(|&x| u32::from(oracle::lower_equal_zero(input.dtype, x)).to_le_bytes()[..DType::Bool(bool_dtype).size()].to_vec())
        .collect();
    for chunk in bytes.chunks(4) {
        let mut b = [0; 4];
        b[..chunk.len()].copy_from_slice(chunk);
        mask_expected.push(u32::from_le_bytes(b));
    }
    mask_expected.push(GUARD);
    unchanged(backend, &mask, &mask_expected)?;
    unchanged(backend, &source, &source_words)?;
    unchanged(backend, &grad, &grad_words)?;
    unchanged(backend, &unused, &[GUARD, POISON, GUARD])?;
    counts.raw += 1;
    Ok(())
}
fn native(backend: &Nvidia, input: &Input, counts: &mut Counts) -> Result<()> {
    let expected = input.expected();
    let shape = input.layout.shape().as_slice();
    for shared in [false, true] {
        let source = input.primitive();
        let alias = shared.then(|| source.clone());
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = Native::relu(source);
        resident(backend, &services, &profile, false)?;
        read(output.clone(), input.dtype, shape, &expected)?;
        if let Some(alias) = alias {
            read(alias, input.dtype, shape, &input.logical)?;
        }
        let gradients = input.gradients();
        let grad = Native::float_from_data(data(input.dtype, shape, &gradients), &NativeDevice);
        let grad_alias = grad.clone();
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let dx = Native::relu_backward(output.clone(), grad);
        resident(backend, &services, &profile, false)?;
        let want: Vec<_> = expected
            .iter()
            .zip(&gradients)
            .map(|(&x, &g)| oracle::backward_bits(input.dtype, x, g))
            .collect();
        read(dx, input.dtype, shape, &want)?;
        read(grad_alias, input.dtype, shape, &gradients)?;
        read(output, input.dtype, shape, &expected)?;
        counts.native += 1;
        counts.backward += 1;
    }
    // Backward also accepts views directly, independently of the dense output
    // produced by forward. Give its gradient the same signed/broadcast geometry.
    let source = input.primitive();
    let alias = source.clone();
    let gradient = input.gradient_view()?;
    let grad = gradient.primitive();
    let grad_alias = grad.clone();
    let services = gpu(backend.service_profile())?;
    let profile = gpu(backend.profile())?;
    let dx = Native::relu_backward(source, grad);
    resident(backend, &services, &profile, false)?;
    let want: Vec<_> = input
        .logical
        .iter()
        .zip(&gradient.logical)
        .map(|(&x, &g)| oracle::backward_bits(input.dtype, x, g))
        .collect();
    read(dx, input.dtype, shape, &want)?;
    read(alias, input.dtype, shape, &input.logical)?;
    read(grad_alias, input.dtype, shape, &gradient.logical)?;
    counts.backward += 1;
    // The inherited method follows the fixed device bool setting. Exercise both
    // storage alternatives explicitly without mutating Burn's global registry.
    for bool_dtype in [BoolDType::U8, BoolDType::U32] {
        let source = input.primitive();
        let alias = source.clone();
        let gradients = input.gradients();
        let grad = Native::float_from_data(data(input.dtype, shape, &gradients), &NativeDevice);
        let grad_alias = grad.clone();
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let mask = Native::float_lower_equal_elem(source, 0.into(), bool_dtype);
        let dx = Native::float_mask_fill(grad, mask, 0.into());
        resident(backend, &services, &profile, false)?;
        let want: Vec<_> = input
            .logical
            .iter()
            .zip(&gradients)
            .map(|(&x, &g)| oracle::backward_bits(input.dtype, x, g))
            .collect();
        read(dx, input.dtype, shape, &want)?;
        read(alias, input.dtype, shape, &input.logical)?;
        read(grad_alias, input.dtype, shape, &gradients)?;
        counts.masks += 1;
    }
    Ok(())
}
fn exercise(backend: &Nvidia, input: &Input, counts: &mut Counts) -> Result<()> {
    for bool_dtype in [BoolDType::U8, BoolDType::U32] {
        raw(backend, input, bool_dtype, counts)?;
    }
    native(backend, input, counts)
}
fn matrices(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, format) in FORMATS {
        let pool = oracle::pool(dtype);
        exercise(backend, &Input::flat(dtype, format, pool.clone())?, counts)?;
        for length in [1, 2, 3, 4, 5, 127, 128, 129, 257] {
            for (step, reverse) in [(1, false), (2, false), (2, true)] {
                let storage: Vec<_> = (0..length * step + 3).map(|i| pool[i % pool.len()]).collect();
                let indices: Vec<_> = (0..length).map(|i| 1 + step * i).collect();
                let logical = if reverse {
                    indices.iter().rev().map(|&i| storage[i]).collect()
                } else {
                    indices.iter().map(|&i| storage[i]).collect()
                };
                let layout = gpu(Layout::from_parts(
                    Shape::new([length]),
                    vec![if reverse { -(step as i64) } else { step as i64 }],
                    if reverse { (1 + step * (length - 1)) as i64 } else { 1 },
                    storage.len(),
                ))?;
                exercise(
                    backend,
                    &Input {
                        dtype,
                        format,
                        storage,
                        logical,
                        layout,
                        view: View::Slice { step, reverse },
                    },
                    counts,
                )?;
            }
        }
        let storage: Vec<_> = (0..24).map(|i| pool[i % pool.len()]).collect();
        let permuted = Input {
            dtype,
            format,
            logical: (0..4)
                .flat_map(|i| (0..2).flat_map(move |j| (0..3).map(move |k| j * 12 + (2 - k) * 4 + 3 - i)))
                .map(|i| storage[i])
                .collect(),
            storage,
            layout: gpu(Layout::contiguous(Shape::new([2, 3, 4]))
                .and_then(|l| l.permute(&[2, 0, 1]))
                .and_then(|l| l.flip(&[0, 2])))?,
            view: View::Permuted,
        };
        exercise(backend, &permuted, counts)?;
        let broadcast = Input {
            dtype,
            format,
            storage: vec![pool[25], pool[15], pool[24]],
            logical: vec![pool[15]; 15],
            layout: gpu(Layout::from_parts(Shape::new([3, 5]), vec![0, 0], 1, 3))?,
            view: View::Broadcast,
        };
        exercise(backend, &broadcast, counts)?;
        for shape in [vec![], vec![1, 1, 1, 1, 1, 1, 1, 5]] {
            let values: Vec<_> = (0..shape.iter().product()).map(|i| pool[i % pool.len()]).collect();
            let mut input = Input::flat(dtype, format, values)?;
            input.layout = gpu(Layout::contiguous(Shape::from(shape)))?;
            exercise(backend, &input, counts)?;
        }
    }
    for (dtype, format) in [FORMATS[2], FORMATS[3]] {
        let input = Input::flat(dtype, format, (0..=u16::MAX).map(u64::from).collect())?;
        exercise(backend, &input, counts)?;
        counts.exhaustive += 1;
        counts.exhaustive_values += input.logical.len();
    }
    Ok(())
}
fn public_and_lifetimes(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    let device = d3os_ai::native_device().map_err(|e| format!("typed ReLU device: {e}"))?;
    for (dtype, format) in FORMATS {
        let input = Input::flat(dtype, format, oracle::pool(dtype))?;
        let source = Tensor::<1>::from_data(
            data(dtype, &[input.logical.len()], &input.logical),
            TensorCreationOptions::new(device.clone()).with_dtype(dtype),
        );
        let alias = source.clone();
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = relu(source);
        resident(backend, &services, &profile, false)?;
        check_data(
            &output.clone().try_into_data().map_err(|e| format!("typed ReLU Dispatch: {e:?}"))?,
            dtype,
            &[input.logical.len()],
            &input.expected(),
        )?;
        let gradients = input.gradients();
        let grad = Tensor::<1>::from_data(
            data(dtype, &[gradients.len()], &gradients),
            TensorCreationOptions::new(device.clone()).with_dtype(dtype),
        );
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let dx = Dispatch::relu_backward(output.into_dispatch(), grad.into_dispatch());
        resident(backend, &services, &profile, false)?;
        let want: Vec<_> = input
            .expected()
            .iter()
            .zip(&gradients)
            .map(|(&x, &g)| oracle::backward_bits(dtype, x, g))
            .collect();
        let dx = try_read_sync(Dispatch::float_into_data(dx))
            .ok_or("typed ReLU Dispatch read pending")?
            .map_err(|e| format!("typed ReLU Dispatch backward: {e:?}"))?;
        check_data(&dx, dtype, &[want.len()], &want)?;
        check_data(
            &alias.try_into_data().map_err(|e| format!("typed ReLU alias: {e:?}"))?,
            dtype,
            &[input.logical.len()],
            &input.logical,
        )?;
        counts.public += 2;
        // The only source owner disappears inside the unfinished command scope.
        let source = input.primitive();
        let grad = Native::float_from_data(data(dtype, &[gradients.len()], &gradients), &NativeDevice);
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let output = gpu(backend.with_commands(|| Ok::<_, Error>(Native::relu(source))))?;
        resident(backend, &services, &profile, false)?;
        read(output.clone(), dtype, &[input.logical.len()], &input.expected())?;
        let services = gpu(backend.service_profile())?;
        let profile = gpu(backend.profile())?;
        let dx = gpu(backend.with_commands(|| Ok::<_, Error>(Native::relu_backward(output, grad))))?;
        resident(backend, &services, &profile, false)?;
        read(dx, dtype, &[want.len()], &want)?;
        counts.queued += 2;
    }
    Ok(())
}
fn empty_and_errors(backend: &Nvidia, counts: &mut Counts) -> Result<()> {
    for (dtype, _) in FORMATS {
        for shape in [vec![0], vec![3, 0], vec![0, 2, 3], vec![1, 1, 1, 1, 0, 1, 1, 3]] {
            let source = Native::float_from_data(data(dtype, &shape, &[]), &NativeDevice);
            let grad = source.clone();
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let output = Native::relu(source);
            let dx = Native::relu_backward(output.clone(), grad);
            no_work(backend, &services, &profile)?;
            read(output, dtype, &shape, &[])?;
            read(dx, dtype, &shape, &[])?;
            counts.empty += 2;
        }
    }
    for dtype in [DType::I32, DType::U8, DType::Bool(BoolStore::U8), DType::Bool(BoolStore::U32)] {
        for length in [0, 3] {
            // Construct a valid primitive of its actual kind, then deliberately
            // call the activation trait with non-floating primitive metadata.
            let value = data(dtype, &[length], &vec![0; length]);
            let source = if dtype.is_bool() {
                Native::bool_from_data(value, &NativeDevice)
            } else {
                Native::int_from_data(value, &NativeDevice)
            };
            let services = gpu(backend.service_profile())?;
            let profile = gpu(backend.profile())?;
            let failed = Native::relu(source);
            no_work(backend, &services, &profile)?;
            if failed.dtype() != dtype || failed.shape().as_slice() != [length] {
                return Err(String::from("typed ReLU failed metadata changed"));
            }
            if !Native::sync(&NativeDevice).is_err_and(|e| !e.is_device_poisoned()) {
                return Err(String::from("typed ReLU missing recoverable error"));
            }
            let failed_read = try_read_sync(Native::float_into_data(failed)).ok_or("typed ReLU failed read pending")?;
            if !failed_read.is_err_and(|e| !e.is_device_poisoned()) {
                return Err(String::from("typed ReLU invalid primitive read succeeded"));
            }
            no_work(backend, &services, &profile)?;
            Native::sync(&NativeDevice).map_err(|e| format!("typed ReLU recovery: {e:?}"))?;
            let pool = oracle::pool(DType::F64);
            let input = Input::flat(DType::F64, Format::F64, pool)?;
            read(Native::relu(input.primitive()), DType::F64, &[input.logical.len()], &input.expected())?;
            counts.rejected += 1;
        }
    }
    Ok(())
}
pub(super) fn run(backend: &Nvidia) -> Result<()> {
    let mut counts = Counts::default();
    matrices(backend, &mut counts)?;
    public_and_lifetimes(backend, &mut counts)?;
    empty_and_errors(backend, &mut counts)?;
    let expected = Counts {
        raw: 260,
        native: 260,
        backward: 390,
        masks: 260,
        public: 8,
        rejected: 8,
        empty: 32,
        queued: 8,
        exhaustive: 2,
        exhaustive_values: 131072,
    };
    if counts != expected {
        return Err(format!("typed ReLU counts {counts:?}, expected {expected:?}"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_TYPED_RELU_OK formats=4 raw={} native={} backward={} masks={} public={} rejected={} empty={} queued={} exhaustive={} exhaustive_values={} exact_bits=true guards=ok tails=zero resident=true native_bool=U32 explicit_mask_stores=2 f32_gate=separate autodiff=not_claimed backend=nvidia",
        counts.raw,
        counts.native,
        counts.backward,
        counts.masks,
        counts.public,
        counts.rejected,
        counts.empty,
        counts.queued,
        counts.exhaustive,
        counts.exhaustive_values
    ));
    Ok(())
}
