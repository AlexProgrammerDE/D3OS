//! Guarded packed assignment and exact wide-index rejection on resident storage.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{BoolStore, DType, Scalar, Shape, TensorData},
        tensor::{IndexingUpdateOp, Slice},
    },
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{
    Backend as _, Error,
    nvidia::{GpuBuffer, Nvidia},
};
use d3os_nvidia::compute::{Kernel, storage::Width};

const GUARD: u32 = 0x6bad_beef;
const INDEX_DTYPES: [DType; 8] = [DType::I64, DType::I32, DType::I16, DType::I8, DType::U64, DType::U32, DType::U16, DType::U8];

fn require(value: bool) -> Result<(), Error> {
    if value { Ok(()) } else { Err(Error::DeviceFault) }
}
fn payload(count: usize, size: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|i| ((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x7ff8_abcd_8000_0001).to_le_bytes()[..size].to_vec())
        .collect()
}
fn pack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|b| {
            let mut word = [0u8; 4];
            word[..b.len()].copy_from_slice(b);
            u32::from_le_bytes(word)
        })
        .collect()
}
fn guarded(bytes: &[u8]) -> Vec<u32> {
    let mut words = vec![GUARD];
    words.extend(pack(bytes));
    words.push(GUARD);
    words
}
fn check(backend: &Nvidia, buffer: &GpuBuffer<u32>, expected: &[u32]) -> Result<(), Error> {
    let mut actual = vec![0; expected.len()];
    backend.read_words_into(buffer, 0, &mut actual)?;
    if let Some((i, (a, b))) = actual.iter().zip(expected).enumerate().find(|(_, (a, b))| a != b) {
        crate::report(format_args!("INDEXED_WORD_MISMATCH at={i} actual={a:#x} expected={b:#x}"));
        return Err(Error::DeviceFault);
    }
    Ok(())
}
fn raw(backend: &Nvidia, width: Width, shape: &[usize], axis: usize, invalid: bool, offset: usize, queued_drop: bool) -> Result<(), Error> {
    let updates = 257;
    let size = width.bytes() as usize;
    let axis_len = shape[axis];
    let mut update_shape = shape.to_vec();
    update_shape[axis] = updates;
    let dense_indices = Layout::contiguous(Shape::from(update_shape))?;
    let count = dense_indices.len();
    let values = payload(count, size);
    let value_words = guarded(&values);
    let value_buffer = backend.upload_words(&value_words)?;
    let mut ids = vec![GUARD; count * 2 + 1];
    for v in 0..count {
        let position = v / dense_indices.strides()[axis] as usize % updates;
        ids[2 * (count - 1 - v) + 1] = if invalid && position == 1 {
            u32::MAX
        } else if invalid && position == updates - 2 {
            axis_len as u32
        } else {
            ((v % 5 + position * 3) % axis_len) as u32
        };
    }
    let id_words = guarded(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>());
    let index_buffer = backend.upload_words(&id_words)?;
    let index_layout = Layout::from_parts(
        dense_indices.shape().clone(),
        dense_indices.strides().iter().map(|s| -2 * s).collect(),
        (2 * count - 1) as i64,
        ids.len(),
    )?;
    let dense = Layout::contiguous(Shape::from(shape.to_vec()))?;
    let layout = Layout::from_parts(dense.shape().clone(), dense.strides().to_vec(), offset as i64, offset + dense.len())?;
    let mut p = layout.gather_parameters(&index_layout, axis)?;
    p[0] |= (width as usize) << 8;
    let initial = payload(layout.storage_elements(), size);
    let mut expected = guarded(&initial);
    // Prefix fields, physical tail fields, and both guard words must survive.
    let tail = initial.len() % 4;
    if tail != 0 {
        let last = expected.len() - 2;
        expected[last] |= 0xa5a5_a5a5 << (tail * 8);
    }
    let output = backend.upload_words(&expected)?;
    let mut expected_bytes: Vec<u8> = expected[1..expected.len() - 1].iter().flat_map(|i| i.to_le_bytes()).collect();
    for v in 0..count {
        let selected = ids[index_layout.element_offset(v)?] as usize;
        if selected < axis_len {
            let mut linear = v;
            let mut at = offset;
            for d in (0..shape.len()).rev() {
                let coord = linear % dense_indices.shape()[d];
                linear /= dense_indices.shape()[d];
                at += if d == axis { selected } else { coord } * dense.strides()[d] as usize;
            }
            expected_bytes[at * size..(at + 1) * size].copy_from_slice(&values[v * size..(v + 1) * size]);
        }
    }
    let len = expected.len();
    expected[1..len - 1].copy_from_slice(&pack(&expected_bytes));
    let status = backend.upload_words(&[GUARD, 0, GUARD])?;
    let before = backend.service_profile()?;
    if queued_drop {
        backend.with_commands(|| {
            backend.dispatch_words(
                Kernel::PackedScatterAssign,
                &[(&value_buffer, 1), (&index_buffer, 1), (&status, 1), (&output, 1)],
                &p,
            )?;
            drop(value_buffer);
            drop(index_buffer);
            Ok::<_, Error>(())
        })?;
        super::dispatch::resident(backend, &before, 0).map_err(|_| Error::DeviceFault)?;
        check(backend, &output, &expected)?;
        check(backend, &status, &[GUARD, u32::from(invalid), GUARD])?;
        return Ok(());
    }
    backend.with_commands(|| {
        backend.dispatch_words(
            Kernel::PackedScatterAssign,
            &[(&value_buffer, 1), (&index_buffer, 1), (&status, 1), (&output, 1)],
            &p,
        )
    })?;
    super::dispatch::resident(backend, &before, 0).map_err(|_| Error::DeviceFault)?;
    check(backend, &output, &expected)?;
    check(backend, &status, &[GUARD, u32::from(invalid), GUARD])?;
    check(backend, &value_buffer, &value_words)?;
    check(backend, &index_buffer, &id_words)?;
    let launches = backend.queue_profile()?.launches;
    for (field, value) in [(0, p[0] | 1 << 16), (16, 0), (2, 0)] {
        let mut bad = p;
        bad[field] = value;
        require(
            backend.dispatch_words(
                Kernel::PackedScatterAssign,
                &[(&value_buffer, 1), (&index_buffer, 1), (&status, 1), (&output, 1)],
                &bad,
            ) == Err(Error::InvalidParameter),
        )?;
    }
    require(
        backend.dispatch_words(
            Kernel::PackedScatterAssign,
            &[(&value_buffer, 1), (&index_buffer, 1), (&status, 1), (&value_buffer, 1)],
            &p,
        ) == Err(Error::InvalidParameter),
    )?;
    require(
        backend.dispatch_words(
            Kernel::PackedScatterAssign,
            &[(&value_buffer, 1), (&index_buffer, 1), (&output, 1), (&output, 1)],
            &p,
        ) == Err(Error::InvalidParameter),
    )?;
    require(
        backend
            .produce_words(
                expected.len() - 2,
                Kernel::PackedScatterAssign,
                &[(&value_buffer, 1), (&index_buffer, 1), (&status, 1)],
                &p,
            )
            .is_err(),
    )?;
    require(backend.queue_profile()?.launches == launches)?;
    check(backend, &output, &expected)?;
    Ok(())
}
fn indices(dtype: DType, values: &[u64], shape: Shape) -> Result<NativeTensor, Error> {
    NativeTensor::from_data(TensorData::from_bytes_vec(
        values.iter().flat_map(|v| v.to_le_bytes()[..dtype.size()].to_vec()).collect(),
        shape,
        dtype,
    ))
}
fn native(backend: &Nvidia, dtype: DType) -> Result<usize, Error> {
    let bytes = payload(10, 8);
    let data = NativeTensor::from_data(TensorData::from_bytes_vec(bytes.clone(), Shape::new([2, 5]), DType::F64))?;
    let ids = indices(dtype, &[99, 4, 99, 0, 99, 2, 99], Shape::new([7]))?.slice(&[Slice::new(1, Some(6), 2)])?;
    let before = backend.service_profile()?;
    let output = backend.with_commands(|| data.select(1, &ids))?;
    super::dispatch::resident(backend, &before, 1).map_err(|_| Error::DeviceFault)?;
    let expected: Vec<u8> = [4usize, 0, 2, 9, 5, 7].iter().flat_map(|&at| bytes[at * 8..(at + 1) * 8].to_vec()).collect();
    require(output.read()?.as_bytes() == expected)?;
    let rhs_bytes = payload(8, 8);
    let rhs = NativeTensor::from_data(TensorData::from_bytes_vec(rhs_bytes.clone(), Shape::new([2, 4]), DType::F64))?
        .slice(&[Slice::full(), Slice::new(1, Some(4), 1)])?
        .flip(&[1])?;
    let before = backend.service_profile()?;
    let output = backend.with_commands(|| data.select_update(1, &ids, &rhs, IndexingUpdateOp::Assign))?;
    super::dispatch::resident(backend, &before, 1).map_err(|_| Error::DeviceFault)?;
    let mut expected = bytes.clone();
    for r in 0..2 {
        for (p, dst) in [4, 0, 2].into_iter().enumerate() {
            let src = r * 4 + 3 - p;
            let dst = r * 5 + dst;
            expected[dst * 8..(dst + 1) * 8].copy_from_slice(&rhs_bytes[src * 8..(src + 1) * 8]);
        }
    }
    require(output.read()?.as_bytes() == expected)?;
    require(data.read()?.as_bytes() == bytes)?;
    let invalids: &[u64] = match dtype {
        DType::I64 => &[u64::MAX, 1 << 32, (1 << 32) + 1, 1 << 63, (-(1i64 << 32)) as u64, i64::MAX as u64],
        DType::U64 => &[u64::MAX, 1 << 32, (1 << 32) + 1, 1 << 63],
        DType::I32 | DType::U32 => &[u32::MAX as u64, 1 << 31],
        DType::I16 | DType::U16 => &[65535, 32768],
        _ => &[255, 128],
    };
    let one = NativeTensor::from_data(TensorData::from_bytes_vec(payload(2, 8), Shape::new([2, 1]), DType::F64))?;
    let mut cases = 2;
    for &bad in invalids {
        let ids = indices(dtype, &[bad], Shape::new([1]))?;
        let before = backend.service_profile()?;
        require(matches!(backend.with_commands(|| data.select(1, &ids)), Err(Error::InvalidParameter)))?;
        super::dispatch::resident(backend, &before, 1).map_err(|_| Error::DeviceFault)?;
        require(matches!(
            data.select_update(1, &ids, &one, IndexingUpdateOp::Assign),
            Err(Error::InvalidParameter)
        ))?;
        let ids = indices(dtype, &[bad, 0], Shape::new([1, 2]))?;
        require(matches!(data.gather_nd(&ids), Err(Error::InvalidParameter)))?;
        cases += 3;
    }
    // An empty non-I32 index must not cause payload work or change ownership.
    let empty = indices(dtype, &[], Shape::new([0]))?;
    let launches = backend.queue_profile()?.launches;
    require(data.select(1, &empty)?.layout().is_empty())?;
    require(backend.queue_profile()?.launches == launches)?;
    require(data.read()?.as_bytes() == bytes)?;
    // Bounds failures remain recoverable for a subsequent packed assignment.
    let valid = indices(dtype, &[1], Shape::new([1]))?;
    let recovered = data.select_update(1, &valid, &one, IndexingUpdateOp::Assign)?;
    let mut expected = bytes.clone();
    let replacement = payload(2, 8);
    for r in 0..2 {
        expected[(r * 5 + 1) * 8..(r * 5 + 2) * 8].copy_from_slice(&replacement[r * 8..(r + 1) * 8]);
    }
    require(recovered.read()?.as_bytes() == expected)?;
    require(data.read()?.as_bytes() == bytes)?;
    Ok(cases + 2)
}
fn view_bytes(tensor: &NativeTensor, storage: &[u8], size: usize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::with_capacity(tensor.layout().len() * size);
    for i in 0..tensor.layout().len() {
        let at = tensor.layout().element_offset(i)?;
        bytes.extend_from_slice(&storage[at * size..(at + 1) * size]);
    }
    Ok(bytes)
}

fn native_narrow(backend: &Nvidia, dtype: DType) -> Result<usize, Error> {
    let size = dtype.size();
    let bytes = payload(10, size);
    let original = NativeTensor::from_data(TensorData::from_bytes_vec(bytes.clone(), Shape::new([5, 2]), dtype))?;
    let data = original.clone().permute(&[1, 0])?.flip(&[1])?;
    let initial = view_bytes(&data, &bytes, size)?;
    let ids = indices(DType::I32, &[4, 99, 4, 99, 0], Shape::new([5]))?
        .slice(&[Slice::new(0, Some(5), 2)])?
        .flip(&[0])?;
    let mut cases = Vec::new();
    for remainder in 0..4 / size {
        // Contiguous views can start at any packed field, even past word zero.
        let offset = 4 / size + remainder;
        let storage = payload(offset + 6, size);
        let rhs = NativeTensor::from_data(TensorData::from_bytes_vec(storage.clone(), Shape::new([offset + 6]), dtype))?
            .slice(&[Slice::new(offset as isize, Some((offset + 6) as isize), 1)])?
            .reshape(Shape::new([2, 3]))?;
        require(rhs.layout().is_contiguous() && rhs.layout().offset() == offset)?;
        cases.push((rhs, storage));
    }
    // This reversed RHS shares the original destination's allocation.
    cases.push((data.clone().slice(&[Slice::full(), Slice::new(0, Some(3), 1)])?.flip(&[1])?, bytes.clone()));
    let storage = payload(6, size);
    let transposed = NativeTensor::from_data(TensorData::from_bytes_vec(storage.clone(), Shape::new([3, 2]), dtype))?.permute(&[1, 0])?;
    cases.push((transposed, storage));
    let storage = payload(2, size);
    let expanded = NativeTensor::from_data(TensorData::from_bytes_vec(storage.clone(), Shape::new([2, 1]), dtype))?.expand(Shape::new([2, 3]))?;
    cases.push((expanded, storage));
    let count = cases.len();
    for (rhs, storage) in cases {
        let rhs_bytes = view_bytes(&rhs, &storage, size)?;
        let mut expected = initial.clone();
        for r in 0..2 {
            for (position, selected) in [0, 4, 4].into_iter().enumerate() {
                let src = (r * 3 + position) * size;
                let dst = (r * 5 + selected) * size;
                expected[dst..dst + size].copy_from_slice(&rhs_bytes[src..src + size]);
            }
        }
        let before = backend.service_profile()?;
        let output = backend.with_commands(|| data.select_update(1, &ids, &rhs, IndexingUpdateOp::Assign))?;
        super::dispatch::resident(backend, &before, 1).map_err(|_| Error::DeviceFault)?;
        require(output.layout().is_contiguous() && output.layout().offset() == 0)?;
        require(output.read()?.as_bytes() == expected)?;
        require(rhs.read()?.as_bytes() == rhs_bytes)?;
        require(original.read()?.as_bytes() == bytes)?;
    }
    Ok(count)
}

fn native_index_max(backend: &Nvidia, dtype: DType, maximum: u64) -> Result<usize, Error> {
    let len = maximum as usize + 1;
    let bytes = payload(len, 2);
    let data = NativeTensor::from_data(TensorData::from_bytes_vec(bytes.clone(), Shape::new([len]), DType::U16))?;
    let ids = indices(dtype, &[maximum, 99, 0, 99, maximum], Shape::new([5]))?
        .slice(&[Slice::new(0, Some(5), 2)])?
        .flip(&[0])?;
    let before = backend.service_profile()?;
    let selected = backend.with_commands(|| data.select(0, &ids))?;
    super::dispatch::resident(backend, &before, 1).map_err(|_| Error::DeviceFault)?;
    let expected: Vec<u8> = [maximum as usize, 0, maximum as usize]
        .iter()
        .flat_map(|&i| bytes[i * 2..i * 2 + 2].to_vec())
        .collect();
    require(selected.read()?.as_bytes() == expected)?;
    let replacement = payload(3, 2);
    let rhs = NativeTensor::from_data(TensorData::from_bytes_vec(replacement.clone(), Shape::new([3]), DType::U16))?;
    let before = backend.service_profile()?;
    let output = backend.with_commands(|| data.select_update(0, &ids, &rhs, IndexingUpdateOp::Assign))?;
    super::dispatch::resident(backend, &before, 1).map_err(|_| Error::DeviceFault)?;
    let mut expected = bytes.clone();
    expected[..2].copy_from_slice(&replacement[2..4]);
    expected[(len - 1) * 2..len * 2].copy_from_slice(&replacement[4..6]);
    require(output.read()?.as_bytes() == expected)?;
    require(data.read()?.as_bytes() == bytes)?;
    Ok(2)
}

fn native_bool_nd(backend: &Nvidia, store: BoolStore) -> Result<usize, Error> {
    let dtype = DType::Bool(store);
    let bytes: Vec<u8> = [0u32, 1, 0, 1, 0, 1].iter().flat_map(|v| v.to_le_bytes()[..dtype.size()].to_vec()).collect();
    let data = NativeTensor::from_data(TensorData::from_bytes_vec(bytes.clone(), Shape::new([2, 3]), dtype))?;
    for (coordinates, shape, expected_shape, expected) in [
        (vec![1, 2], Shape::new([2]), Shape::new([]), bytes[5 * dtype.size()..].to_vec()),
        (vec![1], Shape::new([1, 1]), Shape::new([1, 3]), bytes[3 * dtype.size()..].to_vec()),
    ] {
        let ids = indices(DType::I32, &coordinates, shape)?;
        let before = backend.service_profile()?;
        let output = backend.with_commands(|| data.gather_nd(&ids))?;
        super::dispatch::resident(backend, &before, 1).map_err(|_| Error::DeviceFault)?;
        require(output.layout().shape() == &expected_shape)?;
        let actual = output.read()?;
        require(actual.dtype() == dtype && actual.as_bytes() == expected)?;
    }
    require(data.read()?.as_bytes() == bytes)?;
    Ok(2)
}

fn scalar_fill() -> Result<(), Error> {
    let data = NativeTensor::full_scalar(Shape::new([3]), DType::F64, Scalar::new(0f64, &DType::F64))?;
    let bits = 0x3ff0_0000_0000_0001u64;
    for store in [BoolStore::U8, BoolStore::U32] {
        let dtype = DType::Bool(store);
        let mask = NativeTensor::from_data(TensorData::from_bytes_vec(
            [false, true, false]
                .iter()
                .flat_map(|&v| u32::from(v).to_le_bytes()[..dtype.size()].to_vec())
                .collect(),
            Shape::new([3]),
            dtype,
        ))?;
        let out = data.mask_fill_scalar(&mask, Scalar::new(f64::from_bits(bits), &DType::F64))?;
        let expected: Vec<u8> = [0u64, bits, 0].iter().flat_map(|i| i.to_le_bytes()).collect();
        require(out.read()?.as_bytes() == expected)?;
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let mut raw_cases = 0;
    for width in Width::ALL {
        for rows in [1, 127, 128, 129, 257] {
            for axis in [0, 1] {
                for invalid in [false, true] {
                    raw(backend, width, &if axis == 0 { [5, rows] } else { [rows, 5] }, axis, invalid, 0, false)
                        .map_err(|e| format!("packed indexed raw {width:?} rows={rows} axis={axis} invalid={invalid}: {e}"))?;
                    raw_cases += 1;
                }
            }
        }
    }
    for width in Width::ALL {
        // Rank limits, every rank-eight axis, and an odd extent at a nonzero base.
        for (shape, axes, offset) in [(vec![5], 1, 2), (vec![2; 8], 8, 0)] {
            for axis in 0..axes {
                for invalid in [false, true] {
                    raw(backend, width, &shape, axis, invalid, offset, false).map_err(|e| {
                        format!(
                            "packed indexed rank={} {width:?} axis={axis} offset={offset} invalid={invalid}: {e}",
                            shape.len()
                        )
                    })?;
                    raw_cases += 1;
                }
            }
        }
        // Last input owners disappear inside the open command scope.
        raw(backend, width, &[5], 0, false, 1, true).map_err(|e| format!("packed indexed queued {width:?}: {e}"))?;
        raw_cases += 1;
    }
    for width in [Width::One, Width::Two] {
        // Rows 127/128 share packed words across the 128-lane block boundary.
        for invalid in [false, true] {
            raw(backend, width, &[26, 2, 5], 1, invalid, 0, false).map_err(|e| format!("packed indexed cross-block CAS {width:?} invalid={invalid}: {e}"))?;
            raw_cases += 1;
        }
    }
    let mut native_cases = 0;
    for dtype in INDEX_DTYPES {
        native_cases += native(backend, dtype).map_err(|e| format!("native index {dtype:?}: {e}"))?;
    }
    for dtype in [DType::U8, DType::U16] {
        native_cases += native_narrow(backend, dtype).map_err(|e| format!("native packed views {dtype:?}: {e}"))?;
    }
    for (dtype, maximum) in [(DType::I8, 127), (DType::I16, 32767), (DType::U8, 255), (DType::U16, 65535)] {
        native_cases += native_index_max(backend, dtype, maximum).map_err(|e| format!("native index maximum {dtype:?}: {e}"))?;
    }
    for store in [BoolStore::U8, BoolStore::U32] {
        native_cases += native_bool_nd(backend, store).map_err(|e| format!("native bool ND {store:?}: {e}"))?;
    }
    scalar_fill().map_err(|e| format!("F64 scalar exactness: {e}"))?;
    crate::report(format_args!(
        "NVIDIA_BURN_INDEXED_OK raw={raw_cases} native={native_cases} index_dtypes=8 payload_widths=1,2,4,8 guards=true tails=preserved duplicates=ordered wide_index_aliases=rejected resident=true rank_limits=1,8 cross_block_cas=verified queued_input_drop=verified narrow_offsets=all narrow_index_max=verified native_bool_nd=verified scalar_f64_ulp=verified"
    ));
    Ok(())
}
