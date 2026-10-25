//! Row ownership, guarded writes, ordered duplicates, and recoverable bounds.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Shape, TensorData},
        tensor::{IndexingUpdateOp, Slice},
    },
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, indexing::Scatter};

fn update(a: u32, b: u32, op: Scatter) -> u32 {
    match op {
        Scatter::FloatAssign | Scatter::IntAssign => b,
        Scatter::FloatMultiply => (f32::from_bits(a) * f32::from_bits(b)).to_bits(),
        Scatter::FloatMin => f32::from_bits(a).min(f32::from_bits(b)).to_bits(),
        Scatter::FloatMax => f32::from_bits(a).max(f32::from_bits(b)).to_bits(),
        Scatter::IntMultiply => a.wrapping_mul(b),
        Scatter::IntMin => (a as i32).min(b as i32) as u32,
        Scatter::IntMax => (a as i32).max(b as i32) as u32,
        Scatter::FloatAdd => (f32::from_bits(a) + f32::from_bits(b)).to_bits(),
        Scatter::IntAdd => a.wrapping_add(b),
        _ => unreachable!(),
    }
}

fn selection(backend: &Nvidia) -> Result<(), String> {
    let guard_ids = vec![-88i32, 0, -88, 4, -88, 1, -88, 4, -88];
    let original = NativeTensor::from_data(TensorData::new(guard_ids.clone(), [9])).map_err(|e| format!("{e}"))?;
    let indices = original
        .clone()
        .slice(&[Slice::new(1, Some(8), 2)])
        .and_then(|t| t.flip(&[0]))
        .map_err(|e| format!("{e}"))?;
    for (dtype, modes) in [
        (
            DType::F32,
            [
                Scatter::FloatAdd,
                Scatter::FloatAssign,
                Scatter::FloatMultiply,
                Scatter::FloatMin,
                Scatter::FloatMax,
            ],
        ),
        (
            DType::I32,
            [Scatter::IntAdd, Scatter::IntAssign, Scatter::IntMultiply, Scatter::IntMin, Scatter::IntMax],
        ),
    ] {
        let word = |n: u32| if dtype == DType::F32 { (n as f32).to_bits() } else { n };
        let data = NativeTensor::full(Shape::new([3, 2, 5]), dtype, word(3)).map_err(|e| format!("{e}"))?;
        let values = NativeTensor::full(Shape::new([3, 2, 4]), dtype, word(2))
            .and_then(|t| t.flip(&[0, 2]))
            .map_err(|e| format!("{e}"))?;
        for (update_op, mode) in [
            IndexingUpdateOp::Add,
            IndexingUpdateOp::Assign,
            IndexingUpdateOp::Mul,
            IndexingUpdateOp::Min,
            IndexingUpdateOp::Max,
        ]
        .into_iter()
        .zip(modes)
        {
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let output = backend
                .with_commands(|| data.select_update(2, &indices, &values, update_op))
                .map_err(|e| format!("{e}"))?;
            super::dispatch::resident(backend, &before, 1)?;
            let mut row = vec![word(3); 5];
            for id in [4, 1, 4, 0] {
                row[id] = update(row[id], word(2), mode);
            }
            let actual = output.read().map_err(|e| format!("{e}"))?;
            let expected: Vec<_> = (0..30).map(|i| row[i % 5]).collect();
            if actual.as_bytes() != expected.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>() {
                return Err(format!("native vector selection {mode:?}"));
            }
            for id in [-1i32, 5] {
                let invalid = NativeTensor::from_data(TensorData::new(vec![id; 4], [4])).map_err(|e| format!("{e}"))?;
                if !matches!(data.select_update(2, &invalid, &values, update_op), Err(Error::InvalidParameter)) {
                    return Err(String::from("vector selection accepted an invalid index"));
                }
            }
            let empty = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [0])).map_err(|e| format!("{e}"))?;
            let no_values = NativeTensor::full(Shape::new([3, 2, 0]), dtype, 0).map_err(|e| format!("{e}"))?;
            let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
            let output = data.select_update(2, &empty, &no_values, update_op).map_err(|e| format!("{e}"))?;
            if backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
                || output.read().map_err(|e| format!("{e}"))?.as_bytes() != vec![word(3); 30].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
            {
                return Err(String::from("empty vector selection changed data or launched work"));
            }
        }
    }
    let preserved = original.read().map_err(|e| format!("{e}"))?;
    if preserved.as_slice::<i32>().map_err(|e| format!("{e:?}"))? != guard_ids {
        return Err(String::from("vector selection changed index storage or guards"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_SELECT_UPDATES_OK modes=10 index_view=offset_negative_stride broadcast_indices=true resident=true status_reads=1 invalid_indices=recoverable empty=no_launch"
    ));
    Ok(())
}

fn guarded(backend: &Nvidia) -> Result<(), String> {
    const ROWS: usize = 129;
    const UPDATES: usize = 257;
    const WIDTH: usize = 5;
    const GUARD: u32 = 0x6bad_beef;
    let layout = Layout::contiguous(Shape::new([ROWS, WIDTH])).map_err(|e| format!("{e}"))?;
    let indices = Layout::contiguous(Shape::new([ROWS, UPDATES]))
        .and_then(|l| l.flip(&[1]))
        .map_err(|e| format!("{e}"))?;
    let mut ids = vec![GUARD];
    ids.extend((0..ROWS * UPDATES).map(|i| (i % WIDTH) as u32));
    ids[2] = u32::MAX;
    ids[ROWS * UPDATES] = WIDTH as u32;
    ids.push(GUARD);
    let index_buffer = backend.upload_words(&ids).map_err(|e| format!("{e}"))?;
    for op in Scatter::ALL.into_iter().filter(|op| op.ordered()) {
        let floating = matches!(op, Scatter::FloatAssign | Scatter::FloatMultiply | Scatter::FloatMin | Scatter::FloatMax);
        let mut values = vec![GUARD];
        values.extend((0..ROWS * UPDATES).map(|i| {
            if floating {
                // Dyadic products remain finite while still testing rounding
                // in the order observed by one destination row.
                [1.000_976_6f32, -0.999_023_44, 0.5, 2.][i % 4].to_bits()
            } else {
                [i32::MIN as u32, i32::MAX as u32, 3, (i as u32).wrapping_mul(0x9e37_79b9)][i % 4]
            }
        }));
        values.push(GUARD);
        let values_buffer = backend.upload_words(&values).map_err(|e| format!("{e}"))?;
        let mut expected = vec![GUARD];
        expected.extend((0..ROWS * WIDTH).map(|i| if floating { ((i % 9 + 1) as f32 / 8.).to_bits() } else { i as u32 }));
        expected.push(GUARD);
        let output = backend.upload_words(&expected).map_err(|e| format!("{e}"))?;
        let status = backend.upload_words(&[GUARD, 0, GUARD]).map_err(|e| format!("{e}"))?;
        let p = layout.scatter_parameters(&indices, 1, op).map_err(|e| format!("{e}"))?;
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        backend
            .with_commands(|| backend.dispatch_words(abi::Kernel::Scatter, &[(&values_buffer, 1), (&index_buffer, 1), (&status, 1), (&output, 1)], &p))
            .map_err(|e| format!("{e}"))?;
        super::dispatch::resident(backend, &before, 0)?;
        for row in 0..ROWS {
            for position in 0..UPDATES {
                let selected = ids[1 + row * UPDATES + UPDATES - 1 - position];
                if selected < WIDTH as u32 {
                    let at = 1 + row * WIDTH + selected as usize;
                    expected[at] = update(expected[at], values[1 + row * UPDATES + position], op);
                }
            }
        }
        let mut actual = vec![0; expected.len()];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        let mut error = [0; 3];
        backend.read_words_into(&status, 0, &mut error).map_err(|e| format!("{e}"))?;
        if actual != expected || error != [GUARD, 1, GUARD] {
            return Err(format!("ordered scatter {op:?}: guarded result or invalid index flag mismatch"));
        }
        let mut malformed = p;
        malformed[16] = 0;
        if backend.dispatch_words(
            abi::Kernel::Scatter,
            &[(&values_buffer, 1), (&index_buffer, 1), (&status, 1), (&output, 1)],
            &malformed,
        ) != Err(Error::InvalidParameter)
        {
            return Err(String::from("ordered scatter accepted aliasing destination rows"));
        }
        for (buffer, expected) in [(&values_buffer, &values), (&index_buffer, &ids)] {
            let mut actual = vec![0; expected.len()];
            backend.read_words_into(buffer, 0, &mut actual).map_err(|e| format!("{e}"))?;
            if &actual != expected {
                return Err(String::from("ordered scatter changed its input or input guard"));
            }
        }
    }
    Ok(())
}

fn lifetime(backend: &Nvidia) -> Result<(), String> {
    let data = NativeTensor::from_data(TensorData::new(vec![1f32; 129 * 5], [129, 5])).map_err(|e| format!("{e}"))?;
    let values = NativeTensor::from_data(TensorData::new(vec![2f32; 129 * 257], [129, 257])).map_err(|e| format!("{e}"))?;
    let indices = NativeTensor::from_data(TensorData::new((0..129 * 257).map(|i| i % 5).collect::<Vec<i32>>(), [129, 257])).map_err(|e| format!("{e}"))?;
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let result = backend
        .with_commands(|| {
            let result = data.scatter_update(1, &indices, &values, IndexingUpdateOp::Assign)?;
            drop(data);
            drop(values);
            drop(indices);
            // Queue a consumer after the index-status synchronization, then drop
            // the scatter result before the consumer completes.
            result.scalar(abi::elementwise::Elementwise::Add, 1.)
        })
        .map_err(|e| format!("{e}"))?;
    super::dispatch::resident(backend, &before, 1)?;
    let actual = result.read().map_err(|e| format!("{e}"))?;
    if actual.as_slice::<f32>().map_err(|e| format!("{e:?}"))?.iter().any(|&v| v != 3.) {
        return Err(String::from("ordered scatter queued lifetime result"));
    }
    let data = NativeTensor::from_data(TensorData::new(vec![3i32, 5, 7], [3])).map_err(|e| format!("{e}"))?;
    let values = NativeTensor::from_data(TensorData::new(vec![11i32], [1])).map_err(|e| format!("{e}"))?;
    for update in [IndexingUpdateOp::Assign, IndexingUpdateOp::Mul, IndexingUpdateOp::Min, IndexingUpdateOp::Max] {
        for id in [-1i32, 3] {
            let indices = NativeTensor::from_data(TensorData::new(vec![id], [1])).map_err(|e| format!("{e}"))?;
            if !matches!(data.scatter_update(0, &indices, &values, update), Err(Error::InvalidParameter)) {
                return Err(String::from("ordered scatter did not reject invalid index"));
            }
        }
        let empty_indices = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [0])).map_err(|e| format!("{e}"))?;
        let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
        let output = data.scatter_update(0, &empty_indices, &empty_indices, update).map_err(|e| format!("{e}"))?;
        if backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches
            || output.read().map_err(|e| format!("{e}"))?.as_slice::<i32>().map_err(|e| format!("{e:?}"))? != [3, 5, 7]
        {
            return Err(String::from("ordered scatter empty input launched work or changed data"));
        }
    }
    Ok(())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    guarded(backend)?;
    lifetime(backend)?;
    selection(backend)?;
    crate::report(format_args!(
        "NVIDIA_BURN_ORDERED_SCATTER_OK modes=8 guards=ok duplicates=ordered rows=129 updates=257 invalid_indices=recoverable resident=true status_reads=1 queued_lifetime=verified empty=no_launch"
    ));
    Ok(())
}
