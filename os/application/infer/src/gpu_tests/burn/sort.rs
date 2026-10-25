//! Guarded merge passes, stable indices, and bounded resident sorting storage.
use alloc::{format, string::String, vec, vec::Vec};
use core::cmp::Ordering;
use d3os_ai::{
    burn::backend::{Shape, TensorData},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Backend as _, Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, sort::Mode};

fn compare(a: &(u32, u32), b: &(u32, u32), integer: bool, descending: bool, topk: bool) -> Ordering {
    let order = if integer {
        (a.0 as i32).cmp(&(b.0 as i32))
    } else {
        let (a, b) = (f32::from_bits(a.0), f32::from_bits(b.0));
        if topk {
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.partial_cmp(&b).unwrap(),
            }
        } else {
            a.total_cmp(&b)
        }
    };
    let order = if descending { order.reverse() } else { order };
    order.then(a.1.cmp(&b.1))
}

fn data(integer: bool) -> Vec<u32> {
    (0..3 * 257)
        .map(|i| {
            if integer {
                (i as u32).wrapping_mul(0x9e37_79b9)
            } else {
                [
                    0x0000_0000,
                    0x8000_0000,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7fc0_1234,
                    0xffc0_0035,
                    0x7f80_0001,
                    0xff80_0001,
                    1,
                    0x8000_0001,
                    1f32.to_bits(),
                    (-1f32).to_bits(),
                ][i % 12]
            }
        })
        .collect()
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let shape = Shape::new([3, 257]);
    for mode in Mode::ALL {
        let original = data(mode.integer());
        let layout = if mode.initial() {
            Layout::contiguous(shape.clone()).and_then(|l| l.flip(&[1]))
        } else {
            Layout::contiguous(shape.clone())
        }
        .map_err(|e| format!("{e}"))?;
        let mut pairs: Vec<_> = (0..original.len())
            .map(|i| layout.element_offset(i).map(|offset| (original[offset], (i % 257) as u32)))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{e}"))?;
        let width = if mode.initial() { 1 } else { 128 };
        if !mode.initial() {
            for row in pairs.chunks_mut(257) {
                for run in row.chunks_mut(width) {
                    run.sort_by(|a, b| compare(a, b, mode.integer(), mode.descending(), mode.topk()));
                }
            }
        }
        let mut values = vec![0x6bad_beef; 3];
        values.extend(if mode.initial() {
            original.clone()
        } else {
            pairs.iter().map(|p| p.0).collect()
        });
        values.push(0x6bad_beef);
        let mut input_indices = vec![0x6bad_beef; 5];
        input_indices.extend(pairs.iter().map(|p| p.1));
        input_indices.push(0x6bad_beef);
        let a = backend.upload_words(&values).map_err(|e| format!("{e}"))?;
        let b = backend.upload_words(&input_indices).map_err(|e| format!("{e}"))?;
        let output = backend.upload_words(&vec![0x6bad_beef; original.len() * 2 + 2]).map_err(|e| format!("{e}"))?;
        let p = layout.sort_parameters(1, mode, width).map_err(|e| format!("{e}"))?;
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        backend
            .dispatch_words(Kernel::TensorSort, &[(&a, 3), (&b, 5), (&output, 1)], &p)
            .map_err(|e| format!("{e}"))?;
        let after = backend.service_profile().map_err(|e| format!("{e}"))?;
        if [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE]
            .iter()
            .any(|&op| before[op as usize].calls != after[op as usize].calls)
        {
            return Err(String::from("sort pass transferred or cleared data"));
        }
        let mut invalid = p;
        invalid[5] += original.len();
        if backend.dispatch_words(Kernel::TensorSort, &[(&a, 3), (&b, 5), (&output, 1)], &invalid).is_ok() {
            return Err(String::from("sort accepted invalid source extent"));
        }
        for row in pairs.chunks_mut(257) {
            for run in row.chunks_mut(width * 2) {
                run.sort_by(|a, b| compare(a, b, mode.integer(), mode.descending(), mode.topk()));
            }
        }
        let mut expected = vec![0x6bad_beef];
        expected.extend(pairs.iter().map(|p| p.0));
        expected.extend(pairs.iter().map(|p| p.1));
        expected.push(0x6bad_beef);
        let mut actual = vec![0; expected.len()];
        backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual != expected {
            return Err(format!("guarded sort pass {mode:?} mismatch"));
        }
        for (buffer, expected) in [(&a, &values), (&b, &input_indices)] {
            let mut actual = vec![0; expected.len()];
            backend.read_words_into(buffer, 0, &mut actual).map_err(|e| format!("{e}"))?;
            if &actual != expected {
                return Err(String::from("sort changed source storage or guards"));
            }
        }
    }
    // Low-level unsorted runs and arbitrary index payloads cannot leave holes
    // in a full-writer allocation or turn index values into memory addresses.
    let values: Vec<_> = (0..771).map(|i| if i % 2 == 0 { 0x7fc0_1234 } else { 0x7fc0_0035 }).collect();
    let a = backend.upload_words(&values).map_err(|e| format!("{e}"))?;
    let b = backend.upload_words(&vec![u32::MAX; 771]).map_err(|e| format!("{e}"))?;
    let p = Layout::contiguous(shape.clone())
        .and_then(|l| l.sort_parameters(1, Mode::FloatAscending, 128))
        .map_err(|e| format!("{e}"))?;
    let output = backend
        .produce_words(1542, Kernel::TensorSort, &[(&a, 0), (&b, 0)], &p)
        .map_err(|e| format!("{e}"))?;
    let mut actual = vec![0; 1542];
    backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
    if actual[..771].iter().any(|&v| v != 0x7fc0_1234 && v != 0x7fc0_0035) || actual[771..].iter().any(|&v| v != u32::MAX) {
        return Err(String::from("sort merge did not fully initialize output"));
    }
    let values = data(false);
    let tensor = NativeTensor::from_data(TensorData::new(values.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(), [3, 257]))
        .and_then(|t| t.flip(&[1]))
        .map_err(|e| format!("{e}"))?;
    let mut expected: Vec<_> = (0..771).map(|i| (values[i / 257 * 257 + 256 - i % 257], (i % 257) as u32)).collect();
    for row in expected.chunks_mut(257) {
        row.sort_by(|a, b| compare(a, b, false, false, false));
    }
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let result = backend
        .with_commands(|| {
            let result = tensor.sort(1, false);
            drop(tensor);
            result
        })
        .map_err(|e| format!("{e}"))?;
    let after = backend.service_profile().map_err(|e| format!("{e}"))?;
    if [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 2
    {
        return Err(String::from("native sorting lost bounded resident storage"));
    }
    let sorted = result.0.read().map_err(|e| format!("{e}"))?;
    let indices = result.1.read().map_err(|e| format!("{e}"))?;
    let actual: Vec<_> = sorted.as_bytes().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if actual != expected.iter().map(|p| p.0).collect::<Vec<_>>()
        || indices.as_slice::<i32>().map_err(|e| format!("{e:?}"))? != expected.iter().map(|p| p.1 as i32).collect::<Vec<_>>()
    {
        return Err(String::from("native stable sort or queued lifetime mismatch"));
    }
    let tensor = NativeTensor::from_data(TensorData::new(values.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(), [3, 257]))
        .and_then(|t| t.flip(&[1]))
        .map_err(|e| format!("{e}"))?;
    let before = backend.service_profile().map_err(|e| format!("{e}"))?;
    let result = backend.with_commands(|| tensor.topk(1, 7)).map_err(|e| format!("{e}"))?;
    let after = backend.service_profile().map_err(|e| format!("{e}"))?;
    if [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE]
        .iter()
        .any(|&op| before[op as usize].calls != after[op as usize].calls)
        || after[abi::LAUNCH_ALLOCATE as usize].calls - before[abi::LAUNCH_ALLOCATE as usize].calls != 2
    {
        return Err(String::from("top-k transferred or reallocated tensor data"));
    }
    let mut expected: Vec<_> = (0..771).map(|i| (values[i / 257 * 257 + 256 - i % 257], (i % 257) as u32)).collect();
    for row in expected.chunks_mut(257) {
        row.sort_by(|a, b| compare(a, b, false, true, true));
    }
    let expected: Vec<_> = expected.chunks(257).flat_map(|row| row[..7].iter().copied()).collect();
    let actual = result.0.read().map_err(|e| format!("{e}"))?;
    let actual: Vec<_> = actual.as_bytes().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let index = result.1.read().map_err(|e| format!("{e}"))?;
    if actual != expected.iter().map(|p| p.0).collect::<Vec<_>>()
        || index.as_slice::<i32>().map_err(|e| format!("{e:?}"))? != expected.iter().map(|p| p.1 as i32).collect::<Vec<_>>()
    {
        return Err(String::from("native maximum ordering or top-k prefix view mismatch"));
    }
    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
    let (empty_values, empty_indices) = tensor.topk(1, 0).map_err(|e| format!("{e}"))?;
    if !empty_values.layout().is_empty()
        || !empty_indices.layout().is_empty()
        || before.launches != backend.queue_profile().map_err(|e| format!("{e}"))?.launches
    {
        return Err(String::from("zero top-k launched or returned data"));
    }
    let empty = NativeTensor::from_data(TensorData::new(Vec::<f32>::new(), [3, 0])).map_err(|e| format!("{e}"))?;
    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
    for axis in 0..2 {
        let (v, i) = empty.sort(axis, true).map_err(|e| format!("{e}"))?;
        if !v.layout().is_empty() || !i.layout().is_empty() {
            return Err(String::from("nonempty sort of empty input"));
        }
    }
    if !matches!(empty.sort(2, false), Err(Error::InvalidParameter)) || before.launches != backend.queue_profile().map_err(|e| format!("{e}"))?.launches {
        return Err(String::from("empty sort axis or launch validation"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_SORT_OK modes=10 topk=nan_first stable_indices=true ieee_bits=exact guards=ok full_writer=verified queued_lifetime=verified allocations=2 resident=ok empty=no_launch"
    ));
    Ok(())
}
