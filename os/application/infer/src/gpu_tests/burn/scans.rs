//! Exact guards, IEEE extrema, empty paths, and resident strided scan execution.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::backend::{Shape, TensorData},
    native::{NativeTensor, layout::Layout},
};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute::{self as abi, Kernel, scan::Scan};

fn identity(op: Scan) -> u32 {
    match op {
        Scan::Sum | Scan::IntSum => 0,
        Scan::Product => 1f32.to_bits(),
        Scan::IntProduct => 1,
        Scan::Min => f32::INFINITY.to_bits(),
        Scan::Max => f32::NEG_INFINITY.to_bits(),
        Scan::IntMin => i32::MAX as u32,
        Scan::IntMax => i32::MIN as u32,
    }
}

fn combine(op: Scan, a: u32, b: u32) -> u32 {
    if op.is_integer() {
        let (a, b) = (a as i32, b as i32);
        return match op {
            Scan::IntSum => (i128::from(a) + i128::from(b)) as u32,
            Scan::IntProduct => (i128::from(a) * i128::from(b)) as u32,
            Scan::IntMin => a.min(b) as u32,
            Scan::IntMax => a.max(b) as u32,
            _ => unreachable!(),
        };
    }
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    match op {
        Scan::Sum => (x + y).to_bits(),
        Scan::Product => (x * y).to_bits(),
        _ if x.is_nan() => a,
        _ if y.is_nan() => b,
        _ if x == 0. && y == 0. => {
            if op == Scan::Min {
                a | b
            } else {
                a & b
            }
        }
        Scan::Min => {
            if y < x {
                b
            } else {
                a
            }
        }
        Scan::Max => {
            if y > x {
                b
            } else {
                a
            }
        }
        _ => unreachable!(),
    }
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    for op in Scan::ALL {
        let words: Vec<u32> = (0..3 * 257 * 5)
            .map(|i| match op {
                Scan::Sum => f32::from_bits((i % 7 + 1) as u32).copysign(if i % 3 == 0 { -1. } else { 1. }).to_bits(),
                Scan::Product => {
                    if i % 257 == 0 {
                        1
                    } else {
                        (if i % 7 == 0 { -1f32 } else { 1f32 }).to_bits()
                    }
                }
                Scan::Min | Scan::Max => [0x0000_0000, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 1, 0x8000_0001, 0x7fc0_1234, 0xffc0_0035][i % 8],
                _ => (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(0x7fff_ffff),
            })
            .collect();
        let guard = 0x6bad_beef;
        let mut source_words = vec![guard; 3];
        source_words.extend(&words);
        source_words.push(guard);
        let source = backend.upload_words(&source_words).map_err(|e| format!("{e}"))?;
        let view = Layout::contiguous(Shape::new([3, 257, 5]))
            .and_then(|l| l.permute(&[1, 2, 0]))
            .and_then(|l| l.flip(&[0, 2]))
            .map_err(|e| format!("{e}"))?;
        let logical: Vec<_> = (0..words.len())
            .map(|i| view.element_offset(i).map(|offset| words[offset]))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{e}"))?;
        for axis in 0..3 {
            let output = backend.upload_words(&vec![guard; words.len() + 2]).map_err(|e| format!("{e}"))?;
            let p = view.scan_parameters(axis, op).map_err(|e| format!("{e}"))?;
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            backend
                .dispatch_words(Kernel::TensorScan, &[(&source, 3), (&output, 1)], &p)
                .map_err(|e| format!("{e}"))?;
            let after = backend.service_profile().map_err(|e| format!("{e}"))?;
            if [abi::UPLOAD, abi::DOWNLOAD, abi::ALLOCATE]
                .iter()
                .any(|&op| before[op as usize].calls != after[op as usize].calls)
            {
                return Err(String::from("scan dispatch transferred or cleared tensor data"));
            }
            let mut invalid = p;
            invalid[5] += words.len();
            if backend.dispatch_words(Kernel::TensorScan, &[(&source, 3), (&output, 1)], &invalid).is_ok() {
                return Err(String::from("scan accepted invalid source extent"));
            }
            let suffix = view.shape()[axis + 1..].iter().product::<usize>();
            let mut expected = vec![guard];
            for i in 0..words.len() {
                let coordinate = i / suffix % view.shape()[axis];
                let base = i - coordinate * suffix;
                let value = (0..=coordinate).fold(identity(op), |value, k| combine(op, value, logical[base + k * suffix]));
                expected.push(value);
            }
            expected.push(guard);
            let mut actual = vec![0; expected.len()];
            backend.read_words_into(&output, 0, &mut actual).map_err(|e| format!("{e}"))?;
            if let Some((i, (a, b))) = actual.iter().zip(&expected).enumerate().find(|(_, (a, b))| a != b) {
                return Err(format!("raw scan {op:?} axis={axis}[{i}]: {a:08x} != {b:08x}"));
            }
        }
        let mut actual = vec![0; source_words.len()];
        backend.read_words_into(&source, 0, &mut actual).map_err(|e| format!("{e}"))?;
        if actual != source_words {
            return Err(String::from("scan changed shared source or its guards"));
        }
    }
    let empty = NativeTensor::from_data(TensorData::new(Vec::<i32>::new(), [2, 0, 5])).map_err(|e| format!("{e}"))?;
    let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
    for op in Scan::ALL.into_iter().filter(|op| op.is_integer()) {
        for axis in 0..3 {
            if !empty.scan(axis, op).map_err(|e| format!("{e}"))?.layout().is_empty() {
                return Err(String::from("nonempty native scan for empty input"));
            }
        }
    }
    if !matches!(empty.scan(3, Scan::Sum), Err(Error::InvalidParameter)) {
        return Err(String::from("empty scan accepted invalid axis"));
    }
    if before.launches != backend.queue_profile().map_err(|e| format!("{e}"))?.launches {
        return Err(String::from("empty scan launched a GPU kernel"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_SCAN_OK operations=8 chunks=128 carry=verified ieee=exact guards=ok invalid_bounds=rejected empty=no_launch resident=ok"
    ));
    Ok(())
}
