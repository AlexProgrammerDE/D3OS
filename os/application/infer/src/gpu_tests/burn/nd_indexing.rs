//! Coordinate bounds cannot alias another valid row after linearization.
use alloc::{format, string::String, vec, vec::Vec};
use d3os_ai::{
    burn::{
        backend::{DType, Shape, TensorData},
        tensor::{IndexingUpdateOp, Slice},
    },
    native::{NativeTensor, range::RangePlan},
};
use d3os_compute::{Error, nvidia::Nvidia};
use d3os_nvidia::compute::{elementwise::Elementwise, words::WordOp};

fn read(tensor: &NativeTensor) -> Result<Vec<u32>, String> {
    let data = tensor.read().map_err(|e| format!("{e}"))?;
    Ok(data.as_bytes().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
}

pub(super) fn run(backend: &Nvidia) -> Result<(), String> {
    let updates = [
        IndexingUpdateOp::Add,
        IndexingUpdateOp::Assign,
        IndexingUpdateOp::Mul,
        IndexingUpdateOp::Min,
        IndexingUpdateOp::Max,
    ];
    for dtype in [DType::F32, DType::I32] {
        let word = |n: u32| if dtype == DType::F32 { (n as f32).to_bits() } else { n };
        let mut words = vec![0x7fc0_1299];
        words.extend((1..=6).map(word));
        words.push(0xff80_0001);
        let original = NativeTensor::from_data(TensorData::from_bytes_vec(
            words.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Shape::new([8]),
            dtype,
        ))
        .map_err(|e| format!("{e}"))?;
        let data = original
            .clone()
            .slice(&[Slice::new(1, Some(7), 1)])
            .and_then(|t| t.reshape(Shape::new([2, 3])))
            .map_err(|e| format!("{e}"))?;
        let ids = NativeTensor::from_data(TensorData::new(vec![0i32, 1], [1, 2]))
            .and_then(|t| t.flip(&[1]))
            .map_err(|e| format!("{e}"))?;
        let before = backend.service_profile().map_err(|e| format!("{e}"))?;
        let output = data.gather_nd(&ids).map_err(|e| format!("{e}"))?;
        super::dispatch::resident(backend, &before, 1)?;
        if read(&output)? != [word(4)] {
            return Err(String::from("ND signed indices or source offset"));
        }
        let values = NativeTensor::full(Shape::new([1]), dtype, word(9)).map_err(|e| format!("{e}"))?;
        for update in updates {
            let before = backend.service_profile().map_err(|e| format!("{e}"))?;
            let output = data.scatter_nd(&ids, &values, update).map_err(|e| format!("{e}"))?;
            super::dispatch::resident(backend, &before, 1)?;
            let expected = match update {
                IndexingUpdateOp::Add => 13,
                IndexingUpdateOp::Assign => 9,
                IndexingUpdateOp::Mul => 36,
                IndexingUpdateOp::Min => 4,
                IndexingUpdateOp::Max => 9,
            };
            if read(&output)? != [word(1), word(2), word(3), word(expected), word(5), word(6)] {
                return Err(format!("ND native update {dtype:?} {update:?}"));
            }
            for tuple in [[1, -1], [-1, 3], [0, 3], [i32::MAX, i32::MAX]] {
                let invalid = NativeTensor::from_data(TensorData::new(tuple.to_vec(), [1, 2])).map_err(|e| format!("{e}"))?;
                if !matches!(data.gather_nd(&invalid), Err(Error::InvalidParameter))
                    || !matches!(data.scatter_nd(&invalid, &values, update), Err(Error::InvalidParameter))
                {
                    return Err(format!("ND accepted coordinate alias {tuple:?}"));
                }
            }
        }
        if read(&original)? != words {
            return Err(String::from("ND changed shared source or guards"));
        }
        let point = NativeTensor::from_data(TensorData::new(vec![1i32, 2], [2])).map_err(|e| format!("{e}"))?;
        let scalar = data.gather_nd(&point).map_err(|e| format!("{e}"))?;
        if !scalar.layout().shape().is_empty() || read(&scalar)? != [word(6)] {
            return Err(String::from("ND scalar shape or value"));
        }
        let result = if dtype == DType::F32 {
            scalar.scalar(Elementwise::Add, 2.)
        } else {
            scalar.word_scalar(WordOp::Add, 2)
        }
        .map_err(|e| format!("{e}"))?;
        if !result.layout().shape().is_empty() || read(&result)? != [word(8)] {
            return Err(String::from("scalar ND arithmetic"));
        }
        let values = NativeTensor::full(Shape::new([]), dtype, word(9)).map_err(|e| format!("{e}"))?;
        if read(&data.scatter_nd(&point, &values, IndexingUpdateOp::Assign).map_err(|e| format!("{e}"))?)?
            != [word(1), word(2), word(3), word(4), word(5), word(9)]
        {
            return Err(String::from("scalar ND update"));
        }
        let empty = NativeTensor::empty(Shape::new([2, 0, 3]), dtype).map_err(|e| format!("{e}"))?;
        let ids = NativeTensor::from_data(TensorData::new(vec![0i32, 1], [2, 1])).map_err(|e| format!("{e}"))?;
        let before = backend.queue_profile().map_err(|e| format!("{e}"))?;
        let output = empty.gather_nd(&ids).map_err(|e| format!("{e}"))?;
        for update in updates {
            if !empty.scatter_nd(&ids, &empty, update).map_err(|e| format!("{e}"))?.layout().is_empty() {
                return Err(String::from("empty ND scatter"));
            }
        }
        if !output.layout().is_empty() || backend.queue_profile().map_err(|e| format!("{e}"))?.launches != before.launches {
            return Err(String::from("empty ND launched work"));
        }
    }
    let data = RangePlan::new(0..16, 1)
        .and_then(NativeTensor::arange)
        .and_then(|t| t.reshape(Shape::new([2, 1, 2, 1, 2, 1, 2, 1])))
        .map_err(|e| format!("{e}"))?;
    let ids = NativeTensor::from_data(TensorData::new(vec![1i32, 0, 1, 0, 1, 0, 1, 0], [8])).map_err(|e| format!("{e}"))?;
    if read(&data.gather_nd(&ids).map_err(|e| format!("{e}"))?)? != [15] {
        return Err(String::from("rank-eight ND scalar"));
    }
    let bits = [0u32, 0x8000_0000, 0x7fc0_1234, 0xff80_0001, 0x7f80_0000, 0xff80_0000, 1, 0x8000_0001];
    let source = NativeTensor::from_data(TensorData::from_bytes_vec(
        bits.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Shape::new([2, 4]),
        DType::F32,
    ))
    .map_err(|e| format!("{e}"))?;
    let ids = NativeTensor::from_data(TensorData::new((0..8).flat_map(|i| [i / 4, i % 4]).collect::<Vec<i32>>(), [8, 2])).map_err(|e| format!("{e}"))?;
    let output = source.gather_nd(&ids).map_err(|e| format!("{e}"))?;
    if read(&output)? != bits {
        return Err(String::from("ND gather changed IEEE payloads"));
    }
    let values = output.flip(&[0]).map_err(|e| format!("{e}"))?;
    if read(&source.scatter_nd(&ids, &values, IndexingUpdateOp::Assign).map_err(|e| format!("{e}"))?)? != bits.into_iter().rev().collect::<Vec<_>>() {
        return Err(String::from("ND assignment changed IEEE payloads"));
    }
    crate::report(format_args!(
        "NVIDIA_BURN_ND_INDEXING_OK dtypes=f32,i32 modes=5 coordinate_bounds=independent invalid_aliases=recoverable guards=ok resident=true status_reads=1 scalar=true rank8=true ieee_payloads=exact empty=no_launch"
    ));
    Ok(())
}
