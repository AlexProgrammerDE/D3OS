//! Ordered indexing references and independent derivatives for unique indices.
use alloc::{format, string::String, vec, vec::Vec};
use burn::{
    backend::TensorData,
    tensor::{Device, IndexingUpdateOp, Int, Tensor, kind::Basic},
};

pub const UPDATES: [IndexingUpdateOp; 4] = [IndexingUpdateOp::Assign, IndexingUpdateOp::Mul, IndexingUpdateOp::Min, IndexingUpdateOp::Max];

#[derive(Clone, Copy, Debug)]
enum Indexing {
    Scatter,
    Select,
}

impl Indexing {
    fn updates(self) -> &'static [IndexingUpdateOp] {
        match self {
            Self::Scatter => &UPDATES,
            Self::Select => &[
                IndexingUpdateOp::Add,
                IndexingUpdateOp::Assign,
                IndexingUpdateOp::Mul,
                IndexingUpdateOp::Min,
                IndexingUpdateOp::Max,
            ],
        }
    }

    fn apply<const D: usize, K: Basic>(
        self, data: Tensor<D, K>, axis: usize, indices: Tensor<D, Int>, vector: Tensor<1, Int>, values: Tensor<D, K>, update: IndexingUpdateOp,
    ) -> Tensor<D, K> {
        match self {
            Self::Scatter => data.scatter(axis, indices, values, update),
            Self::Select => data.select_assign(axis, vector, values, update),
        }
    }
}

// Use an offset view as well as a negative stride. The corresponding full-rank
// index reference receives the same axis reversal through view().
fn vector(device: &Device, ids: Vec<i32>, axis: usize) -> Tensor<1, Int> {
    let length = ids.len();
    let mut guarded = vec![-99];
    guarded.extend(ids);
    guarded.push(-99);
    let result =
        Tensor::<1, Int>::from_data(TensorData::new(guarded, [length + 2]), device).slice([burn::tensor::Slice::new(1, Some((length + 1) as isize), 1)]);
    if axis == 1 { result } else { result.flip([0]) }
}

fn view<T: Copy>(input: &[T], shape: [usize; 3], order: [usize; 3]) -> Vec<T> {
    let dims = order.map(|a| shape[a]);
    (0..input.len())
        .map(|i| {
            let coords = [dims[0] - 1 - i / (dims[1] * dims[2]), i / dims[2] % dims[1], dims[2] - 1 - i % dims[2]];
            let mut source = [0; 3];
            for axis in 0..3 {
                source[order[axis]] = coords[axis];
            }
            input[(source[0] * shape[1] + source[1]) * shape[2] + source[2]]
        })
        .collect()
}

fn destination(mut i: usize, input: [usize; 3], output: [usize; 3], axis: usize, selected: i32) -> usize {
    let mut coords = [0; 3];
    for a in (0..3).rev() {
        coords[a] = i % input[a];
        i /= input[a];
    }
    coords[axis] = selected as usize;
    (coords[0] * output[1] + coords[1]) * output[2] + coords[2]
}

pub(super) fn integer(a: u32, b: u32, update: IndexingUpdateOp) -> u32 {
    match update {
        IndexingUpdateOp::Assign => b,
        IndexingUpdateOp::Mul => a.wrapping_mul(b),
        IndexingUpdateOp::Min => (a as i32).min(b as i32) as u32,
        IndexingUpdateOp::Max => (a as i32).max(b as i32) as u32,
        IndexingUpdateOp::Add => a.wrapping_add(b),
    }
}

pub(super) fn float(a: f32, b: f32, update: IndexingUpdateOp) -> f32 {
    match update {
        IndexingUpdateOp::Assign => b,
        IndexingUpdateOp::Mul => a * b,
        IndexingUpdateOp::Min => a.min(b),
        IndexingUpdateOp::Max => a.max(b),
        IndexingUpdateOp::Add => a + b,
    }
}

pub fn run(device: &Device, include_empty: bool, include_ieee: bool) -> Result<(), String> {
    run_indexing(device, include_empty, include_ieee, Indexing::Scatter)
}

pub fn run_select(device: &Device, include_empty: bool, include_ieee: bool) -> Result<(), String> {
    run_indexing(device, include_empty, include_ieee, Indexing::Select)?;
    duplicate_selection(device)?;
    rank_eight_selection(device)
}

fn run_indexing(device: &Device, include_empty: bool, include_ieee: bool, indexing: Indexing) -> Result<(), String> {
    for length in [1, 127, 128, 129] {
        let data_shape = [2, length + 7, 3];
        let update_shape = [2, length, 3];
        let a: Vec<_> = (0..6 * (length + 7)).map(|i| (i as i32 % 97 - 48) as f32 / 4.).collect();
        let b: Vec<_> = (0..6 * length).map(|i| (i as i32 % 31 - 15) as f32 / 8.).collect();
        let ai: Vec<u32> = (0..a.len())
            .map(|i| [i32::MIN as u32, i32::MAX as u32, u32::MAX, (i as u32).wrapping_mul(0x9e37_79b9)][i % 4])
            .collect();
        let bi: Vec<u32> = (0..b.len())
            .map(|i| [i32::MAX as u32, i32::MIN as u32, 3, (i as u32).wrapping_mul(0x85eb_ca6b)][i % 4])
            .collect();
        let ids: Vec<_> = (0..b.len())
            .map(|i| match indexing {
                Indexing::Scatter => ((i / 3 % length + i / (3 * length) + i % 3) % (length + 7)) as i32,
                Indexing::Select => (i / 3 % length + 1) as i32,
            })
            .collect();
        for axis in 0..3 {
            let mut order = [0, 1, 2];
            order.swap(axis, 1);
            let output_dims = order.map(|a| data_shape[a]);
            let dims = order.map(|a| update_shape[a]);
            let logical_a = view(&a, data_shape, order);
            let logical_b = view(&b, update_shape, order);
            let logical_ai = view(&ai, data_shape, order);
            let logical_bi = view(&bi, update_shape, order);
            let logical_ids = view(&ids, update_shape, order);
            let data = Tensor::<3>::from_data(TensorData::new(a.clone(), data_shape), device)
                .permute(order)
                .flip([0, 2]);
            let values = Tensor::<3>::from_data(TensorData::new(b.clone(), update_shape), device)
                .permute(order)
                .flip([0, 2]);
            let indices = Tensor::<3, Int>::from_data(TensorData::new(ids.clone(), update_shape), device)
                .permute(order)
                .flip([0, 2]);
            let data_i = Tensor::<3, Int>::from_data(TensorData::new(ai.iter().map(|&v| v as i32).collect::<Vec<_>>(), data_shape), device)
                .permute(order)
                .flip([0, 2]);
            let values_i = Tensor::<3, Int>::from_data(TensorData::new(bi.iter().map(|&v| v as i32).collect::<Vec<_>>(), update_shape), device)
                .permute(order)
                .flip([0, 2]);
            let vector = vector(device, (0..length).map(|i| (i + 1) as i32).collect(), axis);
            for &update in indexing.updates() {
                let mut expected = logical_a.clone();
                let mut expected_i = logical_ai.clone();
                for (i, &id) in logical_ids.iter().enumerate() {
                    let at = destination(i, dims, output_dims, axis, id);
                    expected[at] = float(expected[at], logical_b[i], update);
                    expected_i[at] = integer(expected_i[at], logical_bi[i], update);
                }
                let result = indexing.apply(data.clone(), axis, indices.clone(), vector.clone(), values.clone(), update);
                if result.dims() != output_dims || result.try_into_vec_as::<f32>().map_err(|e| format!("float scatter {update:?}: {e:?}"))? != expected {
                    return Err(format!("float {indexing:?} length={length} axis={axis} op={update:?}"));
                }
                let result = indexing.apply(data_i.clone(), axis, indices.clone(), vector.clone(), values_i.clone(), update);
                if result.dims() != output_dims
                    || result.try_into_vec_as::<i32>().map_err(|e| format!("integer scatter: {e:?}"))?
                        != expected_i.iter().map(|&v| v as i32).collect::<Vec<_>>()
                {
                    return Err(format!("integer {indexing:?} length={length} axis={axis} op={update:?}"));
                }
            }
            let expected_vector: Vec<_> = if axis == 1 {
                (1..=length).map(|i| i as i32).collect()
            } else {
                (1..=length).rev().map(|i| i as i32).collect()
            };
            if vector.try_into_vec_as::<i32>().map_err(|e| format!("retained index vector: {e:?}"))? != expected_vector {
                return Err(String::from("indexed update changed its index vector"));
            }
            if data.try_into_vec_as::<f32>().map_err(|e| format!("retained data: {e:?}"))? != logical_a
                || values.try_into_vec_as::<f32>().map_err(|e| format!("retained values: {e:?}"))? != logical_b
                || indices.try_into_vec_as::<i32>().map_err(|e| format!("retained indices: {e:?}"))? != logical_ids
            {
                return Err(String::from("scatter changed shared inputs"));
            }
        }
    }
    gradients(device, indexing)?;
    if include_empty {
        let data = Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.]], device);
        let ids = Tensor::<2, Int>::from_data(TensorData::new(Vec::<i32>::new(), [2, 0]), device);
        let values = Tensor::<2>::from_data(TensorData::new(Vec::<f32>::new(), [2, 0]), device);
        let vector = vector(device, Vec::new(), 1);
        for &update in indexing.updates() {
            if indexing
                .apply(data.clone(), 1, ids.clone(), vector.clone(), values.clone(), update)
                .try_into_vec_as::<f32>()
                .map_err(|e| format!("empty scatter: {e:?}"))?
                != [1., 2., 3., 4., 5., 6.]
            {
                return Err(String::from("empty scatter changed data"));
            }
        }
    }
    if include_ieee {
        ieee(device, indexing)?;
    }
    Ok(())
}

fn duplicate_selection(device: &Device) -> Result<(), String> {
    let ids: Vec<_> = (0..257).map(|i| i % 5).collect();
    let indices = vector(device, ids.clone(), 1);
    let a: Vec<_> = (0..15).map(|i| i as f32 / 8. - 1.).collect();
    let b: Vec<_> = (0..3 * 257).map(|i| [0.5f32, -2., -0.5, 2.][i % 4]).collect();
    let ai: Vec<_> = (0..15).map(|i| (i as u32).wrapping_mul(0x9e37_79b9) as i32).collect();
    let bi: Vec<_> = (0..3 * 257).map(|i| (i as u32).wrapping_mul(0x85eb_ca6b) as i32).collect();
    let data = Tensor::<2>::from_data(TensorData::new(a.clone(), [3, 5]), device);
    let values = Tensor::<2>::from_data(TensorData::new(b.clone(), [3, 257]), device);
    let data_i = Tensor::<2, Int>::from_data(TensorData::new(ai.clone(), [3, 5]), device);
    let values_i = Tensor::<2, Int>::from_data(TensorData::new(bi.clone(), [3, 257]), device);
    for update in [IndexingUpdateOp::Add, IndexingUpdateOp::Mul, IndexingUpdateOp::Min, IndexingUpdateOp::Max] {
        let mut expected = a.clone();
        let mut expected_i = ai.clone();
        for row in 0..3 {
            for (i, &id) in ids.iter().enumerate() {
                let at = row * 5 + id as usize;
                expected[at] = float(expected[at], b[row * 257 + i], update);
                expected_i[at] = integer(expected_i[at] as u32, bi[row * 257 + i] as u32, update) as i32;
            }
        }
        if data
            .clone()
            .select_assign(1, indices.clone(), values.clone(), update)
            .try_into_vec_as::<f32>()
            .map_err(|e| format!("duplicate float selection: {e:?}"))?
            != expected
            || data_i
                .clone()
                .select_assign(1, indices.clone(), values_i.clone(), update)
                .try_into_vec_as::<i32>()
                .map_err(|e| format!("duplicate integer selection: {e:?}"))?
                != expected_i
        {
            return Err(format!("duplicate selection {update:?}"));
        }
    }
    Ok(())
}

fn rank_eight_selection(device: &Device) -> Result<(), String> {
    let data = Tensor::<8>::from_data(TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [1, 2, 1, 1, 1, 1, 3, 1]), device).flip([1, 6]);
    let values = Tensor::<8>::from_data(TensorData::new(vec![10f32, 20., 30., 40.], [1, 2, 1, 1, 1, 1, 2, 1]), device).flip([1, 6]);
    let indices = vector(device, vec![2, 0], 6);
    for &update in Indexing::Select.updates() {
        let mut expected = vec![6f32, 5., 4., 3., 2., 1.];
        for (at, value) in [(0, 40.), (2, 30.), (3, 20.), (5, 10.)] {
            expected[at] = float(expected[at], value, update);
        }
        let result = data.clone().select_assign(6, indices.clone(), values.clone(), update);
        if result.dims() != [1, 2, 1, 1, 1, 1, 3, 1] || result.try_into_vec_as::<f32>().map_err(|e| format!("rank-eight selection: {e:?}"))? != expected {
            return Err(format!("rank-eight selection {update:?}"));
        }
    }
    Ok(())
}

fn ieee(device: &Device, indexing: Indexing) -> Result<(), String> {
    let a = [0, 0x8000_0000, 0x7fc0_1234, 1f32.to_bits(), 0x7f80_0001, (-2f32).to_bits(), 0x7f80_0000, 1];
    let b = [
        0x8000_0000,
        0,
        1f32.to_bits(),
        0x7fc0_1234,
        (-2f32).to_bits(),
        0xff80_0001,
        0xff80_0000,
        2f32.to_bits(),
    ];
    let data = Tensor::<2>::from_data(TensorData::new(a.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(), [8, 1]), device);
    let values = Tensor::<2>::from_data(TensorData::new(b.iter().map(|&v| f32::from_bits(v)).collect::<Vec<_>>(), [8, 1]), device);
    let ids = Tensor::<2, Int>::from_data(TensorData::new(vec![0i32; 8], [8, 1]), device);
    let vector = vector(device, vec![0], 1);
    for &update in indexing.updates() {
        let result = indexing
            .apply(data.clone(), 1, ids.clone(), vector.clone(), values.clone(), update)
            .try_into_data()
            .map_err(|e| format!("IEEE scatter: {e:?}"))?;
        for (i, bytes) in result.as_bytes().chunks_exact(4).enumerate() {
            let actual = f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap()));
            let expected = float(f32::from_bits(a[i]), f32::from_bits(b[i]), update);
            if actual.to_bits() != expected.to_bits() && !(actual.is_nan() && expected.is_nan() && update != IndexingUpdateOp::Assign) {
                return Err(format!(
                    "IEEE {indexing:?} op={update:?} index={i} actual=0x{:08x} expected=0x{:08x}",
                    actual.to_bits(),
                    expected.to_bits()
                ));
            }
        }
    }
    Ok(())
}

fn gradients(device: &Device, indexing: Indexing) -> Result<(), String> {
    let auto = device.clone().autodiff();
    for axis in 0..3 {
        let mut order = [0, 1, 2];
        order.swap(axis, 1);
        let data_shape = [2, 5, 3];
        let update_shape = [2, 3, 3];
        let a: Vec<_> = (0..30).map(|i| (i % 9 - 4) as f32).collect();
        let b: Vec<_> = (0..18).map(|i| (i % 7 - 3) as f32).collect();
        let ids: Vec<_> = (0..18).map(|i| (i / 3 % 3) * 2).collect();
        let dims = order.map(|a| data_shape[a]);
        let vdims = order.map(|a| update_shape[a]);
        let la = view(&a, data_shape, order);
        let lb = view(&b, update_shape, order);
        let li = view(&ids, update_shape, order);
        let weights: Vec<_> = (0..30).map(|i| (i % 7 + 1) as f32).collect();
        let vector = vector(&auto, vec![0, 2, 4], axis);
        for &update in indexing.updates() {
            let data = Tensor::<3>::from_data(TensorData::new(a.clone(), data_shape), &auto)
                .permute(order)
                .flip([0, 2])
                .require_grad();
            let values = Tensor::<3>::from_data(TensorData::new(b.clone(), update_shape), &auto)
                .permute(order)
                .flip([0, 2])
                .require_grad();
            let indices = Tensor::<3, Int>::from_data(TensorData::new(ids.clone(), update_shape), &auto)
                .permute(order)
                .flip([0, 2]);
            let mut da = weights.clone();
            let mut db = vec![0.; b.len()];
            for (i, &id) in li.iter().enumerate() {
                let at = destination(i, vdims, dims, axis, id);
                let w = weights[at];
                match update {
                    IndexingUpdateOp::Assign => {
                        da[at] = 0.;
                        db[i] = w;
                    }
                    IndexingUpdateOp::Mul => {
                        da[at] = w * lb[i];
                        db[i] = w * la[at];
                    }
                    IndexingUpdateOp::Min => {
                        da[at] = if la[at] <= lb[i] { w } else { 0. };
                        db[i] = if lb[i] <= la[at] { w } else { 0. };
                    }
                    IndexingUpdateOp::Max => {
                        da[at] = if la[at] >= lb[i] { w } else { 0. };
                        db[i] = if lb[i] >= la[at] { w } else { 0. };
                    }
                    IndexingUpdateOp::Add => db[i] = w,
                }
            }
            let output = indexing.apply(data.clone(), axis, indices, vector.clone(), values.clone(), update);
            let gradients = (output * Tensor::<3>::from_data(TensorData::new(weights.clone(), dims), &auto))
                .sum()
                .backward();
            let actual_a = data.grad(&gradients).ok_or_else(|| String::from("missing scatter data gradient"))?;
            let actual_b = values.grad(&gradients).ok_or_else(|| String::from("missing scatter value gradient"))?;
            if actual_a.try_into_vec_as::<f32>().map_err(|e| format!("scatter data grad: {e:?}"))? != da
                || actual_b.try_into_vec_as::<f32>().map_err(|e| format!("scatter value grad: {e:?}"))? != db
            {
                return Err(format!("{indexing:?} gradient axis={axis} op={update:?}"));
            }
        }
    }
    Ok(())
}
