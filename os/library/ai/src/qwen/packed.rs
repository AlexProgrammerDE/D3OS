//! Burn parameter layout hooks keep records in canonical file order. The
//! resident tensor is repacked only on import or record loading, never during
//! ordinary inference, validation snapshots, or device movement.
use super::{MatrixFormat, MatrixLayout};
use burn::{
    module::{Param, ParamId},
    tensor::{DType, Int, Tensor, TensorData},
};

pub(super) fn parameter(words: Tensor<1, Int>, shape: [usize; 2], format: MatrixFormat, layout: MatrixLayout) -> Param<Tensor<1, Int>> {
    let param = Param::initialized(ParamId::new(), words);
    if layout == MatrixLayout::RowMajor {
        return param;
    }
    param
        .save_mapper(move |tensor| reorder(tensor, shape, format, layout, MatrixLayout::RowMajor))
        .load_mapper(move |tensor| reorder(tensor, shape, format, MatrixLayout::RowMajor, layout))
}

fn reorder(tensor: Tensor<1, Int>, shape: [usize; 2], format: MatrixFormat, from: MatrixLayout, to: MatrixLayout) -> Tensor<1, Int> {
    let device = tensor.device();
    let data = tensor.try_into_data().expect("read packed matrix for record conversion");
    let bytes = from.row_major_bytes(shape, format, data.as_bytes()).expect("validated packed matrix layout");
    let packed = to.pack(shape, format, &bytes).expect("encode packed matrix parameter");
    let count = packed.len() / 4;
    Tensor::from_data(TensorData::from_bytes_vec(packed, [count], DType::I32), &device)
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::qwen::{PackedMatrix, Quantization};
    use burn::{module::Module, tensor::Device};

    #[test]
    fn device_moves_cannot_silently_replace_quantized_arithmetic() {
        let device = Device::flex();
        let shape = [32, 32];
        let format = MatrixFormat::Quantized(Quantization::Q8_0);
        let bytes = alloc::vec![0; format.byte_len(shape).unwrap()];
        let mut matrix = PackedMatrix::from_bytes(shape, format, &bytes, &device).unwrap();
        matrix.prefill_quantization = crate::qwen::PrefillQuantization::Q8_1;
        let moved = matrix.to_device(&device);
        assert_eq!(moved.prefill_quantization, crate::qwen::PrefillQuantization::Q8_1);
        assert!(moved.project(Tensor::ones([2, 32], &device)).is_err());
    }

    #[test]
    fn records_load_between_physical_layouts_without_changing_projection_or_embedding() {
        let device = Device::flex();
        let shape = [64, 96];
        let format = MatrixFormat::Quantized(Quantization::Q8_0);
        let mut bytes = alloc::vec![0u8; format.byte_len(shape).unwrap()];
        for (index, block) in bytes.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&[0x00, 0x3c]);
            for (lane, value) in block[2..].iter_mut().enumerate() {
                *value = (index * 37 + lane * 13) as u8;
            }
        }
        let ordinary = PackedMatrix::from_bytes(shape, format, &bytes, &device).unwrap();
        let packed = MatrixLayout::Q8Columns32.pack(shape, format, &bytes).unwrap();
        let tiled = PackedMatrix {
            words: parameter(
                Tensor::<1, Int>::from_data(TensorData::from_bytes_vec(packed, [bytes.len() / 4], DType::I32), &device),
                shape,
                format,
                MatrixLayout::Q8Columns32,
            ),
            shape,
            format,
            layout: MatrixLayout::Q8Columns32,
            prefill_quantization: crate::qwen::PrefillQuantization::Accurate,
        };
        let expected = ordinary.project(Tensor::ones([2, 96], &device)).unwrap().try_into_vec_as::<f32>().unwrap();
        for restored in [
            ordinary.clone().load_record(tiled.clone().into_record()),
            tiled.load_record(ordinary.clone().into_record()),
        ] {
            let actual = restored.project(Tensor::ones([2, 96], &device)).unwrap().try_into_vec_as::<f32>().unwrap();
            assert_eq!(actual, expected);
            assert_eq!(restored.storage_words(), bytes.len() / 4);
            let actual = restored.embedding(&[0, 31, 32, 63]).unwrap().try_into_vec_as::<f32>().unwrap();
            let expected = ordinary.embedding(&[0, 31, 32, 63]).unwrap().try_into_vec_as::<f32>().unwrap();
            assert_eq!(actual, expected);
        }
    }
}
