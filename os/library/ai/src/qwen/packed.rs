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
        .with_record_shape([format.byte_len(shape).expect("validated canonical matrix size").div_ceil(4)].into())
}

fn reorder(tensor: Tensor<1, Int>, shape: [usize; 2], format: MatrixFormat, from: MatrixLayout, to: MatrixLayout) -> Tensor<1, Int> {
    let device = tensor.device();
    let data = tensor.try_into_data().expect("read packed matrix for record conversion");
    let length = from.byte_len(shape, format).expect("validated packed matrix size");
    let storage = data.as_bytes().get(..length).expect("complete packed matrix storage");
    let bytes = from.row_major_bytes(shape, format, storage).expect("validated packed matrix layout");
    let mut packed = to.pack(shape, format, &bytes).expect("encode packed matrix parameter");
    let count = packed.len().div_ceil(4);
    packed.resize(count * 4, 0);
    Tensor::from_data(TensorData::from_bytes_vec(packed, [count], DType::I32), &device)
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::qwen::{PackedMatrix, Quantization};
    use burn::{module::Module, tensor::Device};

    #[test]
    fn native_layout_selection_preserves_other_devices_formats_and_geometries() {
        let q8 = MatrixFormat::Quantized(Quantization::Q8_0);
        for shape in [[12288, 4096], [4096, 12288], [8192, 4096], [4096, 4096], [248320, 4096]] {
            assert_eq!(crate::qwen::packed_matrix_layout(true, shape, q8), MatrixLayout::Q8Columns32);
            assert_eq!(crate::qwen::packed_matrix_layout(false, shape, q8), MatrixLayout::RowMajor);
            for format in [MatrixFormat::F32, MatrixFormat::Bf16, MatrixFormat::Quantized(Quantization::Q4K)] {
                assert_eq!(crate::qwen::packed_matrix_layout(true, shape, format), MatrixLayout::RowMajor);
            }
        }
        for shape in [[248319, 4096], [248352, 4096], [248320, 4095], [4096, 248320], [32, 32]] {
            assert_eq!(crate::qwen::packed_matrix_layout(true, shape, q8), MatrixLayout::RowMajor);
        }
        let q3 = MatrixFormat::Quantized(Quantization::Q3K);
        assert_eq!(crate::qwen::packed_matrix_layout(true, [17408, 5120], q3), MatrixLayout::Q3Padded112);
        assert_eq!(crate::qwen::packed_matrix_layout(false, [17408, 5120], q3), MatrixLayout::RowMajor);
        for shape in [[17407, 5120], [17408, 5121], [5120, 17408], [32, 256]] {
            assert_eq!(crate::qwen::packed_matrix_layout(true, shape, q3), MatrixLayout::RowMajor);
        }
    }

    #[test]
    fn device_moves_cannot_silently_replace_quantized_arithmetic() {
        let device = Device::flex();
        let shape = [32, 32];
        let format = MatrixFormat::Quantized(Quantization::Q8_0);
        let bytes = alloc::vec![0; format.byte_len(shape).unwrap()];
        let mut matrix = PackedMatrix::from_bytes(shape, format, &bytes, &device).unwrap();
        for mode in [crate::qwen::PrefillQuantization::Q8_1, crate::qwen::PrefillQuantization::Q4KComponents] {
            matrix.prefill_quantization = mode;
            let moved = matrix.clone().to_device(&device);
            assert_eq!(moved.prefill_quantization, mode);
            assert!(moved.project(Tensor::ones([2, 32], &device)).is_err());
        }
    }

    #[test]
    fn q4_selection_cannot_silently_fall_back_after_device_movement() {
        let device = Device::flex();
        let shape = [32, 256];
        let format = MatrixFormat::Quantized(Quantization::Q4K);
        let bytes = alloc::vec![0; format.byte_len(shape).unwrap()];
        let mut matrix = PackedMatrix::from_bytes(shape, format, &bytes, &device).unwrap();
        let input = Tensor::<2>::ones([1, 256], &device);
        let expected = matrix.project(input.clone()).unwrap().try_into_vec_as::<f32>().unwrap();
        for kernel in [
            crate::qwen::Q4VectorKernel::Narrow,
            crate::qwen::Q4VectorKernel::Components,
            crate::qwen::Q4VectorKernel::Aligned,
        ] {
            assert!(matrix.set_q4_vector_kernel(kernel).is_err());
            assert_eq!(matrix.q4_vector_kernel, crate::qwen::Q4VectorKernel::Default);
            matrix.q4_vector_kernel = kernel;
            let moved = matrix.clone().to_device(&device);
            assert_eq!(moved.q4_vector_kernel, kernel);
            assert!(moved.project(input.clone()).is_err());
            matrix.q4_vector_kernel = crate::qwen::Q4VectorKernel::Default;
        }
        matrix.q4_vector_kernel = crate::qwen::Q4VectorKernel::Default;
        assert_eq!(matrix.project(input).unwrap().try_into_vec_as::<f32>().unwrap(), expected);
    }

    #[test]
    fn q3_components_require_native_and_survive_device_moves_without_fallback() {
        let device = Device::flex();
        let shape = [4, 256];
        let format = MatrixFormat::Quantized(Quantization::Q3K);
        let bytes = alloc::vec![0; format.byte_len(shape).unwrap()];
        let mut matrix = PackedMatrix::from_bytes(shape, format, &bytes, &device).unwrap();
        for kernel in [crate::qwen::Q3VectorKernel::Components, crate::qwen::Q3VectorKernel::Aligned] {
            assert!(matrix.set_q3_vector_kernel(kernel).is_err());
            assert_eq!(matrix.q3_vector_kernel, crate::qwen::Q3VectorKernel::Default);
            matrix.q3_vector_kernel = kernel;
            let moved = matrix.clone().to_device(&device);
            assert_eq!(moved.q3_vector_kernel, kernel);
            assert!(moved.project(Tensor::ones([1, 256], &device)).is_err());
            matrix.q3_vector_kernel = crate::qwen::Q3VectorKernel::Default;
        }
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
            q4_vector_kernel: crate::qwen::Q4VectorKernel::Default,
            q3_vector_kernel: crate::qwen::Q3VectorKernel::Default,
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

    #[test]
    fn padded_q3_records_and_device_moves_preserve_odd_blocks_and_canonical_words() {
        let device = Device::flex();
        let format = MatrixFormat::Quantized(Quantization::Q3K);
        for shape in [[1, 256], [3, 768]] {
            let mut bytes = alloc::vec![0; format.byte_len(shape).unwrap()];
            for (index, block) in bytes.chunks_exact_mut(110).enumerate() {
                for (lane, value) in block[..108].iter_mut().enumerate() {
                    *value = (index * 37 + lane * 13) as u8;
                }
                block[108..].copy_from_slice(&0x3800u16.to_le_bytes());
            }
            let ordinary = PackedMatrix::from_bytes(shape, format, &bytes, &device).unwrap();
            let packed = MatrixLayout::Q3Padded112.pack(shape, format, &bytes).unwrap();
            let count = packed.len() / 4;
            let padded = PackedMatrix {
                words: parameter(
                    Tensor::<1, Int>::from_data(TensorData::from_bytes_vec(packed, [count], DType::I32), &device),
                    shape,
                    format,
                    MatrixLayout::Q3Padded112,
                ),
                shape,
                format,
                layout: MatrixLayout::Q3Padded112,
                prefill_quantization: crate::qwen::PrefillQuantization::Accurate,
                q4_vector_kernel: crate::qwen::Q4VectorKernel::Default,
                q3_vector_kernel: crate::qwen::Q3VectorKernel::Default,
            };
            let input = Tensor::from_data(
                TensorData::new((0..2 * shape[1]).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect(), [2, shape[1]]),
                &device,
            );
            let expected = ordinary.project(input.clone()).unwrap().try_into_vec_as::<f32>().unwrap();
            let ids = [0, shape[0] as u32 - 1, 0];
            let embedding = ordinary.embedding(&ids).unwrap().try_into_vec_as::<f32>().unwrap();
            let wrong_shape = [shape[0], shape[1] + 256];
            let wrong = PackedMatrix::from_bytes(wrong_shape, format, &alloc::vec![0; format.byte_len(wrong_shape).unwrap()], &device).unwrap();
            assert!(padded.clone().try_load_record(wrong.into_record()).is_err());
            for restored in [
                padded.clone().to_device(&device),
                ordinary.clone().load_record(padded.clone().into_record()),
                padded.load_record(ordinary.clone().into_record()),
            ] {
                assert_eq!(restored.storage_words(), restored.layout.byte_len(shape, format).unwrap().div_ceil(4));
                let actual = restored.project(input.clone()).unwrap().try_into_vec_as::<f32>().unwrap();
                assert!(actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));
                let actual = restored.embedding(&ids).unwrap().try_into_vec_as::<f32>().unwrap();
                assert!(actual.iter().zip(&embedding).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }
}
