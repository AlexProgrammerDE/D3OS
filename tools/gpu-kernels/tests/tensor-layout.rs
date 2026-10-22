#[path = "../src/tensor.rs"]
mod tensor;

use tensor::{a_coordinate, b_coordinate, output_coordinate};

#[test]
fn warp_fragments_cover_each_matrix_element_once() {
    let mut a = [[0; 16]; 16];
    let mut b = [[0; 8]; 16];
    let mut output = [[0; 8]; 16];
    for lane in 0..32 {
        for element in 0..8 {
            let (row, col) = a_coordinate(lane, element);
            a[row as usize][col as usize] += 1;
        }
        for element in 0..4 {
            let (row, col) = b_coordinate(lane, element);
            b[row as usize][col as usize] += 1;
            let (row, col) = output_coordinate(lane, element);
            output[row as usize][col as usize] += 1;
        }
    }
    assert!(a.into_iter().flatten().all(|visits| visits == 1));
    assert!(b.into_iter().flatten().all(|visits| visits == 1));
    assert!(output.into_iter().flatten().all(|visits| visits == 1));
}

#[test]
fn fragment_gemm_matches_dense_reference_with_partial_tiles() {
    // This exercises the operand distribution, masks, multiple K steps, and
    // output ownership. It does not emulate or validate a hardware instruction.
    for (rows, inner, cols) in [(1, 1, 1), (16, 16, 8), (17, 33, 9), (31, 19, 23), (3, 0, 7)] {
        let a: Vec<f32> = (0..rows * inner).map(|i| ((i * 7 % 23) as f32 - 11.0) / 8.0).collect();
        let b: Vec<f32> = (0..inner * cols).map(|i| ((i * 13 % 19) as f32 - 9.0) / 8.0).collect();
        let mut actual = vec![f32::NAN; (rows * cols) as usize];
        let mut writes = vec![0; actual.len()];
        for row_base in (0..rows).step_by(16) {
            for col_base in (0..cols).step_by(8) {
                let mut accumulators = [[0.0f32; 4]; 32];
                for k_base in (0..inner).step_by(16) {
                    let mut ar = [[0.0f32; 8]; 32];
                    let mut br = [[0.0f32; 4]; 32];
                    for lane in 0..32 {
                        for element in 0..8 {
                            let (row, k) = a_coordinate(lane, element);
                            let (row, k) = (row_base + row, k_base + k);
                            if row < rows && k < inner {
                                ar[lane as usize][element as usize] = a[(row * inner + k) as usize];
                            }
                        }
                        for element in 0..4 {
                            let (k, col) = b_coordinate(lane, element);
                            let (k, col) = (k_base + k, col_base + col);
                            if k < inner && col < cols {
                                br[lane as usize][element as usize] = b[(k * cols + col) as usize];
                            }
                        }
                    }
                    for lane in 0..32 {
                        for element in 0..4 {
                            let (row, col) = output_coordinate(lane, element);
                            for k in 0..16 {
                                // Decode the MMA register layout independently of
                                // the load-coordinate functions under test.
                                let av = ar[(row % 8 * 4 + k % 8 / 2) as usize][(k / 8 * 4 + row / 8 * 2 + k % 2) as usize];
                                let bv = br[(col * 4 + k % 8 / 2) as usize][(k / 8 * 2 + k % 2) as usize];
                                accumulators[lane as usize][element as usize] += av * bv;
                            }
                        }
                    }
                }
                for lane in 0..32 {
                    for element in 0..4 {
                        let (row, col) = output_coordinate(lane, element);
                        let (row, col) = (row_base + row, col_base + col);
                        if row < rows && col < cols {
                            let offset = (row * cols + col) as usize;
                            actual[offset] = accumulators[lane as usize][element as usize];
                            writes[offset] += 1;
                        }
                    }
                }
            }
        }
        for row in 0..rows {
            for col in 0..cols {
                let expected: f32 = (0..inner).map(|k| a[(row * inner + k) as usize] * b[(k * cols + col) as usize]).sum();
                assert_eq!(actual[(row * cols + col) as usize], expected);
            }
        }
        assert!(writes.into_iter().all(|count| count == 1));
    }
}
