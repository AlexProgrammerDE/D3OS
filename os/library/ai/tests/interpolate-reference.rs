use d3os_ai::validation::interpolate::{Mode, backward, forward};
#[test]
fn reference_distinguishes_nearest_coordinates_and_single_output_alignment() {
    let input = [10f32, 20.0];
    let shape = [1, 1, 1, 2];
    assert_eq!(forward(&input, shape, [1, 3], Mode::Nearest, false), [10.0, 10.0, 20.0]);
    assert_eq!(forward(&input, shape, [1, 3], Mode::Exact, false), [10.0, 20.0, 20.0]);
    assert_eq!(forward(&input, shape, [1, 1], Mode::Linear, false), [15.0]);
    assert_eq!(forward(&input, shape, [1, 1], Mode::Linear, true), [10.0]);
}
#[test]
fn reference_cubic_uses_keys_three_quarter_filter_and_its_transpose() {
    let shape = [1, 1, 1, 4];
    assert_eq!(forward(&[0.0, 1.0, 0.0, 0.0], shape, [1, 1], Mode::Cubic, false), [19.0 / 32.0]);
    assert_eq!(
        backward(&[2.0], shape, [1, 1], Mode::Cubic, false),
        [-3.0 / 16.0, 19.0 / 16.0, 19.0 / 16.0, -3.0 / 16.0]
    );
}
