#![cfg(feature = "training")]
use d3os_ai::validation::spatial_convolution::{Case, cases, reference, values};
#[test]
fn scalar_reference_anchors_volume_orientation_and_output_padding() {
    let case = Case {
        input: vec![1, 1, 2, 2, 2],
        weights: vec![1, 1, 2, 2, 2],
        stride: vec![1; 3],
        padding: vec![0; 3],
        dilation: vec![1; 3],
        extra: vec![0; 3],
        groups: 1,
        transpose: false,
    };
    let x = (1..=8).map(|i| i as f32).collect::<Vec<_>>();
    let r = reference(&case, &x, &x, Some(&[3.0]), &[2.0]);
    assert_eq!(r.output, [207.0]);
    assert_eq!(r.dx, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    assert_eq!(r.dw, r.dx);
    assert_eq!(r.db, [2.0]);
    let case = Case {
        input: vec![1, 1, 1, 1],
        weights: vec![1, 1, 2, 2],
        stride: vec![2; 2],
        padding: vec![0; 2],
        dilation: vec![1; 2],
        extra: vec![1; 2],
        groups: 1,
        transpose: true,
    };
    let r = reference(
        &case,
        &[2.0],
        &[1.0, 2.0, 3.0, 4.0],
        Some(&[1.0]),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    assert_eq!(r.output, [3.0, 5.0, 1.0, 7.0, 9.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(r.dx, [37.0]);
    assert_eq!(r.dw, [2.0, 4.0, 8.0, 10.0]);
    assert_eq!(r.db, [45.0]);
}
#[test]
fn analytic_gradients_agree_with_centered_differences_for_every_geometry() {
    for case in cases() {
        let x = values(&case.input, 1);
        let w = values(&case.weights, 2);
        let b = values(&[case.output()[1]], 3);
        let dy = values(&case.output(), 4);
        let r = reference(&case, &x, &w, Some(&b), &dy);
        let loss = |x: &[f32], w: &[f32], b: &[f32]| {
            reference(&case, x, w, Some(b), &dy)
                .output
                .iter()
                .zip(&dy)
                .map(|(&a, &g)| a * f64::from(g))
                .sum::<f64>()
        };
        let epsilon = 1.0 / 32.0;
        for (kind, source, expected) in [(0, &x, &r.dx), (1, &w, &r.dw), (2, &b, &r.db)] {
            for index in [0, source.len() / 2, source.len() - 1] {
                let mut plus = source.clone();
                let mut minus = source.clone();
                plus[index] += epsilon;
                minus[index] -= epsilon;
                let (high, low) = match kind {
                    0 => (loss(&plus, &w, &b), loss(&minus, &w, &b)),
                    1 => (loss(&x, &plus, &b), loss(&x, &minus, &b)),
                    _ => (loss(&x, &w, &plus), loss(&x, &w, &minus)),
                };
                let numerical = (high - low) / f64::from(plus[index] - minus[index]);
                assert!(
                    (numerical - expected[index]).abs() < 1e-8 * (1.0 + expected[index].abs()),
                    "case={case:?} kind={kind} index={index} {numerical} != {}",
                    expected[index]
                );
            }
        }
    }
}
