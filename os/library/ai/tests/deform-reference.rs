#![cfg(feature = "training")]
use d3os_ai::validation::deform::{Case, Pattern, cases, masks, reference, sample, values};
#[test]
fn bilinear_reference_anchors_modulation_and_all_derivatives() {
    let case = Case {
        input: [1, 1, 2, 2],
        weights: [1, 1, 1, 1],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        weight_groups: 1,
        offset_groups: 1,
        pattern: Pattern::Fractional,
    };
    let x = [1.0, 2.0, 3.0, 4.0];
    let offsets = [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5];
    let r = reference(&case, &x, &offsets, &[2.0], Some(&[0.5; 4]), Some(&[1.0]), &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(r.output[0], 3.0);
    assert_eq!(r.dx, [0.375, 0.375, 0.125, 0.125]);
    assert_eq!(r.doffset, [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    assert_eq!(r.dw, [1.0]);
    assert_eq!(r.dmask.unwrap(), [4.0, 0.0, 0.0, 0.0]);
    assert_eq!(r.db, [1.0]);
    let r = reference(&case, &x, &offsets, &[2.0], Some(&[0.0; 4]), None, &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(r.output, [0.0; 4]);
    assert_eq!(r.dx, [0.0; 4]);
    assert_eq!(r.doffset, [0.0; 8]);
    assert_eq!(r.dw, [0.0]);
    assert_eq!(r.dmask.unwrap(), [4.0, 0.0, 0.0, 0.0]);
    let (value, dy, dx, _) = sample(&x, case.input, 0, 0, -1.0, 0.5);
    assert_eq!((value, dy, dx), (0.0, 1.5, 0.0));
    let (value, dy, dx, _) = sample(&x, case.input, 0, 0, -1.01, 0.5);
    assert_eq!((value, dy, dx), (0.0, 0.0, 0.0));
}
#[test]
fn deform_analytic_gradients_agree_with_finite_differences() {
    for case in cases() {
        let x = values(&case.input, 1, true);
        let offset = case.offsets();
        let w = values(&case.weights, 2, true);
        let mask = masks(&case.mask_shape());
        let bias = values(&[case.output()[1]], 3, true);
        let dy = values(&case.output(), 4, true);
        let r = reference(&case, &x, &offset, &w, Some(&mask), Some(&bias), &dy);
        let loss = |x: &[f32], off: &[f32], w: &[f32], m: &[f32], b: &[f32]| {
            reference(&case, x, off, w, Some(m), Some(b), &dy)
                .output
                .iter()
                .zip(&dy)
                .map(|(&v, &g)| v * f64::from(g))
                .sum::<f64>()
        };
        for (kind, source, expected) in [
            (0, &x, &r.dx),
            (1, &offset, &r.doffset),
            (2, &w, &r.dw),
            (3, &mask, r.dmask.as_ref().unwrap()),
            (4, &bias, &r.db),
        ] {
            if kind == 1 && case.pattern == Pattern::Zero {
                continue;
            }
            for index in [0, source.len() / 2, source.len() - 1] {
                let mut plus = source.clone();
                let mut minus = source.clone();
                plus[index] += 1.0 / 256.0;
                minus[index] -= 1.0 / 256.0;
                let (a, b) = match kind {
                    0 => (loss(&plus, &offset, &w, &mask, &bias), loss(&minus, &offset, &w, &mask, &bias)),
                    1 => (loss(&x, &plus, &w, &mask, &bias), loss(&x, &minus, &w, &mask, &bias)),
                    2 => (loss(&x, &offset, &plus, &mask, &bias), loss(&x, &offset, &minus, &mask, &bias)),
                    3 => (loss(&x, &offset, &w, &plus, &bias), loss(&x, &offset, &w, &minus, &bias)),
                    _ => (loss(&x, &offset, &w, &mask, &plus), loss(&x, &offset, &w, &mask, &minus)),
                };
                let numerical = (a - b) / f64::from(plus[index] - minus[index]);
                assert!(
                    (numerical - expected[index]).abs() < 1e-8 * (1.0 + expected[index].abs()),
                    "case={case:?} kind={kind} i={index} {numerical} != {}",
                    expected[index]
                );
            }
        }
    }
}
