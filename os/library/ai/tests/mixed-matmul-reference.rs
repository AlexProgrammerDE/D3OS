#![cfg(feature = "training")]
use d3os_ai::validation::mixed_matmul::{self as m, Class, DType, Group, Options, Policy, Report, RouteKey, Scope, Values};

// Independent integer decoding for the dyadic proof domain. No production
// numerical helper or candidate backend supplies fixture expectations.
fn dyadic(dtype: DType, bits: u64, scale: i32) -> i128 {
    let (eb, fb, bias) = match dtype {
        DType::F16 => (5, 10, 15),
        DType::BF16 => (8, 7, 127),
        DType::F32 | DType::Flex32 => (8, 23, 127),
        DType::F64 => (11, 52, 1023),
        _ => panic!("float fixture"),
    };
    let exp = (bits >> fb) & ((1 << eb) - 1);
    let frac = bits & ((1 << fb) - 1);
    assert_ne!(exp, (1 << eb) - 1);
    let (n, power) = if exp == 0 {
        (frac, 1 - bias - fb + scale)
    } else {
        ((1 << fb) + frac, exp as i32 - bias - fb + scale)
    };
    let n = if power >= 0 {
        (n as i128) << power
    } else {
        assert_eq!(n & ((1u64 << (-power).min(63)) - 1), 0);
        (n as i128) >> (-power).min(127)
    };
    if bits >> (eb + fb) == 0 { n } else { -n }
}
fn at(shape: &[usize], coord: &[usize]) -> usize {
    shape.iter().zip(coord).fold(0, |i, (&n, &c)| {
        assert!(c < n);
        i * n + c
    })
}
fn coord(shape: &[usize], mut i: usize) -> Vec<usize> {
    let mut c = vec![0; shape.len()];
    for a in (0..shape.len()).rev() {
        c[a] = i % shape[a];
        i /= shape[a];
    }
    c
}
fn batch(c: &[usize], shape: &[usize]) -> Vec<usize> {
    c[c.len() - shape.len()..]
        .iter()
        .zip(shape)
        .map(|(&c, &n)| if n == 1 { 0 } else { c })
        .collect()
}
fn fixed(v: Values, scale: i32) -> Vec<i128> {
    v.bits().into_iter().map(|b| dyadic(v.dtype, b, scale)).collect()
}
fn indexed_dot(a: Values, b: Values, out: Values) {
    let (av, bv, expected) = (fixed(a, 4), fixed(b, 4), fixed(out, 8));
    let rank = out.shape.len();
    let k = a.shape[a.shape.len() - 1];
    for (i, &e) in expected.iter().enumerate() {
        let c = coord(out.shape, i);
        let ac = batch(&c[..rank - 2], &a.shape[..a.shape.len() - 2]);
        let bc = batch(&c[..rank - 2], &b.shape[..b.shape.len() - 2]);
        let mut sum = 0;
        let mut abs = 0;
        for t in 0..k {
            let mut ai = ac.clone();
            ai.extend([c[rank - 2], t]);
            let mut bi = bc.clone();
            bi.extend([t, c[rank - 1]]);
            let product = av[at(a.shape, &ai)] * bv[at(b.shape, &bi)];
            sum += product;
            abs += product.abs();
        }
        // Every subset fits the normal F32 2^-8 lattice, independently of
        // order/tree. These bounds also fit the packed destination exactly.
        assert!(abs < 256);
        assert_eq!(sum, e, "index {i}");
    }
}
#[test]
fn corpus_budget_layout_and_raw_refinements_are_frozen() {
    let cases = m::fixtures();
    assert_eq!(cases.len(), 121);
    assert_eq!(m::RESERVED_ROWS, 3);
    let groups = [
        (Group::Geometry, 39),
        (Group::Numeric, 24),
        (Group::Gradient, 30),
        (Group::NativeExtension, 12),
        (Group::Preservation, 8),
        (Group::F64Diagnostic, 8),
    ];
    for (g, n) in groups {
        assert_eq!(cases.iter().filter(|c| c.group == g).count(), n);
    }
    assert_eq!(cases.iter().map(|c| c.numeric.len()).sum::<usize>(), 65);
    assert_eq!(cases.iter().map(|c| c.consumers.len()).sum::<usize>(), 141);
    let ranks: std::collections::BTreeSet<_> = cases
        .iter()
        .filter(|c| c.group == Group::Geometry)
        .map(|c| c.matrix("lhs").unwrap().values.shape.len())
        .collect();
    assert_eq!(ranks, (2..=8).collect());
    assert_eq!(cases.iter().filter(|c| !c.raw_matrices.is_empty()).count(), 3);
    let ids: std::collections::BTreeSet<_> = cases.iter().map(|c| c.id).collect();
    assert_eq!(ids.len(), 121);
    assert!(m::fixture("unknown").is_none());
}
#[test]
fn affine_roots_and_packing_preserve_every_source_bit() {
    for c in m::fixtures() {
        for matrix in c.matrices.iter().chain(c.raw_matrices) {
            let values = matrix.values;
            assert_eq!(m::unpack(values.dtype, &m::pack(values.dtype, &values.bits())).unwrap(), values.bits());
            if let Some(v) = matrix.view {
                let root = v.root.bits();
                let bits = values.bits();
                for (i, b) in bits.iter().enumerate() {
                    let co = coord(values.shape, i);
                    let offset = co.iter().zip(v.strides).fold(v.base as isize, |p, (&a, &s)| p + a as isize * s);
                    assert!(offset >= 0);
                    assert_eq!(root[offset as usize], *b);
                }
                assert_eq!(m::pack(v.root.dtype, &root), v.root.bytes());
            }
        }
    }
    assert!(m::unpack(DType::F16, &[0]).is_err());
}
#[test]
fn every_geometry_cell_and_tile_refinement_has_an_independent_all_order_integer_proof() {
    for c in m::fixtures().iter().filter(|c| c.group == Group::Geometry) {
        indexed_dot(
            c.matrix("lhs").unwrap().values,
            c.matrix("rhs").unwrap().values,
            c.expectation("expected").unwrap().values,
        );
        if !c.raw_matrices.is_empty() {
            indexed_dot(
                c.raw_matrices[0].values,
                c.raw_matrices[1].values,
                c.raw_expected.iter().find(|e| e.name == "tile.output").unwrap().values,
            );
        }
    }
}
#[test]
fn broadcast_gradients_match_an_independent_index_scatter() {
    for c in m::fixtures().iter().filter(|c| c.group == Group::Gradient && c.matrix("lhs").is_some()) {
        let a = c.matrix("lhs").unwrap().values;
        let b = c.matrix("rhs").unwrap().values;
        let g = c.matrix("upstream").unwrap().values;
        let av = fixed(a, 4);
        let bv = fixed(b, 4);
        let gv = fixed(g, 4);
        let mut da = vec![0i128; a.len];
        let mut db = vec![0i128; b.len];
        for (i, &up) in gv.iter().enumerate() {
            let co = coord(g.shape, i);
            let ac = batch(&co[..co.len() - 2], &a.shape[..a.shape.len() - 2]);
            let bc = batch(&co[..co.len() - 2], &b.shape[..b.shape.len() - 2]);
            for t in 0..a.shape[a.shape.len() - 1] {
                let mut ai = ac.clone();
                ai.extend([co[co.len() - 2], t]);
                let mut bi = bc.clone();
                bi.extend([t, co[co.len() - 1]]);
                let ia = at(a.shape, &ai);
                let ib = at(b.shape, &bi);
                da[ia] += up * bv[ib];
                db[ib] += av[ia] * up;
            }
        }
        assert_eq!(da, fixed(c.expectation("dA").unwrap().values, 8), "{} dA", c.id);
        assert_eq!(db, fixed(c.expectation("dB").unwrap().values, 8), "{} dB", c.id);
        if let Some(e) = c.expectation("shared_source_gradient") {
            assert_eq!(da.iter().zip(&db).map(|(a, b)| a + b).collect::<Vec<_>>(), fixed(e.values, 8));
        }
    }
}
#[test]
fn exception_and_subnormal_gates_cannot_be_weakened_by_diagnostics() {
    for dtype in [DType::F16, DType::BF16, DType::Flex32] {
        let nan = match dtype {
            DType::F16 => 0x7e01,
            DType::BF16 => 0x7fc1,
            _ => 0x7fc00001,
        };
        assert_eq!(m::classification(dtype, nan), Class::NaN);
        assert!(m::compare(dtype, Policy::NaNClass, 0, nan));
        assert!(!m::compare(dtype, Policy::NaNClass, 0, 0));
        assert!(!m::compare(dtype, Policy::Exact, 1, 0));
        let sign = 1u64 << (dtype.size() * 8 - 1);
        assert!(m::compare(dtype, Policy::ZeroClass, 0, sign));
        assert!(!m::compare(dtype, Policy::Exact, 0, sign));
        assert!(m::compare(dtype, Policy::Observation, 0, nan));
        assert!(!m::compare(dtype, Policy::ProjectInterval { low: 0, high: 1 }, 0, nan));
    }
    let class_cells = m::fixtures()
        .iter()
        .filter(|c| c.group == Group::Numeric)
        .flat_map(|c| c.expectation("expected").unwrap().policies)
        .filter(|&&p| matches!(p, Policy::NaNClass | Policy::Infinity))
        .count();
    assert!(class_cells >= 12);
    let empirical = m::fixtures()
        .iter()
        .filter(|c| c.group == Group::Numeric)
        .flat_map(|c| c.expectation("expected").unwrap().policies)
        .filter(|p| matches!(p, Policy::ProjectInterval { .. }))
        .count();
    assert_eq!(empirical, 4);
    let c = m::fixture("f16_operand_range_and_output_overflow").unwrap();
    let scratch = m::scratch_expectation(c).unwrap();
    assert_eq!(scratch.values.dtype, DType::F32);
    assert_eq!(m::classification(DType::F32, scratch.values.bits()[0]), Class::Finite);
    assert_eq!(
        m::classification(DType::F16, c.expectation("expected").unwrap().values.bits()[0]),
        Class::Infinity
    );
}
#[test]
fn reporting_rejects_duplicates_missing_routes_and_metadata_changes() {
    let options = Options {
        case_id: Some("f16_dense_odd".into()),
        ..Options::default()
    };
    let c = m::fixture("f16_dense_odd").unwrap();
    let e = c.expectation("expected").unwrap();
    let key = RouteKey {
        case_id: c.id,
        route: "primitive",
        output: e.name,
    };
    let mut report = Report::new(&options, Scope::Shared);
    report.observe(key.clone(), e.values.dtype, e.values.shape, e.values.bits());
    assert_eq!(report.failed, 0);
    report.observe(key.clone(), e.values.dtype, e.values.shape, e.values.bits());
    assert_eq!(report.failed, 1);
    report.observe(
        RouteKey {
            case_id: c.id,
            route: "unexpected",
            output: e.name,
        },
        e.values.dtype,
        e.values.shape,
        e.values.bits(),
    );
    assert_eq!(report.failed, 2);
    report.finish();
    assert!(!report.is_success());
    assert!(report.unobserved > 0);
    let mut wrong = Report::new(&options, Scope::Shared);
    wrong.observe(key, DType::BF16, e.values.shape, e.values.bits());
    assert_eq!(wrong.required_checked, 0);
    assert_eq!(wrong.failed, 1);
    assert!(m::success_footer(&options, Scope::Shared, &report).is_err());
    let mut empty = Report::new(&Options::default(), Scope::Shared);
    empty.finish();
    assert!(!empty.is_success());
    assert_eq!(empty.required_checked, 0);
    assert_eq!(empty.diagnostic_observed, 0);
}
#[test]
fn expected_counts_are_pure_and_external_scopes_stay_separate() {
    let s = m::expected_summary(&Options::default(), Scope::Shared);
    assert_eq!(s.selected_rows, 93);
    assert_eq!(s.external_gates, 0);
    assert_eq!(s.consumer_routes, 141);
    assert_eq!(s.model_values, 8);
    // Source-preservation routes and shared transpose forward values must not
    // enter the gradient population. Count the actual expected AD destinations.
    let gradient_values = m::fixtures()
        .iter()
        .filter(|c| c.group == Group::Gradient)
        .map(|c| {
            let tracked = c
                .expected
                .iter()
                .filter(|e| match e.name {
                    "dA" => !matches!(c.tracking, m::Tracking::Rhs | m::Tracking::Shared),
                    "dB" => !matches!(c.tracking, m::Tracking::Lhs | m::Tracking::Shared),
                    "dx" | "dW" | "db" => true,
                    "shared_source_gradient" | "shared_transpose.gradient" => c.tracking == m::Tracking::Shared,
                    _ => false,
                })
                .map(|e| e.values.len)
                .sum::<usize>();
            tracked
                + c.expected
                    .iter()
                    .filter(|e| e.stage == "source_view_adjoint")
                    .map(|e| e.values.len)
                    .sum::<usize>()
        })
        .sum::<usize>()
        + m::fixtures()
            .iter()
            .filter(|c| c.group == Group::Numeric)
            .filter_map(|c| c.expectation("gradient.dA"))
            .map(|e| e.values.len)
            .sum::<usize>();
    assert_eq!(s.gradient_values, gradient_values);
    assert!(s.diagnostic_values > 0);
    assert_eq!(s.f64_rows, 0);
    assert_eq!(s.f64_values, 0);
    let all = m::expected_summary(&Options::default(), Scope::All);
    assert_eq!(all.selected_rows, 121);
    assert_eq!(all.preservation_rows, 8);
    assert_eq!(all.native_extension_rows, 12);
    assert_eq!(all.f64_rows, 8);
    assert_eq!(s, m::expected_summary(&Options::default(), Scope::Shared));
}
#[cfg(feature = "cpu")]
#[test]
fn actual_public_dispatch_cpu_checks_exact_geometry_tracking_and_linear() {
    use d3os_ai::burn::tensor::{Device, try_read_sync};
    let device = Device::flex();
    for id in [
        "f16_dense_odd",
        "bf16_dense_odd",
        "f16_lhs_tracked_only",
        "f16_rhs_tracked_only",
        "f16_shared_source_both_operands",
        "f16_both_multi_axis_batch_broadcast",
        "f16_transpose_signed_view_adjoints",
        "f16_linear_with_bias",
    ] {
        let options = Options {
            case_id: Some(id.into()),
            ..Options::default()
        };
        let r = try_read_sync(m::run(&device, &options, Scope::Shared)).unwrap();
        assert!(r.is_success(), "{id}: {}", m::report_json(&r));
    }
}
#[test]
fn flattened_linear_gradients_cover_empty_batch_and_zero_output_identities() {
    for c in m::fixtures().iter().filter(|c| c.group == Group::Gradient && c.matrix("input").is_some()) {
        let x = c.matrix("input").unwrap().values;
        let w = c.matrix("weight").unwrap().values;
        let g = c.matrix("upstream").unwrap().values;
        let xv = fixed(x, 4);
        let wv = fixed(w, 4);
        let gv = fixed(g, 4);
        let rows = x.shape[..x.shape.len() - 1].iter().product::<usize>();
        let k = w.shape[0];
        let n = w.shape[1];
        let mut dx = vec![0i128; x.len];
        let mut dw = vec![0i128; w.len];
        let mut db = vec![0i128; n];
        for row in 0..rows {
            for column in 0..n {
                let gradient = gv[row * n + column];
                db[column] += gradient;
                for inner in 0..k {
                    dx[row * k + inner] += gradient * wv[inner * n + column];
                    dw[inner * n + column] += xv[row * k + inner] * gradient;
                }
            }
        }
        assert_eq!(dx, fixed(c.expectation("dx").unwrap().values, 8));
        assert_eq!(dw, fixed(c.expectation("dW").unwrap().values, 8));
        assert_eq!(db, fixed(c.expectation("db").unwrap().values, 4));
    }
}
#[test]
fn source_nan_payload_corruption_and_out_of_width_words_fail() {
    let c = m::fixture("f16_dense_odd").unwrap();
    let matrix = c.matrix("lhs").unwrap();
    let root = matrix.view.unwrap().root;
    let options = Options {
        case_id: Some(c.id.into()),
        ..Options::default()
    };
    let mut report = Report::new(&options, Scope::Shared);
    let mut bits = root.bits();
    assert_eq!(m::classification(root.dtype, bits[0]), Class::NaN);
    bits[0] ^= 1;
    report.observe(
        RouteKey {
            case_id: c.id,
            route: "source_after",
            output: "lhs",
        },
        root.dtype,
        root.shape,
        bits,
    );
    assert!(report.failed > 0);
    assert!(!m::compare(DType::F16, Policy::ZeroClass, 0, 1 << 40));
    let empty = Options {
        case_id: Some("unknown".into()),
        ..Options::default()
    };
    let mut report = Report::new(&empty, Scope::Shared);
    report.finish();
    assert!(!report.is_success());
}

#[test]
fn report_finalization_blocks_every_unobserved_scope_and_is_idempotent() {
    for scope in [Scope::Shared, Scope::NativeExtensions, Scope::Preservation, Scope::F64Diagnostics, Scope::All] {
        let options = Options::default();
        let mut report = Report::new(&options, scope);
        assert!(!report.is_success(), "new {scope:?}");
        assert!(m::success_footer(&options, scope, &report).is_err());
        assert_eq!(report.external_unobserved, report.expected.external_gates);
        report.finish();
        assert!(!report.is_success(), "unobserved {scope:?}");
        assert!(m::success_footer(&options, scope, &report).is_err());
        assert!(report.unobserved > 0 || report.external_unobserved > 0);
        let finalized = (
            report.failed,
            report.unobserved,
            report.external_unobserved,
            report.attempted,
            report.required_checked,
        );
        report.finish();
        assert_eq!(
            finalized,
            (
                report.failed,
                report.unobserved,
                report.external_unobserved,
                report.attempted,
                report.required_checked
            )
        );
        let case = m::fixture("f16_dense_odd").unwrap();
        let expected = case.expectation("expected").unwrap();
        report.observe(
            RouteKey {
                case_id: case.id,
                route: "primitive",
                output: expected.name,
            },
            expected.values.dtype,
            expected.values.shape,
            expected.values.bits(),
        );
        assert_eq!(report.attempted, finalized.3);
        assert_eq!(report.required_checked, finalized.4);
        assert_eq!(report.failed, finalized.0 + 1);
        report.finish();
        assert!(!report.is_success());
        assert!(m::success_footer(&options, scope, &report).is_err());
    }
}
#[test]
fn complete_observations_require_finish_and_late_observations_invalidate_success() {
    let c = m::fixture("f16_dense_odd").unwrap();
    let options = Options {
        case_id: Some(c.id.into()),
        ..Options::default()
    };
    let expected = c.expectation("expected").unwrap();
    let mut report = Report::new(&options, Scope::Shared);
    for route in ["primitive", "public"] {
        report.observe(
            RouteKey {
                case_id: c.id,
                route,
                output: expected.name,
            },
            expected.values.dtype,
            expected.values.shape,
            expected.values.bits(),
        );
    }
    for consumer in c.consumers {
        let e = consumer.expected;
        report.observe(
            RouteKey {
                case_id: c.id,
                route: "consumer",
                output: e.name,
            },
            e.values.dtype,
            e.values.shape,
            e.values.bits(),
        );
    }
    for matrix in c.matrices {
        let root = matrix.view.map_or(matrix.values, |view| view.root);
        for route in ["source_before", "source_after", "alias_after"] {
            report.observe(
                RouteKey {
                    case_id: c.id,
                    route,
                    output: matrix.name,
                },
                root.dtype,
                root.shape,
                root.bits(),
            );
        }
    }
    assert_eq!(report.required_checked, report.expected.required_values);
    assert!(!report.is_success());
    assert!(m::success_footer(&options, Scope::Shared, &report).is_err());
    report.finish();
    assert!(report.is_success());
    let counts = (report.failed, report.attempted, report.required_checked);
    report.finish();
    assert!(report.is_success());
    assert_eq!(counts, (report.failed, report.attempted, report.required_checked));
    report.observe(
        RouteKey {
            case_id: c.id,
            route: "primitive",
            output: expected.name,
        },
        expected.values.dtype,
        expected.values.shape,
        expected.values.bits(),
    );
    assert!(!report.is_success());
    assert_eq!(report.failed, 1);
    assert_eq!(report.attempted, counts.1);
    assert_eq!(report.required_checked, counts.2);
    report.finish();
    assert!(!report.is_success());
    assert!(m::success_footer(&options, Scope::Shared, &report).is_err());
}
fn corrupted_stage_witness(case: &m::Case, route: &'static str, expected: &m::Expected) -> m::NumericWitness {
    let options = Options {
        case_id: Some(case.id.into()),
        ..Options::default()
    };
    let mut report = Report::new(&options, Scope::All);
    let mut actual = expected.values.bits();
    let index = actual
        .iter()
        .position(|&bits| !matches!(m::classification(expected.values.dtype, bits), Class::NaN | Class::Infinity))
        .unwrap();
    actual[index] ^= 1;
    report.observe(
        RouteKey {
            case_id: case.id,
            route,
            output: expected.name,
        },
        expected.values.dtype,
        expected.values.shape,
        actual,
    );
    report.worst.unwrap()
}
fn transpose_reference(values: Values) -> (Vec<usize>, Vec<u64>) {
    let mut shape = values.shape.to_vec();
    let rank = shape.len();
    shape.swap(rank - 2, rank - 1);
    let source = values.bits();
    let bits = (0..values.len)
        .map(|index| {
            let mut c = coord(&shape, index);
            c.swap(rank - 2, rank - 1);
            source[at(values.shape, &c)]
        })
        .collect();
    (shape, bits)
}
#[test]
fn gradient_stage_witnesses_bind_upstream_transposes_and_stored_sum_inputs() {
    let c = m::fixture("f16_destination_ties_both_parities_and_sides").unwrap();
    let witness = corrupted_stage_witness(c, "primitive", c.expectation("gradient.dA_before_sum").unwrap());
    let upstream = c.matrix("gradient.upstream").unwrap().values;
    let (shape, bits) = transpose_reference(c.matrix("gradient.rhs").unwrap().values);
    assert_eq!(witness.stage_witness.operation, "matmul");
    assert_eq!(witness.lhs_shape, upstream.shape);
    assert_eq!(witness.lhs, upstream.bits());
    assert_eq!(witness.rhs_shape, shape);
    assert_eq!(witness.rhs, bits);
    assert_eq!(witness.stage_witness.operands[0].source, "gradient.upstream");
    assert_eq!(witness.stage_witness.output_shape, [3, 1, 1]);
    let sum = corrupted_stage_witness(c, "autodiff_staging", c.expectation("gradient.dA").unwrap());
    assert_eq!(sum.stage_witness.operation, "typed_sum");
    assert_eq!(sum.stage_witness.axis, Some(0));
    assert_eq!(sum.stage_witness.operands.len(), 1);
    assert!(sum.rhs.is_empty());
    assert!(sum.rhs_shape.is_empty());
    assert_eq!(sum.lhs, c.expectation("gradient.dA_before_sum").unwrap().values.bits());
    let c = m::fixture("f16_both_multi_axis_batch_broadcast").unwrap();
    for (name, matrix, upstream_first) in [("dA_before_sum", "rhs", true), ("dB_before_sum", "lhs", false)] {
        let witness = corrupted_stage_witness(c, "primitive", c.expectation(name).unwrap());
        let (shape, bits) = transpose_reference(c.matrix(matrix).unwrap().values);
        let g = c.matrix("upstream").unwrap().values;
        if upstream_first {
            assert_eq!(witness.lhs, g.bits());
            assert_eq!(witness.rhs, bits);
            assert_eq!(witness.rhs_shape, shape);
        } else {
            assert_eq!(witness.lhs, bits);
            assert_eq!(witness.lhs_shape, shape);
            assert_eq!(witness.rhs, g.bits());
        }
    }
    let sum = corrupted_stage_witness(c, "primitive", c.expectation("dA").unwrap());
    assert_eq!(sum.stage_witness.axis, Some(3));
    assert_eq!(sum.stage_witness.operands.len(), 1);
    assert_eq!(sum.lhs, c.expectation("dA_axes.1").unwrap().values.bits());
    assert_eq!(sum.lhs_shape, c.expectation("dA_axes.1").unwrap().values.shape);
}
#[test]
fn linear_consumer_and_repeated_source_witnesses_identify_their_own_graphs() {
    let c = m::fixture("f16_linear_with_bias").unwrap();
    let forward = corrupted_stage_witness(c, "public", c.expectation("forward").unwrap());
    assert_eq!(forward.stage_witness.operation, "typed_add");
    assert_eq!(forward.lhs, c.expectation("stored_forward").unwrap().values.bits());
    assert_eq!(forward.lhs_shape, c.expectation("forward").unwrap().values.shape);
    assert_eq!(forward.rhs, c.matrix("bias").unwrap().values.bits());
    let dx = corrupted_stage_witness(c, "autodiff", c.expectation("dx").unwrap());
    assert_eq!(dx.stage_witness.operation, "matmul");
    assert_eq!(dx.lhs_shape, [6, 7]);
    assert_eq!(dx.rhs_shape, [7, 5]);
    assert_eq!(dx.stage_witness.result_shape, [6, 5]);
    assert_eq!(dx.stage_witness.output_shape, [2, 3, 5]);
    let db = corrupted_stage_witness(c, "autodiff", c.expectation("db").unwrap());
    assert_eq!(db.stage_witness.operation, "typed_sum");
    assert_eq!(db.stage_witness.axis, Some(0));
    assert_eq!(db.stage_witness.operands.len(), 1);
    assert!(db.rhs.is_empty());
    let consumer = c.consumers.iter().find(|consumer| consumer.input == "dx").unwrap();
    let sum = corrupted_stage_witness(c, "consumer", &consumer.expected);
    assert_eq!(sum.stage_witness.operation, "typed_sum");
    assert_eq!(sum.stage_witness.axis, Some(consumer.axis));
    assert_eq!(sum.lhs, c.expectation("dx").unwrap().values.bits());
    let c = m::fixture("f16_shared_source_both_operands").unwrap();
    let join = corrupted_stage_witness(c, "autodiff_transpose", c.expectation("shared_transpose.gradient").unwrap());
    assert_eq!(join.stage_witness.operation, "typed_add");
    assert_eq!(join.stage_witness.operands.len(), 2);
    for operand in &join.stage_witness.operands {
        assert_eq!(operand.origin, "derived_stage");
        assert!(operand.bits.is_none());
        assert_eq!(operand.producer.as_ref().unwrap().operation, "matmul");
    }
    assert_eq!(
        join.stage_witness.operands[0].producer.as_ref().unwrap().operands[0].bits.as_ref().unwrap(),
        &c.matrix("upstream").unwrap().values.bits()
    );
    assert_eq!(join.stage_witness.operands[1].views, [m::OperandView::Transpose(0, 1)]);
}
