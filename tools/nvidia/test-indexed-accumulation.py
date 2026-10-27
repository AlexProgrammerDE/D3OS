#!/usr/bin/env python3
"""Bounded host tests. Synthetic positive records are never GPU evidence.

Run the missing-key phases separately to keep each CPU test phase below 90s:
  python3 -B tools/nvidia/test-indexed-accumulation.py MissingLinux
  python3 -B tools/nvidia/test-indexed-accumulation.py MissingNative
  python3 -B tools/nvidia/test-indexed-accumulation.py EvidenceTests LauncherTests
Optional D3OS_INDEXED_EVIDENCE_CACHE audits retained actual CLI MODEL/inventories.
D3OS_INDEXED_RAW_SOURCE_PINS selects explicit parent-approved Raw positive inputs;
the negative revoked-receipt control keeps the historical cache receipt.
"""

import copy
import hashlib
import importlib.util
import json
import os
import re
import resource
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load("indexed_evidence", "indexed-accumulation-evidence.py")
launcher = load("linux_reference", "linux-reference.py")
CACHE = Path(
    os.environ.get(
        "D3OS_INDEXED_EVIDENCE_CACHE",
        "/home/alex/.cache/d3os/burn-indexed-accumulation-evidence-work",
    )
)


# Source-only synthetic fixture: accepted54-harness.py for-marker inventory,
# SHA256 3846cf1410c226924c96834d0516bd79e836f1ce48dfb535651a1e6229a17ca7.
# These inherited assertions require parent validation and are not GPU evidence.
INHERITED54_MARKERS = (
    "NVIDIA_BURN_PACKED_MOVEMENT_RAW_OK gather=60 mask=400 assign=120 bool_scatter=50 queued=14 widths=1,2,4,8 boundaries=1,127,128,129,257 guards=verified full_writer=gather,mask initialized_partial=assign,bool_scatter tails=verified scalar64=exact rank8=true invalid_metadata=no_launch aliases=no_launch",
    "NVIDIA_BURN_PACKED_MOVEMENT_OK native=165 bool_scatter_argwhere=8 empty_invalid=32 scalar_argwhere=4 dtypes=15 bytes=exact nan_payloads=preserved source_alias=snapshot mask_offsets=4,5,6,7 scalar64=exact compaction_metadata_bytes=4 indexed_status_bytes=4 host_activation_reads=0 f32_i32_fastpaths=verified counters=verified",
    "BURN_MOVEMENT_CONFORMANCE_OK os=d3os backend=native dtypes=15 cases=4793 gather=270 select=270 mask_where=1080 mask_fill=1080 slice_assign=360 cat=270 pad=299 mask_select=1080 argwhere=60 bool_scatter_or=24 source_checks=2994 values=1273900 bool_storage=explicit_u8_u32 exact_bytes=true flex32=direct empty_bool_scatter=checked bool_masks=all_stores views=offset,step,reversed,transposed,broadcast overlap=true tails=1,127,128,129,257 empty=true reference=byte_records",
    "NVIDIA_BURN_PREDICATE_OK formats=15 output_widths=1,4 raw_cases=382 native_cases=382 rejected=431 oracle=independent guards=ok tails=zero broadcast=verified signed_offsets=verified full_writer=verified resident=true queued_lifetime=verified rank0=verified rank8=verified empty=no_launch invalid_inputs=no_launch",
    "BURN_PREDICATE_CONFORMANCE_OK os=d3os backend=native numeric_dtypes=13 bool_dtypes=2 cases=6898 comparisons=1428 scalars=5028 classifications=282 boolean=160 source_checks=270 half_patterns=131072 values=",
    "NVIDIA_BURN_INTEGER_POWER_OK raw=10 guards=ok reference=i128 full_writer=verified resident=true queued_lifetime=verified scalar=true invalid_inputs=no_launch empty=no_launch",
    "BURN_INTEGER_POWER_OK os=d3os backend=native cases=94 reference=i128 wrapping=exact exponent=u32_bits signed_views=true broadcast=true shared=true empty=true",
    "NVIDIA_BURN_DEFORM_OK raw=96 guards=ok full_writers=verified initialized_atomic_output=true subnormal=preserved nonfinite_offsets=zero resident=true queued_lifetime=verified invalid_geometry=no_launch",
    "BURN_DEFORM_OK os=d3os backend=native forward=64 gradients=256 empty=false tf32_exact_inputs=false selected=None backward=true reference=f64 signed_views=true shared=true",
    "BURN_DEFORM_OK os=d3os backend=native forward=64 gradients=256 empty=false tf32_exact_inputs=true selected=None backward=true reference=f64 signed_views=true shared=true",
    "BURN_DEFORM_OK os=d3os backend=native forward=64 gradients=256 empty=true tf32_exact_inputs=false selected=None backward=true reference=f64 signed_views=true shared=true",
    "BURN_SPATIAL_CONVOLUTION_OK os=d3os backend=native forward=56 gradients=140 empty=true selected=None tf32_exact_inputs=false empty_backward=true reference=f64 signed_views=true shared=true",
    "NVIDIA_BURN_SPATIAL_CONVOLUTION_OK raw=56 guards=ok broadcast=true signed_views=true full_writer=verified resident=true queued_lifetime=verified invalid_geometry=no_launch",
    "BURN_SPATIAL_CONVOLUTION_OK os=d3os backend=native forward=56 gradients=140 empty=false selected=None tf32_exact_inputs=false empty_backward=false reference=f64 signed_views=true shared=true",
    "BURN_SPATIAL_CONVOLUTION_OK os=d3os backend=native forward=56 gradients=140 empty=false selected=None tf32_exact_inputs=true empty_backward=false reference=f64 signed_views=true shared=true",
    "BURN_SPATIAL_CONVOLUTION_OK os=d3os backend=native forward=56 gradients=0 empty=true selected=None tf32_exact_inputs=false empty_backward=false reference=f64 signed_views=true shared=true",
    "NVIDIA_BURN_INTERPOLATE_OK cases=40 modes=5 directions=2 axes=2 guards=ok broadcast_reads=true resident=true full_writer=verified queued_lifetime=verified invalid_inputs=no_launch empty=no_launch",
    "BURN_INTERPOLATE_OK os=d3os backend=native modes=5 align_corners=both forward=100 gradient_modes=5 gradients=80 empty=20 ieee=4 reference=f64 signed_views=true wide_coordinates=true",
    "NVIDIA_BURN_INVERSE_OK atan2_pairs=11280 guards=ok ulp=8 full_writer=verified resident=true queued_lifetime=verified scalar=true invalid_inputs=no_launch empty=no_launch",
    "BURN_INVERSE_OK os=d3os backend=native domain=ieee unary_ops=6 unary_inputs=11084 atan2_pairs=11280 ulp=8 gradients=12 signed_views=true broadcast=true empty=true",
    "BURN_INVERSE_OK os=d3os backend=native domain=finite unary_ops=6 unary_inputs=512 atan2_pairs=548 ulp=8 gradients=12 signed_views=true broadcast=true empty=true",
    "NVIDIA_BURN_MATH_OK unary=28",
    "BURN_REMAINDER_OK os=d3os backend=native domain=finite pairs=512 scalar_cases=8 gradients=8 reference=libm_fmod_floor finite_bits=exact zero_sign=unspecified signed_views=true broadcast=true empty=true",
    "NVIDIA_BURN_REMAINDER_OK pairs=53248 guards=ok finite_bits=exact zero_sign=dividend full_writer=verified resident=true queued_lifetime=verified scalar=true invalid_inputs=no_launch empty=no_launch",
    "BURN_REMAINDER_OK os=d3os backend=native domain=ieee pairs=53248 scalar_cases=15 gradients=8 reference=libm_fmod_floor finite_bits=exact zero_sign=unspecified signed_views=true broadcast=true empty=true",
    "NVIDIA_BURN_CONVERSIONS_OK pairs=225 widths=1,2,4,8 rounding=independent_binary guards=ok tails=zero full_writer=verified resident=true queued_lifetime=verified saturation=rust scalar=true empty=no_launch",
    "BURN_CONVERSIONS_OK os=d3os backend=native pairs=195 flex32=true float_to_int=finite_in_range rounding=nearest_even integer_narrowing=low_bits half_float_codes=exhaustive bf16_midpoints=all_intervals gradients=8 signed_views=true empty=true",
    "NVIDIA_BURN_STORAGE_OK widths=1,2,4,8 guards=ok tails=zero full_writer=verified queued_lifetime=verified resident=true unsupported_arithmetic=no_launch",
    "BURN_STORAGE_FLEX32_DIRECT_OK os=d3os backend=native",
    "BURN_STORAGE_CONFORMANCE_OK os=d3os backend=native dtypes=15 flex32=direct bytes=1,2,4,8 views=signed,broadcast,overlap fill=true bool_storage=explicit_backend reload=true half_bits=exhaustive empty=true scope=storage",
    "NVIDIA_BURN_ND_INDEXING_OK dtypes=f32,i32 modes=5 coordinate_bounds=independent invalid_aliases=recoverable guards=ok resident=true status_reads=1 scalar=true rank8=true ieee_payloads=exact empty=no_launch",
    "BURN_ND_INDEXING_OK os=d3os backend=native dtypes=f32,i32 scatter_modes=5 widths=1,2,3 index_ranks=1,2,3 scatter_gradient_cases=15 gather_gradient_cases=3 scalar=true exact_i32=true duplicates=true empty_slices=true shared_views=true",
    "NVIDIA_BURN_SELECT_UPDATES_OK modes=10 index_view=offset_negative_stride broadcast_indices=true resident=true status_reads=1 invalid_indices=recoverable empty=no_launch",
    "BURN_SELECT_UPDATES_OK os=d3os backend=native dtypes=f32,i32 modes=5 axes=all unique_gradient_indices=true gradient_cases=15 duplicates=add,mul,min,max ranks=3,8 ieee=true empty=true shared_views=true",
    "NVIDIA_BURN_ORDERED_SCATTER_OK modes=8 guards=ok duplicates=ordered rows=129 updates=257 invalid_indices=recoverable resident=true status_reads=1 queued_lifetime=verified empty=no_launch",
    "BURN_SCATTER_UPDATES_OK os=d3os backend=native dtypes=f32,i32 modes=4 axes=all unique_indices=true gradient_cases=12 ieee=true empty=true shared_views=true",
    "BURN_QUANTIZED_EMPTY_RESULT os=d3os backend=native case=direct status=ok bytes=[0, 0, 128, 63]",
    "NVIDIA_BURN_COMPACTION_OK guards=ok malformed_prefix=full_writer queued_lifetime=verified argwhere_reads=1 argwhere_read_bytes=4 mask_select_reads=2 empty_input=no_launch resident=true",
    "BURN_COMPACTION_CONFORMANCE_OK os=d3os backend=native dtypes=f32,i32,bool,q8 argwhere=true nonzero=true mask_select=true gradients=4 ranks=1,3,8 quantized_domain=compact_tails empty=true shared_views=true",
    "NVIDIA_BURN_CROSS_OK resident=ok queued_lifetime=verified empty=no_launch invalid_inputs=rejected",
    "BURN_CROSS_CONFORMANCE_OK os=d3os backend=native dtype=f32 axes=all signed_views=true broadcasting=true gradients=6 empty=true",
    "NVIDIA_BURN_SORT_OK modes=10 topk=nan_first stable_indices=true ieee_bits=exact guards=ok full_writer=verified queued_lifetime=verified allocations=2 resident=ok empty=no_launch",
    "BURN_SORT_CONFORMANCE_OK os=d3os backend=native sorting=gpu topk_order=nan_first float_domains=ieee,finite dtypes=f32,i32 sort=true argsort=true topk=true gradients=6 empty=true shared_views=true",
    "NVIDIA_BURN_SCAN_OK operations=8 chunks=128 carry=verified ieee=exact guards=ok invalid_bounds=rejected empty=no_launch resident=ok",
    "BURN_SCAN_CONFORMANCE_OK os=d3os backend=native operations=8 reference=i128,f64 lengths=1,127,128,129,257,531 gradients=4 empty=true shared_views=true",
    "NVIDIA_BURN_I32_MATMUL_OK reference=i128 wrapping=exact guards=ok bounds=ok signed_batches=ok resident=ok",
    "BURN_INTEGER_MATMUL_CONFORMANCE_OK os=d3os backend=native cases=8 domain=all reference=i128 wrapping=exact shared_views=true empty=true",
    "BURN_INTEGER_CONFORMANCE_OK os=d3os backend=native operations=div,remainder tensor=true scalar=true shared_views=true",
    "NVIDIA_BURN_WORDS_OK operations=32",
    "NVIDIA_BURN_QUANTIZATION_OK api=public integer_bits=8,4,2 formats=9 minifloat=e4m3,e5m2,e2m1 boundaries=exhaustive scale_dtypes=5 levels=1,2",
    "BURN_CAPABILITIES_OK os=d3os backend=native rows=226 flags_only=true",
    "BURN_ERF_CONFORMANCE_OK os=d3os backend=native erf_ulp=8 erf=forward,backward gelu=forward,backward shared_views=verified",
    "INFERENCE_OK self-test backend=nvidia",
)


def encode(number, dtype):
    # An independent encoding path for the reference's exact small integers.
    if dtype in ("F16", "F32", "Flex32", "F64", "BF16"):
        raw = struct.pack(
            {"F16": "<e", "F32": "<f", "Flex32": "<f", "F64": "<d", "BF16": "<f"}[
                dtype
            ],
            number,
        )
        if dtype == "BF16":
            raw = raw[2:]
    else:
        raw = (number % (1 << int(dtype[1:]))).to_bytes(int(dtype[1:]) // 8, "little")
    return "0x" + raw[::-1].hex()


def snapshot(dtype, shape, bits):
    return {"dtype": dtype, "shape": list(shape), "bits": list(bits)}


def synthetic_report(summary, population, backend):
    """Test fixture only, built separately from the production report validator."""
    stages, seeds, events = [], [], []
    for case in population["cases"]:
        row, expected = case["frozen_row"], case["frozen_row"]["expected"]
        identifier, dtype = row["id"], row["dtype"]
        views = {}
        for role, source in row["sources"].items():
            views[role] = snapshot(
                source["dtype"],
                source["shape"],
                [source["storage_bits"][i] for i in source["logical_to_storage"]],
            )
        observed = {}

        def input_operand(role, identifier=identifier, views=views):
            return {
                "key": {"id": identifier, "check": "InputView", "subject": role},
                "actual": copy.deepcopy(views[role]),
                "unavailable": None,
            }

        def stage_operand(name, identifier=identifier, observed=observed):
            return {
                "key": {"id": identifier, "check": "PublicStage", "subject": name},
                "actual": copy.deepcopy(observed[name]),
                "unavailable": None,
            }

        def add(check, name, value, operands, identifier=identifier, observed=observed):
            stages.append(
                {
                    "key": {"id": identifier, "check": check, "subject": name},
                    "expected": copy.deepcopy(value),
                    "actual": copy.deepcopy(value),
                    "read_error": None,
                    "actual_operands": operands,
                }
            )
            observed[name] = value

        if expected["kind"] == "value":
            add(
                "Output",
                "result",
                snapshot(dtype, expected["shape"], expected["bits"]),
                [input_operand(r) for r in ("destination", "values", "indices")],
            )
        elif expected["kind"] == "error":
            add(
                "Recovery",
                "same_context_valid_operation",
                snapshot(dtype, [2], [encode(n, dtype) for n in (2, 1)]),
                [],
            )
            operation = (
                "float_scatter"
                if dtype.startswith("F") or dtype == "BF16"
                else "int_scatter"
            )
            events.append(
                {
                    "id": identifier,
                    "expected": {
                        "recoverable_index_range": "IndexRange",
                        "recoverable_coordinate_bounds": "CoordinateBounds",
                    }[expected["semantic_error"]],
                    "actual_error": f"An error happened during execution\nCaused by:\n  D3OS native Burn {operation}: invalid operation parameter",
                    "output_metadata_valid": True,
                    "accepted": True,
                }
            )
        else:
            m = row["public_metadata"]
            add(
                "PublicStage",
                "forward",
                snapshot(
                    dtype,
                    expected["forward"]["shape"],
                    expected["forward"]["storage_bits"],
                ),
                [input_operand(r) for r in ("operand", "indices")],
            )
            product = [
                encode(a * b, dtype)
                for a, b in zip(
                    m["output"]["storage_values_flat"],
                    m["upstream_coefficient"]["storage_values_flat"],
                    strict=True,
                )
            ]
            total = sum(
                a * b
                for a, b in zip(
                    m["output"]["storage_values_flat"],
                    m["upstream_coefficient"]["storage_values_flat"],
                    strict=True,
                )
            )
            branch_sums = []
            for branch in range(1, m["graph"]["branches"] + 1):
                product_name, sum_name = (
                    f"branch_{branch}_product",
                    f"branch_{branch}_sum",
                )
                add(
                    "PublicStage",
                    product_name,
                    snapshot(dtype, m["output"]["shape"], product),
                    [stage_operand("forward"), input_operand("upstream_coefficient")],
                )
                add(
                    "PublicStage",
                    sum_name,
                    snapshot(dtype, [1], [encode(total, dtype)]),
                    [stage_operand(product_name)],
                )
                branch_sums.append(sum_name)
            add(
                "PublicStage",
                "root",
                snapshot(dtype, expected["root"]["shape"], expected["root"]["bits"]),
                [stage_operand(n) for n in branch_sums],
            )
            gradient = expected["dense_original_leaf_gradient"]
            add(
                "PublicStage",
                "original_leaf_gradient",
                snapshot(dtype, gradient["shape"], gradient["storage_bits"]),
                [
                    input_operand(r)
                    for r in ("operand", "indices", "upstream_coefficient")
                ]
                + [stage_operand("root")],
            )
            seeds.append(
                {
                    "id": identifier,
                    "status": "expected_only_source_backed",
                    "expected": snapshot(
                        dtype, expected["seed"]["shape"], expected["seed"]["bits"]
                    ),
                    "actual": None,
                }
            )
    model_results = [
        {"key": m["key"], "passed": True, "evidence": "MODEL"}
        for m in population["models"]
    ]
    model_results.sort(key=lambda m: (m["key"]["template"], m["key"]["content_sha256"]))
    return {
        "corpus_version": 1,
        "scope": summary["scope"],
        "dtype_filter": population["dtype_filter"],
        "backend": backend,
        "completion": "Finished",
        "finalized": True,
        "ok": True,
        "FROZEN": False,
        "source_binding_approved": False,
        "numerical_freeze_sha256": population["numerical_freeze_sha256"],
        "fixtures_sha256": population["fixtures_sha256"],
        "device_attempted": summary["cases"],
        "device_passed": summary["cases"],
        "device_failed": 0,
        "model_attempted": summary["attachments"],
        "model_passed": summary["attachments"],
        "successful_observations": copy.deepcopy(population["required_observations"]),
        "failed_observations": [],
        "missing_observations": [],
        "failure_count": 0,
        "omitted_failure_details": 0,
        "failures": [],
        "stage_observations": stages,
        "seeds": seeds,
        "model_results": model_results,
        "synchronized": summary["scope"] != "ModelReference",
        "device_poisoned": False,
        "source_observation": "retained_clone_lineage_logical_reads_after_whole_graph",
        "physical_tails_ownership_residency_proven": False,
        "full_parity": False,
        "structured_policies": copy.deepcopy(population["structured_policies"]),
        "semantic_events": events,
        "selected_rows": [
            {
                "id": c["frozen_row"]["id"],
                "row_sha256": c["row_sha256"],
                "policy": c["execution_policy"],
            }
            for c in population["cases"]
        ],
        "attempted_ids": sorted(c["frozen_row"]["id"] for c in population["cases"]),
    }


def selection(scope="LinuxCommon", dtype="I8"):
    summary, population = evidence.reference_selection(scope, dtype)
    backend = (
        "host_model_no_device"
        if scope == "ModelReference"
        else evidence.LINUX_BACKEND
        if scope == "LinuxCommon"
        else evidence.NATIVE_BACKEND
    )
    return summary, population, synthetic_report(summary, population, backend)


def check(report, summary, population):
    return evidence.validate_report(
        report,
        summary,
        population,
        execution_os="linux"
        if population.get("scope") in ("LinuxCommon", "ModelReference")
        else "d3os",
        backend=report.get("backend"),
    )


def serial_for(report, summary):
    # Test-only wire encoding; the footer is independently spelled here.
    prefix = (
        "BURN_INDEXED_ACCUMULATION_REPORT"
        if summary["scope"] in ("LinuxCommon", "ModelReference")
        else "NVIDIA_BURN_INDEXED_ACCUMULATION_SHARED"
    )
    if summary["scope"] == "ModelReference":
        footer = f"BURN_INDEXED_ACCUMULATION_MODEL_OK evidence=MODEL templates={summary['templates']} attachments={summary['attachments']} device_attempted=0 device_passed=0"
    else:
        footer = f"BURN_INDEXED_ACCUMULATION_OK scope={summary['scope']} attempted={summary['cases']} passed={summary['cases']} source_roles={summary['source_roles']} source_storages={summary['source_storages']} observations={summary['observation_keys']} logical_only=true full_parity=false"
    return (
        prefix + " " + json.dumps(report, separators=(",", ":")) + "\n" + footer + "\n"
    )


class EvidenceTests(unittest.TestCase):
    def test_strict_json(self):
        for text in (
            '{"a":1,"a":2}',
            '{"a":{"b":0,"b":1}}',
            "NaN",
            "Infinity",
            "-Infinity",
            "1e999",
            "{} false",
            "[]\x00",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evidence.strict_json(text)
        self.assertEqual(
            evidence.strict_json('{"a":[0,false,null]}'), {"a": [0, False, None]}
        )

    def test_synthetic_positive_all_scopes_and_dtypes(self):
        # These are parser fixtures, never observations of an actual backend.
        for scope in ("LinuxCommon", "NativeValues", "NativeErrors", "ModelReference"):
            summary, population, report = selection(scope, None)
            receipt = check(report, summary, population)
            self.assertFalse(
                receipt["all_physical_source_words_independently_revalidated"]
            )
            self.assertFalse(receipt["full_parity"])
            serial = serial_for(report, summary)
            evidence.validate(
                serial,
                summary,
                population,
                execution_os="d3os" if scope.startswith("Native") else "linux",
                backend=report["backend"],
            )
        for dtype in evidence.DTYPES:
            summary, population, report = selection("NativeValues", dtype)
            check(report, summary, population)
            for number in (-7, 0, 1, 2, 48):
                self.assertEqual(
                    evidence._integer_bits(number, dtype), encode(number, dtype)
                )

    def test_actual_host_only_exports_when_available(self):
        model = CACHE / "actual-model.stdout"
        if not model.is_file():
            self.skipTest("retained device-free CLI exports are optional")
        summary, population = evidence.reference_selection("ModelReference")
        receipt = evidence.validate(
            model.read_text(),
            summary,
            population,
            execution_os="linux",
            backend="host_model_no_device",
        )
        self.assertEqual(
            (receipt["model_passed"], receipt["device_attempted"]), (283, 0)
        )
        inventory = CACHE / "inventory"
        for scope, label in (
            ("LinuxCommon", "linux-common"),
            ("NativeValues", "native-values"),
            ("NativeErrors", "native-errors"),
            ("ModelReference", "model-reference"),
        ):
            if inventory.is_dir():
                evidence.validate_selection(
                    evidence.strict_json(
                        (inventory / f"{label}-summary.json").read_text()
                    ),
                    evidence.strict_json(
                        (inventory / f"{label}-population.json").read_text()
                    ),
                )

    def test_closed_report_and_inventory_fields(self):
        summary, population, report = selection()
        for field in report:
            bad = dict(report)
            del bad[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(bad, summary, population)
        for field in summary:
            bad = dict(summary)
            del bad[field]
            with self.subTest(summary=field), self.assertRaises(ValueError):
                check(report, bad, population)
        for field in population:
            bad = dict(population)
            del bad[field]
            with self.subTest(population=field), self.assertRaises(ValueError):
                check(report, summary, bad)
        for target in (report, summary, population):
            bad = dict(target, extra=False)
            with self.assertRaises(ValueError):
                check(
                    bad if target is report else report,
                    bad if target is summary else summary,
                    bad if target is population else population,
                )

    def test_native_backend_exact_identity_and_primary_sources(self):
        summary, population, report = selection("NativeErrors", None)
        self.assertEqual(report["backend"], "dispatch<d3os-native-nvidia>")
        check(report, summary, population)
        for alias in (
            "d3os-native-nvidia",
            "prefix_dispatch<d3os-native-nvidia>",
            "dispatch<d3os-native-nvidia>_suffix",
            "dispatch<dispatch<d3os-native-nvidia>>",
            "dispatch<fusion<d3os-native-nvidia>>",
            "dispatch<fusion<cubecl<cuda>>>",
            "dispatch<d3os-native-nvidia> ",
        ):
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    check(dict(report, backend=alias), summary, population)
                with self.assertRaises(ValueError):
                    evidence.validate_report(
                        report, summary, population, execution_os="d3os", backend=alias
                    )
        sources = evidence.validate_backend_sources()
        for source in (
            "third_party/burn-dispatch/src/backend.rs",
            "os/library/burn-native/src/backend/mod.rs",
        ):
            self.assertIn(source, sources)
            pins = {**evidence.BACKEND_SOURCE_PINS, source: "0" * 64}
            with (
                mock.patch.object(evidence, "BACKEND_SOURCE_PINS", pins),
                self.assertRaises(ValueError),
            ):
                check(report, summary, population)
        linux_summary, linux_population, linux_report = selection("LinuxCommon", "I8")
        self.assertEqual(linux_report["backend"], "dispatch<fusion<cubecl<cuda>>>")
        check(linux_report, linux_summary, linux_population)
        with self.assertRaises(ValueError):
            evidence.validate_report(
                report,
                summary,
                population,
                execution_os="linux",
                backend=evidence.NATIVE_BACKEND,
            )

    def test_revoked_raw_receipt_remains_closed(self):
        receipt = CACHE / "raw-source-pins.json"
        if not receipt.is_file():
            self.skipTest("retained prior Raw receipt is optional")
        pins = evidence.strict_json(receipt.read_text())
        current = hashlib.sha256(
            (evidence.ROOT / evidence.RAW_SOURCE).read_bytes()
        ).hexdigest()
        if pins["sources"][evidence.RAW_SOURCE] == current:
            self.skipTest("Raw source has a current renewed binding")
        with self.assertRaises(ValueError):
            evidence.validate_raw("", {}, source_pins=pins)

    def test_status_counts_ids_and_provenance(self):
        summary, population, report = selection()
        changes = {
            "device_attempted": (None, False, 0, 13.0),
            "device_passed": (True, 0, 13.0),
            "device_failed": (False, 0.0, 1),
            "model_attempted": (1, False),
            "model_passed": (1,),
            "failure_count": (False, 1),
            "omitted_failure_details": (1,),
            "completion": ("Running", "Rejected", "Aborted"),
            "ok": (False,),
            "finalized": (False,),
            "synchronized": (False,),
            "device_poisoned": (True,),
            "FROZEN": (True,),
            "source_binding_approved": (True,),
            "full_parity": (True,),
            "physical_tails_ownership_residency_proven": (True,),
            "backend": ("cuda", "dispatch<cubecl<cuda>>", "host_model_no_device"),
            "dtype_filter": ("I16",),
            "fixtures_sha256": ("0" * 64,),
            "numerical_freeze_sha256": ("0" * 64,),
            "source_observation": ("all_physical_bytes",),
        }
        for field, values in changes.items():
            for value in values:
                bad = dict(report, **{field: value})
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    check(bad, summary, population)
        for field in ("attempted_ids", "selected_rows", "successful_observations"):
            for values in (
                report[field][:-1],
                report[field] + report[field][:1],
                list(reversed(report[field])),
            ):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    check(dict(report, **{field: values}), summary, population)
        for field in (
            "failures",
            "failed_observations",
            "missing_observations",
            "semantic_events",
            "model_results",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(dict(report, **{field: [{}]}), summary, population)
        for field in ("frozen_row", "row_sha256", "execution_policy"):
            bad = copy.deepcopy(population)
            bad["cases"][0][field] = {} if field == "frozen_row" else "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(report, summary, bad)
        bad = copy.deepcopy(population)
        bad["structured_policies"]["FROZEN"] = True
        with self.assertRaises(ValueError):
            check(report, summary, bad)
        with (
            mock.patch.object(
                evidence,
                "SOURCE_PINS",
                {
                    **evidence.SOURCE_PINS,
                    "tools/indexed-accumulation-reference/provenance.json": "0" * 64,
                },
            ),
            self.assertRaises(ValueError),
        ):
            evidence.reference_selection("LinuxCommon")

    def test_emitted_stages_and_operands(self):
        for scope, dtype in (
            ("LinuxCommon", "F16"),
            ("NativeErrors", None),
            ("NativeValues", "U64"),
        ):
            summary, population, report = selection(scope, dtype)
            for index, stage in enumerate(report["stage_observations"]):
                for field in stage:
                    bad = copy.deepcopy(report)
                    del bad["stage_observations"][index][field]
                    with (
                        self.subTest(scope=scope, stage=index, missing=field),
                        self.assertRaises(ValueError),
                    ):
                        check(bad, summary, population)
                bad = copy.deepcopy(report)
                del bad["stage_observations"][index]
                with self.assertRaises(ValueError):
                    check(bad, summary, population)
            stage_index = next(
                (
                    i
                    for i, s in enumerate(report["stage_observations"])
                    if s["actual_operands"]
                ),
                0,
            )
            for field, value in (
                ("shape", [True]),
                ("shape", [1.0]),
                ("shape", [0]),
                ("dtype", "F64"),
                ("bits", [1]),
                ("bits", ["0x1"]),
                ("bits", ["0X00"]),
                ("bits", ["0x" + "0" * 32]),
            ):
                bad = copy.deepcopy(report)
                target = bad["stage_observations"][stage_index]
                target["actual"][field] = value
                target["expected"][field] = copy.deepcopy(value)
                with (
                    self.subTest(scope=scope, field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    check(bad, summary, population)
            for field, value in (
                ("actual", None),
                ("read_error", "failed"),
                ("extra", False),
            ):
                bad = copy.deepcopy(report)
                bad["stage_observations"][0][field] = value
                with self.assertRaises(ValueError):
                    check(bad, summary, population)
            if report["stage_observations"][stage_index]["actual_operands"]:
                for mutate in (
                    lambda x: x.clear(),
                    lambda x: x.append(copy.deepcopy(x[0])),
                    lambda x: x.reverse(),
                    lambda x: x[0].update(actual=None),
                    lambda x: x[0].update(unavailable="missing"),
                    lambda x: x[0]["key"].update(subject="wrong"),
                    lambda x: x[0]["actual"]["bits"].__setitem__(0, "0x0"),
                ):
                    bad = copy.deepcopy(report)
                    mutate(bad["stage_observations"][stage_index]["actual_operands"])
                    with self.assertRaises(ValueError):
                        check(bad, summary, population)

    def test_nan_source_wide_indices_seed_and_recovery(self):
        summary, population, report = selection("NativeValues", None)
        for stage in report["stage_observations"]:
            for operand in stage["actual_operands"]:
                if (
                    operand["key"]["subject"] == "indices"
                    and operand["actual"]["dtype"] == "U64"
                ):
                    bad = copy.deepcopy(report)
                    target = bad["stage_observations"][
                        report["stage_observations"].index(stage)
                    ]["actual_operands"][stage["actual_operands"].index(operand)]
                    target["actual"]["bits"] = [
                        f"0x{int(b, 16) & 0xFFFFFFFF:08x}"
                        for b in target["actual"]["bits"]
                    ]
                    with self.assertRaises(ValueError):
                        check(bad, summary, population)
                    break
            else:
                continue
            break
        for seed in (None, {"dtype": "F16", "shape": [1], "bits": ["0x3c00"]}):
            bad = copy.deepcopy(report)
            bad["seeds"][0]["actual"] = seed
            bad["seeds"][0]["status"] = "observed"
            with self.assertRaises(ValueError):
                check(bad, summary, population)
        summary, population, report = selection("NativeErrors", None)
        for index, event in enumerate(report["semantic_events"]):
            for field, value in (
                ("actual_error", None),
                ("actual_error", "invalid operation parameter"),
                (
                    "expected",
                    "CoordinateBounds"
                    if event["expected"] == "IndexRange"
                    else "IndexRange",
                ),
                ("accepted", False),
                ("output_metadata_valid", False),
                ("extra", False),
            ):
                bad = copy.deepcopy(report)
                bad["semantic_events"][index][field] = value
                with self.assertRaises(ValueError):
                    check(bad, summary, population)
            bad = copy.deepcopy(report)
            bad["stage_observations"][index]["actual"]["bits"].reverse()
            with self.assertRaises(ValueError):
                check(bad, summary, population)
        # Source NaN payloads are never normalized to a result-class oracle.
        source = {
            "dtype": "F16",
            "shape": [2],
            "strides_elements": [-1],
            "offset_elements": 1,
            "logical_to_storage": [1, 0],
            "storage_bits": ["0x7c01", "0xfe23"],
        }
        self.assertEqual(
            evidence._source_snapshot(source)["bits"], ["0xfe23", "0x7c01"]
        )
        source["logical_to_storage"] = [0, 1]
        with self.assertRaises(ValueError):
            evidence._source_snapshot(source)

    def test_model_exact_membership_and_zero_device(self):
        summary, population, report = selection("ModelReference", None)
        for field, value in (
            ("device_attempted", 1),
            ("device_passed", 1),
            ("device_failed", None),
            ("synchronized", True),
            ("model_passed", 282),
            ("successful_observations", [{}]),
            ("stage_observations", [{}]),
        ):
            with self.assertRaises(ValueError):
                check(dict(report, **{field: value}), summary, population)
        for index in range(len(report["model_results"])):
            bad = dict(
                report,
                model_results=report["model_results"][:index]
                + report["model_results"][index + 1 :],
            )
            with self.assertRaises(ValueError):
                check(bad, summary, population)
        for mutate in (
            lambda r: r[0].update(passed=1),
            lambda r: r[0].update(passed=False),
            lambda r: r[0]["key"].update(content_sha256="0" * 64),
            lambda r: r.append(r[0]),
            lambda r: r.reverse(),
        ):
            bad = copy.deepcopy(report)
            mutate(bad["model_results"])
            with self.assertRaises(ValueError):
                check(bad, summary, population)

    def test_marker_os_order_and_contamination(self):
        summary, population, report = selection()
        good = serial_for(report, summary)
        kwargs = {"execution_os": "linux", "backend": evidence.LINUX_BACKEND}
        for bad in (
            "NOT_" + good,
            "junk " + good,
            good.replace("REPORT ", "REPORTX "),
            good + good,
            "\n".join(reversed(good.splitlines())),
            good.replace("logical_only=true", "logical_only=true junk"),
            good.replace("REPORT {", 'REPORT {"corpus_version":1,'),
            good.replace("REPORT", "SHARED"),
            good + "NOT_BURN_INDEXED_ACCUMULATION_OK false\n",
            good + "thread panicked\n",
            good.replace("}\nBURN_", "} garbage\nBURN_"),
        ):
            with self.subTest(bad=bad[:60]), self.assertRaises(ValueError):
                evidence.validate(bad, summary, population, **kwargs)
        evidence.validate(good.replace("\n", "\r\n"), summary, population, **kwargs)
        summary, population, report = selection("NativeErrors", None)
        bare = serial_for(report, summary)
        for prefix in (
            "[1.234][INF][sys_logger.rs@020] [infer.rs@050] ",
            "\x1b[36m[1.234]\x1b[94m[INF]\x1b[35m[sys_logger.rs@020]\x1b[39m [infer.rs@050] ",
        ):
            logged = "\n".join(prefix + line for line in bare.splitlines()) + "\n"
            evidence.validate(
                logged,
                summary,
                population,
                execution_os="d3os",
                backend=evidence.NATIVE_BACKEND,
            )
            with self.assertRaises(ValueError):
                evidence.validate(
                    logged,
                    summary,
                    population,
                    execution_os="linux",
                    backend=evidence.NATIVE_BACKEND,
                )
        for prefix in ("[fake] ", "[1.234][INF][wrong.rs@020] [infer.rs@050] ", "NOT_"):
            with self.assertRaises(ValueError):
                evidence.validate(
                    prefix + bare,
                    summary,
                    population,
                    execution_os="d3os",
                    backend=evidence.NATIVE_BACKEND,
                )

    def test_raw_fail_closed_without_final_pins(self):
        for pins in (
            {},
            {"approved": False},
            {
                "approved": True,
                "final": False,
                "sources": {},
                "raw_plan": {},
                "recipes": {},
            },
        ):
            with self.assertRaises(ValueError):
                evidence.validate_raw("", {}, source_pins=pins)


class RawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("D3OS_INDEXED_RAW_SOURCE_PINS")
        pins_path = (
            Path(configured)
            if configured is not None
            else CACHE / "raw-source-pins.json"
        )
        if not pins_path.is_file():
            if configured is not None:
                raise ValueError(
                    "explicit parent-approved Raw source pins file is missing"
                )
            raise unittest.SkipTest("parent-approved final Raw inputs are optional")
        cls.pins = evidence.strict_json(pins_path.read_text())
        try:
            evidence._raw_sources(cls.pins)
        except ValueError as error:
            if configured is not None:
                raise
            raise unittest.SkipTest(
                f"Raw final source binding is unavailable: {error}"
            ) from error
        cls.plan = evidence.strict_json(Path(cls.pins["raw_plan"]["path"]).read_text())
        cls.recipes = evidence.strict_json(
            Path(cls.pins["recipes"]["path"]).read_text()
        )
        cls.lines = [
            "NVIDIA_BURN_INDEXED_ACCUMULATION_PLAN protocol=55 profiles=89 services=14 population="
            + json.dumps(
                {
                    "FROZEN": True,
                    "raw_recipes": 138,
                    "valid_GPU_recipes": 98,
                    "invalid_status_GPU_recipes": 8,
                    "host_metadata_rejects": 32,
                    "raw_launches": 228,
                    "C3_vectors": 16,
                    "C3_unique_keys": 132,
                    "MODEL_device_passes": 0,
                    "main_rows_added": 0,
                    "device_observation": "NOT_EXECUTED",
                },
                separators=(",", ":"),
            )
        ]
        # Only the parser fixture uses the reviewed recipes. These are synthetic
        # success assertions, never a run of Raw or an observation of hardware.
        for recipe in cls.recipes:
            for trial in (0, 1):
                for key in recipe["keys"]:
                    cls.lines.append(
                        f"NVIDIA_BURN_INDEXED_ACCUMULATION_RAW_C3 evidence=RAW_HARDWARE template={key['template']} contentsha={key['sha256']} vector={recipe['name']} field={key['lane']} initialized_base_trial={trial} matched=true MODEL_device_passes=0"
                    )
        cls.lines.append(
            "NVIDIA_BURN_INDEXED_ACCUMULATION_RAW_PASS raw_recipes=138 raw_launches=228 counts=Counts { valid_trials: 196, invalid_trials: 16, recoveries: 16, c3_original: 132, c3_changed_base: 132, native_physical: 99, native_errors: 21, empty_aliases: 33, queued: 11, public_lifetimes: 5, native_metadata_rejects: 26 } main_rows_added=0 MODEL_device_passes=0 synchronous_status_wait=true"
        )
        shared, footers = [], []
        for scope in ("NativeValues", "NativeErrors"):
            summary, _, report = selection(scope, None)
            wire = serial_for(report, summary).splitlines()
            shared.append(wire[0])
            footers.append(wire[1])
        cls.lines.extend(shared + footers)
        cls.lines.append(
            "NVIDIA_BURN_INDEXED_ACCUMULATION_STRICT_OK protocol=55 profiles=89 raw_launches=228 shared_values=184 shared_errors=8 MODEL_device_passes=0 os=d3os backend=native qualification=project_contract full_parity=false"
        )
        cls.serial = "\n".join(cls.lines) + "\n"
        # Existing source-owned success lines, not runtime captures or GPU proof.
        source = evidence.ROOT / "os/application/infer/src/gpu_tests/burn.rs"
        source_text = source.read_text()
        inherited = re.findall(
            r'"(NVIDIA_BURN_(?:STORAGE|ELEMENTWISE|MATRIX|REDUCTION)_OK [^"\n]+)"',
            source_text,
        )
        if len(inherited) != 4:
            raise ValueError("inherited Native success source inventory differs")
        cls.inherited_lines = [*INHERITED54_MARKERS, *inherited]
        if len(cls.inherited_lines) != 57 or len(set(cls.inherited_lines)) != 57:
            raise ValueError("inherited source-derived success inventory differs")
        separate_source = (
            evidence.ROOT / "os/application/infer/src/gpu_tests/burn/mixed_matmul.rs"
        )
        if (
            "NVIDIA_BURN_MIXED_MATMUL_SEPARATE scope={scope:?} qualification=false {}"
            not in separate_source.read_text()
        ):
            raise ValueError("inherited separate diagnostic source format differs")
        cls.inherited_lines.append(
            "NVIDIA_BURN_MIXED_MATMUL_SEPARATE scope=F64Diagnostics qualification=false "
            + json.dumps(
                {
                    "diagnostic": "synthetic caught operation panicked",
                    "full_public_qualification": False,
                }
            )
        )

    @classmethod
    def combined_capture(cls, indexed):
        # Every physical indexed line is kept intact and in order. No caller
        # filtering: the complete constructed capture goes directly to the gate.
        inherited = iter(cls.inherited_lines)
        pieces = []
        for index, line in enumerate(indexed.splitlines(keepends=True)):
            if index % 5 == 0:
                outside = next(inherited, None)
                if outside is not None:
                    pieces.append(outside + "\n")
            pieces.append(line)
            if not line.endswith("\n"):
                pieces.append("\n")
        pieces.extend(line + "\n" for line in inherited)
        return "".join(pieces)

    def rejects(self, serial=None, *, plan=None, pins=None, **options):
        indexed = self.serial if serial is None else serial
        for capture in (indexed, self.combined_capture(indexed)):
            with self.assertRaises(ValueError):
                evidence.validate_raw(
                    capture,
                    self.plan if plan is None else plan,
                    source_pins=self.pins if pins is None else pins,
                    **options,
                )

    def test_synthetic_complete_native_contract(self):
        result = evidence.validate_raw(self.serial, self.plan, source_pins=self.pins)
        self.assertEqual([s["device_passed"] for s in result["shared"]], [184, 8])
        self.assertTrue(result["shared_reports_independently_validated"])
        self.assertTrue(result["requires_parent_inherited_suite_validation"])
        self.assertFalse(result["unrelated_lines_validated"])
        self.assertFalse(result["host_per_word_revalidation"])
        self.assertFalse(result["physical_tails_ownership_residency_proven"])
        self.assertEqual(result["native_control_read_bytes_allowed"], 4)
        prefix = "[1.234][INF][sys_logger.rs@020] [infer.rs@050] "
        logged = "\n".join(prefix + line for line in self.lines) + "\n"
        evidence.validate_raw(logged, self.plan, source_pins=self.pins)

    def test_whole_capture_keeps_inherited_scope_parent_owned(self):
        combined = self.combined_capture(self.serial)
        self.assertIn("panicked", combined)
        self.assertIn("qualification=false", combined)
        self.assertEqual(len(self.inherited_lines), 58)
        for inherited in self.inherited_lines:
            self.assertEqual(combined.splitlines().count(inherited), 1)
        with mock.patch.object(
            evidence, "_records", wraps=evidence._records
        ) as records:
            result = evidence.validate_raw(combined, self.plan, source_pins=self.pins)
        records.assert_called_once_with(combined, "d3os", allow_inherited=True)
        self.assertEqual([s["device_passed"] for s in result["shared"]], [184, 8])
        self.assertTrue(result["requires_parent_inherited_suite_validation"])
        for flag in (
            "unrelated_lines_validated",
            "host_per_word_revalidation",
            "physical_tails_ownership_residency_proven",
            "full_parity",
            "vendor_parity",
            "performance",
        ):
            self.assertFalse(result[flag])
        for extra in (
            "NVIDIA_INFO protocol=54 kernels=88 services=14\n",
            'UNRELATED_DIAGNOSTIC {"message":"panicked", "qualification":false}\n',
        ):
            scoped = evidence.validate_raw(
                extra + combined, self.plan, source_pins=self.pins
            )
            self.assertFalse(scoped["unrelated_lines_validated"])
            self.assertTrue(scoped["requires_parent_inherited_suite_validation"])
        for os_name, scope in (("linux", "LinuxCommon"), ("d3os", "NativeErrors")):
            summary, population, report = selection(scope, None)
            single = serial_for(report, summary)
            for inherited in self.inherited_lines:
                capture = inherited + "\n" + single
                if "BURN_" in inherited or "panicked" in inherited:
                    with self.assertRaises(ValueError):
                        evidence.validate(
                            capture,
                            summary,
                            population,
                            execution_os=os_name,
                            backend=report["backend"],
                        )
                else:
                    # Preserve the existing allowance for plain inherited log
                    # lines such as INFERENCE_OK in single-program validation.
                    evidence.validate(
                        capture,
                        summary,
                        population,
                        execution_os=os_name,
                        backend=report["backend"],
                    )

    def test_combined_capture_rejects_indexed_namespace_aliases(self):
        for index in range(len(self.lines)):
            lines = list(self.lines)
            lines[index] = lines[index].replace(
                "INDEXED_ACCUMULATION", "indexed_accumulation", 1
            )
            self.rejects("\n".join(lines))
        for token in (
            "indexedaccumulation",
            "Indexed-Accumulation",
            "INDEXED ACCUMULATION",
            "INDEXED\tACCUMULATION",
            "INDEXED__ACCUMULATION",
            "INDEXED.ACCUMULATION",
        ):
            self.rejects(
                self.serial.replace("INDEXED_ACCUMULATION_PLAN", token + "_PLAN", 1)
            )
            self.rejects(
                self.serial + "NVIDIA_BURN_" + token + "_UNKNOWN synthetic=true\n"
            )
        self.rejects(self.serial + self.lines[0].lower() + "\n")
        for contamination in ("NOT_", "prefix_", " ", "[fake] "):
            self.rejects(contamination + self.serial)
        self.rejects(
            self.serial.replace("ACCUMULATION_PLAN ", "ACCUMULATION_PLAN_SUFFIX ", 1)
        )
        self.rejects(
            self.serial.replace("ACCUMULATION_PLAN ", "ACCUMULATION_PLAN\t", 1)
        )
        for diagnostic in (
            "INDEXED_ACCUMULATION_FAILED reason=synthetic",
            "indexed accumulation: operation panicked",
            "indexed-accumulation failed",
            'NVIDIA_BURN_OTHER_DIAGNOSTIC {"message":"indexed accumulation failed"}',
        ):
            self.rejects(self.serial + diagnostic + "\n")

    def test_every_missing_or_duplicate_c3_trial_and_order(self):
        for index in range(1, 265):
            self.rejects("\n".join(self.lines[:index] + self.lines[index + 1 :]))
        lines = list(self.lines)
        lines[2] = lines[1]
        self.rejects("\n".join(lines))
        lines = list(self.lines)
        lines[1:3] = reversed(lines[1:3])
        self.rejects("\n".join(lines))
        for fragment, replacement in (
            ("template=B19", "template=B18"),
            ("contentsha=", "contentsha=0"),
            ("field=0 ", "field=1 "),
            ("initialized_base_trial=0", "initialized_base_trial=1"),
            ("matched=true", "matched=false"),
            ("MODEL_device_passes=0", "MODEL_device_passes=1"),
            ("evidence=RAW_HARDWARE", "evidence=MODEL"),
            ("vector=C3_", "vector=UNKNOWN_"),
        ):
            lines = list(self.lines)
            lines[1] = lines[1].replace(fragment, replacement)
            self.rejects("\n".join(lines))

    def test_raw_plan_counts_pins_and_os(self):
        for field in self.plan:
            plan = dict(self.plan)
            del plan[field]
            self.rejects(plan=plan)
        for field, value in (
            ("approved", False),
            ("approved", 1),
            ("final", False),
            ("extra", 0),
        ):
            self.rejects(pins=dict(self.pins, **{field: value}))
        for section in ("sources", "raw_plan", "recipes"):
            pins = copy.deepcopy(self.pins)
            if section == "sources":
                pins[section][evidence.RAW_SOURCE] = "0" * 64
            else:
                pins[section]["sha256"] = "0" * 64
            self.rejects(pins=pins)
        for field in evidence.RAW_COUNTS:
            lines = list(self.lines)
            wanted = evidence.RAW_COUNTS[field]
            lines[265] = lines[265].replace(
                f"{field}: {wanted}", f"{field}: {wanted + 1}"
            )
            self.rejects("\n".join(lines))
        for fragment, replacement in (
            ("protocol=55", "protocol=54"),
            ("profiles=89", "profiles=90"),
            ("services=14", "services=15"),
            ('"FROZEN":true', '"FROZEN":false'),
            ('"raw_launches":228', '"raw_launches":228.0'),
            ('"MODEL_device_passes":0', '"MODEL_device_passes":1'),
            ('"device_observation":"NOT_EXECUTED"', '"device_observation":"PASSED"'),
        ):
            lines = list(self.lines)
            lines[0] = lines[0].replace(fragment, replacement)
            self.rejects("\n".join(lines))
        for index in (0, 265, 266, 267, 268, 269, 270):
            self.rejects("\n".join(self.lines[:index] + self.lines[index + 1 :]))
        self.rejects("NOT_" + self.serial)
        self.rejects(self.serial + self.lines[265])
        self.rejects(protocol=54)
        self.rejects(profiles=88)
        self.rejects(protocol=True)
        self.rejects(self.serial.replace("os=d3os", "os=linux"))
        self.rejects(self.serial.replace("full_parity=false", "full_parity=true"))
        self.rejects(
            self.serial.replace(
                "synchronous_status_wait=true", "synchronous_status_wait=false"
            )
        )

    def test_native_shared_data_is_independently_checked(self):
        for index in (266, 267):
            report = json.loads(self.lines[index].partition(" ")[2])
            report["stage_observations"][0]["actual"]["bits"][0] = "0x00"
            lines = list(self.lines)
            lines[index] = "NVIDIA_BURN_INDEXED_ACCUMULATION_SHARED " + json.dumps(
                report
            )
            self.rejects("\n".join(lines))
        lines = list(self.lines)
        lines[266], lines[267] = lines[267], lines[266]
        self.rejects("\n".join(lines))


class MissingLinux(unittest.TestCase):
    def test_every_missing_linux_observation(self):
        self.missing("LinuxCommon")

    def missing(self, scope):
        summary, population, report = selection(scope, None)
        for index in range(len(report["successful_observations"])):
            bad = dict(
                report,
                successful_observations=report["successful_observations"][:index]
                + report["successful_observations"][index + 1 :],
            )
            with (
                self.subTest(scope=scope, missing=index),
                self.assertRaises(ValueError),
            ):
                check(bad, summary, population)


class MissingNative(MissingLinux):
    def test_every_missing_linux_observation(self):
        self.missing("NativeErrors")

    def test_every_missing_native_observation(self):
        self.missing("NativeValues")


class LauncherTests(unittest.TestCase):
    def test_linux_rejects_other_scopes(self):
        for scope in ("NativeErrors", "NativeValues", "ModelReference"):
            summary, population, report = selection(scope, None)
            with self.assertRaises(ValueError):
                launcher.validate_indexed_accumulation(
                    serial_for(report, summary), summary, population
                )

    def test_inventory_bytes_archived_before_staging(self):
        summary, population = evidence.reference_selection("LinuxCommon", "I8")
        exports = [
            json.dumps(x, indent=1).encode() + b"\n" for x in (summary, population)
        ]
        directory = mock.Mock()
        destinations = {}
        directory.__truediv__ = mock.Mock(
            side_effect=lambda name: destinations.setdefault(name, mock.Mock())
        )
        with (
            mock.patch.object(
                launcher, "indexed_accumulation_evidence", return_value=evidence
            ),
            mock.patch.object(evidence, "validate_sources", return_value={}),
            mock.patch.object(
                launcher.subprocess, "check_output", side_effect=exports
            ) as command,
        ):
            launcher.indexed_accumulation_inventory(
                directory,
                Path("/binary with spaces"),
                ["--scope=linux-common", "--dtype=i8"],
            )
        self.assertEqual(
            command.call_args_list[0].args[0],
            [
                "/binary with spaces",
                "--expected-summary",
                "--scope=linux-common",
                "--dtype=i8",
            ],
        )
        self.assertEqual(command.call_args_list[1].args[0][1], "--population")
        for suffix, exported in zip(("expected", "population"), exports, strict=True):
            destinations[
                f"burn-indexed_accumulation-{suffix}.json"
            ].write_bytes.assert_called_once_with(exported)
        for arguments in (
            [],
            ["--models"],
            ["--scope=native-errors"],
            ["--scope=linux-common", "--models"],
            ["--scope=linux-common", "--dtype=bad"],
        ):
            with (
                mock.patch.object(evidence, "validate_sources", return_value={}),
                self.assertRaises(ValueError),
            ):
                launcher.indexed_accumulation_inventory(
                    directory, Path("/unused"), arguments
                )

    def test_failed_inventory_is_retained_and_never_staged(self):
        import subprocess

        directory = mock.Mock()
        destinations = {}
        directory.__truediv__ = mock.Mock(
            side_effect=lambda name: destinations.setdefault(name, mock.Mock())
        )
        failure = subprocess.CalledProcessError(
            1,
            ["synthetic inventory"],
            output=b"rejected inventory",
            stderr=b"retained failure",
        )
        with (
            mock.patch.object(
                launcher, "indexed_accumulation_evidence", return_value=evidence
            ),
            mock.patch.object(evidence, "validate_sources", return_value={}),
            mock.patch.object(launcher.subprocess, "check_output", side_effect=failure),
            mock.patch.object(launcher, "stage_burn_program") as stage,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            launcher.indexed_accumulation_inventory(
                directory, Path("/synthetic"), ["--scope=linux-common"]
            )
        destinations[
            "burn-indexed_accumulation-expected.json"
        ].write_bytes.assert_called_once_with(b"rejected inventory")
        destinations[
            "burn-indexed_accumulation-expected.stderr"
        ].write_bytes.assert_called_once_with(b"retained failure")
        stage.assert_not_called()

    def test_burn_probe_stages_inventory_and_quotes_exact_arguments(self):
        import tempfile

        summary, population = evidence.reference_selection("LinuxCommon", "I8")
        exports = [json.dumps(x).encode() + b"\n" for x in (summary, population)]
        CACHE.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="launcher-synthetic-", dir=CACHE))
        root = directory / "root"
        root.mkdir()
        binary = directory / "binary with spaces"
        binary.write_bytes(b"synthetic test fixture, never executed")
        libdevice = directory / "synthetic-libdevice"
        libdevice.write_bytes(b"synthetic test fixture, never compiled")
        with (
            mock.patch.object(
                launcher, "indexed_accumulation_evidence", return_value=evidence
            ),
            mock.patch.object(evidence, "validate_sources", return_value={}),
            mock.patch.object(
                launcher.subprocess, "check_output", side_effect=exports
            ) as command,
            mock.patch.object(
                launcher, "stage_burn_program", return_value=libdevice
            ) as stage,
        ):
            launcher.burn_probe(
                root,
                binary,
                "indexed_accumulation",
                indexed_accumulation_arguments=["--scope=linux-common", "--dtype=i8"],
            )
        self.assertEqual(command.call_count, 2)
        stage.assert_called_once_with(root, binary, "/burn-indexed_accumulation")
        script = (root / "benchmark").read_text()
        self.assertIn(
            "exec /burn-indexed_accumulation --scope=linux-common --dtype=i8\n", script
        )
        manifest = json.loads(
            (directory / "burn-indexed_accumulation-manifest.json").read_text()
        )
        self.assertEqual(manifest["arguments"], ["--scope=linux-common", "--dtype=i8"])
        self.assertEqual(
            manifest["inputs"]["expected_population"]["sha256"],
            hashlib.sha256(exports[1]).hexdigest(),
        )
        self.assertIn("evidence_validator", manifest["inputs"])
        self.assertEqual(
            (directory / "burn-indexed_accumulation-population.json").read_bytes(),
            exports[1],
        )

    def test_parser_filters_and_mutual_exclusion_before_work(self):
        for arguments in (
            ("--burn-indexed-accumulation-dtype=i8",),
            ("--burn-indexed-accumulation=/missing/binary",),
            ("--burn-indexed-accumulation-dtype=bool",),
            ("--burn-indexed-no-flex32",),
            ("--burn-mixed-matmul-dtype=f16",),
            ("--burn-indexed-accumulation=/bin/true", "--burn-indexed=/bin/true"),
            (
                "--burn-indexed-accumulation=/bin/true",
                "--quantized-profile",
                "/profile",
            ),
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch.object(sys, "argv", ["linux-reference.py", *arguments]),
                mock.patch("sys.stderr"),
                mock.patch("sys.stdout"),
                self.assertRaises(SystemExit) as error,
            ):
                launcher.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    unittest.main()
