#!/usr/bin/env python3
"""CPU-only evidence mutations using bounded, independently generated plans.

Set D3OS_MIXED_MATMUL_SHARED_CACHE to audit real frozen CPU exports as well.
No test creates a device, compiles code or treats synthetic records as GPU proof.
"""

import copy
import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evidence = load_module("mixed_matmul_evidence", "mixed-matmul-evidence.py")
generator = load_module("mixed_matmul_generator", "generate-mixed-matmul-fixtures.py")


def policy_export(policy):
    if isinstance(policy, dict):
        low, high = policy["ProjectInterval"]
        return {"kind": "project_interval", "low": low, "high": high}
    return {
        "kind": {
            "Exact": "exact",
            "ZeroClass": "zero_class",
            "NaNClass": "nan_class",
            "Infinity": "infinity",
            "Observation": "observation",
        }[policy]
    }


def stage_for(row, expected):
    name, shape = expected["name"], expected["shape"]

    def operand(source, origin="matrix", transpose=False):
        field, bits_field = (
            ("matrices", "logical_bits")
            if origin == "matrix"
            else ("expected", "expected_bits")
        )
        m = next(m for m in row[field] if m["name"] == source)
        dims, bits, views = m["shape"], m[bits_field], []
        if transpose:
            tensor = generator.Tensor(
                tuple(dims),
                [generator.decode(b, generator.storage(m["dtype"])) for b in bits],
                m["dtype"],
            )
            transformed = tensor.transpose().export()
            dims, bits = transformed["shape"], [int(b, 16) for b in transformed["bits"]]
            views = [
                {"kind": "transpose", "first": len(dims) - 2, "second": len(dims) - 1}
            ]
        return {
            "origin": origin,
            "source": source,
            "dtype": m["dtype"],
            "shape": dims,
            "bits": bits,
            "views": views,
            "producer": None,
        }

    axis = None
    if name == "gradient.forward":
        operands, operation = (
            [operand("gradient.lhs"), operand("gradient.rhs")],
            "matmul",
        )
    elif name == "gradient.dA_before_sum":
        operands, operation = (
            [operand("gradient.upstream"), operand("gradient.rhs", transpose=True)],
            "matmul",
        )
    elif name == "gradient.dA":
        operands, operation, axis = (
            [operand("gradient.dA_before_sum", "expected_stage")],
            "typed_sum",
            0,
        )
    elif name == "linear.stored":
        operands, operation = (
            [operand("linear.input"), operand("linear.weight")],
            "matmul",
        )
    elif name == "linear.output":
        operands, operation = (
            [operand("linear.stored", "expected_stage"), operand("linear.bias")],
            "typed_add",
        )
    else:
        operands, operation = [operand("lhs"), operand("rhs")], "matmul"
    return {
        "operation": operation,
        "operands": operands,
        "axis": axis,
        "result_shape": shape,
        "output_shape": shape,
        "source_view": None,
        "bits_origin": "immutable_fixture_not_device_capture",
    }


def bounded_plans():
    ids = {
        "f16_exact_finite_dyadic",
        "flex32_exact_finite_dyadic",
        "f16_cancellation_order",
        "f16_live_nan_inf_and_masked_padding",
        "f16_zero_M",
        "f16_destination_ties_both_parities_and_sides",
    }
    descriptions, population = [], []
    for seed in generator.BUDGET["fixtures"]:
        if seed["key"] not in ids:
            continue
        row = (
            generator.numeric(seed)
            if seed["group"] == "numeric"
            else generator.geometry(seed)
        )
        runtime = generator.runtime_case(row)
        description = {
            k: runtime[k] for k in ("id", "group", "dtype", "operation", "numeric")
        }
        description.update(
            raw_matrices=len(runtime["raw_matrices"]),
            raw_expected=len(runtime["raw_expected"]),
            matrices=[],
            expected=[],
            status="not_device_observed",
        )
        for matrix in runtime["matrices"]:
            value = matrix["values"]
            description["matrices"].append(
                {
                    "name": matrix["name"],
                    "dtype": value["dtype"],
                    "shape": value["shape"],
                    "logical_bits": [int(b, 16) for b in value["bits"]],
                }
            )
        for expected in runtime["expected"]:
            value = expected["values"]
            description["expected"].append(
                {
                    "name": expected["name"],
                    "dtype": value["dtype"],
                    "shape": value["shape"],
                    "expected_bits": [int(b, 16) for b in value["bits"]],
                    "stored_stage": expected["stage"],
                }
            )
            for route in ("primitive", "public"):
                if route == "public" and expected["name"] != "expected":
                    continue
                policies = [policy_export(p) for p in expected["policies"]]
                diagnostic = sum(p["kind"] == "observation" for p in policies)
                population.append(
                    {
                        "key": {
                            "case_id": runtime["id"],
                            "route": route,
                            "output": expected["name"],
                        },
                        "dtype": value["dtype"],
                        "shape": value["shape"],
                        "values": len(policies),
                        "expected_bits": [int(b, 16) for b in value["bits"]],
                        "policies": policies,
                        "required_values": len(policies) - diagnostic,
                        "diagnostic_values": diagnostic,
                        "model_values": sum(
                            p["kind"] == "project_interval" for p in policies
                        ),
                        "stored_stage": expected["stage"],
                        "source": False,
                    }
                )
        for matrix in description["matrices"]:
            # Payload-bearing source guards make byte identity a separate gate.
            nan = 0x7E35 if matrix["dtype"] == "F16" else 0x7FC5A53C
            bits = [nan, *matrix["logical_bits"], nan]
            for route in ("source_before", "source_after", "alias_after"):
                population.append(
                    {
                        "key": {
                            "case_id": runtime["id"],
                            "route": route,
                            "output": matrix["name"],
                        },
                        "dtype": matrix["dtype"],
                        "shape": [len(bits)],
                        "values": len(bits),
                        "expected_bits": bits,
                        "policies": [{"kind": "exact"} for _ in bits],
                        "required_values": len(bits),
                        "diagnostic_values": 0,
                        "model_values": 0,
                        "stored_stage": "untouched_source_bytes",
                        "source": True,
                    }
                )
        for plan in population:
            if plan["key"]["case_id"] == runtime["id"]:
                plan["stage_witness"] = (
                    None
                    if plan["source"]
                    else stage_for(
                        description,
                        next(
                            e
                            for e in description["expected"]
                            if e["name"] == plan["key"]["output"]
                        ),
                    )
                )
        descriptions.append(description)
    return descriptions, population


def summary_for(population, descriptions):
    selected = {p["key"]["case_id"] for p in population}
    groups = dict.fromkeys(
        (
            "geometry",
            "numeric",
            "gradient",
            "native_extensions",
            "preservation",
            "f64_diagnostics",
        ),
        0,
    )
    for row in descriptions:
        if row["id"] in selected:
            groups[{"Geometry": "geometry", "Numeric": "numeric"}[row["group"]]] += 1
    counts = {
        k: sum(p[k] for p in population)
        for k in ("required_values", "diagnostic_values", "model_values")
    }
    return {
        "version": 1,
        "corpus_rows": 121,
        "reserves": 3,
        "numeric_cells": 65,
        "corpus_last_drop_consumers": 141,
        "selected_rows": len(selected),
        **counts,
        "required_routes": sum(
            p["required_values"] > 0 or p["values"] == 0 for p in population
        ),
        "diagnostic_routes": sum(p["diagnostic_values"] > 0 for p in population),
        "source_values": sum(p["values"] for p in population if p["source"]),
        "gradient_values": 0,
        "consumer_routes": 0,
        "external_gates": 0,
        "groups": groups,
        "dtype_values": {
            dt: sum(p["values"] for p in population if p["dtype"] == dt)
            for dt in ("F16", "BF16", "Flex32", "F64")
        },
        "conditional_model": "project_empirical_conditional_normal_tree_not_vendor_guarantee",
        "vendor_rounding_guarantee": False,
        "device_observations": 0,
        "physical_proof": False,
    }


def decoded(dtype, bits):
    return generator.decode(bits, generator.storage(dtype))


def number(dtype, bits):
    value = decoded(dtype, bits)
    return float(value.q) if value.kind == "finite" else None


def rank(dtype, bits):
    width = {"F16": 16, "BF16": 16, "Flex32": 32}[dtype]
    return (
        bits | (1 << (width - 1))
        if bits < 1 << (width - 1)
        else (1 << width) - bits - 1
    )


def errors(actual, ideal):
    absolute = abs(actual - ideal) if actual is not None and ideal is not None else None
    return absolute, absolute / abs(ideal) if absolute is not None and ideal else None


def approximation(text):
    if text in ("nan", "inf", "-inf"):
        return None
    numerator, _, denominator = text.partition("/")
    return float(numerator) / float(denominator or "1")


def report_for(summary, population, descriptions, changes=None):
    """Render a synthetic record independently of the validator's helpers."""
    rows = {r["id"]: r for r in descriptions}
    observations, measurements, worst = [], [], None
    for p in population:
        key, dt = p["key"], p["dtype"]
        bits = list(p["expected_bits"])
        for i in range(len(bits)):
            bits[i] = (changes or {}).get(
                (key["case_id"], key["route"], key["output"], i), bits[i]
            )
        zeros = sum(
            decoded(dt, a).kind == decoded(dt, e).kind == "finite"
            and decoded(dt, a).q == decoded(dt, e).q == 0
            and a != e
            for a, e in zip(bits, p["expected_bits"], strict=True)
        )
        observations.append(
            {
                "key": key,
                "dtype": dt,
                "shape": p["shape"],
                "values": bits,
                "required_checked": p["required_values"],
                "diagnostic_observed": p["diagnostic_values"],
                "model_checked": p["model_values"],
                "zero_sign_differences": zeros,
            }
        )
        row = rows[key["case_id"]]
        for i, a in enumerate(bits):
            cell = (
                row["numeric"][i]
                if key["output"] == "expected" and row["numeric"]
                else None
            )
            e = p["expected_bits"][i]
            if (
                not p["source"]
                and number(dt, a) is not None
                and number(dt, e) is not None
            ):
                distance = abs(rank(dt, a) - rank(dt, e))
                if worst is None or distance > worst["storage_ULP_distance"]:
                    stage = p["stage_witness"]
                    absolute, relative = errors(
                        number(dt, a),
                        approximation(cell["ideal_rational"])
                        if cell
                        else number(dt, e),
                    )
                    worst = {
                        "key": key,
                        "operation": stage["operation"],
                        "cell": i,
                        "stored_stage": p["stored_stage"],
                        "dtype": dt,
                        "actual": a,
                        "expected": e,
                        "storage_ULP_distance": distance,
                        "approximate_absolute_error": absolute,
                        "approximate_relative_error": relative,
                        "exact_ideal_rational": cell["ideal_rational"] if cell else "",
                        "sum_abs_products": cell["sum_abs_products"] if cell else "",
                        **{
                            f"{side}_{field}": stage["operands"][index][field]
                            if len(stage["operands"]) > index
                            else None
                            for index, side in enumerate(("lhs", "rhs"))
                            for field in ("shape", "bits")
                        },
                        "stage_witness": stage,
                        "metrics_are_acceptance": False,
                    }
            if cell:
                policy = p["policies"][i]
                policy_text = {
                    "exact": "Exact",
                    "zero_class": "ZeroClass",
                    "nan_class": "NaNClass",
                    "infinity": "Infinity",
                    "observation": "Observation",
                }
                text = policy_text.get(policy["kind"])
                if text is None:
                    text = f"ProjectInterval {{ low: {policy['low']}, high: {policy['high']} }}"
                value = decoded(dt, a)
                cls = (
                    "NaN"
                    if value.kind == "nan"
                    else "Infinity"
                    if value.kind == "inf"
                    else "Zero"
                    if not value.q
                    else "Subnormal"
                    if abs(value.q) < generator.pow2(-14 if dt == "F16" else -126)
                    else "Finite"
                )
                absolute, relative = errors(
                    number(dt, a), approximation(cell["ideal_rational"])
                )
                measurements.append(
                    {
                        "key": key,
                        "cell": i,
                        "name": cell["name"],
                        "policy": text,
                        "required": policy["kind"] != "observation",
                        "actual_class": cls,
                        "actual_bits": a,
                        "ideal_stored_bits": cell["ideal_bits"],
                        "serial_diagnostic_bits": cell["serial_bits"],
                        "exact_operand_dot": cell["ideal_rational"],
                        "sum_abs_products": cell["sum_abs_products"],
                        "conditional_scratch_interval": [
                            cell["scratch_low"],
                            cell["scratch_high"],
                        ],
                        "approximate_absolute_error": absolute,
                        "approximate_relative_error": relative,
                        "storage_ULP_distance": abs(
                            rank(dt, a) - rank(dt, cell["ideal_bits"])
                        )
                        if number(dt, a) is not None
                        and number(dt, cell["ideal_bits"]) is not None
                        else None,
                        "metrics_are_acceptance": False,
                    }
                )
    return {
        "expected": summary,
        "attempted_routes": len(population),
        "passed_required_routes": summary["required_routes"],
        "failed": 0,
        "required_checked": summary["required_values"],
        "diagnostic_observed": summary["diagnostic_values"],
        "model_checked": summary["model_values"],
        "checked_required_routes": summary["required_routes"],
        "observed_diagnostic_routes": summary["diagnostic_routes"],
        "unobserved_routes": 0,
        "external_unobserved": 0,
        "tracking_failures": 0,
        "logical_success": True,
        "finalized": True,
        "failures": [],
        "observations": observations,
        "worst_numeric_witness": worst,
        "numeric_measurements": measurements,
        "diagnostic_measurements": [m for m in measurements if not m["required"]],
        "full_public_qualification": False,
        "physical_proof": False,
    }


def serialized(report, execution_os="linux", backend="flex_cpu", fusion="disabled"):
    s = report["expected"]
    marker = (
        "BURN_MIXED_MATMUL_REPORT"
        if execution_os == "linux"
        else "NVIDIA_BURN_MIXED_MATMUL_SHARED"
    )
    footer = (
        f"BURN_MIXED_MATMUL_LOGICAL_OK version=1 scope=Shared rows={s['selected_rows']} "
        f"required_checked={s['required_values']} diagnostic_observed={s['diagnostic_values']} "
        f"model_checked={s['model_values']} physical_proof=false full_public_qualification=false "
        f"conditional_model=project_empirical_conditional_normal_tree_not_vendor_guarantee "
        f"os={execution_os} backend={backend} fusion={fusion}"
    )
    return f"{marker} {json.dumps(report, allow_nan=False)}\n{footer}\n"


def logger_record(record, *, colored=True):
    if colored:
        return (
            "\x1b[36m[291.613]\x1b[94m[INF]\x1b[35m[sys_logger.rs@020]\x1b[39m [infer.rs@050] "
            + record
        )
    return "[291.613][INF][sys_logger.rs@020] [infer.rs@050] " + record


def contaminated_records(record):
    token, _, payload = record.partition(" ")
    yield "negated", "NOT_" + record
    yield "identifier_prefix", "OTHER_" + record
    yield "identifier_suffix", token + "_EXTRA " + payload
    yield "token_suffix", token + "X " + payload
    yield "embedded", "unrelated text " + record
    yield "leading_space", " " + record
    yield "unknown_prefix", "[unknown] " + record
    yield (
        "unknown_level",
        logger_record(record, colored=False).replace("[INF]", "[DBG]"),
    )
    yield (
        "unknown_file",
        logger_record(record, colored=False).replace("sys_logger.rs", "other.rs"),
    )
    yield (
        "wrong_kernel_line",
        logger_record(record, colored=False).replace("@020", "@021"),
    )
    yield (
        "wrong_user_file",
        logger_record(record, colored=False).replace("infer.rs", "other.rs"),
    )
    yield (
        "wrong_user_line",
        logger_record(record, colored=False).replace("@050", "@051"),
    )
    yield "unknown_SGR", logger_record(record).replace("\x1b[94m", "\x1b[31m")
    yield "partial_SGR", logger_record(record).replace("\x1b[35m", "")
    yield "suffix_SGR", record + "\x1b[39m"
    yield "suffix_text", record + " contamination"
    yield "embedded_CR", "unrelated\r" + record
    yield "embedded_VT", "unrelated\v" + record


class MixedMatmulEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.descriptions, cls.population = bounded_plans()
        cls.summary = summary_for(cls.population, cls.descriptions)
        cls.record = report_for(cls.summary, cls.population, cls.descriptions)

    def check(self, report=None, population=None, summary=None, descriptions=None):
        return evidence.validate_report(
            self.record if report is None else report,
            self.summary if summary is None else summary,
            self.population if population is None else population,
            self.descriptions if descriptions is None else descriptions,
        )

    def test_complete_record_derived_counts_and_unqualified_scope(self):
        result = self.check()
        self.assertEqual(result["observed_routes"], len(self.population))
        self.assertEqual(result["model_checked"], self.summary["model_values"])
        self.assertFalse(result["physical_proof"])
        self.assertFalse(result["full_public_qualification"])
        self.assertFalse(result["vendor_rounding_guarantee"])

    def test_strict_json_rejects_duplicates_nonfinite_and_overflow(self):
        for text in (
            '{"a":1,"a":2}',
            '{"a":{"b":0,"b":1}}',
            '{"a":NaN}',
            "[Infinity]",
            "[-Infinity]",
            "[1e999]",
            "{} {}",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evidence.strict_json(text)
        self.assertEqual(
            evidence.strict_json('{"a":[0,18446744073709551615]}')["a"][1], 2**64 - 1
        )

    def test_route_coverage_cannot_be_shifted_or_forged(self):
        for mutation in (
            "duplicate",
            "missing",
            "unexpected",
            "shift",
            "wrong_shape",
            "wrong_dtype",
            "truncated",
            "missing_empty",
            "missing_diagnostic",
            "none",
        ):
            r = copy.deepcopy(self.record)
            if mutation == "duplicate":
                r["observations"].append(copy.deepcopy(r["observations"][0]))
            elif mutation == "missing":
                r["observations"].pop()
            elif mutation == "unexpected":
                r["observations"][0]["key"]["route"] = "unknown"
            elif mutation == "shift":
                r["observations"][0]["key"]["case_id"] = "unknown"
            elif mutation == "wrong_shape":
                r["observations"][0]["shape"] = [r["observations"][0]["shape"][0] + 1]
            elif mutation == "wrong_dtype":
                r["observations"][0]["dtype"] = "BF16"
            elif mutation == "truncated":
                next(o for o in r["observations"] if o["values"])["values"].pop()
            elif mutation == "missing_empty":
                r["observations"] = [o for o in r["observations"] if o["values"]]
            elif mutation == "missing_diagnostic":
                r["observations"] = [
                    o for o in r["observations"] if not o["diagnostic_observed"]
                ]
            else:
                r["observations"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(r)

    def test_known_route_permutation_and_contradictory_observation_counters(self):
        r = copy.deepcopy(self.record)
        candidates = [(i, p) for i, p in enumerate(self.population) if p["source"]]
        pair = next(
            (a, b)
            for i, (a, p) in enumerate(candidates)
            for b, q in candidates[i + 1 :]
            if p["dtype"] == q["dtype"]
            and p["shape"] == q["shape"]
            and p["expected_bits"] != q["expected_bits"]
        )
        a, b = pair
        r["observations"][a]["key"], r["observations"][b]["key"] = (
            r["observations"][b]["key"],
            r["observations"][a]["key"],
        )
        with self.assertRaises(ValueError):
            self.check(r)
        for field in (
            "required_checked",
            "diagnostic_observed",
            "model_checked",
            "zero_sign_differences",
        ):
            r = copy.deepcopy(self.record)
            r["observations"][0][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(r)

    def test_storage_and_count_types_are_exact(self):
        for value in (True, 0.0, -1, 1 << 64, 1 << 16):
            r = copy.deepcopy(self.record)
            next(o for o in r["observations"] if o["values"])["values"][0] = value
            with self.subTest(bits=value), self.assertRaises(ValueError):
                self.check(r)
        for field in (
            "attempted_routes",
            "required_checked",
            "diagnostic_observed",
            "model_checked",
            "failed",
            "tracking_failures",
            "external_unobserved",
        ):
            for value in (True, float(self.record[field]), self.record[field] + 1):
                r = copy.deepcopy(self.record)
                r[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    self.check(r)
        for location in ("shape", "counter", "summary", "cell"):
            r = copy.deepcopy(self.record)
            if location == "shape":
                r["observations"][0]["shape"][0] = True
            elif location == "counter":
                r["observations"][0]["required_checked"] = float(
                    r["observations"][0]["required_checked"]
                )
            elif location == "summary":
                r["expected"]["selected_rows"] = float(self.summary["selected_rows"])
            else:
                r["numeric_measurements"][0]["cell"] = 0.0
            with self.subTest(location=location), self.assertRaises(ValueError):
                self.check(r)

    def test_required_policy_corruptions_and_source_nan_payload(self):
        for kind in (
            "exact",
            "zero_class",
            "nan_class",
            "infinity",
            "project_interval",
        ):
            r = copy.deepcopy(self.record)
            for plan, observation in zip(
                self.population, r["observations"], strict=True
            ):
                indices = [
                    i
                    for i, p in enumerate(plan["policies"])
                    if p["kind"] == kind and not plan["source"]
                ]
                if indices:
                    i = indices[0]
                    old = observation["values"][i]
                    observation["values"][i] = (
                        old ^ 1
                        if kind == "exact"
                        else 0x3C00
                        if kind != "infinity"
                        else old ^ 0x8000
                    )
                    break
            else:
                self.fail(f"missing fixture policy {kind}")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.check(r)
        r = copy.deepcopy(self.record)
        source = next(
            o
            for p, o in zip(self.population, r["observations"], strict=True)
            if p["source"]
        )
        source["values"][0] ^= 1
        with self.assertRaises(ValueError):
            self.check(r)

    def test_zero_sign_nan_payload_and_interval_endpoints(self):
        changes = {}
        for plan in self.population:
            if plan["source"]:
                continue
            for i, policy in enumerate(plan["policies"]):
                bits = plan["expected_bits"][i]
                if policy["kind"] == "zero_class":
                    changes[(*plan["key"].values(), i)] = bits ^ 0x8000
                if policy["kind"] == "nan_class":
                    changes[(*plan["key"].values(), i)] = bits ^ 1
        self.check(
            report_for(self.summary, self.population, self.descriptions, changes)
        )
        for endpoint in ("low", "high"):
            changes = {
                (*p["key"].values(), i): policy[endpoint]
                for p in self.population
                for i, policy in enumerate(p["policies"])
                if policy["kind"] == "project_interval"
            }
            self.check(
                report_for(self.summary, self.population, self.descriptions, changes)
            )
        p = next(
            p for p in self.population if p["dtype"] == "Flex32" and p["model_values"]
        )
        i = next(
            i
            for i, policy in enumerate(p["policies"])
            if policy["kind"] == "project_interval"
        )
        for bits in (
            p["policies"][i]["low"] - 1,
            p["policies"][i]["high"] + 1,
            0x7FC00000,
            0x7F800000,
        ):
            r = report_for(
                self.summary,
                self.population,
                self.descriptions,
                {(*p["key"].values(), i): bits},
            )
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                self.check(r)

    def test_diagnostic_nonfinite_is_observed_without_finite_reference_invention(self):
        plan = next(p for p in self.population if p["diagnostic_values"])
        i = next(
            i for i, p in enumerate(plan["policies"]) if p["kind"] == "observation"
        )
        changes = {(*plan["key"].values(), i): 0x7E01}
        r = report_for(self.summary, self.population, self.descriptions, changes)
        self.check(r)
        m = next(m for m in r["diagnostic_measurements"] if m["actual_bits"] == 0x7E01)
        self.assertIsNone(m["approximate_absolute_error"])
        self.assertIsNone(m["storage_ULP_distance"])
        m["approximate_absolute_error"] = 0
        with self.assertRaises(ValueError):
            self.check(r)

    def test_numeric_metrics_bind_all_claims_and_both_worst_operands(self):
        fields = (
            "name",
            "policy",
            "required",
            "actual_class",
            "actual_bits",
            "ideal_stored_bits",
            "serial_diagnostic_bits",
            "exact_operand_dot",
            "sum_abs_products",
            "conditional_scratch_interval",
            "approximate_absolute_error",
            "approximate_relative_error",
            "storage_ULP_distance",
            "metrics_are_acceptance",
            "cell",
        )
        for field in fields:
            r = copy.deepcopy(self.record)
            m = r["numeric_measurements"][0]
            m[field] = (
                True
                if field in ("required", "metrics_are_acceptance")
                else "corrupt"
                if isinstance(m[field], str)
                else []
                if isinstance(m[field], list)
                else 123
            )
            if field == "required":
                m[field] = not self.record["numeric_measurements"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(r)
        for field in (
            "lhs_bits",
            "rhs_bits",
            "lhs_shape",
            "rhs_shape",
            "actual",
            "expected",
            "dtype",
            "stored_stage",
            "operation",
            "cell",
            "storage_ULP_distance",
            "exact_ideal_rational",
            "sum_abs_products",
            "approximate_absolute_error",
            "metrics_are_acceptance",
        ):
            r = copy.deepcopy(self.record)
            w = r["worst_numeric_witness"]
            if isinstance(w[field], list):
                w[field][0] += 1
            elif type(w[field]) is str:
                w[field] += "corrupt"
            elif type(w[field]) is bool:
                w[field] = True
            else:
                w[field] += 1
            with self.subTest(worst=field), self.assertRaises(ValueError):
                self.check(r)
        for field in (
            "numeric_measurements",
            "diagnostic_measurements",
            "worst_numeric_witness",
        ):
            r = copy.deepcopy(self.record)
            r[field] = [] if field.endswith("measurements") else None
            with self.subTest(omitted=field), self.assertRaises(ValueError):
                self.check(r)

    def test_staged_gradient_sum_and_bias_witnesses_use_their_own_operands(self):
        for output in (
            "gradient.dA_before_sum",
            "gradient.dA",
            "linear.stored",
            "linear.output",
        ):
            p = next(p for p in self.population if p["key"]["output"] == output)
            ordered = [p, *[q for q in self.population if q is not p]]
            r = report_for(self.summary, ordered, self.descriptions)
            self.check(r)
            witness = r["worst_numeric_witness"]
            self.assertEqual(witness["key"]["output"], output)
            self.assertEqual(witness["stage_witness"], p["stage_witness"])
            if output == "gradient.dA_before_sum":
                self.assertEqual(witness["lhs_shape"], [3, 1, 2])
                self.assertEqual(witness["rhs_shape"], [3, 2, 1])
            if output == "gradient.dA":
                self.assertIsNone(witness["rhs_shape"])
                self.assertIsNone(witness["rhs_bits"])
            for field in ("source", "shape", "bits", "views"):
                corrupted = copy.deepcopy(r)
                operand = corrupted["worst_numeric_witness"]["stage_witness"][
                    "operands"
                ][0]
                operand[field] = (
                    "unknown"
                    if field == "source"
                    else [{"kind": "invalid"}]
                    if field == "views"
                    else []
                )
                with (
                    self.subTest(output=output, field=field),
                    self.assertRaises(ValueError),
                ):
                    self.check(corrupted)
        for corruption in (
            "unknown_matrix",
            "wrong_bits",
            "bad_view",
            "wrong_result",
            "wrong_origin",
        ):
            pop = copy.deepcopy(self.population)
            p = next(p for p in pop if p["key"]["output"] == "gradient.dA_before_sum")
            w = p["stage_witness"]
            if corruption == "unknown_matrix":
                w["operands"][0]["source"] = "unknown_matrix"
            elif corruption == "wrong_bits":
                w["operands"][0]["bits"][0] ^= 1
            elif corruption == "bad_view":
                w["operands"][1]["views"][0]["first"] = True
            elif corruption == "wrong_result":
                w["result_shape"][0] += 1
            else:
                w["bits_origin"] = "device_capture"
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                self.check(population=pop)

    def test_nonfinite_negative_bool_and_unrepresentable_metrics_reject(self):
        for metric in (True, -1.0, float("nan"), float("inf"), 1 << 4096, None):
            r = copy.deepcopy(self.record)
            r["numeric_measurements"][0]["approximate_absolute_error"] = metric
            with (
                self.subTest(type=type(metric).__name__),
                self.assertRaises(ValueError),
            ):
                self.check(r)
        descriptions = copy.deepcopy(self.descriptions)
        next(row for row in descriptions if row["numeric"])["numeric"][0][
            "ideal_rational"
        ] = "unknown_finite_reference"
        with self.assertRaises(ValueError):
            self.check(descriptions=descriptions)

    def test_empty_selection_external_scopes_and_forged_oracle_metadata(self):
        with self.assertRaises(ValueError):
            self.check(population=[], summary=summary_for([], self.descriptions))
        for change in (
            "source_policy",
            "interval_reversed",
            "interval_nonfinite",
            "interval_float",
            "population_count",
            "unknown_case",
            "external_group",
            "summary_bool",
        ):
            pop, desc, summary = (
                copy.deepcopy(self.population),
                copy.deepcopy(self.descriptions),
                copy.deepcopy(self.summary),
            )
            if change == "source_policy":
                next(p for p in pop if p["source"])["policies"][0] = {
                    "kind": "nan_class"
                }
            elif change.startswith("interval"):
                policy = next(
                    policy
                    for p in pop
                    for policy in p["policies"]
                    if policy["kind"] == "project_interval"
                )
                if change == "interval_reversed":
                    policy["low"] = policy["high"] + 1
                elif change == "interval_nonfinite":
                    policy["high"] = 0x7C00
                else:
                    policy["low"] = float(policy["low"])
            elif change == "population_count":
                pop[0]["required_values"] += 1
            elif change == "unknown_case":
                pop[0]["key"]["case_id"] = "unregistered"
            elif change == "external_group":
                desc[0]["group"] = "NativeExtension"
            else:
                summary["device_observations"] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check(population=pop, summary=summary, descriptions=desc)

    def test_os_specific_prefix_and_footer_binding(self):
        for execution_os, backend in (("linux", "flex_cpu"), ("d3os", "native")):
            text = serialized(self.record, execution_os, backend)
            result = evidence.validate(
                text,
                self.summary,
                self.population,
                self.descriptions,
                execution_os=execution_os,
                backend=backend,
                fusion="disabled",
            )
            self.assertEqual(result["execution_os"], execution_os)
            lines = text.splitlines()
            malformed = [
                text + lines[-1],
                text + lines[0],
                lines[-1] + "\n" + lines[0],
                text.replace(f"os={execution_os}", "os=other"),
                text.replace(f"backend={backend}", "backend=other"),
                text.replace("fusion=disabled", "fusion=enabled"),
                text.replace("version=1", "version=2"),
                text.replace("scope=Shared", "scope=All"),
                text.replace(evidence.MODEL, "vendor_guaranteed"),
                text + "MIXED_MATMUL_FAILURE observed failure\n",
                text.replace("physical_proof=false", "physical_proof=true"),
            ]
            other_marker = (
                "NVIDIA_BURN_MIXED_MATMUL_SHARED"
                if execution_os == "linux"
                else "BURN_MIXED_MATMUL_REPORT"
            )
            malformed.append(text.replace(lines[0].split()[0], other_marker))
            for serial in malformed:
                with (
                    self.subTest(execution_os=execution_os, serial_tail=serial[-100:]),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate(
                        serial,
                        self.summary,
                        self.population,
                        self.descriptions,
                        execution_os=execution_os,
                        backend=backend,
                        fusion="disabled",
                    )
        with self.assertRaises(ValueError):
            evidence.validate(
                serialized(self.record),
                self.summary,
                self.population,
                self.descriptions,
                execution_os="linux",
                backend="flex_cpu",
                fusion="disabled",
                scope="All",
            )

    def test_complete_shared_record_tokens_and_production_logger_prefixes(self):
        for execution_os, backend, fusion in (
            ("linux", "flex_cpu", "disabled"),
            ("linux", "cuda", "enabled_materialized_chain_steps"),
            ("d3os", "native", "disabled"),
        ):
            text = serialized(self.record, execution_os, backend, fusion)
            lines = text.splitlines()
            for index in (0, 1):
                for label, corrupted in contaminated_records(lines[index]):
                    changed = list(lines)
                    changed[index] = corrupted
                    with (
                        self.subTest(os=execution_os, index=index, corruption=label),
                        self.assertRaises(ValueError),
                    ):
                        evidence.validate(
                            "\n".join(changed),
                            self.summary,
                            self.population,
                            self.descriptions,
                            execution_os=execution_os,
                            backend=backend,
                            fusion=fusion,
                        )
            if execution_os == "d3os":
                for colored in (True, False):
                    logged = (
                        "\r\n".join(
                            logger_record(line, colored=colored) for line in lines
                        )
                        + "\r\n"
                    )
                    result = evidence.validate(
                        logged,
                        self.summary,
                        self.population,
                        self.descriptions,
                        execution_os=execution_os,
                        backend=backend,
                        fusion=fusion,
                    )
                    self.assertTrue(result["logical_success"])
                    self.assertFalse(result["physical_proof"])
                    # Bare/logged duplicates must still count as two records.
                    for line in lines:
                        with self.assertRaises(ValueError):
                            evidence.validate(
                                logged + line + "\n",
                                self.summary,
                                self.population,
                                self.descriptions,
                                execution_os=execution_os,
                                backend=backend,
                                fusion=fusion,
                            )
            else:
                with self.assertRaises(ValueError):
                    evidence.validate(
                        "\n".join(logger_record(line) for line in lines),
                        self.summary,
                        self.population,
                        self.descriptions,
                        execution_os=execution_os,
                        backend=backend,
                        fusion=fusion,
                    )
            footer = lines[-1]
            for suffix in (" ", "\t", "\r", " [infer.rs@050]"):
                with (
                    self.subTest(os=execution_os, footer_suffix=repr(suffix)),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate(
                        lines[0] + "\n" + footer + suffix,
                        self.summary,
                        self.population,
                        self.descriptions,
                        execution_os=execution_os,
                        backend=backend,
                        fusion=fusion,
                    )

    def test_mixed_os_detection_and_separate_failure_bypass_require_real_tokens(self):
        linux = serialized(self.record)
        d3os = serialized(self.record, "d3os", "native")
        for execution_os, backend, text, foreign in (
            ("linux", "flex_cpu", linux, d3os.splitlines()[0]),
            ("d3os", "native", d3os, linux.splitlines()[0]),
        ):
            for extra in (foreign, "NOT_" + foreign, "unrelated " + foreign):
                with (
                    self.subTest(os=execution_os, prefix=extra[:60]),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate(
                        text + extra + "\n",
                        self.summary,
                        self.population,
                        self.descriptions,
                        execution_os=execution_os,
                        backend=backend,
                        fusion="disabled",
                    )
        for marker in ("PLAN", "OBSERVED", "STRICT_OK", "SEPARATE", "RAW_NUMERIC"):
            for prefix in ("", "NOT_", "unrelated "):
                foreign = prefix + "NVIDIA_BURN_MIXED_MATMUL_" + marker + " parser_unit"
                with (
                    self.subTest(foreign=marker, prefix=prefix),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate(
                        linux + foreign + "\n",
                        self.summary,
                        self.population,
                        self.descriptions,
                        execution_os="linux",
                        backend="flex_cpu",
                        fusion="disabled",
                    )
        for prefix in ("NOT_", "prefix ", "[unknown] "):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                evidence.validate(
                    linux
                    + prefix
                    + "NVIDIA_BURN_MIXED_MATMUL_SEPARATE panicked UNFROZEN\n",
                    self.summary,
                    self.population,
                    self.descriptions,
                    execution_os="linux",
                    backend="flex_cpu",
                    fusion="disabled",
                )
        numeric = "NVIDIA_BURN_MIXED_MATMUL_RAW_NUMERIC case=parser_unit"
        for record in (
            numeric,
            logger_record(numeric),
            logger_record(numeric, colored=False),
        ):
            self.assertEqual(
                evidence._serial_lines(record, execution_os="d3os"), [numeric]
            )
        for label, record in contaminated_records(numeric):
            if label == "suffix_text":
                continue  # Numeric payload semantics remain outside this parser.
            with self.subTest(numeric=label), self.assertRaises(ValueError):
                evidence._serial_lines(record, execution_os="d3os")

    def test_success_claims_cannot_hide_failures_or_upgrade_qualification(self):
        for field, value in (
            ("logical_success", False),
            ("finalized", False),
            ("finalized", 1),
            ("physical_proof", True),
            ("full_public_qualification", True),
            ("failures", [{"message": "failed"}]),
            ("unobserved_routes", 1),
        ):
            r = copy.deepcopy(self.record)
            r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(r)

    def test_raw_incomplete_records_and_numeric_types_fail_closed(self):
        for text in (
            "",
            serialized(self.record),
            "NVIDIA_BURN_MIXED_MATMUL_STRICT_OK scope=shared93_plus_raw\n",
        ):
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                evidence.validate_raw(text, evidence.RAW_COUNTS)
        for field in evidence.RAW_COUNTS:
            counts = dict(evidence.RAW_COUNTS)
            counts[field] = float(counts[field])
            with self.subTest(field=field), self.assertRaises(ValueError):
                evidence.validate_raw("", counts)
        for counts in (
            evidence.RAW_COUNTS | {"queued": 0},
            evidence.RAW_COUNTS | {"physical_proof": 1},
        ):
            with self.assertRaises(ValueError):
                evidence.validate_raw("", counts)


@unittest.skipUnless(
    os.environ.get("D3OS_MIXED_MATMUL_SHARED_CACHE"),
    "real CPU exports were not requested",
)
class RealCpuExportTests(unittest.TestCase):
    def test_every_successful_real_cpu_case_and_real_cli_serial(self):
        cache = Path(os.environ["D3OS_MIXED_MATMUL_SHARED_CACHE"])
        descriptions = evidence.strict_json(
            (cache / "cli-diagnostics.json").read_text()
        )
        population = evidence.strict_json(
            (cache / "expected-population.json").read_text()
        )
        results = evidence.strict_json(
            (cache / "cpu-isolated-results.json").read_text()
        )
        checked, rejected = 0, 0
        for result in results:
            report_path = cache / "cpu-isolated" / f"{result['id']}.json"
            if not report_path.is_file() or not report_path.stat().st_size:
                continue
            r = evidence.strict_json(report_path.read_text())
            pop = [p for p in population if p["key"]["case_id"] == result["id"]]
            if result["success"]:
                self.assertEqual(result["exit"], 0)
                validated = evidence.validate_report(
                    r, r["expected"], pop, descriptions
                )
                checked += (
                    validated["required_checked"] + validated["diagnostic_observed"]
                )
            else:
                with self.assertRaises(ValueError):
                    evidence.validate_report(r, r["expected"], pop, descriptions)
                rejected += 1
        self.assertGreater(checked, 0)
        self.assertGreater(rejected, 0)
        serial_path = os.environ.get("D3OS_MIXED_MATMUL_CPU_SERIAL")
        if serial_path:
            serial = Path(serial_path).read_text()
            report_line = next(
                line
                for line in serial.splitlines()
                if line.startswith("BURN_MIXED_MATMUL_REPORT ")
            )
            report = evidence.strict_json(report_line.split(" ", 1)[1])
            ids = {o["key"]["case_id"] for o in report["observations"]}
            evidence.validate(
                serial,
                report["expected"],
                [p for p in population if p["key"]["case_id"] in ids],
                descriptions,
                execution_os="linux",
                backend="flex_cpu",
                fusion="disabled",
            )
        print(
            json.dumps(
                {
                    "real_CPU_values_validated": checked,
                    "failed_CPU_records_rejected": rejected,
                }
            )
        )

    def test_R1_six_saved_cpu_alias_reproductions_and_root_wrapper(self):
        cache = Path(os.environ["D3OS_MIXED_MATMUL_SHARED_CACHE"])
        descriptions = evidence.strict_json(
            (cache / "cli-diagnostics.json").read_text()
        )
        report = evidence.strict_json(
            (cache / "cpu-isolated/f16_exact_finite_dyadic.json").read_text()
        )
        population = [
            p
            for p in descriptions["expected_population"]
            if p["key"]["case_id"] == "f16_exact_finite_dyadic"
        ]
        filtered = dict(descriptions, expected_population=population)
        launcher = load_module(
            "mixed_matmul_marker_linux_wrapper", "linux-reference.py"
        )
        original = (cache / "real-cpu-numeric.stdout").read_text()
        controls, wrapper_controls = 0, 0
        for execution_os, backend, fusion in (
            ("linux", "flex_cpu", "disabled"),
            ("linux", "cuda", "enabled_materialized_chain_steps"),
            ("d3os", "native", "disabled"),
        ):
            text = (
                original.replace("os=linux", "os=" + execution_os)
                .replace("backend=flex_cpu", "backend=" + backend)
                .replace("fusion=disabled", "fusion=" + fusion)
            )
            token = "BURN_MIXED_MATMUL_REPORT"
            if execution_os == "d3os":
                text = text.replace(token, "NVIDIA_BURN_MIXED_MATMUL_SHARED")
                token = "NVIDIA_BURN_MIXED_MATMUL_SHARED"
            for marker in (token, "BURN_MIXED_MATMUL_LOGICAL_OK"):
                malformed = text.replace(marker, "NOT_" + marker)
                with (
                    self.subTest(os=execution_os, backend=backend, marker=marker),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate(
                        malformed,
                        report["expected"],
                        population,
                        descriptions,
                        execution_os=execution_os,
                        backend=backend,
                        fusion=fusion,
                    )
                controls += 1
                if backend == "cuda":
                    with self.assertRaises(ValueError):
                        launcher.validate_mixed_matmul(
                            malformed,
                            report["expected"],
                            population,
                            filtered,
                            case_id="f16_exact_finite_dyadic",
                        )
                    wrapper_controls += 1
            if backend == "cuda":
                self.assertTrue(
                    launcher.validate_mixed_matmul(
                        text,
                        report["expected"],
                        population,
                        filtered,
                        case_id="f16_exact_finite_dyadic",
                    )["logical_success"]
                )
            if execution_os == "d3os" and os.environ.get(
                "D3OS_MIXED_MATMUL_LOGGER_PREFIX"
            ):
                # The prefix was read from accepted protocol-53 serial bytes.
                # Runtime relabels here are parser fixtures, never GPU evidence.
                prefix = evidence.strict_json(
                    Path(os.environ["D3OS_MIXED_MATMUL_LOGGER_PREFIX"]).read_text()
                )["exact_prefix"]
                logged = (
                    "\r\n".join(prefix + line for line in text.splitlines()) + "\r\n"
                )
                self.assertTrue(
                    evidence.validate(
                        logged,
                        report["expected"],
                        population,
                        descriptions,
                        execution_os="d3os",
                        backend="native",
                        fusion="disabled",
                    )["logical_success"]
                )
        self.assertEqual(controls, 6)
        self.assertEqual(wrapper_controls, 2)


@unittest.skipUnless(
    os.environ.get("D3OS_MIXED_MATMUL_SHARED_CACHE")
    and os.environ.get("D3OS_MIXED_MATMUL_RAW_CACHE"),
    "frozen Raw plan exports were not requested",
)
class RawMarkerExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared_cache = Path(os.environ["D3OS_MIXED_MATMUL_SHARED_CACHE"])
        raw_cache = Path(os.environ["D3OS_MIXED_MATMUL_RAW_CACHE"])
        plan_bytes = (raw_cache / "plan.json").read_bytes()
        if hashlib.sha256(plan_bytes).hexdigest() != evidence.RAW_PLAN_SHA256:
            raise ValueError("Raw test input does not match the current reviewed plan")
        cls.plan = evidence.strict_json(plan_bytes.decode())
        cls.counts = evidence.strict_json((raw_cache / "counts.json").read_text())
        cls.descriptions = evidence.strict_json(
            (shared_cache / "cli-diagnostics.json").read_text()
        )
        cls.population = evidence.strict_json(
            (shared_cache / "expected-population.json").read_text()
        )
        # Synthetic parser fixtures stay in memory. They are never runtime or
        # GPU evidence, and the real CLI tests use only actual CPU observations.
        cls.shared = report_for(
            cls.plan["shared"], cls.population, cls.descriptions["rows"]
        )
        counts = (
            "Counts { " + ", ".join(f"{k}: {v}" for k, v in cls.counts.items()) + " }"
        )
        pins = (
            "["
            + ", ".join(
                f"({json.dumps(k)}, {json.dumps(v)})"
                for k, v in evidence.SOURCE_PINS.items()
            )
            + "]"
        )
        shared_lines = serialized(cls.shared, "d3os", "native").splitlines()
        cls.lines = [
            (
                f"NVIDIA_BURN_MIXED_MATMUL_PLAN frozen=true protocol=54 kernels=88 source_pins={pins} raw={counts} "
                f"shared={json.dumps(cls.plan['shared'])} population={json.dumps(cls.plan)} "
                f"conditional_model={evidence.MODEL} full_public_qualification=false"
            ),
            shared_lines[0],
        ]
        for scope, rows, external in (
            ("NativeExtensions", 12, 12),
            ("Preservation", 8, 8),
            ("F64Diagnostics", 8, 1),
        ):
            expected = summary_for([], [])
            expected["selected_rows"], expected["external_gates"] = rows, external
            group = {
                "NativeExtensions": "native_extensions",
                "Preservation": "preservation",
                "F64Diagnostics": "f64_diagnostics",
            }[scope]
            expected["groups"][group] = rows
            descriptor = {
                "expected": expected,
                "attempted_routes": 0,
                "passed_required_routes": 0,
                "failed": 1,
                "required_checked": 0,
                "diagnostic_observed": 0,
                "model_checked": 0,
                "checked_required_routes": 0,
                "observed_diagnostic_routes": 0,
                "unobserved_routes": 0,
                "external_unobserved": external,
                "tracking_failures": 0,
                "logical_success": False,
                "finalized": True,
                "failures": [
                    {
                        "key": {
                            "case_id": "unqualified_scope",
                            "route": "execution",
                            "output": "case",
                        },
                        "message": "backend panicked: separate diagnostic",
                    }
                ],
                "observations": [],
                "worst_numeric_witness": None,
                "numeric_measurements": [],
                "diagnostic_measurements": [],
                "physical_proof": False,
                "full_public_qualification": False,
            }
            cls.lines.append(
                f"NVIDIA_BURN_MIXED_MATMUL_SEPARATE scope={scope} qualification=false {json.dumps(descriptor)}"
            )
        cls.lines.extend(
            [
                (
                    f"NVIDIA_BURN_MIXED_MATMUL_OBSERVED accepted=true expected={counts} observed={counts} failed=0 examples=[] "
                    "external_extension_evidence=raw_separate preservation=separate f64=diagnostic_only full_public_qualification=false"
                ),
                shared_lines[1],
                (
                    f"NVIDIA_BURN_MIXED_MATMUL_STRICT_OK scope=shared93_plus_raw protocol=54 kernels=88 observed={counts} "
                    "original_allocation_tail=checked full_public_qualification=false"
                ),
            ]
        )
        cls.serial = "\n".join(cls.lines) + "\n"

    def test_raw_complete_contract_keeps_provenance_and_value_limits(self):
        result = evidence.validate_raw(self.serial, self.counts)
        self.assertTrue(result["raw_marker_consistency"])
        for field in (
            "requires_independent_shared_validation",
            "requires_execution_provenance",
            "requires_source_image_compiler_provenance",
        ):
            self.assertTrue(result[field])
        for field in (
            "physical_proof",
            "queued_residency_proven",
            "raw_per_word_values_revalidated",
            "full_public_qualification",
        ):
            self.assertFalse(result[field])
        evidence.validate(
            self.serial,
            self.plan["shared"],
            self.population,
            self.descriptions,
            execution_os="d3os",
            backend="native",
            fusion="disabled",
        )

    def test_append_only_indexed_protocol_binds_every_kernel_slot(self):
        plan = copy.deepcopy(self.plan)
        plan["protocol"], plan["kernel_count"] = 55, 89
        arrays = []
        for group, fields in (
            ("raw_recipes", ("guarded_scratch_kernels", "guarded_cast_kernels")),
            ("native_forward_recipes", ("kernels",)),
            ("public_backward_recipes", ("kernels",)),
        ):
            for recipe in plan[group]:
                for field in fields:
                    self.assertEqual(len(recipe[field]), 88)
                    recipe[field].append(0)
                    arrays.append(recipe[field])
        self.assertEqual(len(arrays), 177)

        def serial_for(population):
            before = " population=" + json.dumps(self.plan) + " conditional_model="
            after = " population=" + json.dumps(population) + " conditional_model="
            return self.serial.replace(before, after).replace(
                "protocol=54 kernels=88", "protocol=55 kernels=89"
            )

        serial = serial_for(plan)
        self.assertTrue(
            evidence.validate_raw(serial, self.counts, protocol=55, profiles=89)[
                "raw_marker_consistency"
            ]
        )
        for protocol, profiles in (
            (54, 88),
            (55, 88),
            (54, 89),
            (True, 89),
            (55, 89.0),
        ):
            with (
                self.subTest(protocol=protocol, profiles=profiles),
                self.assertRaises(ValueError),
            ):
                evidence.validate_raw(
                    serial, self.counts, protocol=protocol, profiles=profiles
                )
        with self.assertRaises(ValueError):
            evidence.validate_raw(self.serial, self.counts, protocol=55, profiles=89)
        for index, counters in enumerate(arrays):
            for slot in (0, 88):
                counters[slot] += 1
                with (
                    self.subTest(recipe=index, slot=slot),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate_raw(
                        serial_for(plan), self.counts, protocol=55, profiles=89
                    )
                counters[slot] -= 1
            counters.pop()
            with (
                self.subTest(recipe=index, missing_slot=True),
                self.assertRaises(ValueError),
            ):
                evidence.validate_raw(
                    serial_for(plan), self.counts, protocol=55, profiles=89
                )
            counters.append(0)

    def test_raw_population_binds_every_current_packed_writer_field(self):
        def with_population(plan):
            before = " population=" + json.dumps(self.plan) + " conditional_model="
            after = " population=" + json.dumps(plan) + " conditional_model="
            self.assertEqual(self.serial.count(before), 1)
            return self.serial.replace(before, after)

        reordered = dict(reversed(list(self.plan.items())))
        self.assertTrue(
            evidence.validate_raw(with_population(reordered), self.counts)[
                "raw_marker_consistency"
            ]
        )
        fields = (
            "cast_trial_order",
            "cast_poison_masks",
            "cast_physical_overwrite_all_live_elements",
            "cast_fresh_cross_trial_coherence",
            "cast_unused_tail_zero",
        )
        historical = copy.deepcopy(self.plan)
        for field in fields:
            del historical[field]
            missing = copy.deepcopy(self.plan)
            del missing[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                evidence.validate_raw(with_population(missing), self.counts)
            mutated = copy.deepcopy(self.plan)
            value = mutated[field]
            if type(value) is bool:
                mutated[field] = False
            elif field == "cast_trial_order":
                mutated[field] = list(reversed(value))
            else:
                mutated[field][0] ^= 1
            with self.subTest(mutated=field), self.assertRaises(ValueError):
                evidence.validate_raw(with_population(mutated), self.counts)
        with self.assertRaisesRegex(ValueError, "Raw population differs"):
            evidence.validate_raw(with_population(historical), self.counts)

    def test_raw_marker_aliases_logger_formats_and_complete_records(self):
        for colored in (True, False):
            logged = (
                "\r\n".join(logger_record(line, colored=colored) for line in self.lines)
                + "\r\n"
            )
            self.assertTrue(
                evidence.validate_raw(logged, self.counts)["raw_marker_consistency"]
            )
        for index, line in enumerate(self.lines):
            for label, corrupted in contaminated_records(line):
                lines = list(self.lines)
                lines[index] = corrupted
                with (
                    self.subTest(index=index, corruption=label),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate_raw("\n".join(lines), self.counts)
            with self.subTest(duplicate=index), self.assertRaises(ValueError):
                evidence.validate_raw(
                    self.serial + logger_record(line) + "\n", self.counts
                )
        for marker in ("BURN_MIXED_MATMUL_REPORT", "NOT_BURN_MIXED_MATMUL_REPORT"):
            with self.subTest(foreign=marker), self.assertRaises(ValueError):
                evidence.validate_raw(self.serial + marker + " {}\n", self.counts)
        # Full footer comparison must retain suffix whitespace after normalization.
        for suffix in (" ", "\t", "\x1b[39m"):
            lines = list(self.lines)
            lines[-2] = logger_record(lines[-2]) + suffix
            with (
                self.subTest(footer_suffix=repr(suffix)),
                self.assertRaises(ValueError),
            ):
                evidence.validate_raw("\n".join(lines), self.counts)

    def test_separate_scope_reports_stay_complete_unobserved_and_unqualified(self):
        for index in (2, 3, 4):
            prefix, payload = self.lines[index].split(" qualification=false ", 1)
            descriptor = evidence.strict_json(payload)
            for mutation in (
                "missing_observations",
                "premature",
                "external_observed",
                "summary_physical",
                "vendor",
                "bad_count",
            ):
                d = copy.deepcopy(descriptor)
                if mutation == "missing_observations":
                    del d["observations"]
                elif mutation == "premature":
                    d["finalized"] = False
                elif mutation == "external_observed":
                    d["external_unobserved"] = 0
                elif mutation == "summary_physical":
                    d["expected"]["physical_proof"] = True
                elif mutation == "vendor":
                    d["expected"]["vendor_rounding_guarantee"] = True
                else:
                    d["expected"]["selected_rows"] = float(
                        d["expected"]["selected_rows"]
                    )
                lines = list(self.lines)
                lines[index] = prefix + " qualification=false " + json.dumps(d)
                with (
                    self.subTest(index=index, mutation=mutation),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate_raw("\n".join(lines), self.counts)

    def test_raw_markers_pins_counts_plan_and_external_gates_cannot_be_injected(self):
        for i in range(len(self.lines)):
            for lines in (
                self.lines[:i] + self.lines[i + 1 :],
                self.lines + [self.lines[i]],
            ):
                with (
                    self.subTest(marker=i, length=len(lines)),
                    self.assertRaises(ValueError),
                ):
                    evidence.validate_raw("\n".join(lines), self.counts)
        changes = (
            ("frozen=true", "frozen=false"),
            ("protocol=54", "protocol=53"),
            ("kernels=88", "kernels=87"),
            (evidence.SOURCE_PINS["shared"], "0" * 64),
            ("source_words: 45116", "source_words: 45115"),
            ("accepted=true", "accepted=false"),
            ("failed=0 examples=[]", "failed=1 examples=[]"),
            ("failed=0 examples=[]", 'failed=0 examples=["failed"]'),
            ("original_allocation_tail=checked", "original_allocation_tail=unchecked"),
            ("qualification=false", "qualification=true"),
            ('"external_unobserved": 12', '"external_unobserved": 0'),
            ('"logical_success": false', '"logical_success": true'),
            ('"service_count": 14', '"service_count": 13'),
            ('"batches": 1', '"batches": true'),
            ("os=d3os", "os=linux"),
            (
                '"scratch_physical_overwrite_all_words": true',
                '"scratch_physical_overwrite_all_words": false',
            ),
            (
                '"fresh_cross_trial_coherence": true',
                '"fresh_cross_trial_coherence": false',
            ),
        )
        for before, after in changes:
            self.assertNotEqual(before, after)
            mutated = self.serial.replace(before, after)
            self.assertNotEqual(mutated, self.serial)
            with self.subTest(before=before), self.assertRaises(ValueError):
                evidence.validate_raw(mutated, self.counts)
        for field, value in (
            ("protocol", True),
            ("profiles", 88.0),
            ("source_pins", evidence.SOURCE_PINS | {"shared": "0" * 64}),
        ):
            with self.subTest(argument=field), self.assertRaises(ValueError):
                evidence.validate_raw(self.serial, self.counts, **{field: value})
        lines = list(self.lines)
        lines[2], lines[3] = lines[3], lines[2]
        with self.assertRaises(ValueError):
            evidence.validate_raw("\n".join(lines), self.counts)


if __name__ == "__main__":
    unittest.main()
