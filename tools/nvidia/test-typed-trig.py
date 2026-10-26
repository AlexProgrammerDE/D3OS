#!/usr/bin/env python3
"""Check aggregate qualification gates without initializing a GPU."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "linux_reference", Path(__file__).with_name("linux-reference.py")
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class TrigQualification(unittest.TestCase):
    def setUp(self):
        self.fixtures = {
            "version": 1,
            "policy_sha256": "a" * 64,
            "fixture_sha256": "b" * 64,
        }
        self.expected = {"values": 1}
        self.plan = [[1, 0, 0], *[[0, 0, 0] for _ in range(4)]]
        self.scope = (
            "no_flex32=false skip_rank_zero=false finite=false dtype=all op=all"
        )
        counters = [
            "values",
            "failures",
            "classification_failures",
            "numerical_failures",
            "exact_special_failures",
            "half_envelope_failures",
            "codomain_failures",
            "unreferenced_results",
            "passing_finite_pairs",
            "finite_pairs",
            "ideal_finite_pairs",
            "sign_mismatches",
            "nonzero_to_zero",
            "subnormal_preserved",
            "nan_payload_differences",
            "quiet_nan_outputs",
            "negative_nan_outputs",
        ]
        cells = []
        for dtype in ("f64", "f32", "flex32", "f16", "bf16"):
            for operation in ("tanh", "sin", "cos"):
                cell = dict.fromkeys(counters, 0)
                cell.update(
                    dtype=dtype,
                    op=operation,
                    observed_classes=[0] * 5,
                    transitions=[[0] * 5 for _ in range(5)],
                    histogram=[0] * 7,
                    ideal_histogram=[0] * 7,
                    passing_histogram=[0] * 7,
                    max_ulp=None,
                    max_ideal_ulp=None,
                    passing_max_ulp=None,
                    worst_input_bits=None,
                    worst_actual_bits=None,
                )
                cells.append(cell)
        # tanh(+infinity) is exactly one. All other cells have no observations.
        cells[0].update(
            values=1,
            finite_pairs=1,
            ideal_finite_pairs=1,
            passing_finite_pairs=1,
            max_ulp=0,
            max_ideal_ulp=0,
            passing_max_ulp=0,
            worst_input_bits="7ff0000000000000",
            worst_actual_bits="3ff0000000000000",
        )
        cells[0]["transitions"][2][2] = 1
        cells[0]["observed_classes"][2] = 1
        for key in ("histogram", "ideal_histogram", "passing_histogram"):
            cells[0][key][0] = 1
        self.accuracy = {"failures": 0, "cells": cells}

    def marker(self, accuracy, execution_os="linux", backend="flex_cpu", totals=None):
        return (
            f"BURN_TYPED_TRIG_OK os={execution_os} backend={backend} fusion=disabled corpus=1 "
            f"totals={json.dumps(self.expected if totals is None else totals)} accuracy={json.dumps(accuracy)} "
            f"policy_sha256={self.fixtures['policy_sha256']} fixture_sha256={self.fixtures['fixture_sha256']} "
            "numerical_policy=project_qualification_targets staged_half=f32_neighbor_envelope "
            "no_ftz=strict preserved_inputs=exact_bytes chain=materialized_known_refs "
            f"poison=retained_allocation scope=strict_native {self.scope}"
        )

    def validate(self, text, execution_os="linux", backend="flex_cpu"):
        return reference.validate_typed_trig(
            text,
            self.expected,
            self.plan,
            self.fixtures,
            execution_os=execution_os,
            backend=backend,
            fusion="disabled",
            scope=self.scope,
        )

    def test_linux_and_native_log_framing(self):
        self.assertEqual(self.validate(self.marker(self.accuracy)), self.accuracy)
        native = "\x1b[36m[1.23]\x1b[0m[INF][infer.rs@050] " + self.marker(
            self.accuracy, "d3os", "native"
        )
        self.assertEqual(self.validate(native, "d3os", "native"), self.accuracy)

    def test_reject_contradictory_observations(self):
        for key, value in (
            ("values", True),
            ("failures", 1),
            ("codomain_failures", 1),
            ("unreferenced_results", 1),
            ("passing_finite_pairs", 0),
            ("quiet_nan_outputs", 1),
            ("max_ulp", 1),
            ("observed_classes", [1, 0, 0, 0, 0]),
            ("worst_actual_bits", "4000000000000000"),
            ("worst_actual_bits", "4000000000000000ffffffffffffffff"),
        ):
            with self.subTest(field=key, value=value), self.assertRaises(RuntimeError):
                accuracy = copy.deepcopy(self.accuracy)
                accuracy["cells"][0][key] = value
                self.validate(self.marker(accuracy))

    def test_reject_plausible_histogram_outside_operation_budget(self):
        accuracy = copy.deepcopy(self.accuracy)
        for key in ("max_ulp", "max_ideal_ulp", "passing_max_ulp"):
            accuracy["cells"][0][key] = 2
        for key in ("histogram", "ideal_histogram", "passing_histogram"):
            accuracy["cells"][0][key] = [0, 0, 1, 0, 0, 0, 0]
        with self.assertRaises(RuntimeError):
            self.validate(self.marker(accuracy))

    def test_reject_missing_operation_and_noninteger_plan(self):
        accuracy = copy.deepcopy(self.accuracy)
        accuracy["cells"][1], accuracy["cells"][2] = (
            accuracy["cells"][2],
            accuracy["cells"][1],
        )
        with self.assertRaises(RuntimeError):
            self.validate(self.marker(accuracy))
        for value in (True, 1.0):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.validate(self.marker(self.accuracy, totals={"values": value}))

    def test_half_ideal_observations_cannot_disappear(self):
        self.expected = {"values": 2}
        for fi, dtype, infinity, one in (
            (3, "f16", "7c00", "3c00"),
            (4, "bf16", "7f80", "3f80"),
        ):
            with self.subTest(dtype=dtype):
                accuracy = copy.deepcopy(self.accuracy)
                source = accuracy["cells"][0]
                empty = copy.deepcopy(accuracy["cells"][fi * 3])
                accuracy["cells"][0] = empty | {"dtype": "f64", "op": "tanh"}
                cell = source | {
                    "dtype": dtype,
                    "values": 2,
                    "finite_pairs": 2,
                    "ideal_finite_pairs": 2,
                    "passing_finite_pairs": 2,
                    "worst_input_bits": infinity.zfill(16),
                    "worst_actual_bits": one.zfill(16),
                }
                accuracy["cells"][fi * 3] = cell
                cell["observed_classes"][2] = 2
                cell["transitions"][2][2] = 2
                for key in ("histogram", "ideal_histogram", "passing_histogram"):
                    cell[key][0] = 2
                self.plan = [[0] * 3 for _ in range(5)]
                self.plan[fi][0] = 2
                self.assertEqual(self.validate(self.marker(accuracy)), accuracy)
                for remaining in (0, 1):
                    cell.update(
                        ideal_finite_pairs=remaining,
                        ideal_histogram=[remaining, 0, 0, 0, 0, 0, 0],
                        max_ideal_ulp=0 if remaining else None,
                    )
                    with self.subTest(remaining=remaining), self.assertRaises(
                        RuntimeError
                    ):
                        self.validate(self.marker(accuracy))

    def test_reject_ambiguous_serialization_and_relaxed_scope(self):
        marker = self.marker(self.accuracy)
        for text in (
            marker + "\n" + marker,
            marker.replace('"failures": 0', '"failures": 0, "failures": 0', 1),
            marker.replace('"max_ulp": 0', '"max_ulp": NaN', 1),
            marker.replace("no_ftz=strict", "no_ftz=relaxed"),
        ):
            with self.subTest(text=text), self.assertRaises((RuntimeError, ValueError)):
                self.validate(text)


if __name__ == "__main__":
    unittest.main()
