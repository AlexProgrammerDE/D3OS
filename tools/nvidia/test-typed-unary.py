#!/usr/bin/env python3
"""Exercise unary observation accounting and serial evidence without a GPU."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "evidence", Path(__file__).with_name("typed-unary-evidence.py")
)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class UnaryQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures, cls.bounds = evidence.load_bounds(
            Path(__file__).with_name("typed-unary-manifest.json")
        )

    def setUp(self):
        self.plan = [[0] * 4 for _ in range(5)]
        self.reference_plan = [
            {"staged": [0] * 5, "ideal": [0] * 5, "slack": 0} for _ in range(20)
        ]
        self.cells = []
        for dtype in evidence.DTYPES:
            for op in evidence.OPERATIONS:
                cell = dict.fromkeys(
                    (
                        "values",
                        "failures",
                        "classification_failures",
                        "numerical_failures",
                        "exact_special_failures",
                        "half_envelope_failures",
                        "codomain_failures",
                        "unreferenced_results",
                        "sign_mismatches",
                        "nonzero_to_zero",
                        "finite_pairs",
                        "ideal_finite_pairs",
                        "passing_finite_pairs",
                        "subnormal_preserved",
                        "quiet_nan_outputs",
                        "negative_nan_outputs",
                        "nan_payload_differences",
                        "reference_class_disagreements",
                        "reference_nonzero_finite_disagreements",
                    ),
                    0,
                )
                cell.update(
                    dtype=dtype,
                    op=op,
                    observed_classes=[0] * 5,
                    unreferenced_observed_classes=[0] * 5,
                    transitions=[[0] * 5 for _ in range(5)],
                    ideal_transitions=[[0] * 5 for _ in range(5)],
                    histogram=[0] * 7,
                    ideal_histogram=[0] * 7,
                    passing_histogram=[0] * 7,
                    max_ulp=None,
                    max_ideal_ulp=None,
                    passing_max_ulp=None,
                    worst_input_bits=None,
                    worst_actual_bits=None,
                    worst_ideal_input_bits=None,
                    worst_ideal_actual_bits=None,
                )
                self.cells.append(cell)
        self.accuracy = {"failures": 0, "cells": self.cells}
        self.scope = (
            "no_flex32=false skip_rank_zero=false finite=false dtype=all op=all"
        )

    def finite(self, index, steps=0):
        fi, oi = divmod(index, 4)
        width = (64, 32, 32, 16, 16)[fi]
        sign = 1 << (width - 1)
        exponent = (0x7FF0000000000000, 0x7F800000, 0x7F800000, 0x7C00, 0x7F80)[fi]
        one = (0x3FF0000000000000, 0x3F800000, 0x3F800000, 0x3C00, 0x3F80)[fi]
        source = one
        ideal, staged, low, high = self.bounds[index]["rows"][source]
        actual = staged + steps
        if fi >= 3:
            for source, (ideal, staged, low, high) in self.bounds[index][
                "rows"
            ].items():
                if evidence.classification(source, sign, exponent) in (0, 3, 4):
                    continue
                for actual in (low, high):
                    if (
                        all(
                            evidence.classification(bits, sign, exponent) == 2
                            for bits in (ideal, staged, actual)
                        )
                        and abs(
                            evidence.ordered(actual, sign)
                            - evidence.ordered(staged, sign)
                        )
                        == steps
                        and abs(
                            evidence.ordered(actual, sign)
                            - evidence.ordered(ideal, sign)
                        )
                        == steps
                    ):
                        break
                else:
                    continue
                break
            else:
                self.fail("no valid finite witness for requested distance")
        self.plan[fi][oi] = 1
        cell = self.cells[index]
        cell.update(
            values=1,
            finite_pairs=1,
            ideal_finite_pairs=1,
            passing_finite_pairs=1,
            max_ulp=steps,
            max_ideal_ulp=steps,
            passing_max_ulp=steps,
        )
        cell["observed_classes"][2] = 1
        for key in ("transitions", "ideal_transitions"):
            cell[key][2][2] = 1
        for key in ("histogram", "ideal_histogram", "passing_histogram"):
            cell[key][evidence.bucket(steps)] = 1
        for prefix in ("worst", "worst_ideal"):
            cell[f"{prefix}_input_bits"] = f"{source:016x}"
            cell[f"{prefix}_actual_bits"] = f"{actual:016x}"
        for key in ("staged", "ideal"):
            self.reference_plan[index][key][2] = 1

    def validate(self, accuracy=None):
        return evidence.validate_accuracy(
            self.accuracy if accuracy is None else accuracy,
            self.plan,
            sum(map(sum, self.plan)),
            self.bounds,
            reference_plan=self.reference_plan,
        )

    def test_all_cell_identities_and_operation_budgets(self):
        for index, limit in enumerate(self.bounds):
            self.finite(index, min(limit["staged"], limit["ideal"]))
        self.assertEqual(self.validate(), self.accuracy)
        for index, limit in enumerate(self.bounds):
            bad = copy.deepcopy(self.accuracy)
            cell = bad["cells"][index]
            maximum = max(limit["staged"], limit["ideal"]) + 1
            for h, m in (
                ("histogram", "max_ulp"),
                ("ideal_histogram", "max_ideal_ulp"),
                ("passing_histogram", "passing_max_ulp"),
            ):
                cell[m] = maximum
                cell[h] = [int(i == evidence.bucket(maximum)) for i in range(7)]
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                self.validate(bad)

    def test_dual_reference_observations_cannot_disappear(self):
        self.finite(13)
        self.validate()
        for reference_key in ("ideal_transitions", "transitions"):
            bad = copy.deepcopy(self.accuracy)
            cell = bad["cells"][13]
            cell[reference_key][2][2] = 0
            cell[reference_key][0][2] = 1
            cell[
                (
                    "ideal_finite_pairs"
                    if reference_key == "ideal_transitions"
                    else "finite_pairs"
                )
            ] = 0
            with self.subTest(reference=reference_key), self.assertRaises(RuntimeError):
                self.validate(bad)
        for key in (
            "reference_class_disagreements",
            "reference_nonzero_finite_disagreements",
        ):
            bad = copy.deepcopy(self.accuracy)
            bad["cells"][13][key] = 1
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate(bad)

    def test_both_reference_matrices_cannot_relabel_finite_inputs_as_nan(self):
        self.finite(0)
        bad = copy.deepcopy(self.accuracy)
        cell = bad["cells"][0]
        cell.update(
            finite_pairs=0,
            ideal_finite_pairs=0,
            passing_finite_pairs=0,
            max_ulp=None,
            max_ideal_ulp=None,
            passing_max_ulp=None,
            worst_input_bits=None,
            worst_actual_bits=None,
            worst_ideal_input_bits=None,
            worst_ideal_actual_bits=None,
        )
        for key in ("transitions", "ideal_transitions"):
            cell[key] = [[0] * 5 for _ in range(5)]
            cell[key][4][4] = 1
        cell["observed_classes"] = [0, 0, 0, 0, 1]
        for key in ("histogram", "ideal_histogram", "passing_histogram"):
            cell[key] = [0] * 7
        with self.assertRaises(RuntimeError):
            self.validate(bad)

    def test_witnesses_require_actual_reference_and_distance(self):
        self.finite(4)
        for key, bits in (
            ("worst_input_bits", "000000003f801234"),
            ("worst_actual_bits", "00000000bf800000"),
            ("worst_actual_bits", "000000003f800000"),
            ("worst_ideal_input_bits", "000000003f801234"),
            ("worst_ideal_actual_bits", "00000000bf800000"),
        ):
            bad = copy.deepcopy(self.accuracy)
            bad["cells"][4][key] = bits
            with self.subTest(key=key, bits=bits), self.assertRaises(RuntimeError):
                self.validate(bad)

    def test_cosh_domain_and_independent_ideal_witness(self):
        self.finite(6)
        for key, bits in (
            ("worst_actual_bits", "00000000bf800000"),
            ("worst_actual_bits", "000000003f000000"),
            ("worst_ideal_actual_bits", "000000007f800000"),
            ("worst_ideal_input_bits", None),
        ):
            bad = copy.deepcopy(self.accuracy)
            bad["cells"][6][key] = bits
            with self.subTest(key=key, bits=bits), self.assertRaises(RuntimeError):
                self.validate(bad)

    def test_counter_histogram_and_class_contradictions(self):
        self.finite(7, 4)
        self.validate()
        for key, value in (
            ("values", True),
            ("values", 0),
            ("passing_finite_pairs", 0),
            ("codomain_failures", 1),
            ("quiet_nan_outputs", 1),
            ("max_ulp", 2),
            ("observed_classes", [0, 1, 0, 0, 0]),
            ("unreferenced_observed_classes", [1, 0, 0, 0, 0]),
        ):
            bad = copy.deepcopy(self.accuracy)
            bad["cells"][7][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate(bad)

    def marker(self, os="linux", backend="flex_cpu"):
        expected = {
            "schema_version": 3,
            "values": sum(map(sum, self.plan)),
            "expected_value_counts": self.plan,
            "expected_staged_class_counts": [
                [self.reference_plan[fi * 4 + oi]["staged"] for oi in range(4)]
                for fi in range(5)
            ],
            "expected_ideal_class_counts": [
                [self.reference_plan[fi * 4 + oi]["ideal"] for oi in range(4)]
                for fi in range(5)
            ],
            "chain_value_counts": [[0] * 4 for _ in range(5)],
        }
        return expected, (
            f"BURN_TYPED_UNARY_OK os={os} backend={backend} fusion=disabled corpus=1 totals={json.dumps(expected)} accuracy={json.dumps(self.accuracy)}"
            f" policy_sha256={self.fixtures['policy_sha256']} fixture_sha256={self.fixtures['fixture_sha256']}"
            " numerical_policy=project_qualification_targets staged_half=f32_neighbor_envelope no_ftz=strict preserved_inputs=exact_bytes"
            f" chain=materialized_known_refs poison=retained_allocation scope=strict_native {self.scope}"
        )

    def test_terminal_framing_and_unambiguous_serialization(self):
        self.finite(0)
        expected, marker = self.marker()

        def validate(text, os="linux", backend="flex_cpu"):
            return evidence.validate(
                text,
                expected,
                self.fixtures,
                self.bounds,
                execution_os=os,
                backend=backend,
                fusion="disabled",
                scope=self.scope,
            )

        self.assertEqual(validate(marker), self.accuracy)
        _, native = self.marker("d3os", "native")
        self.assertEqual(
            validate(
                "\x1b[32m[1.2][INF][infer.rs@55] \x1b[0m" + native, "d3os", "native"
            ),
            self.accuracy,
        )
        for bad in (
            marker + "\n" + marker,
            marker.replace('"failures": 0', '"failures": 0, "failures": 0', 1),
            marker.replace('"max_ulp": 0', '"max_ulp": NaN', 1),
            marker.replace('"schema_version": 3', '"schema_version": true', 1),
            marker.replace("no_ftz=strict", "no_ftz=relaxed"),
        ):
            with self.subTest(bad=bad[:80]), self.assertRaises(
                (RuntimeError, ValueError)
            ):
                validate(bad)

    def test_mixed_chain_plan_is_explicit(self):
        self.finite(0)
        self.finite(5)
        expected, _ = self.marker()
        self.assertEqual(evidence.validate_plan(expected), self.plan)
        for invalid in (True, 1.0, -1):
            bad = copy.deepcopy(expected)
            bad["expected_value_counts"][0][0] = invalid
            with self.subTest(value=invalid), self.assertRaises(RuntimeError):
                evidence.validate_plan(bad)


if __name__ == "__main__":
    unittest.main()
