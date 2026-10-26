#!/usr/bin/env python3
"""Exercise Power evidence accounting without creating a GPU device."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "power_evidence", Path(__file__).with_name("typed-power-evidence.py")
)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class PowerEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.references, cls.limits = evidence.load_references(
            Path(__file__).with_name("typed-power-manifest.json")
        )

    def setUp(self):
        self.summary = {
            "values": 0,
            "nonchain_values": 0,
            "chain_values": 0,
            "cells": [],
        }
        self.accuracy = {"failures": 0, "cells": []}
        self.scope = "no_flex32=false skip_rank_zero=false finite=false common_linux=false dtype=all"

    def install(
        self,
        dtype="f32",
        schedule="mathematical",
        steps=0,
        predicate=None,
        allow_special=False,
    ):
        fi = evidence.DTYPES.index(dtype)
        for (kind, operation, _, _), row in self.references.items():
            if (
                kind != dtype
                or operation != schedule
                or (predicate and not predicate(row))
            ):
                continue
            if schedule == "mathematical" and steps:
                if row.exact:
                    continue
                candidates = (row.low, row.high) if fi >= 3 else (row.staged + steps,)
            else:
                candidates = (row.staged,)
            for actual in candidates:
                if (
                    evidence.distance(fi, actual, row.staged) == steps
                    or (allow_special and actual == row.staged)
                ) and evidence.passing(fi, schedule, row, actual):
                    break
            else:
                continue
            break
        else:
            self.fail("no eligible immutable witness")
        route = "tensor" if schedule == "mathematical" else "powf_scalar"
        key = {"dtype": dtype, "route": route, "schedule": schedule, "chain": False}
        expected = key | {
            "values": 1,
            "reference_classes": [0] * 5,
            "ideal_classes": [0] * 5,
        }
        expected["reference_classes"][evidence.classify(fi, row.staged)] = 1
        expected["ideal_classes"][evidence.classify(fi, row.ideal)] = 1
        self.summary["cells"].append(expected)
        self.summary["values"] += 1
        self.summary["nonchain_values"] += 1
        c = dict.fromkeys(
            (
                "values",
                "failures",
                "classification_failures",
                "numerical_failures",
                "exact_special_failures",
                "unreferenced_results",
                "finite_pairs",
                "ideal_finite_pairs",
                "normal_subnormal_transitions",
                "sign_mismatches",
            ),
            0,
        )
        c.update(
            values=1,
            observed_classes=[0] * 5,
            unreferenced_observed_classes=[0] * 5,
            transitions=[[0] * 5 for _ in range(5)],
            ideal_transitions=[[0] * 5 for _ in range(5)],
            histogram=[0] * 7,
            ideal_histogram=[0] * 7,
            max_ulp=None,
            max_ideal_ulp=None,
            worst=None,
            worst_ideal=None,
        )
        observed = evidence.classify(fi, actual)
        c["observed_classes"][observed] = 1
        c["transitions"][evidence.classify(fi, row.staged)][observed] = 1
        c["ideal_transitions"][evidence.classify(fi, row.ideal)][observed] = 1
        c["normal_subnormal_transitions"] = int(
            evidence.classify(fi, row.staged) in (1, 2)
            and observed in (1, 2)
            and evidence.classify(fi, row.staged) != observed
        )
        witness = {"schedule": schedule} | {
            name: f"0x{value:016x}"
            for name, value in (
                ("lhs", row.lhs),
                ("rhs", row.rhs),
                ("actual", actual),
                ("staged", row.staged),
                ("ideal", row.ideal),
            )
        }
        for wanted, count, hist, maximum, example in (
            (row.staged, "finite_pairs", "histogram", "max_ulp", "worst"),
            (
                row.ideal,
                "ideal_finite_pairs",
                "ideal_histogram",
                "max_ideal_ulp",
                "worst_ideal",
            ),
        ):
            distance = evidence.distance(fi, actual, wanted)
            if distance is not None:
                c[count] = 1
                c[hist][evidence.bucket(distance)] = 1
                c[maximum] = distance
                c[example] = witness.copy()
        self.accuracy["cells"].append(key | {"accuracy": c})
        return c

    def validate(self, accuracy=None):
        return evidence.validate_accuracy(
            self.accuracy if accuracy is None else accuracy,
            self.summary,
            self.references,
            self.limits,
        )

    def test_all_formats_and_stored_shortcut_schedules(self):
        for dtype in evidence.DTYPES:
            for schedule in evidence.SCHEDULES:
                self.install(dtype, schedule)
        self.assertEqual(self.validate(), self.accuracy)

    def test_four_step_histogram_bucket_and_operation_budget(self):
        cell = self.install(steps=4)
        self.assertEqual(cell["histogram"], [0, 0, 0, 1, 0, 0, 0])
        self.validate()
        for maximum, histogram in (
            (5, [0, 0, 0, 0, 1, 0, 0]),
            (2, [0, 0, 0, 1, 0, 0, 0]),
        ):
            bad = copy.deepcopy(self.accuracy)
            bad["cells"][0]["accuracy"].update(max_ulp=maximum, histogram=histogram)
            with self.subTest(maximum=maximum), self.assertRaises(RuntimeError):
                self.validate(bad)

    def test_stored_wrapper_class_can_differ_from_mathematical_power(self):
        self.install(
            "f64",
            "square_reciprocal",
            allow_special=True,
            predicate=lambda r: r.staged == 0
            and evidence.classify(0, r.ideal) in (1, 2),
        )
        self.validate()
        c = self.accuracy["cells"][0]["accuracy"]
        self.assertEqual(c["finite_pairs"], 0)
        self.assertEqual(c["ideal_finite_pairs"], 0)
        bad = copy.deepcopy(self.accuracy)
        bad["cells"][0]["accuracy"]["ideal_transitions"] = copy.deepcopy(
            c["transitions"]
        )
        with self.assertRaises(RuntimeError):
            self.validate(bad)

    def test_both_reference_matrices_cannot_discard_finite_observations(self):
        self.install()
        bad = copy.deepcopy(self.accuracy)
        c = bad["cells"][0]["accuracy"]
        c.update(
            observed_classes=[0, 0, 0, 0, 1],
            finite_pairs=0,
            ideal_finite_pairs=0,
            max_ulp=None,
            max_ideal_ulp=None,
            worst=None,
            worst_ideal=None,
            histogram=[0] * 7,
            ideal_histogram=[0] * 7,
        )
        for matrix in ("transitions", "ideal_transitions"):
            c[matrix] = [[0] * 5 for _ in range(5)]
            c[matrix][4][4] = 1
        with self.assertRaises(RuntimeError):
            self.validate(bad)

    def test_chain_populations_retain_their_frozen_finite_reference_classes(self):
        c = self.install(
            predicate=lambda r: r.lhs == 0x40000000 and r.rhs == 0x40400000
        )
        for collection in (self.summary, self.accuracy):
            collection["cells"][0]["chain"] = True
        self.summary.update(chain_values=1, nonchain_values=0)
        self.validate()
        for name in ("transitions", "ideal_transitions"):
            c[name] = [[0] * 5 for _ in range(5)]
            c[name][4][4] = 1
        c.update(
            observed_classes=[0, 0, 0, 0, 1],
            finite_pairs=0,
            ideal_finite_pairs=0,
            histogram=[0] * 7,
            ideal_histogram=[0] * 7,
            max_ulp=None,
            max_ideal_ulp=None,
            worst=None,
            worst_ideal=None,
        )
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_valid_worst_reference_must_occur_in_reported_transitions(self):
        c = self.install(predicate=lambda r: evidence.classify(1, r.staged) == 2)
        row = self.references["f32", "mathematical", 1, 0x3F7FFFFF]
        witness = {"schedule": "mathematical"} | {
            name: f"0x{value:016x}"
            for name, value in (
                ("lhs", row.lhs),
                ("rhs", row.rhs),
                ("actual", row.staged),
                ("staged", row.staged),
                ("ideal", row.ideal),
            )
        }
        c.update(worst=witness, worst_ideal=witness.copy())
        self.assertEqual(c["transitions"][1][1], 0)
        self.assertEqual(c["ideal_transitions"][1][1], 0)
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_both_operand_and_schedule_witnesses_are_bound(self):
        self.install()
        for witness, key, value in (
            ("worst", "rhs", "0x000000003f801234"),
            ("worst", "schedule", "square"),
            ("worst", "actual", "0x00000000bf800000"),
            ("worst_ideal", "staged", "0x0000000000000000"),
            ("worst_ideal", "lhs", "0xffffffffffffffff"),
        ):
            bad = copy.deepcopy(self.accuracy)
            bad["cells"][0]["accuracy"][witness][key] = value
            with self.subTest(witness=witness, key=key), self.assertRaises(
                RuntimeError
            ):
                self.validate(bad)

    def test_count_identity_and_transition_corruptions_reject(self):
        self.install()
        for key, value in (
            ("values", True),
            ("failures", 1),
            ("unreferenced_results", 1),
            ("finite_pairs", 0),
            ("normal_subnormal_transitions", 1),
            ("observed_classes", [0, 1, 0, 0, 0]),
        ):
            bad = copy.deepcopy(self.accuracy)
            bad["cells"][0]["accuracy"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate(bad)
        for cell in (
            self.accuracy["cells"][0] | {"chain": 0},
            self.accuracy["cells"][0] | {"route": "unknown"},
        ):
            with self.subTest(cell=cell), self.assertRaises(RuntimeError):
                self.validate({"failures": 0, "cells": [cell]})
        with self.assertRaises(RuntimeError):
            self.validate({"failures": 0, "cells": self.accuracy["cells"] * 2})

    def test_identical_references_require_matching_error_metrics(self):
        c = self.install(steps=4)
        self.validate()
        c["worst"]["actual"] = c["worst"]["staged"]
        c.update(max_ulp=0, histogram=[1, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(RuntimeError):
            self.validate()

    def marker(self, execution_os="linux", backend="flex_cpu"):
        return (
            f"BURN_TYPED_POWER_OK os={execution_os} backend={backend} fusion=disabled corpus={self.manifest['corpus_version']} "
            f"policy_sha256={self.manifest['policy_sha256']} fixture_sha256={self.manifest['fixture_sha256']} qualification_passed=true "
            f"totals={json.dumps(self.summary)} accuracy={json.dumps(self.accuracy)}"
            " reference=directed_mpfr_4.2.2_stored_inputs_f64_2steps_f32_4steps_half_f32_neighbors4 wrapper=stored_rne_schedule"
            " scalar=original_variants_public_float_f64 tensor_powi=int_into_float_then_pow nan=class_identity_alias_exact_bytes"
            f" chain=materialized_each_primitive {self.scope}"
        )

    def test_linux_native_framing_and_strict_serialization(self):
        self.install()

        def validate(text, os="linux", backend="flex_cpu"):
            return evidence.validate(
                text,
                self.summary,
                self.manifest,
                self.references,
                self.limits,
                execution_os=os,
                backend=backend,
                fusion="disabled",
                scope=self.scope,
            )

        marker = self.marker()
        self.assertEqual(validate(marker), self.accuracy)
        self.assertEqual(
            validate(
                "\x1b[36m[1.2][INF][infer.rs@50] \x1b[0m"
                + self.marker("d3os", "native"),
                "d3os",
                "native",
            ),
            self.accuracy,
        )
        for bad in (
            marker + "\n" + marker,
            marker.replace('"failures": 0', '"failures": 0, "failures": 0', 1),
            marker.replace('"max_ulp": 0', '"max_ulp": NaN', 1),
            marker.replace("qualification_passed=true", "qualification_passed=false"),
            marker.replace("half_f32_neighbors4", "half_f32_neighbors8"),
        ):
            with self.subTest(marker=bad[:100]), self.assertRaises(
                (RuntimeError, ValueError)
            ):
                validate(bad)


if __name__ == "__main__":
    unittest.main()
