#!/usr/bin/env python3
"""Bounded host-only Erf evidence tests. No backend process is launched."""
import copy
import importlib.util
import itertools
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("erf_evidence", Path(__file__).with_name("typed-erf-evidence.py"))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".cache/d3os/burn-typed-erf-evidence-tests"

# Pure work descriptions from the frozen public --expected-summary route.
# These contain no observed values, histograms, maxima or success claims.
PUBLIC_DEFAULT = {'schema_version': 3,
 'policy_sha256': '5d67c74ea3d9404264eacaac819693bad43e2fa0a953b56bfdbdd0bd310af11c',
 'fixture_sha256': '661fec5a11dabfd9969f356dbbd38d8025b3704793f39d42e49f91eecab5a742',
 'options': {'no_flex32': False,
             'skip_rank_zero': False,
             'finite': False,
             'dtype': 'all',
             'op': 'all'},
 'dtypes': 5,
 'operations': 5,
 'cases': 680,
 'numerical_cases': 10,
 'layout_cases': 600,
 'public_cases': 305,
 'chain_cases': 70,
 'chain_steps': 210,
 'tensor_rows': 230618,
 'selected_dtype_operation_counts': [[48362], [25592], [25592], [65536], [65536]],
 'expected_value_counts': [[106486], [60946], [60946], [140834], [140834]],
 'values': 510046,
 'expected_staged_class_counts': [[[76, 9732, 96604, 0, 32]],
                                  [[76, 10360, 50436, 0, 32]],
                                  [[76, 10360, 50436, 0, 32]],
                                  [[76, 3838, 132448, 0, 4430]],
                                  [[76, 496, 139706, 0, 514]]],
 'expected_ideal_class_counts': [[[76, 9732, 96604, 0, 32]],
                                 [[76, 10360, 50436, 0, 32]],
                                 [[76, 10360, 50436, 0, 32]],
                                 [[76, 3838, 132448, 0, 4430]],
                                 [[76, 496, 139706, 0, 514]]],
 'chain_value_counts': [[42], [42], [42], [42], [42]],
 'expected_oddness_checks': [[48344], [25574], [25574], [63488], [65280]],
 'source_checks': 680,
 'alias_checks': 890,
 'poison_checks': 680,
 'metadata_checks': 3070,
 'empty_cases': 130}

FINITE_F32 = {'schema_version': 3,
 'policy_sha256': '5d67c74ea3d9404264eacaac819693bad43e2fa0a953b56bfdbdd0bd310af11c',
 'fixture_sha256': '661fec5a11dabfd9969f356dbbd38d8025b3704793f39d42e49f91eecab5a742',
 'options': {'no_flex32': False,
             'skip_rank_zero': True,
             'finite': True,
             'dtype': 'f32',
             'op': 'all'},
 'dtypes': 1,
 'operations': 1,
 'cases': 130,
 'numerical_cases': 2,
 'layout_cases': 118,
 'public_cases': 60,
 'chain_cases': 10,
 'chain_steps': 30,
 'tensor_rows': 25574,
 'selected_dtype_operation_counts': [[0], [25574], [0], [0], [0]],
 'expected_value_counts': [[0], [60896], [0], [0], [0]],
 'values': 60896,
 'expected_staged_class_counts': [[[0, 0, 0, 0, 0]],
                                  [[74, 10348, 50444, 0, 0]],
                                  [[0, 0, 0, 0, 0]],
                                  [[0, 0, 0, 0, 0]],
                                  [[0, 0, 0, 0, 0]]],
 'expected_ideal_class_counts': [[[0, 0, 0, 0, 0]],
                                 [[74, 10348, 50444, 0, 0]],
                                 [[0, 0, 0, 0, 0]],
                                 [[0, 0, 0, 0, 0]],
                                 [[0, 0, 0, 0, 0]]],
 'chain_value_counts': [[0], [30], [0], [0], [0]],
 'expected_oddness_checks': [[0], [25574], [0], [0], [0]],
 'source_checks': 130,
 'alias_checks': 160,
 'poison_checks': 130,
 'metadata_checks': 570,
 'empty_cases': 26}

def selection(fi=None):
    result = copy.deepcopy(PUBLIC_DEFAULT)
    if fi is None:
        return result
    result["options"]["dtype"] = e.DTYPES[fi]
    for key in ("selected_dtype_operation_counts", "expected_value_counts", "expected_staged_class_counts", "expected_ideal_class_counts", "chain_value_counts", "expected_oddness_checks"):
        result[key] = [copy.deepcopy(value) if i == fi else ([[0] * 5] if key in ("expected_staged_class_counts", "expected_ideal_class_counts") else [0]) for i, value in enumerate(result[key])]
    for key in ("cases", "numerical_cases", "layout_cases", "public_cases", "chain_cases", "chain_steps", "source_checks", "alias_checks", "poison_checks", "metadata_checks", "empty_cases", "dtypes", "operations"):
        result[key] //= 5
    result["values"] = result["expected_value_counts"][fi][0]
    result["tensor_rows"] = result["selected_dtype_operation_counts"][fi][0]
    return result


def synthetic_accuracy(expected, bounds, raw=False):
    """Comparator observations at immutable staged references, not GPU evidence."""
    contexts = e._raw_context(expected, bounds) if raw else e._shared_context(expected, bounds)
    oddness = expected["oddness_checks"] if raw else expected["expected_oddness_checks"]
    cells = []
    for fi, context in enumerate(contexts):
        cell = dict.fromkeys(e.COUNTERS, 0)
        cell.update(dtype=e.DTYPES[fi], op="erf", observed_classes=context["staged"].copy(), unreferenced_observed_classes=[0] * 5,
                    transitions=[[0] * 5 for _ in range(5)], ideal_transitions=[[0] * 5 for _ in range(5)],
                    histogram=[0] * 7, passing_histogram=[0] * 7, ideal_histogram=[0] * 7,
                    max_ulp=None, max_ideal_ulp=None, passing_max_ulp=None,
                    worst_input_bits=None, worst_actual_bits=None, worst_ideal_input_bits=None, worst_ideal_actual_bits=None)
        cell["values"] = sum(context["staged"])
        cell["oddness_checks"] = oddness[fi][0]
        cell["quiet_nan_outputs"] = context["staged"][4]
        for cls, count in enumerate(context["staged"]):
            cell["transitions"][cls][cls] = count
            cell["ideal_transitions"][cls][cls] = count
        cell["subnormal_preserved"] = context["staged"][1]
        finite = context["staged"][1] + context["staged"][2]
        cell["finite_pairs"] = cell["ideal_finite_pairs"] = cell["passing_finite_pairs"] = finite
        if finite:
            cell["histogram"][0] = cell["passing_histogram"][0] = finite
            for distance, count in context["reference_deltas"].items():
                cell["ideal_histogram"][e.bucket(distance)] += count
            cell["max_ulp"] = cell["passing_max_ulp"] = 0
            cell["max_ideal_ulp"] = max(context["reference_deltas"])
            eligible = [row for b, row in bounds[fi]["rows"].items() if b in context["selected_inputs"] and e.classify(fi, row.staged) in (1, 2)]
            for prefix, distance in (("worst", 0), ("worst_ideal", cell["max_ideal_ulp"])):
                row = next(r for r in eligible if e.distance(fi, r.staged, r.ideal if prefix == "worst_ideal" else r.staged) == distance)
                cell[prefix + "_input_bits"] = f"{row.input:016x}"
                cell[prefix + "_actual_bits"] = f"{row.staged:016x}"
        cells.append(cell)
    return {"failures": 0, "cells": cells}


def public_serial(expected, accuracy, platform=("linux", "flex_cpu", "disabled")):
    os_name, backend, fusion = platform
    common = f"os={os_name} backend={backend} fusion={fusion} corpus=1"
    data = e.canonical(accuracy)
    totals = e.canonical(expected)
    header = f"BURN_TYPED_ERF_ACCURACY {common} policy_sha256={e.POLICY_SHA256} fixture_sha256={e.FIXTURE_SHA256} qualification_passed=true totals={totals} accuracy={data}"
    footer = f"BURN_TYPED_ERF_OK {common} totals={totals} accuracy={data} policy_sha256={e.POLICY_SHA256} fixture_sha256={e.FIXTURE_SHA256} numerical_policy=project_qualification_targets staged_half=f32_neighbor_envelope no_ftz=strict preserved_inputs=exact_bytes chain=materialized_known_refs poison=retained_allocation scope=forward_erf_only {e.option_scope(expected)}"
    return header + "\n" + footer + "\n"


def raw_serial(expected, accuracy):
    counts = e._raw_counts(expected)
    routes = json.dumps(list(e.ROUTES))
    rc = json.dumps(expected["route_counts"])
    line = f"NVIDIA_BURN_TYPED_ERF_ACCURACY policy_sha256={e.POLICY_SHA256} fixture_sha256={e.FIXTURE_SHA256} frozen=true qualification_passed=true coherent=true comparisons={expected['comparisons']} counts={counts} route_names={routes} route_counts={rc} accuracy={e.canonical(accuracy)}"
    c = expected["counts"]
    footer = f"NVIDIA_BURN_TYPED_ERF_OK operations=1 formats=5 corpus_rows={c['corpus_rows']} corpus_batches={c['corpus_batches']} chunk=4096 raw_poison_variants=2 comparisons={expected['comparisons']} rows={json.dumps(expected['row_counts'])} counts={counts} route_names={routes} route_counts={rc} oddness_checks={json.dumps(expected['oddness_checks'])} policy_sha256={e.POLICY_SHA256} fixture_sha256={e.FIXTURE_SHA256} corpus_version=1 sdk_operations=25 protocol=50 profiles=82 kernel=81 guards=ok tails=zero raw_poison_independence=exact_bytes source_nan_payloads=preserved resident=true full_writer=verified raw_f32=FloatArithmetic native_f32=TensorElementwise queued_last_owner=verified errors=recoverable qualification=project_targets steps=2 half=f32_neighbor_image no_ftz=strict oddness=exact_sign_xor scope=forward_erf_only linux_parity=not_claimed backend=nvidia"
    return line + "\n" + footer + "\n"


RAW_DEFAULT = {'counts': {'corpus_rows': 230618,
            'corpus_batches': 58,
            'geometry': 200,
            'raw': 516,
            'fresh': 526,
            'native': 405,
            'public': 605,
            'rejected': 541,
            'empty': 100,
            'queued': 20,
            'f32_routes': 202,
            'typed_routes': 808},
 'row_counts': [[48362], [25592], [25592], [65536], [65536]],
 'cell_values': [[225988], [134908], [134908], [294684], [294684]],
 'comparisons': 1085172,
 'staged_classes': [[[3279, 19900, 198011, 0, 4798]],
                    [[3279, 21209, 106432, 0, 3988]],
                    [[3279, 21209, 106432, 0, 3988]],
                    [[3279, 8706, 269781, 0, 12918]],
                    [[3279, 2298, 283357, 0, 5750]]],
 'ideal_classes': [[[3279, 19900, 198011, 0, 4798]],
                   [[3279, 21209, 106432, 0, 3988]],
                   [[3279, 21209, 106432, 0, 3988]],
                   [[3279, 8706, 269781, 0, 12918]],
                   [[3279, 2298, 283357, 0, 5750]]],
 'oddness_checks': [[96994], [51454], [51454], [127282], [130866]],
 'full_finite_pairs': [24172, 12787, 12787, 31744, 32640],
 'route_counts': [[104, 104, 40, 40, 40, 40, 40, 1, 2, 1],
                  [94, 94, 40, 40, 40, 40, 40, 1, 2, 1],
                  [94, 94, 40, 40, 40, 40, 40, 1, 2, 1],
                  [112, 112, 40, 40, 40, 40, 40, 1, 2, 1],
                  [112, 112, 40, 40, 40, 40, 40, 1, 2, 1]],
 'policy_sha256': '5d67c74ea3d9404264eacaac819693bad43e2fa0a953b56bfdbdd0bd310af11c',
 'fixture_sha256': '661fec5a11dabfd9969f356dbbd38d8025b3704793f39d42e49f91eecab5a742'}


FINITE_ALL = {'schema_version': 3,
 'policy_sha256': '5d67c74ea3d9404264eacaac819693bad43e2fa0a953b56bfdbdd0bd310af11c',
 'fixture_sha256': '661fec5a11dabfd9969f356dbbd38d8025b3704793f39d42e49f91eecab5a742',
 'options': {'no_flex32': False,
             'skip_rank_zero': False,
             'finite': True,
             'dtype': 'all',
             'op': 'all'},
 'dtypes': 5,
 'operations': 5,
 'cases': 660,
 'numerical_cases': 10,
 'layout_cases': 600,
 'public_cases': 305,
 'chain_cases': 50,
 'chain_steps': 150,
 'tensor_rows': 228260,
 'selected_dtype_operation_counts': [[48344], [25574], [25574], [63488], [65280]],
 'expected_value_counts': [[106438], [60898], [60898], [136726], [140310]],
 'values': 505270,
 'expected_staged_class_counts': [[[76, 9738, 96594, 0, 0]],
                                  [[76, 10348, 50444, 0, 0]],
                                  [[76, 10348, 50444, 0, 0]],
                                  [[76, 3862, 132758, 0, 0]],
                                  [[76, 448, 139756, 0, 0]]],
 'expected_ideal_class_counts': [[[76, 9738, 96594, 0, 0]],
                                 [[76, 10348, 50444, 0, 0]],
                                 [[76, 10348, 50444, 0, 0]],
                                 [[76, 3862, 132758, 0, 0]],
                                 [[76, 448, 139756, 0, 0]]],
 'chain_value_counts': [[30], [30], [30], [30], [30]],
 'expected_oddness_checks': [[48344], [25574], [25574], [63488], [65280]],
 'source_checks': 660,
 'alias_checks': 810,
 'poison_checks': 660,
 'metadata_checks': 2890,
 'empty_cases': 130}


def filtered_selection(fi=None, *, finite=False, no_flex32=False, skip_rank_zero=False, op="all"):
    """Compose pure filter descriptions from retained independent public plans."""
    plan = copy.deepcopy(FINITE_ALL if finite else PUBLIC_DEFAULT)
    kept = [i for i in range(5) if (fi is None or fi == i) and not (no_flex32 and i == 2)]
    if skip_rank_zero:
        for i in range(5):
            plan["expected_value_counts"][i][0] -= 2
            plan["expected_staged_class_counts"][i][0][0] -= 2
            plan["expected_ideal_class_counts"][i][0][0] -= 2
        for key, amount in (("cases", 10), ("layout_cases", 10), ("public_cases", 5), ("source_checks", 10), ("alias_checks", 10), ("poison_checks", 10), ("metadata_checks", 40)):
            plan[key] -= amount
    for key in ("selected_dtype_operation_counts", "expected_value_counts", "expected_staged_class_counts", "expected_ideal_class_counts", "chain_value_counts", "expected_oddness_checks"):
        plan[key] = [value if i in kept else ([[0] * 5] if key in ("expected_staged_class_counts", "expected_ideal_class_counts") else [0]) for i, value in enumerate(plan[key])]
    for key in ("cases", "numerical_cases", "layout_cases", "public_cases", "chain_cases", "chain_steps", "source_checks", "alias_checks", "poison_checks", "metadata_checks", "empty_cases", "dtypes", "operations"):
        plan[key] = plan[key] // 5 * len(kept)
    plan["values"] = sum(row[0] for row in plan["expected_value_counts"])
    plan["tensor_rows"] = sum(row[0] for row in plan["selected_dtype_operation_counts"])
    plan["options"] = {"no_flex32": no_flex32, "skip_rank_zero": skip_rank_zero, "finite": finite, "dtype": "all" if fi is None else e.DTYPES[fi], "op": op}
    return plan


def change_observation(accuracy, fi, row, actual):
    """Move one genuine staged-reference observation to an admitted output."""
    assert e.passing(fi, row, actual)
    cell = accuracy["cells"][fi]
    old_class, new_class = e.classify(fi, row.staged), e.classify(fi, actual)
    cell["observed_classes"][old_class] -= 1
    cell["observed_classes"][new_class] += 1
    for mkey, hkey, ref, maximum, prefix in (("transitions", "histogram", row.staged, "max_ulp", "worst"), ("ideal_transitions", "ideal_histogram", row.ideal, "max_ideal_ulp", "worst_ideal")):
        rc = e.classify(fi, ref)
        cell[mkey][rc][old_class] -= 1
        cell[mkey][rc][new_class] += 1
        old_distance = e.distance(fi, row.staged, ref)
        new_distance = e.distance(fi, actual, ref)
        cell[hkey][e.bucket(old_distance)] -= 1
        cell[hkey][e.bucket(new_distance)] += 1
        if new_distance > cell[maximum]:
            cell[maximum] = new_distance
            cell[prefix + "_input_bits"] = f"{row.input:016x}"
            cell[prefix + "_actual_bits"] = f"{actual:016x}"
    cell["passing_histogram"] = cell["histogram"].copy()
    cell["passing_max_ulp"] = cell["max_ulp"]
    cell["subnormal_preserved"] = cell["transitions"][1][1]


class ErfEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.bounds = e.load_bounds(ROOT / "tools/nvidia/typed-erf-manifest.json")
        cls.default = PUBLIC_DEFAULT
        cls.all_accuracy = synthetic_accuracy(cls.default, cls.bounds)
        cls.f16 = selection(3)
        cls.f16_accuracy = synthetic_accuracy(cls.f16, cls.bounds)
        cls.f32 = FINITE_F32
        cls.f32_accuracy = synthetic_accuracy(cls.f32, cls.bounds)
        CACHE.mkdir(parents=True, exist_ok=True)

    def check(self, accuracy=None, expected=None):
        expected = self.f32 if expected is None else expected
        accuracy = self.f32_accuracy if accuracy is None else accuracy
        return e.validate_accuracy(accuracy, e.validate_plan(expected), expected["values"], self.bounds,
                                   reference_plan=e.summary_reference_plan(expected), oddness_expected=expected["expected_oddness_checks"])

    def check_case(self, accuracy, expected, raw):
        if raw:
            return e.validate_raw(raw_serial(expected, accuracy), expected, self.manifest, self.bounds)
        return self.check(accuracy, expected)

    def reject(self, mutate, accuracy=None, expected=None):
        bad = copy.deepcopy(self.f32_accuracy if accuracy is None else accuracy)
        mutate(bad)
        with self.assertRaises(RuntimeError):
            self.check(bad, expected)

    def test_frozen_rows_images_closure_and_direct_staged_metrics(self):
        self.assertEqual(sum(len(x["rows"]) for x in self.bounds), 230618)
        self.assertEqual([(x["staged"], x["ideal"]) for x in self.bounds], [(2, 2), (2, 2), (2, 2), (1, 1), (0, 0)])
        self.assertEqual(self.check(self.all_accuracy, self.default), self.all_accuracy)
        for source in (0x1612, 0x9612):
            row = self.bounds[3]["rows"][source]
            self.assertNotEqual(row.ideal, row.staged)
            self.assertEqual(e.distance(3, row.ideal, row.staged), 1)
        context = e._shared_context(self.f16, self.bounds)[3]
        self.assertGreaterEqual(context["reference_deltas"][1], 4)
        self.assertEqual(e.canonical(self.manifest), e.canonical(e.strict_json((ROOT / "tools/nvidia/typed-erf-manifest.json").read_text())))

    def test_counter_types_failed_gates_and_oddness_cannot_hide_in_aggregate(self):
        for key in e.FAILURES:
            with self.subTest(key=key):
                self.reject(lambda a, key=key: a["cells"][1].__setitem__(key, 1))
        for key in e.COUNTERS:
            for value in (-1, True, 0.0):
                with self.subTest(key=key, value=value):
                    self.reject(lambda a, key=key, value=value: a["cells"][1].__setitem__(key, value))
        for key in ("oddness_checks", "values", "finite_pairs", "ideal_finite_pairs", "passing_finite_pairs"):
            self.reject(lambda a, key=key: a["cells"][1].__setitem__(key, a["cells"][1][key] - 1))
        self.reject(lambda a: a.__setitem__("failures", 1))
        self.reject(lambda a: a["cells"][1].__setitem__("oddness_checks", 0))

    def test_keys_excluded_zero_cells_and_histograms_bind(self):
        self.reject(lambda a: a["cells"].reverse())
        self.reject(lambda a: a["cells"].pop())
        self.reject(lambda a: a["cells"].append(copy.deepcopy(a["cells"][1])))
        self.reject(lambda a: a["cells"][1].__setitem__("op", "power"))
        self.reject(lambda a: a["cells"][1].__setitem__("unplanned", 0))
        for index in (0, 2, 3, 4):
            self.reject(lambda a, index=index: a["cells"][index].__setitem__("max_ulp", 0))
            self.reject(lambda a, index=index: a["cells"][index].__setitem__("values", 1))
        for key in ("histogram", "passing_histogram", "ideal_histogram"):
            for index in (0, 6):
                self.reject(lambda a, key=key, index=index: a["cells"][1][key].__setitem__(index, a["cells"][1][key][index] + 1))
        for key in e.MAXIMA:
            for value in (None, True, 3):
                self.reject(lambda a, key=key, value=value: a["cells"][1].__setitem__(key, value))

    def test_row_and_column_class_populations_and_chain_classes_are_exact(self):
        for key in ("transitions", "ideal_transitions"):
            self.reject(lambda a, key=key: a["cells"][1][key][2].__setitem__(2, a["cells"][1][key][2][2] + 1))
            def move_row(a, key=key):
                a["cells"][1][key][1][1] -= 1
                a["cells"][1][key][2][1] += 1
            self.reject(move_row)
        # Every internal class sum can remain coherent while every immutable
        # reference class is relabeled. Frozen closure forbids this escape.
        forged = copy.deepcopy(self.all_accuracy)
        for cell in forged["cells"]:
            n = cell["values"]
            for k in e.COUNTERS:
                cell[k] = 0
            cell["values"] = n
            fi = e.DTYPES.index(cell["dtype"])
            cell["oddness_checks"] = self.default["expected_oddness_checks"][fi][0]
            cell["observed_classes"] = [0, 0, 0, 0, n]
            cell["transitions"] = cell["ideal_transitions"] = [[0] * 5 for _ in range(5)]
            cell["transitions"][4][4] = n
            for k in ("histogram", "passing_histogram", "ideal_histogram"):
                cell[k] = [0] * 7
            for k in (*e.MAXIMA, *e.WITNESSES):
                cell[k] = None
        with self.assertRaises(RuntimeError):
            self.check(forged, self.default)
        def relabel_chains(a):
            cell = a["cells"][1]
            for key in ("transitions", "ideal_transitions"):
                cell[key][2][2] -= 1
                cell[key][4][2] += 1
        self.reject(relabel_chains)

    def test_witness_domain_class_sign_zero_inf_codomain_and_both_metric_checks(self):
        for key in e.WITNESSES:
            for value in (None, "0", "f" * 16, "0x0000000000000001", 1, True):
                self.reject(lambda a, key=key, value=value: a["cells"][1].__setitem__(key, value))
        for bits in (0, 0x80000000, 0x7F800000, 0x7FC00000, 0x3F800001, 0xBF000000):
            self.reject(lambda a, bits=bits: a["cells"][1].__setitem__("worst_actual_bits", f"{bits:016x}"))
        # Infinity input has an exact finite output and a finite-pair distance,
        # but is excluded by --finite even when both metric tuples are forged.
        def excluded(a):
            for prefix in ("worst", "worst_ideal"):
                a["cells"][1][prefix + "_input_bits"] = "000000007f800000"
                a["cells"][1][prefix + "_actual_bits"] = "000000003f800000"
        self.reject(excluded)
        # A witness of one maximum must also fit the other histogram. Equal
        # classes alone cannot establish compatibility of distinct references.
        bad = copy.deepcopy(self.f16_accuracy)
        cell = bad["cells"][3]
        self.assertEqual((cell["max_ulp"], cell["max_ideal_ulp"]), (0, 1))
        cell["ideal_histogram"] = [0, cell["ideal_finite_pairs"], 0, 0, 0, 0, 0]
        with self.assertRaises(RuntimeError):
            self.check(bad, self.f16)
        bad = copy.deepcopy(self.f16_accuracy)
        bad["cells"][3]["worst_ideal_actual_bits"] = bad["cells"][3]["worst_actual_bits"]
        with self.assertRaises(RuntimeError):
            self.check(bad, self.f16)

    def test_half_joint_reference_histograms_and_nan_diagnostics_cannot_be_relabelled(self):
        for key in ("reference_class_disagreements", "reference_nonzero_finite_disagreements", "subnormal_preserved"):
            self.reject(lambda a, key=key: a["cells"][3].__setitem__(key, a["cells"][3][key] + 1), self.f16_accuracy, self.f16)
        for key in ("quiet_nan_outputs", "negative_nan_outputs", "nan_payload_differences"):
            self.reject(lambda a, key=key: a["cells"][3].__setitem__(key, a["cells"][3]["observed_classes"][4] + 1), self.f16_accuracy, self.f16)
        bad = copy.deepcopy(self.f16_accuracy)
        cell = bad["cells"][3]
        cell["ideal_histogram"][0] += cell["ideal_histogram"][1] - 1
        cell["ideal_histogram"][1] = 1
        with self.assertRaises(RuntimeError):
            self.check(bad, self.f16)
        bad = copy.deepcopy(self.f16_accuracy)
        bad["cells"][3]["unreferenced_observed_classes"][4] = 1
        with self.assertRaises(RuntimeError):
            self.check(bad, self.f16)

    def test_pure_plan_rejects_observations_bool_counts_or_coherent_forged_populations(self):
        for key, value in (("max_ulp", 0), ("values", True), ("schema_version", 20)):
            bad = copy.deepcopy(self.f32)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                e.summary_reference_plan(bad)
        bad = copy.deepcopy(self.f32)
        for key in ("expected_staged_class_counts", "expected_ideal_class_counts"):
            bad[key][1][0][1] -= 1
            bad[key][1][0][2] += 1
        plan = e.summary_reference_plan(bad)
        with self.assertRaises(RuntimeError):
            e.validate_accuracy(self.f32_accuracy, e.validate_plan(bad), bad["values"], self.bounds, reference_plan=plan, oddness_expected=bad["expected_oddness_checks"])

    def test_exact_platform_scope_footer_and_single_marker_contract(self):
        serial = public_serial(self.f32, self.f32_accuracy)
        validate = lambda text, **kw: e.validate(text, self.f32, self.manifest, self.bounds, execution_os=kw.get("execution_os", "linux"), backend=kw.get("backend", "flex_cpu"), fusion=kw.get("fusion", "disabled"), scope=kw.get("scope"))
        self.assertEqual(validate(serial), self.f32_accuracy)
        # Native log prefixes and SGR escapes are supported only for d3os.
        native = public_serial(self.f32, self.f32_accuracy, ("d3os", "native", "disabled"))
        native = "\n".join("\x1b[32m[infer] [gpu] " + line + "\x1b[0m" for line in native.splitlines())
        self.assertEqual(validate(native, execution_os="d3os", backend="native"), self.f32_accuracy)
        for text in (
            serial + serial,
            serial.splitlines()[1] + "\n",
            " " + serial.splitlines()[0] + "\n" + serial.splitlines()[1] + "\n",
            "prefix " + serial.splitlines()[0] + "\n" + serial,
            serial + serial.splitlines()[1] + "\n",
            serial.replace("qualification_passed=true", "qualification_passed=false"),
            serial.replace("preserved_inputs=exact_bytes", "preserved_inputs=canonical_nan"),
            serial.replace("no_ftz=strict", "no_ftz=relaxed"),
            serial.replace("scope=forward_erf_only", "scope=full_burn"),
            serial.replace("skip_rank_zero=true", "skip_rank_zero=false"),
            serial.replace("fixture_sha256=" + e.FIXTURE_SHA256, "fixture_sha256=" + "0" * 64),
            serial + "BURN_TYPED_ERF_FAILURE observed failure\n",
            serial.rstrip() + " unplanned=true\n",
        ):
            with self.subTest(size=len(text)), self.assertRaises(RuntimeError):
                validate(text)
        for kwargs in ({"backend": "cuda"}, {"fusion": "enabled"}, {"execution_os": "d3os", "backend": "native"}, {"scope": "finite=false"}):
            with self.assertRaises(RuntimeError):
                validate(serial, **kwargs)
        # A self-consistent footer cannot hide contradictory preceding accuracy.
        header, footer = serial.splitlines()
        bad_accuracy = copy.deepcopy(self.f32_accuracy)
        bad_accuracy["cells"][1]["oddness_failures"] = 1
        wrong_header = header.replace(e.canonical(self.f32_accuracy), e.canonical(bad_accuracy))
        with self.assertRaises(RuntimeError):
            validate(wrong_header + "\n" + footer)

    def test_strict_json_rejects_duplicates_nonfinite_exponents_and_wrong_counter_types(self):
        serial = public_serial(self.f32, self.f32_accuracy)
        validate = lambda text: e.validate(text, self.f32, self.manifest, self.bounds, execution_os="linux", backend="flex_cpu", fusion="disabled")
        duplicate = serial.replace('"failures":0', '"failures":0,"failures":0', 1)
        with self.assertRaises(RuntimeError):
            validate(duplicate)
        for value in (float("nan"), float("inf"), -float("inf")):
            bad = copy.deepcopy(self.f32_accuracy)
            bad["cells"][1]["max_ulp"] = value
            payload = json.dumps(bad, separators=(",", ":"))
            text = serial.replace(e.canonical(self.f32_accuracy), payload)
            with self.assertRaises(RuntimeError):
                validate(text)
        for exponent in ("1e9999", "-1e9999"):
            with self.assertRaises(RuntimeError):
                validate(serial.replace('"max_ulp":null', '"max_ulp":' + exponent, 1))
        with self.assertRaises(RuntimeError):
            e.strict_json('{"nested":{"counter":1,"counter":2}}')

    def test_raw_geometry_plan_routes_oddness_and_classes_use_no_shared_slack(self):
        expected = copy.deepcopy(RAW_DEFAULT)
        accuracy = synthetic_accuracy(expected, self.bounds, raw=True)
        serial = raw_serial(expected, accuracy)
        self.assertEqual(e.validate_raw(serial, expected, self.manifest, self.bounds), accuracy)
        plan = e.raw_reference_plan(expected)
        for fi in range(5):
            bad = copy.deepcopy(accuracy)
            bad["cells"][fi]["oddness_checks"] -= 1
            with self.assertRaises(RuntimeError):
                e.validate_accuracy(bad, expected["cell_values"], expected["comparisons"], self.bounds, reference_plan=plan, oddness_expected=expected["oddness_checks"])
        for key in ("cell_values", "oddness_checks", "route_counts", "staged_classes", "ideal_classes"):
            bad = copy.deepcopy(expected)
            if key in ("staged_classes", "ideal_classes"):
                bad[key][0][0][0] += 1
            else:
                bad[key][0][0] += 1
            with self.assertRaises(RuntimeError):
                e.raw_reference_plan(bad)
        for text in (
            serial.replace("qualification_passed=true", "qualification_passed=false"),
            serial.replace("coherent=true", "coherent=false"),
            serial.replace("source_nan_payloads=preserved", "source_nan_payloads=canonicalized"),
            serial.replace("raw_poison_independence=exact_bytes", "raw_poison_independence=unchecked"),
            serial.replace("protocol=50", "protocol=49"),
            serial.replace("kernel=81", "kernel=82"),
            serial.replace('"Native.float_unary"', '"untracked"'),
            serial + "NVIDIA_BURN_TYPED_ERF_FAILURE observed failure\n",
        ):
            with self.assertRaises(RuntimeError):
                e.validate_raw(text, expected, self.manifest, self.bounds)
        with self.assertRaises(RuntimeError):
            e.validate_raw(serial, expected, self.manifest, self.bounds, execution_os="linux")

    def test_append_only_protocol_requires_explicit_matching_profile_identity(self):
        expected = copy.deepcopy(RAW_DEFAULT)
        accuracy = synthetic_accuracy(expected, self.bounds, raw=True)
        old = raw_serial(expected, accuracy)
        current = old.replace("protocol=50 profiles=82", "protocol=51 profiles=83")
        self.assertEqual(e.validate_raw(current, expected, self.manifest, self.bounds, protocol=51, profiles=83), accuracy)
        for text, protocol, profiles in ((old, 51, 83), (current, 50, 82), (current, 51, 82), (current, 50, 83), (current, 52, 84), (current, True, 83)):
            with self.subTest(protocol=protocol, profiles=profiles), self.assertRaises(RuntimeError):
                e.validate_raw(text, expected, self.manifest, self.bounds, protocol=protocol, profiles=profiles)

    def test_reference_special_rules_allow_nan_output_payloads_but_keep_sources_exact(self):
        for fi in range(5):
            for source in (0, e.sign(fi), e.EXPONENTS[fi], e.EXPONENTS[fi] | e.sign(fi)):
                row = self.bounds[fi]["rows"][source]
                self.assertTrue(e.passing(fi, row, row.staged))
                self.assertFalse(e.passing(fi, row, row.staged ^ e.sign(fi)))
            nan = self.bounds[fi]["rows"][e.EXPONENTS[fi] | 1]
            self.assertTrue(e.passing(fi, nan, e.EXPONENTS[fi] | e.sign(fi) | 2))
            self.assertFalse(e.passing(fi, nan, e.ONES[fi]))
            tiny = self.bounds[fi]["rows"][1]
            self.assertFalse(e.passing(fi, tiny, 0))
            self.assertFalse(e.passing(fi, tiny, tiny.staged ^ e.sign(fi)))
            at_unit = self.bounds[fi]["rows"][e.EXPONENTS[fi] - 1]
            self.assertFalse(e.passing(fi, at_unit, e.ONES[fi] + 1))
        row = next(r for r in self.bounds[1]["rows"].values() if e.classify(1, r.staged) == 1 and e.classify(1, r.staged + 2) == 2)
        self.assertTrue(e.passing(1, row, row.staged + 2))
        self.assertEqual(e.convert(0x3ADB3000, 1, 3), 0x16DA)
        for fi in (3, 4):
            for b in range(65536):
                if e.classify(fi, b) <= 2:
                    self.assertEqual(e.convert(e.convert(b, fi, 1), 1, fi), b)

    def test_manifest_policy_size_binary_hash_and_index_are_frozen(self):
        with tempfile.TemporaryDirectory(prefix="burn-typed-erf-evidence-", dir=CACHE) as temporary:
            root = Path(temporary)
            files = ("tools/nvidia/typed-erf-manifest.json", "tools/nvidia/typed-erf-generate.py", "os/library/ai/src/validation/typed_erf_fixtures.rs", "os/library/ai/src/validation/typed_erf_fixtures.bin")
            for name in files:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            manifest_path = root / files[0]
            binary = root / files[3]
            original = binary.read_bytes()
            for data in (original[:-1], original + b"\0", bytes([original[0] ^ 1]) + original[1:]):
                binary.write_bytes(data)
                with self.assertRaises(RuntimeError):
                    e.load_bounds(manifest_path)
            binary.write_bytes(original)
            index = root / files[2]
            index.write_bytes(index.read_bytes() + b"\n")
            with self.assertRaises(RuntimeError):
                e.load_bounds(manifest_path)
            shutil.copyfile(ROOT / files[2], index)
            changed = copy.deepcopy(self.manifest)
            changed["policy"]["f64_f32_flex32_ordered_steps"] += 1
            manifest_path.write_text(json.dumps(changed))
            with self.assertRaises(RuntimeError):
                e.load_bounds(manifest_path)

    def test_loaded_manifest_cannot_be_replaced_by_two_matching_hash_fields(self):
        serial = public_serial(self.f32, self.f32_accuracy)
        for key, value in (("corpus_version", 2), ("fixture_bytes", 0), ("scope", "full_burn")):
            changed = copy.deepcopy(self.manifest)
            changed[key] = value
            with self.assertRaises(RuntimeError):
                e.validate(serial, self.f32, changed, self.bounds, execution_os="linux", backend="flex_cpu", fusion="disabled")

    def test_actual_cpu_public_record_when_available(self):
        record = Path(os.environ.get("TYPED_ERF_CPU_RECORD", Path.home() / ".cache/d3os/burn-typed-erf-evidence-work/cpu-public.txt"))
        if not record.is_file():
            self.skipTest("CPU public capture unavailable; test never launches a backend")
        accuracy = e.validate(record.read_text(), FINITE_F32, self.manifest, self.bounds, execution_os="linux", backend="flex_cpu", fusion="disabled")
        self.assertEqual(accuracy["cells"][1]["values"], FINITE_F32["values"])
        self.assertEqual(accuracy["failures"], 0)


    def test_r2_exact_zero_distance_class_changes_are_rejected_public_and_raw(self):
        cases = ((FINITE_F32, 1, False), (selection(4), 4, False), (RAW_DEFAULT, 4, True))
        for expected, fi, raw in cases:
            accuracy = synthetic_accuracy(expected, self.bounds, raw=raw)
            cell = accuracy["cells"][fi]
            for key in ("transitions", "ideal_transitions"):
                cell[key][1][1] -= 1
                cell[key][1][2] += 1
                cell[key][2][2] -= 1
                cell[key][2][1] += 1
            cell["subnormal_preserved"] -= 1
            self.assertEqual((cell["max_ulp"], cell["max_ideal_ulp"]), (0, 0))
            with self.subTest(dtype=e.DTYPES[fi], raw=raw), self.assertRaises(RuntimeError):
                if raw:
                    e.validate_raw(raw_serial(expected, accuracy), expected, self.manifest, self.bounds)
                else:
                    e.validate(public_serial(expected, accuracy), expected, self.manifest, self.bounds, execution_os="linux", backend="flex_cpu", fusion="disabled")

    def test_r2_exact_public_and_raw_half_histogram_capacity_repros_are_rejected(self):
        for expected, raw, changed, repetitions in ((selection(3), False, 20000, 2), (RAW_DEFAULT, True, 40000, 4)):
            accuracy = synthetic_accuracy(expected, self.bounds, raw=raw)
            cell = accuracy["cells"][3]
            for key in ("histogram", "passing_histogram", "ideal_histogram"):
                cell[key][0] -= changed
                cell[key][1] += changed
            cell["max_ulp"] = cell["passing_max_ulp"] = 1
            row = next(r for r in self.bounds[3]["rows"].values() if r.low != r.high and r.staged == r.ideal)
            actual = next(v for v in (row.low, row.high) if v != row.staged)
            cell["worst_input_bits"] = f"{row.input:016x}"
            cell["worst_actual_bits"] = f"{actual:016x}"
            flexible = sum(r.low != r.high for r in self.bounds[3]["rows"].values())
            conservative = repetitions * flexible + cell["values"] - repetitions * len(self.bounds[3]["rows"])
            self.assertEqual(conservative, 32580 if raw else 9782)
            self.assertGreater(changed, conservative)
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                if raw:
                    e.validate_raw(raw_serial(expected, accuracy), expected, self.manifest, self.bounds)
                else:
                    e.validate(public_serial(expected, accuracy), expected, self.manifest, self.bounds, execution_os="linux", backend="flex_cpu", fusion="disabled")

    def test_legal_normal_subnormal_boundary_changes_and_flexible_half_images_pass(self):
        accuracy = synthetic_accuracy(FINITE_F32, self.bounds)
        for source_class, actual_class in ((1, 2), (2, 1)):
            row, actual = next((r, v) for r in self.bounds[1]["rows"].values() for v in e.successors(1, r)
                               if e.classify(1, r.input) <= 2 and e.classify(1, r.staged) == source_class
                               and e.classify(1, v) == actual_class and e.distance(1, v, r.staged) == 1)
            change_observation(accuracy, 1, row, actual)
        self.assertEqual(sum(accuracy["cells"][1]["histogram"][1:]), 2)
        self.assertEqual(self.check(accuracy), accuracy)
        row = next(r for r in self.bounds[3]["rows"].values() if r.low != r.high and r.staged == r.ideal)
        actual = next(v for v in (row.low, row.high) if v != row.staged)
        for expected, raw in ((selection(3), False), (RAW_DEFAULT, True)):
            accuracy = synthetic_accuracy(expected, self.bounds, raw=raw)
            change_observation(accuracy, 3, row, actual)
            if raw:
                self.assertEqual(e.validate_raw(raw_serial(expected, accuracy), expected, self.manifest, self.bounds), accuracy)
            else:
                self.assertEqual(self.check(accuracy, expected), accuracy)

    def test_joint_pair_assignment_matches_bounded_exhaustive_integer_choices(self):
        # This checks the Hall interval calculation against actual assignment,
        # rather than another implementation of its inequalities.
        options = (frozenset({(0, 0)}), frozenset({(0, 0), (1, 1)}), frozenset({(0, 1), (1, 0)}), frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}))
        for left, right in itertools.product(options, repeat=2):
            for a, b in ((1, 1), (1, 2), (2, 2)):
                n = a + b
                assignments = {(sum(s for s, i in chosen), sum(i for s, i in chosen)) for chosen in itertools.product(*([left] * a + [right] * b))}
                groups = Counter()
                groups[left] += a
                groups[right] += b
                context = {"distance_pairs": groups, "transition_caps": {name: Counter({(2, 2): n}) for name in ("staged", "ideal")}}
                for staged_one, ideal_one in itertools.product(range(n + 1), repeat=2):
                    cell = {"finite_pairs": n, "ideal_finite_pairs": n, "histogram": [n - staged_one, staged_one, 0, 0, 0, 0, 0], "ideal_histogram": [n - ideal_one, ideal_one, 0, 0, 0, 0, 0], "transitions": [[0] * 5 for _ in range(5)], "ideal_transitions": [[0] * 5 for _ in range(5)]}
                    cell["transitions"][2][2] = cell["ideal_transitions"][2][2] = n
                    try:
                        e._validate_image_capacities(3, cell, context)
                    except RuntimeError:
                        accepted = False
                    else:
                        accepted = True
                    self.assertEqual(accepted, (staged_one, ideal_one) in assignments)

    def test_materialized_chain_closure_is_one_capacity_slot_per_stage(self):
        context = e._shared_context(PUBLIC_DEFAULT, self.bounds)[3]
        static_finite = sum(n for source, n in context["row_weights"].items() if e.classify(3, self.bounds[3]["rows"][source].staged) in (1, 2))
        chain_finite = sum(any(e.classify(3, self.bounds[3]["rows"][b].staged) in (1, 2) for b in sources) for sources in context["chain_slots"])
        self.assertEqual(len(context["chain_slots"]), 42)
        self.assertEqual(sum(context["distance_pairs"].values()), static_finite + chain_finite)
        sources = tuple(r.input for r in self.bounds[3]["rows"].values() if r.low != r.high)
        single_slot = {"row_weights": Counter(), "chain_slots": [sources]}
        e._image_capacities(self.bounds[3], single_slot)
        self.assertEqual(sum(single_slot["distance_pairs"].values()), 1)
        for name in ("staged", "ideal"):
            self.assertLessEqual(max(single_slot["transition_caps"][name].values()), 1)

    def test_all_public_filters_and_platform_scopes_keep_image_capacities(self):
        configurations = [{"fi": fi} for fi in range(5)] + [{"finite": True}, {"no_flex32": True, "skip_rank_zero": True}, {"finite": True, "no_flex32": True, "skip_rank_zero": True, "op": "erf"}, {"fi": 1, "finite": True, "skip_rank_zero": True}, {"fi": 2, "no_flex32": True}]
        for options in configurations:
            expected = filtered_selection(**options)
            accuracy = synthetic_accuracy(expected, self.bounds)
            for platform in (("linux", "flex_cpu", "disabled"), ("linux", "cuda", "enabled_materialized_chain_steps"), ("d3os", "native", "disabled")):
                # Synthetic scope records exercise parsing only; no backend runs.
                self.assertEqual(e.validate(public_serial(expected, accuracy, platform), expected, self.manifest, self.bounds, execution_os=platform[0], backend=platform[1], fusion=platform[2]), accuracy)

    def test_transition_crossings_share_exact_distance_bin_capacity(self):
        for expected, raw in ((selection(1), False), (RAW_DEFAULT, True)):
            context = (e._raw_context(expected, self.bounds) if raw else e._shared_context(expected, self.bounds))[1]
            eligible = []
            for source, weight in context["row_weights"].items():
                row = self.bounds[1]["rows"][source]
                outputs = [v for v in e.successors(1, row) if e.classify(1, row.staged) == 2 and e.classify(1, v) == 1 and e.distance(1, v, row.staged) == 1]
                if outputs:
                    eligible.append((row, outputs[0], weight))
            one_capacity = sum(weight for _, _, weight in eligible)
            requested = context["transition_caps"]["staged"][2, 1]
            self.assertEqual(one_capacity, 8 if raw else 4)
            self.assertGreater(requested, one_capacity)
            bad = synthetic_accuracy(expected, self.bounds, raw=raw)
            cell = bad["cells"][1]
            for key in ("transitions", "ideal_transitions"):
                cell[key][2][2] -= requested
                cell[key][2][1] += requested
            cell["observed_classes"][2] -= requested
            cell["observed_classes"][1] += requested
            for key in ("histogram", "passing_histogram", "ideal_histogram"):
                cell[key][0] -= requested
                cell[key][1] += requested
            for key in ("max_ulp", "passing_max_ulp", "max_ideal_ulp"):
                cell[key] = 1
            row, actual, _ = eligible[0]
            for prefix in ("worst", "worst_ideal"):
                cell[prefix + "_input_bits"] = f"{row.input:016x}"
                cell[prefix + "_actual_bits"] = f"{actual:016x}"
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                self.check_case(bad, expected, raw)
            legal = synthetic_accuracy(expected, self.bounds, raw=raw)
            for row, actual, weight in eligible:
                for _ in range(weight):
                    change_observation(legal, 1, row, actual)
            self.assertEqual(self.check_case(legal, expected, raw), legal)

    def test_distinct_worst_events_consume_joint_half_image_budgets(self):
        for expected, raw in ((selection(3), False), (RAW_DEFAULT, True)):
            context = (e._raw_context(expected, self.bounds) if raw else e._shared_context(expected, self.bounds))[3]
            mandatory = context["reference_deltas"][1]
            self.assertGreater(mandatory, 0)
            self.assertEqual(mandatory % 2, 0)
            row = next(r for r in self.bounds[3]["rows"].values() if r.staged == r.ideal and r.low != r.high)
            actual = next(v for v in e.successors(3, row) if v != row.staged)
            bad = synthetic_accuracy(expected, self.bounds, raw=raw)
            cell = bad["cells"][3]
            for key in ("histogram", "passing_histogram", "ideal_histogram"):
                cell[key] = [cell["finite_pairs"] - mandatory // 2, mandatory // 2, 0, 0, 0, 0, 0]
            for key in ("max_ulp", "passing_max_ulp", "max_ideal_ulp"):
                cell[key] = 1
            for prefix in ("worst", "worst_ideal"):
                cell[prefix + "_input_bits"] = f"{row.input:016x}"
                cell[prefix + "_actual_bits"] = f"{actual:016x}"
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                self.check_case(bad, expected, raw)
            legal = synthetic_accuracy(expected, self.bounds, raw=raw)
            change_observation(legal, 3, row, actual)
            self.assertEqual(self.check_case(legal, expected, raw), legal)

    def test_completed_native_archive_replays_without_new_gpu_execution_when_available(self):
        archive = Path(os.environ.get("TYPED_ERF_COMPLETED_NATIVE_ARCHIVE", Path.home() / ".cache/d3os/burn-typed-erf-harness-first"))
        if not (archive / "serial.log").is_file():
            self.skipTest("completed Native archive unavailable; test never launches or watches a VM")
        expected = e.strict_json((archive / "typed-erf-expected.json").read_text())
        raw = e.strict_json((archive / "typed-erf-raw-plan.json").read_text())
        serial = (archive / "serial.log").read_text()
        public_accuracy = e.validate(serial, expected, self.manifest, self.bounds, execution_os="d3os", backend="native", fusion="disabled")
        raw_accuracy = e.validate_raw(serial, raw, self.manifest, self.bounds)
        self.assertEqual(sum(c["values"] for c in public_accuracy["cells"]) + sum(c["values"] for c in raw_accuracy["cells"]), 1595218)


if __name__ == "__main__":
    unittest.main()
