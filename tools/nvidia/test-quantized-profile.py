#!/usr/bin/env python3
"""Validate counter reports without a GPU or the Nsight package."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("linux_reference", Path(__file__).with_name("linux-reference.py"))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class QuantizedProfile(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "quantized-profile-manifest.json").write_text(json.dumps({
            "cases": [{"format": 11, "outputs": 5}],
        }))
        profiles = self.directory / "root/profiles"
        profiles.mkdir(parents=True)
        (profiles / "case0.ncu-repz").write_bytes(b"retained report")
        self.csv = profiles / "case0.csv"
        self.rows = [
            ["Kernel Name", "Block Size", "Grid Size", "gpu__time_duration.sum", "dram__bytes_read.sum",
             "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed"],
            ["", "", "", "ns", "byte", "%"],
            ["vector_project_q3k", "(128, 1, 1)", "(2, 1, 1)", "169,472", "38328192", "84.69"],
        ]

    def validate(self):
        with self.csv.open("w", newline="") as file:
            csv.writer(file).writerows(self.rows)
        return reference.validate_quantized_profile(self.directory)

    def test_parse_counter_units_and_thousands_separators(self):
        result = self.validate()
        self.assertEqual(result["cases"][0]["metrics"]["gpu__time_duration.sum"], {"value": 169472, "unit": "ns"})
        self.assertTrue(result["instrumented"])
        self.assertFalse(result["ordinary_performance_comparison"])

    def test_reject_incompatible_units_or_missing_counters(self):
        for unit, value in (("us", "169.472"), ("ns", "N/A"), ("ns", "0"), ("ns", "-1"), ("ns", "nan"), ("ns", "inf")):
            with self.subTest(unit=unit, value=value):
                self.rows[1][3], self.rows[2][3] = unit, value
                with self.assertRaises(ValueError):
                    self.validate()

    def test_reject_different_kernel_or_geometry(self):
        for column, value in ((0, "vector_project_q4k"), (1, "(64, 1, 1)"), (2, "(1, 1, 1)")):
            with self.subTest(column=column):
                previous = self.rows[2][column]
                self.rows[2][column] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.rows[2][column] = previous

    def test_reject_duplicate_or_truncated_columns(self):
        original = self.rows[0][3]
        self.rows[0][3] = self.rows[0][4]
        with self.assertRaises(ValueError):
            self.validate()
        self.rows[0][3] = original
        self.rows[2].pop()
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_more_than_one_profiled_launch(self):
        self.rows.append(self.rows[2].copy())
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
