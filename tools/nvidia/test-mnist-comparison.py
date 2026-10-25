#!/usr/bin/env python3
"""Verify the numerical error metric used for MNIST parity comparisons."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).with_name("compare-mnist.py"))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class NumericalComparison(unittest.TestCase):
    def test_absolute_and_relative_error_share_the_existing_convolution_bound(self):
        result = comparison.difference([0.0, 3.0, -9.0], [0.00001, 3.00004, -8.9997])
        self.assertEqual(result["values"], 3)
        self.assertEqual(result["tolerance_failures"], 1)
        self.assertAlmostEqual(result["max_abs"], 0.0003)
        self.assertAlmostEqual(result["max_scaled_error"], 1.5)

    def test_invalid_lengths_and_nonfinite_values_cannot_pass(self):
        for expected, actual in (([], []), ([0.0], []), ([float("nan")], [0.0]), ([0.0], [float("inf")])):
            with self.subTest(expected=expected, actual=actual):
                with self.assertRaises(RuntimeError):
                    comparison.difference(expected, actual)


if __name__ == "__main__":
    unittest.main()
