#!/usr/bin/env python3
"""Check multiplexed hardware-counter intervals without GPU or perf permissions."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compiler_profile", Path(__file__).with_name("profile-compiler.py")
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def reading(value, enabled, running):
    return {"value": value, "enabled_ns": enabled, "running_ns": running}


class CounterIntervals(unittest.TestCase):
    def test_scale_only_the_observed_interval(self):
        # Previous counts used a different multiplexing ratio.
        before = reading(100, 100, 50)
        after = reading(175, 200, 75)
        self.assertEqual(profile.counter_delta(before, after), 300)

    def test_unscheduled_idle_counter_contributes_zero(self):
        self.assertEqual(profile.counter_delta(reading(40, 80, 10), reading(40, 180, 10)), 0)

    def test_reject_reset_and_inconsistent_time(self):
        before = reading(100, 100, 50)
        for after in (reading(99, 200, 100), reading(101, 110, 90), reading(101, 200, 50)):
            with self.subTest(after=after), self.assertRaises(ValueError):
                profile.counter_delta(before, after)


if __name__ == "__main__":
    unittest.main()
