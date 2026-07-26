"""Unit tests for marker parsing and statistical helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_dcm_pilot import classify_trials, holm_adjust, marker_code


class MarkerParsingTests(unittest.TestCase):
    def test_marker_class_is_not_ignored(self) -> None:
        self.assertEqual(marker_code("Stimulus/S  1", ("Stimulus/",)), 1)
        self.assertIsNone(marker_code("Response/R  1", ("Stimulus/",)))
        self.assertEqual(
            marker_code("Response/R  2", ("Stimulus/", "Response/")), 2
        )

    def test_trial_state_machine_handles_all_outcomes(self) -> None:
        events = [
            (100, "Stimulus/S  1"),
            (200, "Stimulus/S  3"),
            (300, "Response/R  2"),
            (400, "Stimulus/S  1"),
            (500, "Response/R  4"),
            (600, "Stimulus/S  1"),
            (700, "Stimulus/S  1"),
        ]
        trials = classify_trials(events, sfreq=100.0)
        self.assertEqual(
            [trial.outcome for trial in trials],
            [
                "response_2",
                "response_4",
                "missing_before_next_onset",
                "missing_at_recording_end",
            ],
        )
        self.assertTrue(trials[0].picture_offset_seen)
        self.assertAlmostEqual(trials[0].response_latency_seconds, 2.0)
        self.assertFalse(trials[1].is_correct)


class StatisticalHelperTests(unittest.TestCase):
    def test_holm_adjustment_is_monotonic_in_rank(self) -> None:
        p_values = [0.03, 0.001, 0.02]
        adjusted = holm_adjust(p_values)
        order = np.argsort(p_values)
        ranked = np.asarray(adjusted)[order]
        self.assertTrue(np.all(np.diff(ranked) >= 0))
        self.assertEqual(adjusted[1], 0.003)
        self.assertEqual(adjusted[2], 0.04)
        self.assertEqual(adjusted[0], 0.04)


if __name__ == "__main__":
    unittest.main()
