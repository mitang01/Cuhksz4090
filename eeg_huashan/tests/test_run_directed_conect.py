"""Unit tests for directed-connectivity helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_directed_conect import (
    PLANNED_EDGES,
    frequency_bands,
    net_time_reversed_gc,
    ordered_pairs,
)


class FrequencyBandTests(unittest.TestCase):
    def test_short_picture_window_omits_low_frequency_bands(self) -> None:
        self.assertEqual(
            frequency_bands(0.48, 30.0),
            {"alpha": (8.0, 12.0), "beta": (13.0, 30.0)},
        )

    def test_two_second_rest_includes_theta(self) -> None:
        self.assertIn("theta", frequency_bands(2.0, 45.0))


class TimeReversalTests(unittest.TestCase):
    def test_net_trgc_uses_both_directions(self) -> None:
        pairs = ordered_pairs()
        pair_index = {pair: index for index, pair in enumerate(pairs)}
        gc = np.zeros((len(pairs), 1))
        gc_tr = np.zeros_like(gc)
        source, target = PLANNED_EDGES[0]
        gc[pair_index[(source, target)], 0] = 0.8
        gc[pair_index[(target, source)], 0] = 0.2
        gc_tr[pair_index[(source, target)], 0] = 0.3
        gc_tr[pair_index[(target, source)], 0] = 0.1
        result = net_time_reversed_gc(gc, gc_tr)
        self.assertAlmostEqual(result[0, 0], 0.4)


if __name__ == "__main__":
    unittest.main()
