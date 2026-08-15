#!/usr/bin/env python3
"""
Bombcell auto-curation for MountainSort4 v3 session-level outputs.

This script reuses the maintained curation pipeline in
Auto_sorting_curation.py, but applies the stricter requested rules:
  - num_spikes > 200
  - isi_violations_ratio < 0.01

All other Bombcell thresholds are disabled. No interactive plotting is used.
"""

from __future__ import annotations

import numpy as np

import Auto_sorting_curation as base


# SpikeInterface's threshold API uses inclusive >= / <= comparisons. Since
# num_spikes is integer, >=201 implements >200. nextafter implements strict
# floating-point <0.01 rather than <=0.01.
base.DEFAULT_SESSION_GLOB = "sorting_results_*_v3"
base.BOMBCELL_THRESHOLDS = {
    "noise": {},
    "mua": {
        "num_spikes": {"greater": 201, "less": None},
        "isi_violations_ratio": {
            "greater": None,
            "less": float(np.nextafter(0.01, -np.inf)),
        },
    },
    "non-somatic": {},
}
base.BOMBCELL_LABELING_RULES = {
    "num_spikes": "> 200",
    "isi_violations_ratio": "< 0.01",
}


if __name__ == "__main__":
    base.main()
