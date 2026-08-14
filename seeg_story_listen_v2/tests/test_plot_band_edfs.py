from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_band_edfs as plotter


def test_plotter_combines_available_bands_in_one_png(tmp_path: Path) -> None:
    assert plotter.parse_args([]).scale_mv == 0.005
    input_dir = tmp_path / "processed"
    subject_dir = input_dir / "sub001"
    output_dir = tmp_path / "pictures"
    subject_dir.mkdir(parents=True)
    sfreq = 128.0
    times = np.arange(round(2 * sfreq)) / sfreq
    data = np.vstack(
        [
            np.sin(2 * np.pi * 2 * times) * 50e-6,
            np.sin(2 * np.pi * 6 * times) * 80e-6,
            np.sin(2 * np.pi * 10 * times) * 100e-6,
        ]
    )
    for band in ("delta", "high_gamma"):
        raw = mne.io.RawArray(
            data,
            mne.create_info(["LA1", "LA2", "LA3"], sfreq, ch_types="seeg"),
            verbose="ERROR",
        )
        mne.export.export_raw(
            subject_dir / f"sub001_prepocessed_{band}.edf",
            raw,
            fmt="edf",
            physical_range="channelwise",
            overwrite=True,
            verbose="ERROR",
        )
        raw.close()

    result = plotter.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--max-time-bins",
            "64",
            "--dpi",
            "60",
        ]
    )

    destination = (
        output_dir / "sub001" / "sub001_prepocessed_all_bands_full_length.png"
    )
    assert result == 0
    assert matplotlib.get_backend().casefold() == "agg"
    assert destination.is_file()
    image = mpimg.imread(destination)
    assert image.shape[0] > 100
    assert image.shape[1] > image.shape[0]
