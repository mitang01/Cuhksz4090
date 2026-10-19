from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import print_edf_info as printer


def _write_edf(path: Path, *, subject: str, sfreq: float = 500.0) -> None:
    info = mne.create_info(["LA1", "LA2", "LA3"], sfreq, ch_types="seeg")
    info["subject_info"] = {"his_id": subject}
    data = np.vstack(
        [
            np.sin(2 * np.pi * 8 * np.arange(int(sfreq)) / sfreq),
            np.sin(2 * np.pi * 12 * np.arange(int(sfreq)) / sfreq),
            np.sin(2 * np.pi * 20 * np.arange(int(sfreq)) / sfreq),
        ]
    ) * 1e-6
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(
        path,
        raw,
        fmt="edf",
        physical_range="channelwise",
        overwrite=True,
        verbose="ERROR",
    )
    raw.close()


def test_print_edf_info_reports_raw_and_info(tmp_path: Path) -> None:
    assert printer.parse_args([]).input_dir == printer.DEFAULT_INPUT

    root = tmp_path / "story_listen_v2"
    output = tmp_path / "edf_info.txt"
    raw_edf = root / "sub001" / "ses01" / "recording.edf"
    processed_edf = root / "sub001" / "ses01" / "recording_prepocessed_gamma.edf"
    _write_edf(raw_edf, subject="sub001")
    _write_edf(processed_edf, subject="sub001")

    result = printer.main(
        ["--input-dir", str(root), "--output", str(output)]
    )
    text = output.read_text(encoding="utf-8")

    assert result == 0
    assert "files discovered: 1" in text
    assert str(raw_edf.resolve()) in text
    assert "print(raw):" in text
    assert "print(raw.info):" in text
    assert "sfreq: 500.0 Hz" in text
    assert "ch_names: LA1, LA2, LA3" in text
    assert "chs: 3 EEG" in text
    assert "highpass: 0.0 Hz" in text
    assert "lowpass: 250.0 Hz" in text
    assert "subject_info:" in text
    assert "his_id: sub001" in text
    assert "prepocessed" not in text

    result_all = printer.main(
        [
            "--input-dir",
            str(root),
            "--output",
            str(tmp_path / "all.txt"),
            "--include-processed",
        ]
    )
    all_text = (tmp_path / "all.txt").read_text(encoding="utf-8")
    assert result_all == 0
    assert "files discovered: 2" in all_text
    assert "prepocessed" in all_text


def test_print_edf_info_missing_directory_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert printer.main(["--input-dir", str(missing)]) == 2
