from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preprocess_seeg as prep


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_laplacian_matrix_uses_immediate_neighbors() -> None:
    names = ["LA1", "LA2", "LA3", "LB01", "LB02", "ECG"]
    matrix, output_names, dropped = prep.laplacian_matrix(names, prep.NON_SEEG_DEFAULT)

    assert output_names == names[:-1]
    np.testing.assert_allclose(matrix[0, :3], [1, -1, 0])
    np.testing.assert_allclose(matrix[1, :3], [-0.5, 1, -0.5])
    np.testing.assert_allclose(matrix[2, :3], [0, -1, 1])
    assert dropped == ["ECG"]


def test_textgrid_reads_nonempty_intervals_from_first_tier(tmp_path: Path) -> None:
    textgrid = tmp_path / "story1.TextGrid"
    textgrid.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "syllables"
        xmin = 0
        xmax = 2
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 1
            text = "hel"
        intervals [3]:
            xmin = 1
            xmax = 2
            text = "lo"
    item [2]:
        class = "IntervalTier"
        name = "other"
        xmin = 0
        xmax = 2
        intervals: size = 1
        intervals [1]:
            xmin = 0.25
            xmax = 2
            text = "ignored"
""",
        encoding="utf-8",
    )

    assert prep.parse_first_interval_tier(textgrid) == [0.5, 1.0]


def test_speech_responsiveness_detects_positive_response() -> None:
    sfreq = 128.0
    token_onsets = np.arange(2.0, 62.0, 2.0)
    rng = np.random.default_rng(12)
    data = rng.normal(0, 0.25, (2, round(64 * sfreq)))
    for onset in token_onsets:
        start = round((onset + 0.05) * sfreq)
        stop = round((onset + 0.2) * sfreq)
        data[0, start:stop] += 3.0

    rows, responsive, times, erps = prep.speech_responsiveness(
        data,
        sfreq,
        token_onsets,
        epoch_start=-1,
        epoch_end=2,
        baseline_start=-0.2,
        baseline_end=-0.05,
        response_start=0.05,
        response_end=0.2,
        fdr_q=0.01,
    )

    assert responsive.tolist() == [0]
    assert rows[0]["fdr_p_value"] < 0.01
    assert rows[1]["responsive"] is False
    assert erps.shape == (2, len(times))


def test_dry_run_validates_audio_events_and_excludes_story18(tmp_path: Path) -> None:
    source = tmp_path / "input"
    output = tmp_path / "output"
    (source / "sub001").mkdir(parents=True)
    (source / "sub001" / "recording.edf").touch()
    write_csv(
        source / "event_stimuli.csv",
        ["trigger", "stimulus"],
        [
            {"trigger": "1", "stimulus": "story1.wav"},
            {"trigger": "2", "stimulus": "story1.wav"},
            {"trigger": "3", "stimulus": "story18.wav"},
            {"trigger": "4", "stimulus": "story18.wav"},
        ],
    )
    write_csv(
        source / "001_event.csv",
        ["time", "trigger"],
        [
            {"time": 2, "trigger": 1},
            {"time": 4, "trigger": 2},
            {"time": 6, "trigger": 3},
            {"time": 8, "trigger": 4},
        ],
    )
    wav_dir = source / "stimuli_wav"
    wav_dir.mkdir()
    silence = np.zeros(2000, dtype=np.int16)
    wavfile.write(wav_dir / "story1.wav", 1000, silence)
    wavfile.write(wav_dir / "story18.wav", 1000, silence)
    textgrid_dir = source / "stimuli_textgrid"
    textgrid_dir.mkdir()
    (textgrid_dir / "story1.TextGrid").write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"
item []:
    item [1]:
        class = "IntervalTier"
        name = "syllables"
        xmin = 0
        xmax = 2
        intervals: size = 1
        intervals [1]:
            xmin = 0.5
            xmax = 1
            text = "token"
""",
        encoding="utf-8",
    )

    result = prep.main(
        [
            "--input-dir",
            str(source),
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )

    assert result == 0
    qc = output / "sub001" / "recording_qc"
    duration_rows = list(csv.DictReader((qc / "audio_trigger_duration_qc.csv").open()))
    token_rows = list(csv.DictReader((qc / "textgrid_token_qc.csv").open()))
    assert all(row["within_tolerance"] == "True" for row in duration_rows)
    assert {row["stimulus"]: row["status"] for row in token_rows} == {
        "story1": "ok",
        "story18": "excluded_story18",
    }
