from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mne
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
    (source / "sub001" / "sub001.edf").touch()
    write_csv(
        source / "event_stimuli.csv",
        ["trigger_label", " stimuli_filename", " silence"],
        [
            {
                "trigger_label": "onset_1",
                " stimuli_filename": "story1.wav",
                " silence": 0.5,
            },
            {
                "trigger_label": "offset_1",
                " stimuli_filename": "story1.wav",
                " silence": 0.5,
            },
            {
                "trigger_label": "onset_18",
                " stimuli_filename": "story18.wav",
                " silence": 0.25,
            },
            {
                "trigger_label": "offset_18",
                " stimuli_filename": "story18.wav",
                " silence": 0.25,
            },
        ],
    )
    event_csv = source / "sub001" / "sub001_event.csv"
    event_csv.write_text(
        '"onset_1, 2.5, 0"\n'
        '"offset_1, 3.5, 0"\n'
        '"onset_18, 6.25, 0"\n'
        '"offset_18, 7.75, 0"\n',
        encoding="utf-8",
    )
    (source / "sub002").mkdir()
    (source / "sub002" / "sub002_event.csv").write_text(
        "onset_1, 100, 0\noffset_1, 102, 0\n", encoding="utf-8"
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
    assert prep.find_event_csv(source / "sub001" / "sub001.edf", source) == event_csv
    qc = output / "sub001" / "sub001_qc"
    duration_rows = list(csv.DictReader((qc / "audio_trigger_duration_qc.csv").open()))
    token_rows = list(csv.DictReader((qc / "textgrid_token_qc.csv").open()))
    assert all(row["within_tolerance"] == "True" for row in duration_rows)
    story1_duration = next(
        row for row in duration_rows if row["stimulus"] == "story1"
    )
    assert float(story1_duration["onset_trigger_time"]) == 2.5
    assert float(story1_duration["onset_silence_s"]) == 0.5
    assert float(story1_duration["onset"]) == 2.0
    assert float(story1_duration["offset_trigger_time"]) == 3.5
    assert float(story1_duration["offset_silence_s"]) == 0.5
    assert float(story1_duration["offset"]) == 4.0
    assert {row["stimulus"]: row["status"] for row in token_rows} == {
        "story1": "ok",
        "story18": "excluded_story18",
    }


def test_full_high_gamma_pipeline_exports_valid_edfs(tmp_path: Path) -> None:
    source = tmp_path / "input"
    output = tmp_path / "output"
    subject_dir = source / "sub002"
    subject_dir.mkdir(parents=True)
    sfreq = 512.0
    duration = 40.0
    times = np.arange(round(duration * sfreq)) / sfreq
    rng = np.random.default_rng(31)
    responsive_signal = rng.normal(0, 0.1e-6, len(times))
    relative_tokens = np.arange(1.0, 31.0)
    for token in relative_tokens:
        mask = (times >= 3 + token + 0.05) & (times < 3 + token + 0.2)
        responsive_signal[mask] += np.sin(2 * np.pi * 100 * times[mask]) * 8e-6
    data = np.vstack(
        [
            responsive_signal,
            rng.normal(0, 1e-6, len(times)),
            rng.normal(0, 1e-6, len(times)),
            rng.normal(0, 1e-5, len(times)),
        ]
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(["LA1", "LA2", "LA3", "ECG"], sfreq, ch_types="eeg"),
        verbose="ERROR",
    )
    source_edf = subject_dir / "synthetic.edf"
    mne.export.export_raw(
        source_edf,
        raw,
        fmt="edf",
        physical_range="channelwise",
        overwrite=True,
        verbose="ERROR",
    )
    raw.close()

    write_csv(
        source / "event_stimuli.csv",
        ["trigger", "stimulus"],
        [
            {"trigger": "11", "stimulus": "story2.wav"},
            {"trigger": "12", "stimulus": "story2.wav"},
        ],
    )
    write_csv(
        source / "002_event.csv",
        ["time", "trigger"],
        [{"time": 3, "trigger": 11}, {"time": 38, "trigger": 12}],
    )
    wav_dir = source / "stimuli_wav"
    wav_dir.mkdir()
    wavfile.write(wav_dir / "story2.wav", 1000, np.zeros(35000, dtype=np.int16))
    textgrid_dir = source / "stimuli_textgrid"
    textgrid_dir.mkdir()
    intervals = "\n".join(
        f"""        intervals [{index}]:
            xmin = {onset}
            xmax = {onset + 0.5}
            text = "syllable{index}\""""
        for index, onset in enumerate(relative_tokens, start=1)
    )
    (textgrid_dir / "story2.TextGrid").write_text(
        f"""File type = "ooTextFile"
Object class = "TextGrid"
item []:
    item [1]:
        class = "IntervalTier"
        name = "syllables"
        xmin = 0
        xmax = 35
        intervals: size = {len(relative_tokens)}
{intervals}
""",
        encoding="utf-8",
    )

    result = prep.main(
        [
            "--input-dir",
            str(source),
            "--output-dir",
            str(output),
            "--bands",
            "high_gamma",
            "delta",
        ]
    )

    assert result == 0
    processed = output / "sub002" / "synthetic_prepocessed_high_gamma.edf"
    processed_delta = output / "sub002" / "synthetic_prepocessed_delta.edf"
    responsive = output / "sub002" / "synthetic_responsive_high_gamma.edf"
    responsive_delta = output / "sub002" / "synthetic_responsive_delta.edf"
    assert processed.is_file()
    assert processed_delta.is_file()
    assert responsive.is_file()
    assert not responsive_delta.exists()
    processed_raw = mne.io.read_raw_edf(processed, preload=False, verbose="ERROR")
    responsive_raw = mne.io.read_raw_edf(responsive, preload=False, verbose="ERROR")
    try:
        assert processed_raw.info["sfreq"] == 128
        assert processed_raw.ch_names == ["LA1", "LA2", "LA3"]
        assert 0 < len(responsive_raw.ch_names) <= 3
    finally:
        processed_raw.close()
        responsive_raw.close()
    qc = output / "sub002" / "synthetic_qc"
    assert (qc / "high_gamma_responsive_token_epochs.npz").is_file()
    assert (qc / "delta_no_responsive_channels.txt").is_file()
    metadata = json.loads((qc / "processing_metadata.json").read_text())
    assert metadata["line_frequencies_hz"] == [50, 100, 150, 200, 250]
    assert metadata["responsive_electrode_selection"] == "independent_per_band"
    diagnostics = json.loads(
        (qc / "high_gamma_speech_response_diagnostics.json").read_text()
    )
    assert diagnostics["n_responsive_channels"] == len(responsive_raw.ch_names)
    delta_diagnostics = json.loads(
        (qc / "delta_speech_response_diagnostics.json").read_text()
    )
    assert delta_diagnostics["n_responsive_channels"] == 0
