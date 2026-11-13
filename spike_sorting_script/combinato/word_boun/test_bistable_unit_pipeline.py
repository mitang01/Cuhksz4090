"""Synthetic recording for the bistable unit pipeline.

One HG unit fires after the reported turbo boundary and is quiet for boater.
One ATL unit fires the same way after both percepts. say/ace is one syllable,
so both words stay on the TTL and neither unit should show a percept effect.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bistable_unit_pipeline as pipe


def test_rising_edge_is_found_across_a_chunk_boundary():
    data = np.zeros((25, 1), dtype=np.uint8)
    data[5, 0] = 1
    data[10, 0] = 1
    data[11, 0] = 1
    edges = pipe.rising_edges(data, 0, 0, block=10)
    assert list(edges) == [5, 10]


def test_button_names_map_onto_words():
    row = {"hand": "left", "left_word": "turbo.wav", "right_word": "boater.wav"}
    assert pipe.map_percept(row, "left", list(row)) == "turbo"
    assert pipe.map_percept(row, "right", list(row)) == "boater"


def test_two_syllable_shift_follows_the_token_name():
    shifts, rule = pipe.shift_for_pair(
        "turbo",
        ("turbo", "boater"),
        cycle_s=1.0,
        syllables=2,
        two_syllable=set(),
        explicit_ms={},
    )
    assert rule == "half-cycle"
    assert shifts["turbo"] == 0.0
    assert shifts["boater"] == 0.5
    onset, onset_rule = pipe.shift_for_pair(
        "say",
        ("say", "ace"),
        cycle_s=1.0,
        syllables=1,
        two_syllable=set(),
        explicit_ms={},
    )
    assert onset_rule == "onset"
    assert onset == {"say": 0.0, "ace": 0.0}


def _write_unit(folder: Path, spikes_ms: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(folder / f"data_{folder.name}.h5", "w") as handle:
        handle.create_dataset(f"pos/times", data=np.asarray(spikes_ms, dtype=np.float64))
    sort_dir = folder / "sort_pos_simple"
    sort_dir.mkdir()
    with h5py.File(sort_dir / "sort_cat.h5", "w") as handle:
        handle.attrs["sign"] = "pos"
        count = int(spikes_ms.size)
        handle.create_dataset("index", data=np.arange(count, dtype=np.uint32))
        handle.create_dataset("classes", data=np.ones(count, dtype=np.uint16))
        handle.create_dataset("groups", data=np.array([[1, 1]], dtype=np.int16))
        handle.create_dataset("types", data=np.array([[1, 2]], dtype=np.int16))


def _schedule() -> list[list[float]]:
    clock = 1.0
    blocks = []
    for _ in range(4):
        anchors = []
        for _trial in range(20):
            anchors.append(clock)
            clock += 1.0
        blocks.append(anchors)
        clock += 3.0
    return blocks


def _write_recording(path: Path, blocks: list[list[float]], sample_rate: int) -> None:
    last = blocks[-1][-1]
    n_samples = int((last + 4.0) * sample_rate)
    digital = np.zeros((n_samples, 2), dtype=np.uint8)
    digital[sample_rate : sample_rate + 5, 0] = 1
    digital[sample_rate * 2 : sample_rate * 2 + 5, 0] = 1
    digital[sample_rate * 3 : sample_rate * 3 + 5, 0] = 1
    for anchors in blocks:
        for anchor in anchors:
            start = int(round(anchor * sample_rate))
            digital[start : start + 8, 1] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("board_dig_in_data", data=digital)
        handle.create_dataset("board_dig_in_sample_rate", data=np.array([sample_rate], dtype=np.float64))
        handle.create_dataset("channel_number", data=np.array([17, 65], dtype=np.int32))
        text = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "brain_region",
            data=np.array(["ATL", "HG"], dtype=object),
            dtype=text,
        )


def _write_csvs(log_dir: Path, blocks: list[list[float]]) -> None:
    log_dir.mkdir()
    specs = [
        ("block01_turbo.csv", "turbo", 2, ("turbo", "boater")),
        ("block02_turbo.csv", "turbo", 2, ("turbo", "boater")),
        ("block03_say.csv", "say", 1, ("say", "ace")),
        ("block04_say.csv", "say", 1, ("say", "ace")),
    ]
    for (name, stimulus, syllables, percepts), anchors in zip(specs, blocks):
        with (log_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stimulus", "percept", "syllables"])
            writer.writeheader()
            for index, _anchor in enumerate(anchors):
                writer.writerow(
                    {
                        "stimulus": stimulus,
                        "percept": percepts[index % 2],
                        "syllables": syllables,
                    }
                )


def _spikes_ms(blocks: list[list[float]], selective: bool) -> np.ndarray:
    spikes = []
    specs = [("turbo", ("turbo", "boater")), ("turbo", ("turbo", "boater")), ("say", ("say", "ace")), ("say", ("say", "ace"))]
    for (stimulus, percepts), anchors in zip(specs, blocks):
        for index, anchor in enumerate(anchors):
            percept = percepts[index % 2]
            if stimulus == "turbo" and percept == "boater":
                if selective:
                    continue
                onset = anchor + 0.5 + 0.12
            else:
                onset = anchor + 0.12
            spikes.extend([onset * 1000.0] * 6)
    return np.asarray(spikes, dtype=np.float64)


def test_pipeline_recovers_the_perceived_word_effect(tmp_path: Path):
    sample_rate = 30000
    blocks = _schedule()
    result_root = tmp_path / "combinato"
    subject = result_root / "sub-007"
    _write_unit(subject / "A-065", _spikes_ms(blocks, selective=True))
    _write_unit(subject / "A-017", _spikes_ms(blocks, selective=False))
    (subject / "A-033").mkdir(parents=True)
    merged = tmp_path / "sub-007.mat"
    _write_recording(merged, blocks, sample_rate)
    log_dir = tmp_path / "logs"
    _write_csvs(log_dir, blocks)
    output = tmp_path / "out"

    summary = pipe.main(
        [
            "--subject",
            "sub-007",
            "--result-root",
            str(result_root),
            "--merged",
            str(merged),
            "--log-dir",
            str(log_dir),
            "--output-dir",
            str(output),
            "--min-spikes",
            "20",
            "--n-permutations",
            "199",
            "--seed",
            "0",
        ]
    )

    by_key = {(item["unit_id"], item["stimulus"]): item for item in summary["results"]}
    selective = by_key[("A-065-g1", "turbo")]
    matched = by_key[("A-017-g1", "turbo")]
    say = by_key[("A-065-g1", "say")]
    assert selective["region"] == "HG"
    assert matched["region"] == "ATL"
    assert selective["evoked"] and selective["significant"]
    assert matched["evoked"] and not matched["significant"]
    assert say["evoked"] and not say["significant"]
    assert selective["pair"] == ("turbo", "boater")

    boater = [trial for trial in summary["trials"] if trial.percept == "boater"]
    ace = [trial for trial in summary["trials"] if trial.percept == "ace"]
    assert boater and all(trial.shift_rule == "half-cycle" and abs(trial.shift_s - 0.5) < 1e-9 for trial in boater)
    assert ace and all(trial.shift_s == 0.0 and trial.shift_rule == "onset" for trial in ace)
    assert len(summary["trials"]) == 80

    for name in (
        "01_percept_reports.png",
        "02_alignment.png",
        "03_evoked_response.png",
        "04_percept_ranksum.png",
        "05_region_summary.png",
        "trials.csv",
        "units.csv",
        "tests.csv",
        "summary.txt",
    ):
        written = output / name
        assert written.is_file() and written.stat().st_size > 100, name
    text = (output / "summary.txt").read_text()
    assert "trigger_channel_1based" in text
    assert "A-033" in text
