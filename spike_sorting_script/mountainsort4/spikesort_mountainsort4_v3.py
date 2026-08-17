#!/usr/bin/env python3
"""
Session-level MountainSort4 sorting for raw Intan MAT chunks.

Each subject is processed as one session. Before concatenation, amplifier
channels are matched by their labels (for example A-017), reordered into a
common order, and restricted to the intersection present in every MAT file.
Thus, if one split is missing a channel, that channel is excluded from every
split rather than padded with synthetic data.

No interactive plotting is used.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# Set thread limits before importing numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd

# MountainSort4 still uses the legacy spikeextractors package, whose runtime
# spike-train validation references np.Inf. NumPy 2 removed that alias.
if not hasattr(np, "Inf"):
    setattr(np, "Inf", np.inf)

import spikeinterface.full as si
from probeinterface import generate_linear_probe
from scipy.io import loadmat
from spikeinterface.metrics.quality.quality_metrics import ComputeQualityMetrics

warnings.simplefilter("ignore")


OUTPUT_ROOT = Path("/share/home/mitan/spike_sorting/mountainsort4")
SAMPLING_RATE = 30000.0
GAIN_TO_UV = 0.195
SORTER_DETECT_THRESHOLD = 7.0
MAX_SPIKEINTERFACE_JOBS = 4

# Existing sorter QC outputs are retained for compatibility. Final v2 Bombcell
# curation applies the requested stricter num_spikes/ISI-only rules.
QC_MIN_SNR = 5.0
QC_MAX_ISI_VIOLATION = 0.02
QC_MIN_NUM_SPIKES = 200
QC_MIN_FIRING_RATE_HZ = 0.18
QC_MIN_PRESENCE_RATIO = 0.75
QC_REQUIRE_ISI_METRIC = False

SESSION_CONFIGS = [
    {
        "subject": "sub7",
        "session": "ses01",
        "mat_files": [
            Path("/share/workspace3/ieeg/micro/word_boun_perce_v2/sub-007/ses-01/Temp_260322_134414.mat"),
            Path("/share/workspace3/ieeg/micro/word_boun_perce_v2/sub-007/ses-01/Temp_260322_135414.mat"),
        ],
        "regions": {
            "ATL": (17, 32),
            "VMPFC": (33, 48),
            "HG": (65, 80),
            "Amygdala": (97, 112),
        },
    },
    {
        "subject": "sub8",
        "session": "ses01",
        "mat_files": [
            Path("/share/workspace3/ieeg/micro/word_boun_perce_v2/sub-008/ses-01/Temp_260404_093808.mat"),
        ],
        "regions": {
            "ATL": (17, 32),
            "Amygdala": (33, 48),
            "HG": (49, 64),
            "VMPFC": (97, 112),
        },
    },
    {
        "subject": "sub9",
        "session": "ses01",
        "mat_files": [
            Path("/share/workspace3/ieeg/micro/word_boun_perce_v2/sub-009/ses-01/Temp_260416_120045.mat"),
            Path("/share/workspace3/ieeg/micro/word_boun_perce_v2/sub-009/ses-01/Temp_260416_121045.mat"),
        ],
        "regions": {
            "Amygdala": (1, 16),
            "VMPFC": (17, 32),
            "HG": (33, 48),
        },
    },
    {
        "subject": "sub11",
        "session": "ses02",
        "mat_files": [
            Path("/share/workspace3/ieeg/micro/word_boun_perce_v2/sub-011/ses-02/Temp_260615_105330.mat"),
            Path("/share/workspace3/ieeg/micro/word_boun_perce_v2/sub-011/ses-02/Temp_260615_110330.mat"),
            Path("/share/workspace3/ieeg/micro/word_boun_perce_v2/sub-011/ses-02/Temp_260615_111330.mat"),
        ],
        "regions": {
            "HG": (17, 32),
            "ATL": (33, 48),
        },
    },
]

METRIC_OUTPUT_CANDIDATES = {
    "num_spikes": ["num_spikes"],
    "snr": ["snr"],
    "isi_violation": [
        "isi_violations_ratio",
        "isi_violation",
        "isi_violation_ratio",
        "isi_violations",
        "isi_violations_rate",
    ],
    "firing_rate": ["firing_rate"],
    "presence_ratio": ["presence_ratio"],
}

UNIT_META_COLUMNS = [
    "region",
    "unit_id",
    "global_key",
    "best_channel_label",
    "best_channel_number",
    "n_spikes",
    "mean_fr_hz",
    "snr",
    "isi_violation",
    "firing_rate",
    "presence_ratio",
    "num_spikes_qm",
    "auto_qc_pass",
    "auto_qc_fail_reasons",
]


@dataclass
class MatMetadata:
    path: Path
    source_labels: list[str]
    labels: list[str]
    shape: tuple[int, int]
    n_samples: int
    t_first: float | None
    t_last: float | None


def prepare_parallel_settings() -> None:
    try:
        available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 2))
    except ValueError:
        available_cpus = os.cpu_count() or 2
    n_jobs = 1 if available_cpus <= 2 else min(available_cpus - 2, MAX_SPIKEINTERFACE_JOBS)

    os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    si.set_global_job_kwargs(
        n_jobs=n_jobs,
        chunk_duration="1s",
        progress_bar=True,
        max_threads_per_worker=1,
        pool_engine="process",
        mp_context="fork",
    )
    print(f"[INFO] Parallel setting: n_jobs={n_jobs}, available_cpus={available_cpus}")


def channel_number(label: str) -> int | None:
    match = re.search(r"(\d+)$", str(label).strip())
    return int(match.group(1)) if match else None


def canonical_label(label: str) -> str:
    number = channel_number(label)
    if number is None:
        return ""
    return f"A-{number:03d}"


def label_sort_key(label: str) -> tuple[int, str]:
    number = channel_number(label)
    return (number if number is not None else 10**9, label)


def decode_hdf5_name_dataset(h5f: h5py.File, ref: Any) -> str:
    try:
        arr = np.asarray(h5f[ref])
    except Exception:
        return ""
    flat = arr.ravel()
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(x) for x in flat).strip()
    if np.issubdtype(flat.dtype, np.integer):
        return "".join(chr(int(x)) for x in flat if 0 < int(x) < 256).strip()
    return ""


def decode_hdf5_channel_names(h5f: h5py.File, group_key: str) -> list[str]:
    if group_key not in h5f:
        return []
    group = h5f[group_key]
    for key in ("native_channel_name", "custom_channel_name"):
        if key not in group:
            continue
        refs = np.asarray(group[key], dtype=object).ravel()
        return [decode_hdf5_name_dataset(h5f, ref) for ref in refs]
    return []


def extract_legacy_channel_name(obj: Any, index: int) -> str:
    for field in ("native_channel_name", "custom_channel_name"):
        if hasattr(obj, field):
            value = getattr(obj, field)
            if isinstance(value, str):
                return value.strip()
            arr = np.asarray(value).ravel()
            if arr.dtype.kind in {"U", "S"}:
                return "".join(str(x) for x in arr).strip()
            if np.issubdtype(arr.dtype, np.integer):
                return "".join(chr(int(x)) for x in arr if int(x) != 0).strip()
        if isinstance(obj, np.void) and obj.dtype.names and field in obj.dtype.names:
            arr = np.asarray(obj[field]).ravel()
            if arr.dtype.kind in {"U", "S"}:
                return "".join(str(x) for x in arr).strip()
    return f"UNKNOWN-{index:03d}"


def orient_samples_channels(
    data: np.ndarray,
    n_labels: int,
    mat_path: Path,
) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected amplifier_data shape {arr.shape} in {mat_path}")
    if arr.shape[1] == n_labels:
        return arr
    if arr.shape[0] == n_labels:
        return arr.T
    raise ValueError(
        f"{mat_path.name}: amplifier_data shape {arr.shape} does not match "
        f"{n_labels} amplifier channel labels"
    )


def read_mat_metadata(mat_path: Path) -> MatMetadata:
    if h5py.is_hdf5(str(mat_path)):
        with h5py.File(mat_path, "r") as h5f:
            if "amplifier_data" not in h5f:
                raise KeyError(f"'amplifier_data' not found in {mat_path}")
            labels = decode_hdf5_channel_names(h5f, "amplifier_channels")
            raw_shape = tuple(int(x) for x in h5f["amplifier_data"].shape)
            t = np.asarray(h5f["t_amplifier"]).squeeze() if "t_amplifier" in h5f else None
    else:
        mat = loadmat(
            mat_path,
            variable_names=["amplifier_channels", "amplifier_data", "t_amplifier"],
            squeeze_me=True,
            struct_as_record=False,
        )
        if "amplifier_data" not in mat:
            raise KeyError(f"'amplifier_data' not found in {mat_path}")
        channel_objs = np.asarray(mat.get("amplifier_channels", [])).ravel()
        labels = [extract_legacy_channel_name(obj, i) for i, obj in enumerate(channel_objs, 1)]
        raw_shape = tuple(int(x) for x in np.asarray(mat["amplifier_data"]).shape)
        t = np.asarray(mat["t_amplifier"]).squeeze() if "t_amplifier" in mat else None

    source_labels = [str(label).strip() for label in labels]
    labels = [canonical_label(label) for label in source_labels]
    if not labels or any(not label for label in labels):
        raise ValueError(f"{mat_path.name}: amplifier channel labels could not be decoded reliably")
    if len(set(labels)) != len(labels):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(
            f"{mat_path.name}: source labels collapse to duplicate canonical labels: {duplicates}"
        )

    if raw_shape[1] == len(labels):
        n_samples = raw_shape[0]
    elif raw_shape[0] == len(labels):
        n_samples = raw_shape[1]
    else:
        raise ValueError(
            f"{mat_path.name}: amplifier shape {raw_shape} does not match {len(labels)} labels"
        )

    t_first = float(t[0]) if t is not None and t.ndim == 1 and t.size else None
    t_last = float(t[-1]) if t is not None and t.ndim == 1 and t.size else None
    return MatMetadata(
        path=mat_path,
        source_labels=source_labels,
        labels=labels,
        shape=raw_shape,
        n_samples=n_samples,
        t_first=t_first,
        t_last=t_last,
    )


def load_amplifier_data(mat_path: Path, n_labels: int) -> np.ndarray:
    if h5py.is_hdf5(str(mat_path)):
        with h5py.File(mat_path, "r") as h5f:
            data = np.asarray(h5f["amplifier_data"])
    else:
        mat = loadmat(mat_path, variable_names=["amplifier_data"])
        data = np.asarray(mat["amplifier_data"])
    return orient_samples_channels(data, n_labels=n_labels, mat_path=mat_path)


def inspect_channel_alignment(
    subject: str,
    session: str,
    metadata: list[MatMetadata],
    region_ranges: Dict[str, tuple[int, int]],
    session_output: Path,
) -> tuple[list[str], Dict[str, list[str]]]:
    label_sets = [set(item.labels) for item in metadata]
    common_labels = sorted(set.intersection(*label_sets), key=label_sort_key)
    union_labels = sorted(set.union(*label_sets), key=label_sort_key)
    excluded_labels = sorted(set(union_labels) - set(common_labels), key=label_sort_key)

    if not common_labels:
        raise ValueError(f"{subject}/{session}: no common amplifier channels across MAT files")

    region_labels: Dict[str, list[str]] = {}
    region_missing: Dict[str, list[str]] = {}
    empty_regions = []
    for region, (start, end) in region_ranges.items():
        expected = [f"A-{number:03d}" for number in range(start, end + 1)]
        retained = [label for label in expected if label in common_labels]
        missing = [label for label in expected if label not in common_labels]
        if not retained:
            empty_regions.append(region)
        region_labels[region] = retained
        region_missing[region] = missing

    reference_order = metadata[0].labels
    files_report = []
    mismatch = False
    for item in metadata:
        missing_vs_union = sorted(set(union_labels) - set(item.labels), key=label_sort_key)
        extra_vs_first = sorted(set(item.labels) - set(reference_order), key=label_sort_key)
        common_in_file_order = [label for label in item.labels if label in common_labels]
        order_matches = common_in_file_order == common_labels
        file_mismatch = bool(missing_vs_union or extra_vs_first or not order_matches)
        mismatch = mismatch or file_mismatch
        files_report.append(
            {
                "path": str(item.path),
                "channel_count": len(item.labels),
                "sample_count": item.n_samples,
                "source_to_canonical_labels": [
                    {"source_label": source, "canonical_label": canonical}
                    for source, canonical in zip(item.source_labels, item.labels)
                ],
                "labels": item.labels,
                "missing_labels_relative_to_session_union": missing_vs_union,
                "extra_labels_relative_to_first_file": extra_vs_first,
                "common_order_matches_canonical": order_matches,
                "t_amplifier_first": item.t_first,
                "t_amplifier_last": item.t_last,
            }
        )

    report = {
        "subject": subject,
        "session": session,
        "has_channel_mismatch": mismatch,
        "strategy": "intersection_by_channel_label; no zero padding",
        "union_channel_count": len(union_labels),
        "common_channel_count": len(common_labels),
        "union_labels": union_labels,
        "retained_common_labels": common_labels,
        "excluded_session_labels": excluded_labels,
        "region_retained_labels": region_labels,
        "region_missing_labels": region_missing,
        "files": files_report,
    }
    report_path = session_output / "channel_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    csv_rows = []
    for file_report in files_report:
        csv_rows.append(
            {
                "subject": subject,
                "session": session,
                "mat_file": file_report["path"],
                "channel_count": file_report["channel_count"],
                "sample_count": file_report["sample_count"],
                "missing_labels": ";".join(file_report["missing_labels_relative_to_session_union"]),
                "extra_vs_first": ";".join(file_report["extra_labels_relative_to_first_file"]),
                "order_matches": file_report["common_order_matches_canonical"],
                "session_excluded_labels": ";".join(excluded_labels),
            }
        )
    pd.DataFrame(csv_rows).to_csv(session_output / "channel_alignment_report.csv", index=False)
    label_map_rows = [
        {
            "subject": subject,
            "session": session,
            "mat_file": str(item.path),
            "channel_index": channel_index,
            "source_label": source_label,
            "canonical_label": canonical_label_value,
        }
        for item in metadata
        for channel_index, (source_label, canonical_label_value) in enumerate(
            zip(item.source_labels, item.labels)
        )
    ]
    pd.DataFrame(label_map_rows).to_csv(session_output / "channel_label_map.csv", index=False)

    if mismatch:
        print(
            f"[WARN] {subject}/{session}: channel mismatch detected; "
            f"excluded from all chunks: {excluded_labels}"
        )
    else:
        print(f"[INFO] {subject}/{session}: channel labels match across all MAT files")
    if empty_regions:
        raise ValueError(
            f"{subject}/{session}: no retained channels for region(s) "
            f"{', '.join(empty_regions)}; inspect {session_output / 'channel_label_map.csv'}"
        )
    return common_labels, region_labels


def build_aligned_concatenated_recording(
    metadata: list[MatMetadata],
    common_labels: list[str],
    session_output: Path,
):
    cache_root = session_output / "_aligned_recording_cache"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)

    saved_recordings = []
    timeline_rows = []
    concatenated_start = 0
    for index, item in enumerate(metadata):
        print(f"[LOAD] Aligning {item.path}")
        amplifier_uv = load_amplifier_data(item.path, n_labels=len(item.labels))
        label_to_column = {label: col for col, label in enumerate(item.labels)}
        columns = [label_to_column[label] for label in common_labels]
        aligned_uv = np.asarray(amplifier_uv[:, columns], dtype=np.float32)
        adc_float = np.rint(aligned_uv / GAIN_TO_UV)
        clipped_count = int(np.count_nonzero((adc_float < -32768) | (adc_float > 32767)))
        amplifier_int16 = np.clip(adc_float, -32768, 32767).astype(np.int16)

        recording = si.NumpyRecording(
            traces_list=[amplifier_int16],
            sampling_frequency=SAMPLING_RATE,
            channel_ids=common_labels,
        )
        recording.set_channel_gains(GAIN_TO_UV)
        recording.set_channel_offsets(0.0)
        chunk_folder = cache_root / f"chunk_{index + 1:02d}"
        saved = recording.save(
            format="binary",
            folder=str(chunk_folder),
            n_jobs=1,
            chunk_duration="1s",
        )
        saved_recordings.append(saved)

        n_samples = saved.get_num_samples()
        timeline_rows.append(
            {
                "chunk_index": index + 1,
                "mat_file": str(item.path),
                "concatenated_start_sample": concatenated_start,
                "concatenated_end_sample_exclusive": concatenated_start + n_samples,
                "concatenated_start_s": concatenated_start / SAMPLING_RATE,
                "concatenated_end_s": (concatenated_start + n_samples) / SAMPLING_RATE,
                "source_t_first": item.t_first,
                "source_t_last": item.t_last,
                "int16_clipped_values": clipped_count,
            }
        )
        concatenated_start += n_samples
        del amplifier_uv, aligned_uv, adc_float, amplifier_int16, recording

    pd.DataFrame(timeline_rows).to_csv(session_output / "chunk_timeline.csv", index=False)
    return si.concatenate_recordings(saved_recordings, ignore_times=True)


def get_metric_request_info() -> tuple[list[str], Dict[str, str]]:
    available = set(ComputeQualityMetrics.get_available_metric_names())
    requested: list[str] = []
    metric_map: Dict[str, str] = {}
    for logical_name, candidates in METRIC_OUTPUT_CANDIDATES.items():
        selected = next((candidate for candidate in candidates if candidate in available), None)
        if selected:
            requested.append(selected)
            metric_map[logical_name] = selected
    return list(dict.fromkeys(requested)), metric_map


def resolve_metric_map(qm: pd.DataFrame, metric_map: Dict[str, str]) -> Dict[str, str]:
    resolved = dict(metric_map)
    for logical_name, candidates in METRIC_OUTPUT_CANDIDATES.items():
        current = resolved.get(logical_name)
        if current in qm.columns:
            continue
        selected = next((candidate for candidate in candidates if candidate in qm.columns), None)
        if selected:
            resolved[logical_name] = selected
    return resolved


def metric_value(qm_row: pd.Series, column: str | None, default: float = np.nan) -> float:
    if column is None:
        return default
    value = qm_row.get(column, default)
    return float(value) if value is not None else default


def evaluate_qc(
    qm_row: pd.Series,
    metric_map: Dict[str, str],
    n_spikes: int,
    mean_fr_hz: float,
) -> tuple[bool, str]:
    snr = metric_value(qm_row, metric_map.get("snr"))
    isi = metric_value(qm_row, metric_map.get("isi_violation"))
    firing_rate = metric_value(qm_row, metric_map.get("firing_rate"))
    presence = metric_value(qm_row, metric_map.get("presence_ratio"))
    num_spikes_qm = metric_value(qm_row, metric_map.get("num_spikes"))
    reasons = []
    if np.isnan(snr) or snr < QC_MIN_SNR:
        reasons.append(f"snr<{QC_MIN_SNR}")
    if np.isnan(isi):
        if QC_REQUIRE_ISI_METRIC:
            reasons.append("isi_missing")
    elif isi > QC_MAX_ISI_VIOLATION:
        reasons.append(f"isi>{QC_MAX_ISI_VIOLATION}")
    if n_spikes < QC_MIN_NUM_SPIKES:
        reasons.append(f"n_spikes<{QC_MIN_NUM_SPIKES}")
    if mean_fr_hz < QC_MIN_FIRING_RATE_HZ:
        reasons.append(f"fr<{QC_MIN_FIRING_RATE_HZ}")
    if np.isnan(presence) or presence < QC_MIN_PRESENCE_RATIO:
        reasons.append(f"presence<{QC_MIN_PRESENCE_RATIO}")
    if not np.isnan(firing_rate) and firing_rate < QC_MIN_FIRING_RATE_HZ:
        reasons.append("firing_rate_qm_low")
    if not np.isnan(num_spikes_qm) and num_spikes_qm < QC_MIN_NUM_SPIKES:
        reasons.append("num_spikes_qm_low")
    reasons = list(dict.fromkeys(reasons))
    return not reasons, ";".join(reasons)


def sort_one_region(
    recording_full,
    region_name: str,
    region_labels: list[str],
    total_duration_s: float,
    session_output: Path,
) -> tuple[dict, dict, list[dict]]:
    print(f"[INFO] Sorting {region_name}: {region_labels}")
    region_root = session_output / region_name
    sort_dir = region_root / "sorting"
    analyzer_dir = region_root / "analyzer"
    phy_dir = region_root / "phy"
    spike_dir = region_root / "spike_times"
    spike_dir.mkdir(parents=True, exist_ok=True)
    for folder in (sort_dir, analyzer_dir, phy_dir):
        if folder.exists():
            shutil.rmtree(folder)

    recording_sub = recording_full.select_channels(channel_ids=region_labels)
    probe = generate_linear_probe(num_elec=len(region_labels), ypitch=1000)
    probe.set_device_channel_indices(np.arange(len(region_labels)))
    recording_sub = recording_sub.set_probe(probe)
    recording_sub.set_channel_groups(np.arange(len(region_labels)))
    recording_bp = si.bandpass_filter(recording_sub, freq_min=300, freq_max=6000)
    recording_cmr = si.common_reference(recording_bp, reference="global", operator="median")

    sorting = si.run_sorter(
        sorter_name="mountainsort4",
        recording=recording_cmr,
        folder=str(sort_dir),
        verbose=True,
        detect_threshold=SORTER_DETECT_THRESHOLD,
        freq_min=300,
        freq_max=6000,
        adjacency_radius=-1,
    )
    unit_ids = sorting.get_unit_ids()
    if len(unit_ids) == 0:
        print(f"[WARN] {region_name}: no units found")
        return {}, {}, []

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording_cmr,
        format="binary_folder",
        folder=str(analyzer_dir),
        return_in_uV=True,
    )
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
    analyzer.compute("templates")
    analyzer.compute("noise_levels")
    analyzer.compute("spike_amplitudes")
    analyzer.compute("unit_locations")
    analyzer.compute("correlograms")
    analyzer.compute("template_similarity")

    requested_metrics, metric_map = get_metric_request_info()
    analyzer.compute("quality_metrics", metric_names=requested_metrics)
    qm = analyzer.get_extension("quality_metrics").get_data()
    metric_map = resolve_metric_map(qm, metric_map)
    si.export_to_phy(
        analyzer,
        output_folder=str(phy_dir),
        compute_pc_features=False,
        compute_amplitudes=True,
    )

    templates = analyzer.get_extension("templates").get_data()
    all_spikes: dict[str, np.ndarray] = {}
    good_spikes: dict[str, np.ndarray] = {}
    metadata_rows = []
    for unit_index, unit_id in enumerate(unit_ids):
        spike_samples = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)
        spike_seconds = spike_samples / SAMPLING_RATE
        global_key = f"{region_name}_unit{unit_id}"
        all_spikes[global_key] = spike_seconds
        np.save(spike_dir / f"{global_key}_spikes_sec.npy", spike_seconds)

        try:
            template = templates[unit_index]
            ptp = template.max(axis=0) - template.min(axis=0)
            best_label = region_labels[int(np.argmax(ptp))]
        except Exception:
            best_label = ""

        qm_row = qm.loc[unit_id] if unit_id in qm.index else pd.Series(dtype=float)
        mean_fr_hz = len(spike_samples) / total_duration_s
        passed, reasons = evaluate_qc(qm_row, metric_map, len(spike_samples), mean_fr_hz)
        if passed:
            good_spikes[global_key] = spike_seconds

        metadata_rows.append(
            {
                "region": region_name,
                "unit_id": unit_id,
                "global_key": global_key,
                "best_channel_label": best_label,
                "best_channel_number": channel_number(best_label) if best_label else -1,
                "n_spikes": int(len(spike_samples)),
                "mean_fr_hz": round(mean_fr_hz, 4),
                "snr": round(metric_value(qm_row, metric_map.get("snr")), 4),
                "isi_violation": round(metric_value(qm_row, metric_map.get("isi_violation")), 4),
                "firing_rate": round(metric_value(qm_row, metric_map.get("firing_rate")), 4),
                "presence_ratio": round(metric_value(qm_row, metric_map.get("presence_ratio")), 4),
                "num_spikes_qm": metric_value(qm_row, metric_map.get("num_spikes"), -1),
                "auto_qc_pass": passed,
                "auto_qc_fail_reasons": reasons,
            }
        )
    print(f"[INFO] {region_name}: units={len(unit_ids)}, QC-passing={len(good_spikes)}")
    return all_spikes, good_spikes, metadata_rows


def process_session(config: dict) -> None:
    subject = config["subject"]
    session = config["session"]
    mat_files = [Path(path) for path in config["mat_files"]]
    missing_files = [path for path in mat_files if not path.exists()]
    if missing_files:
        print(f"[ERROR] {subject}/{session}: missing MAT files: {missing_files}")
        return

    session_output = OUTPUT_ROOT / f"sorting_results_{subject}_{session}_v3"
    session_output.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print(f"[SESSION] {subject}/{session}: {len(mat_files)} MAT file(s)")
    print("=" * 88)

    metadata = [read_mat_metadata(path) for path in mat_files]
    common_labels, region_labels = inspect_channel_alignment(
        subject=subject,
        session=session,
        metadata=metadata,
        region_ranges=config["regions"],
        session_output=session_output,
    )
    recording = build_aligned_concatenated_recording(
        metadata=metadata,
        common_labels=common_labels,
        session_output=session_output,
    )
    total_duration_s = recording.get_total_duration()
    print(
        f"[INFO] Concatenated recording: channels={recording.get_num_channels()}, "
        f"duration={total_duration_s / 60:.2f} min"
    )

    all_spikes: dict[str, np.ndarray] = {}
    all_good_spikes: dict[str, np.ndarray] = {}
    all_metadata: list[dict] = []
    for region_name, labels in region_labels.items():
        region_spikes, good_spikes, rows = sort_one_region(
            recording_full=recording,
            region_name=region_name,
            region_labels=labels,
            total_duration_s=total_duration_s,
            session_output=session_output,
        )
        all_spikes.update(region_spikes)
        all_good_spikes.update(good_spikes)
        all_metadata.extend(rows)

    summary = pd.DataFrame(all_metadata, columns=UNIT_META_COLUMNS)
    summary.to_csv(session_output / "all_regions_units_summary.csv", index=False)
    if summary.empty:
        pd.DataFrame(columns=UNIT_META_COLUMNS).to_csv(
            session_output / "good_units_summary.csv", index=False
        )
    else:
        summary[summary["auto_qc_pass"]].to_csv(
            session_output / "good_units_summary.csv", index=False
        )
    with (session_output / "all_spike_times.pkl").open("wb") as file:
        pickle.dump(all_spikes, file, protocol=4)
    with (session_output / "good_units_spike_times.pkl").open("wb") as file:
        pickle.dump(all_good_spikes, file, protocol=4)
    print(f"[DONE] Session output: {session_output}")


def main() -> None:
    prepare_parallel_settings()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for config in SESSION_CONFIGS:
        try:
            process_session(config)
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[ERROR] {config['subject']}/{config['session']}: "
                f"{type(exc).__name__}: {exc}"
            )
    print("[DONE] All requested v3 sessions finished.")


if __name__ == "__main__":
    main()
