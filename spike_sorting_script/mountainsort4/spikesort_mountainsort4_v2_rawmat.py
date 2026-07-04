"""
Raw-MAT variant of spikesort_mountainsort4_v2.py
------------------------------------------------
Changes from the original script:
  - only processes sub5/sub6;
  - reads each raw .mat file directly (no merged binary input);
  - runs sorting per .mat file found under configured raw folders.

All sorting/QC parameters are kept identical to the original script.

Channel handling for sub5/sub6 raw MAT files:
  - nominal 128-channel files: include 1-based channels 64-128 for spike sorting;
  - 127-channel files (missing one recorded channel): include 1-based 63-127.
"""

import os
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any

# Set thread caps before importing numpy/scipy stacks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd
import spikeinterface.full as si
from probeinterface import generate_linear_probe
from scipy.io import loadmat
from spikeinterface.metrics.quality.quality_metrics import ComputeQualityMetrics

warnings.simplefilter("ignore")


# Output directory (each MAT file gets its own sorting_results_<name>_v2_rawmat folder)
OUTPUT_ROOT = Path("/share/home/mitan/spike_sorting/mountainsort4")

# Recording metadata
SAMPLING_RATE = 30000.0
GAIN_TO_UV = 0.195

# Raw folders requested by user. The script recursively discovers *.mat under each.
RAW_FOLDERS = [
    {"subject": "sub5", "path": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5")},
    {"subject": "sub6", "path": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_1")},
    {"subject": "sub6", "path": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_1")},
    {"subject": "sub6", "path": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_1")},
]

MAT_GLOB = "*.mat"


# Sorting/QC thresholds (identical to original)
SORTER_DETECT_THRESHOLD = 7.0
QC_MIN_SNR = 5.0
QC_MAX_ISI_VIOLATION = 0.02
QC_MIN_NUM_SPIKES = 200
QC_MIN_FIRING_RATE_HZ = 0.18
QC_MIN_PRESENCE_RATIO = 0.75
QC_REQUIRE_ISI_METRIC = False
MAX_SPIKEINTERFACE_JOBS = 4

METRIC_OUTPUT_CANDIDATES = {
    "num_spikes": ["num_spikes"],
    "snr": ["snr"],
    "isi_violation": [
        "isi_violations_ratio",
        "isi_violation",
        "isi_violation_ratio",
        "isi_violations",
        "isi_violations_rate",
        "isi_violation_rate",
    ],
    "firing_rate": ["firing_rate"],
    "presence_ratio": ["presence_ratio"],
}


def prepare_parallel_settings():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            available_cpus = int(slurm_cpus)
        except ValueError:
            available_cpus = os.cpu_count() or 2
    else:
        available_cpus = os.cpu_count() or 2

    n_jobs = max(1, min(available_cpus - 2, MAX_SPIKEINTERFACE_JOBS))
    if available_cpus <= 2:
        n_jobs = 1

    os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    si.set_global_job_kwargs(
        n_jobs=n_jobs,
        chunk_duration="1s",
        progress_bar=True,
        max_threads_per_worker=1,
        pool_engine="process",
        mp_context="fork",
    )
    print(
        "[INFO] Parallel setting: "
        f"n_jobs={n_jobs} (available_cpus={available_cpus}, BLAS threads=1)"
    )


def get_metric_request_info():
    available_metrics = set(ComputeQualityMetrics.get_available_metric_names())
    requested = []
    metric_map = {}
    for logical_name, candidates in METRIC_OUTPUT_CANDIDATES.items():
        selected = next((c for c in candidates if c in available_metrics), None)
        if selected is not None:
            requested.append(selected)
            metric_map[logical_name] = selected
    requested = list(dict.fromkeys(requested))
    return requested, metric_map


def resolve_metric_map_from_qm_columns(
    metric_map: Dict[str, str],
    qm: pd.DataFrame,
) -> Dict[str, str]:
    if qm.empty:
        return dict(metric_map)

    qm_columns = list(qm.columns)
    qm_column_set = set(qm_columns)
    resolved = dict(metric_map)

    for logical_name, candidates in METRIC_OUTPUT_CANDIDATES.items():
        current = resolved.get(logical_name)
        if current in qm_column_set:
            continue

        selected = next((c for c in candidates if c in qm_column_set), None)
        if selected is None and logical_name == "isi_violation":
            selected = next(
                (c for c in qm_columns if "isi" in c.lower() and "ratio" in c.lower()),
                None,
            )
        if selected is None and logical_name == "isi_violation":
            selected = next((c for c in qm_columns if "isi" in c.lower()), None)
        if selected is not None:
            resolved[logical_name] = selected
    return resolved


def evaluate_qc_pass(
    qm_row: pd.Series,
    metric_map: Dict[str, str],
    n_spikes: int,
    mean_fr_hz: float,
) -> tuple[bool, str]:
    fail_reasons = []
    snr_col = metric_map.get("snr")
    isi_col = metric_map.get("isi_violation")
    fr_col = metric_map.get("firing_rate")
    pr_col = metric_map.get("presence_ratio")
    ns_col = metric_map.get("num_spikes")

    snr_val = qm_row.get(snr_col, np.nan) if snr_col else np.nan
    isi_val = qm_row.get(isi_col, np.nan) if isi_col else np.nan
    fr_val = qm_row.get(fr_col, np.nan) if fr_col else np.nan
    pr_val = qm_row.get(pr_col, np.nan) if pr_col else np.nan
    ns_qm_val = qm_row.get(ns_col, np.nan) if ns_col else np.nan

    if np.isnan(snr_val) or float(snr_val) < QC_MIN_SNR:
        fail_reasons.append(f"snr<{QC_MIN_SNR}")
    if np.isnan(isi_val):
        if QC_REQUIRE_ISI_METRIC:
            fail_reasons.append("isi_missing")
    elif float(isi_val) > QC_MAX_ISI_VIOLATION:
        fail_reasons.append(f"isi>{QC_MAX_ISI_VIOLATION}")
    if n_spikes < QC_MIN_NUM_SPIKES:
        fail_reasons.append(f"n_spikes<{QC_MIN_NUM_SPIKES}")
    if mean_fr_hz < QC_MIN_FIRING_RATE_HZ:
        fail_reasons.append(f"fr<{QC_MIN_FIRING_RATE_HZ}")
    if np.isnan(pr_val) or float(pr_val) < QC_MIN_PRESENCE_RATIO:
        fail_reasons.append(f"presence<{QC_MIN_PRESENCE_RATIO}")
    if not np.isnan(fr_val) and float(fr_val) < QC_MIN_FIRING_RATE_HZ:
        fail_reasons.append("firing_rate_qm_low")
    if not np.isnan(ns_qm_val) and float(ns_qm_val) < QC_MIN_NUM_SPIKES:
        fail_reasons.append("num_spikes_qm_low")

    fail_reasons = list(dict.fromkeys(fail_reasons))
    return len(fail_reasons) == 0, ";".join(fail_reasons)


def orient_samples_channels(data: np.ndarray, mat_path: Path) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected amplifier_data shape {arr.shape} in {mat_path}")

    # Prefer orientation where channel count is plausible for these datasets.
    plausible = {127, 128}
    if arr.shape[1] in plausible:
        return arr
    if arr.shape[0] in plausible and arr.shape[1] not in plausible:
        return arr.T

    # Fallback: samples dimension is usually much larger than channels.
    return arr if arr.shape[0] > arr.shape[1] else arr.T


def load_amplifier_from_mat(mat_path: Path) -> np.ndarray:
    """Load amplifier_data from one MAT file as samples x channels float32 (uV)."""
    if h5py.is_hdf5(str(mat_path)):
        with h5py.File(mat_path, "r") as f:
            if "amplifier_data" not in f:
                raise KeyError(f"'amplifier_data' not found in {mat_path}")
            amp_data = np.array(f["amplifier_data"])
    else:
        mat = loadmat(mat_path)
        if "amplifier_data" not in mat:
            raise KeyError(f"'amplifier_data' not found in {mat_path}")
        amp_data = np.array(mat["amplifier_data"])

    amp = orient_samples_channels(amp_data, mat_path=mat_path).astype(np.float32, copy=False)
    if amp.shape[1] < 127:
        raise ValueError(
            f"{mat_path.name}: amplifier channels={amp.shape[1]} (<127), "
            "cannot apply requested sub5/sub6 channel policy."
        )
    if amp.shape[1] > 128:
        print(
            f"[WARN] {mat_path.name}: amplifier channels={amp.shape[1]} (>128). "
            "Truncating to first 128 channels."
        )
        amp = amp[:, :128]
    return amp


def get_region_channel_map_for_raw(subject: str, n_channels: int) -> Dict[str, tuple]:
    """
    Build region map for raw MAT channels using requested policy:
      - if n_channels==128: spike bank is 1-based 64..128;
      - if n_channels==127: spike bank is 1-based 63..127.
    """
    if subject not in {"sub5", "sub6"}:
        raise ValueError(f"Unsupported subject for raw script: {subject}")

    if n_channels == 128:
        spike_start_1idx = 64
        spike_end_1idx = 128
    elif n_channels == 127:
        spike_start_1idx = 63
        spike_end_1idx = 127
    else:
        raise ValueError(f"Unsupported channel count {n_channels}; expected 127 or 128.")

    ch_start = spike_start_1idx - 1  # python index inclusive
    ch_end_exclusive = spike_end_1idx  # python index exclusive
    total_spike_ch = ch_end_exclusive - ch_start
    if total_spike_ch < 64:
        raise ValueError(
            f"Spike channel bank too small ({total_spike_ch}) for {subject} n_channels={n_channels}."
        )

    # Keep original 16-channel region widths, assign any remainder to final region.
    offsets = [0, 16, 32, 48, total_spike_ch]
    if subject == "sub5":
        order = ["ATL", "HG", "VMPFC", "Amygdala"]
    else:
        order = ["ATL", "HG", "Amygdala", "VMPFC"]

    region_map = {}
    for i, region in enumerate(order):
        rs = ch_start + offsets[i]
        re = ch_start + offsets[i + 1]
        region_map[region] = (rs, re)
    return region_map


def build_recording_from_amplifier(amplifier_uv: np.ndarray):
    """
    Build SpikeInterface recording from amplifier traces.
    To keep scaling close to the original binary flow, convert to int16 ADC units
    using the same GAIN_TO_UV that was used there.
    """
    amplifier_int16 = np.round(amplifier_uv / GAIN_TO_UV).astype(np.int16, copy=False)
    recording = si.NumpyRecording(traces_list=[amplifier_int16], sampling_frequency=SAMPLING_RATE)
    return recording


def make_recording_sortable(recording_mem, session_output: Path):
    """
    Mountainsort4 requires a serializable recording object.
    Persist in-memory traces to a binary folder and return that extractor.
    """
    recording_cache_dir = session_output / "_recording_cache"
    if recording_cache_dir.exists():
        shutil.rmtree(recording_cache_dir)
    print(f"[INFO] Saving temporary binary recording cache: {recording_cache_dir}")
    recording_saved = recording_mem.save(
        format="binary",
        folder=str(recording_cache_dir),
        n_jobs=1,
        chunk_duration="1s",
    )
    return recording_saved


def sort_one_region(
    recording_full,
    total_duration_s: float,
    region_name: str,
    ch_start: int,
    ch_end: int,
    session_output: Path,
):
    print(f"[INFO] Sorting region: {region_name} (channels {ch_start + 1}-{ch_end})")
    n_ch = ch_end - ch_start
    if n_ch <= 0:
        print(f"[WARN] Empty region {region_name} [{ch_start}:{ch_end}], skipping.")
        return {}, {}, []

    region_root = session_output / region_name
    sort_dir = region_root / "sorting"
    analyzer_dir = region_root / "analyzer"
    phy_dir = region_root / "phy"
    spike_dir = region_root / "spike_times"
    spike_dir.mkdir(parents=True, exist_ok=True)

    for folder in (sort_dir, analyzer_dir, phy_dir):
        if folder.exists():
            shutil.rmtree(folder)

    channel_ids = recording_full.get_channel_ids()[ch_start:ch_end]
    recording_sub = recording_full.select_channels(channel_ids=channel_ids)

    probe = generate_linear_probe(num_elec=n_ch, ypitch=1000)
    probe.set_device_channel_indices(np.arange(n_ch))
    recording_sub = recording_sub.set_probe(probe)
    recording_sub.set_channel_groups(np.arange(n_ch))

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

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording_cmr,
        format="binary_folder",
        folder=str(analyzer_dir),
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
    if requested_metrics:
        if "isi_violation" in metric_map:
            analyzer.compute("quality_metrics", metric_names=requested_metrics)
        else:
            print(
                "[WARN] ISI metric name not matched in available metric list; "
                "computing default quality metrics for column inference."
            )
            analyzer.compute("quality_metrics")
        qm = analyzer.get_extension("quality_metrics").get_data()
        metric_map = resolve_metric_map_from_qm_columns(metric_map=metric_map, qm=qm)
    else:
        qm = pd.DataFrame()

    si.export_to_phy(
        analyzer,
        output_folder=str(phy_dir),
        compute_pc_features=False,
        compute_amplitudes=True,
    )

    unit_ids = sorting.get_unit_ids()
    templates_data = analyzer.get_extension("templates").get_data()
    region_spike_times = {}
    region_good_spike_times = {}
    region_meta = []

    for unit_id in unit_ids:
        spike_samples = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)
        spike_times_s = spike_samples / SAMPLING_RATE
        np.save(spike_dir / f"{region_name}_unit{unit_id}_spikes_sec.npy", spike_times_s)

        global_key = f"{region_name}_unit{unit_id}"
        region_spike_times[global_key] = spike_times_s

        try:
            unit_idx = list(unit_ids).index(unit_id)
            template_unit = templates_data[unit_idx]
            ptp_per_ch = template_unit.max(axis=0) - template_unit.min(axis=0)
            best_ch_local = int(np.argmax(ptp_per_ch))
            best_ch_global_1idx = ch_start + best_ch_local + 1
        except Exception:
            best_ch_global_1idx = -1

        qm_row = qm.loc[unit_id] if unit_id in qm.index else pd.Series(dtype=float)
        mean_fr_hz = len(spike_samples) / total_duration_s
        auto_qc_pass, auto_qc_fail_reasons = evaluate_qc_pass(
            qm_row=qm_row,
            metric_map=metric_map,
            n_spikes=int(len(spike_samples)),
            mean_fr_hz=mean_fr_hz,
        )
        if auto_qc_pass:
            region_good_spike_times[global_key] = spike_times_s

        region_meta.append(
            {
                "region": region_name,
                "unit_id": unit_id,
                "global_key": global_key,
                "best_channel_1idx": best_ch_global_1idx,
                "n_spikes": int(len(spike_samples)),
                "mean_fr_hz": round(mean_fr_hz, 4),
                "snr": round(float(qm_row.get(metric_map.get("snr"), np.nan)), 4),
                "isi_violation": round(float(qm_row.get(metric_map.get("isi_violation"), np.nan)), 4),
                "firing_rate": round(float(qm_row.get(metric_map.get("firing_rate"), np.nan)), 4),
                "presence_ratio": round(float(qm_row.get(metric_map.get("presence_ratio"), np.nan)), 4),
                "num_spikes_qm": int(qm_row.get(metric_map.get("num_spikes"), -1)),
                "auto_qc_pass": auto_qc_pass,
                "auto_qc_fail_reasons": auto_qc_fail_reasons,
            }
        )

    print(
        f"[INFO] Region {region_name}: found {len(unit_ids)} units, "
        f"QC-passing {len(region_good_spike_times)} units."
    )
    return region_spike_times, region_good_spike_times, region_meta


def sort_one_mat_file(subject: str, mat_path: Path):
    if not mat_path.exists():
        print(f"[WARN] Input file not found: {mat_path}. Skipping.")
        return

    session_name = f"{subject}_{mat_path.parent.name}_{mat_path.stem}"
    print("=" * 88)
    print(f"[START] MAT file {session_name}: {mat_path}")
    print("=" * 88)

    try:
        amplifier_uv = load_amplifier_from_mat(mat_path)
    except Exception as exc:
        print(f"[WARN] Failed loading {mat_path}: {type(exc).__name__}: {exc}")
        return

    n_channels = amplifier_uv.shape[1]
    region_channel_map = get_region_channel_map_for_raw(subject=subject, n_channels=n_channels)
    print(f"[INFO] amplifier_data shape: samples={amplifier_uv.shape[0]}, channels={n_channels}")
    print(f"[INFO] region map for this file: {region_channel_map}")

    session_output = OUTPUT_ROOT / f"sorting_results_{session_name}_v2_rawmat"
    session_output.mkdir(parents=True, exist_ok=True)

    recording_mem = build_recording_from_amplifier(amplifier_uv=amplifier_uv)
    recording_full = make_recording_sortable(recording_mem=recording_mem, session_output=session_output)
    total_duration_s = recording_full.get_num_samples() / SAMPLING_RATE
    print(f"[INFO] Recording duration: {total_duration_s / 60:.2f} min")

    all_spike_times = {}
    all_good_spike_times = {}
    all_units_meta = []

    for region_name, (ch_start, ch_end) in region_channel_map.items():
        region_spikes, region_good_spikes, region_meta = sort_one_region(
            recording_full=recording_full,
            total_duration_s=total_duration_s,
            region_name=region_name,
            ch_start=ch_start,
            ch_end=ch_end,
            session_output=session_output,
        )
        all_spike_times.update(region_spikes)
        all_good_spike_times.update(region_good_spikes)
        all_units_meta.extend(region_meta)

    summary_df = pd.DataFrame(all_units_meta)
    summary_csv = session_output / "all_regions_units_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    good_summary_csv = session_output / "good_units_summary.csv"
    summary_df[summary_df["auto_qc_pass"]].to_csv(good_summary_csv, index=False)

    all_spikes_pkl = session_output / "all_spike_times.pkl"
    with all_spikes_pkl.open("wb") as f:
        pickle.dump(all_spike_times, f, protocol=4)
    good_spikes_pkl = session_output / "good_units_spike_times.pkl"
    with good_spikes_pkl.open("wb") as f:
        pickle.dump(all_good_spike_times, f, protocol=4)

    print(f"[DONE] MAT file {session_name} outputs:")
    print(f"       - {summary_csv}")
    print(f"       - {good_summary_csv}")
    print(f"       - {all_spikes_pkl}")
    print(f"       - {good_spikes_pkl}")


def discover_mat_jobs():
    jobs = []
    seen = set()
    for cfg in RAW_FOLDERS:
        subject = cfg["subject"]
        root = Path(cfg["path"])
        key = (subject, str(root))
        if key in seen:
            continue
        seen.add(key)

        if not root.exists():
            print(f"[WARN] Raw folder missing: {root}")
            continue
        mats = sorted(root.rglob(MAT_GLOB))
        if not mats:
            print(f"[WARN] No MAT files found under {root}")
            continue
        print(f"[INFO] Discovered {len(mats)} MAT files under {root} for {subject}")
        for mat_path in mats:
            jobs.append({"subject": subject, "path": mat_path})
    return jobs


def main():
    prepare_parallel_settings()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = discover_mat_jobs()
    if not jobs:
        print("[DONE] No MAT files to process.")
        return

    for job in jobs:
        sort_one_mat_file(subject=job["subject"], mat_path=job["path"])
    print("[DONE] All requested raw MAT files finished.")


if __name__ == "__main__":
    main()

