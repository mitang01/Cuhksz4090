#   ATL:      Ch 32-47
#   HG:       Ch 16-31
#   VMPFC:    Ch 48-63
#   Amygdala: Ch 0-15


# Other notes:
# Bandpass: 300–6000 Hz
# Sorter: mountainsort4
# SORTER_DETECT_THRESHOLD = 7.0
# Quality metrics: num_spikes, snr, isi_violation, firing_rate, presence_ratio
# QC thresholds: snr>=5.0, isi<=0.02, n_spikes>=200, fr>=0.18, presence>=0.75
# QC pass: all metrics pass
# QC fail: any metric fails (see auto_qc_fail_reasons)

import os
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Dict

# Set thread caps before importing numpy/scipy stacks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import spikeinterface.full as si
from probeinterface import generate_linear_probe
from spikeinterface.metrics.quality.quality_metrics import ComputeQualityMetrics

warnings.simplefilter("ignore")


# Input files produced by merge.py
MERGED_RECORDINGS = {
    "session3": Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4/bistable_sub4_session3"),
    "session5": Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4/bistable_sub4_session5"),
    "session2": Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4/bistable_sub4_session2"),
}

# Output directory requested by user
OUTPUT_ROOT = Path("/share/home/mitan/spike_sorting/mountainsort4")

# Recording metadata
SAMPLING_RATE = 30000.0
NUM_CHANNELS_TOTAL = 128
GAIN_TO_UV = 0.195

# Brain-region channels (0-indexed, end-exclusive)
REGION_CHANNEL_MAP = {
    "ATL": (32, 48),       # channels 32-47
    "HG": (16, 32),        # channels 16-31
    "VMPFC": (48, 64),     # channels 48-63
    "Amygdala": (0, 16)    # channels 0-15
}

# Sorting/QC thresholds (conservative but not all-rejecting)
SORTER_DETECT_THRESHOLD = 7.0
QC_MIN_SNR = 5.0
QC_MAX_ISI_VIOLATION = 0.02
QC_MIN_NUM_SPIKES = 200
QC_MIN_FIRING_RATE_HZ = 0.18
QC_MIN_PRESENCE_RATIO = 0.75
# Some runs export no ISI metric; when missing, do not auto-fail every unit.
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

    # Keep worker count conservative on clusters with strict RLIMIT_NPROC.
    n_jobs = max(1, min(available_cpus - 2, MAX_SPIKEINTERFACE_JOBS))
    if available_cpus <= 2:
        n_jobs = 1

    # Keep BLAS/OpenMP single-threaded in each worker to prevent thread explosion.
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
    metric_candidates = METRIC_OUTPUT_CANDIDATES

    requested = []
    metric_map = {}
    for logical_name, candidates in metric_candidates.items():
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
    """Resolve metric names against actual quality-metric output columns."""
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
            # If alias matching misses ISI in this SpikeInterface version, compute
            # default quality metrics and infer an ISI-like column from output.
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


def sort_one_session(session_name: str, recording_path: Path):
    if not recording_path.exists():
        print(f"[WARN] Input file not found: {recording_path}. Skipping {session_name}.")
        return

    print("=" * 72)
    print(f"[START] Session {session_name}: {recording_path}")
    print("=" * 72)

    session_output = OUTPUT_ROOT / f"sorting_results_{session_name}_v2"
    session_output.mkdir(parents=True, exist_ok=True)

    recording_full = si.read_binary(
        file_paths=str(recording_path),
        num_channels=NUM_CHANNELS_TOTAL,
        dtype="int16",
        sampling_frequency=SAMPLING_RATE,
        gain_to_uV=GAIN_TO_UV,
        time_axis=0,
    )
    total_duration_s = recording_full.get_num_samples() / SAMPLING_RATE
    print(f"[INFO] Recording duration: {total_duration_s / 60:.2f} min")

    all_spike_times = {}
    all_good_spike_times = {}
    all_units_meta = []

    for region_name, (ch_start, ch_end) in REGION_CHANNEL_MAP.items():
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

    print(f"[DONE] Session {session_name} outputs:")
    print(f"       - {summary_csv}")
    print(f"       - {good_summary_csv}")
    print(f"       - {all_spikes_pkl}")
    print(f"       - {good_spikes_pkl}")


def main():
    prepare_parallel_settings()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for session_name, recording_path in MERGED_RECORDINGS.items():
        sort_one_session(session_name=session_name, recording_path=recording_path)
    print("[DONE] All requested sessions finished.")


if __name__ == "__main__":
    main()

