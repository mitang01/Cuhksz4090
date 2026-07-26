#!/usr/bin/env python3
"""Run the two-participant ERP and Python source-connectivity pilot.

This preserves the trial, ERP, sensor-statistics, and template warnings from
``run_dcm_pilot.py``. The SPM DCM stage is replaced by source-space state-space
Granger causality and time-reversed Granger causality from MNE-Connectivity.
The results are directed connectivity, not Dynamic Causal Modelling.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import mne_connectivity
import networkx as nx
import numpy as np
from mne.minimum_norm import (
    apply_inverse_epochs,
    make_inverse_operator,
    write_inverse_operator,
)
from mne_connectivity import spectral_connectivity_epochs
from run_dcm_pilot import (
    SCRIPT_DIR,
    annotation_events,
    classify_trials,
    dcm_pre_speech_end_ms,
    descriptive_erp_statistics,
    epoch_picture,
    epoch_rest,
    load_datasets,
    plot_erps,
    preprocess_raw,
    raw_audit,
    select_roi_channels,
    temporal_cluster_statistics,
    write_csv,
    write_trial_table,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "directed_connectivity_results"
NODE_NAMES = ("OT", "pMTG", "ATL", "IFG")
NODE_INDEX = {name: index for index, name in enumerate(NODE_NAMES)}

# The same hypothesis edges as F1-F3, but these are edge sets rather than DCMs.
FAMILY_EDGES = {
    "F1_feedforward": (
        ("OT", "pMTG"),
        ("OT", "ATL"),
        ("pMTG", "IFG"),
        ("ATL", "IFG"),
    ),
    "F2_feedback": (
        ("OT", "pMTG"),
        ("OT", "ATL"),
        ("pMTG", "IFG"),
        ("ATL", "IFG"),
        ("IFG", "pMTG"),
        ("IFG", "ATL"),
    ),
    "F3_direct_route": (
        ("OT", "pMTG"),
        ("OT", "ATL"),
        ("pMTG", "IFG"),
        ("ATL", "IFG"),
        ("OT", "IFG"),
    ),
}
PLANNED_EDGES = tuple(
    dict.fromkeys(edge for edges in FAMILY_EDGES.values() for edge in edges)
)
LABEL_COMPONENTS = {
    "OT": ("lateraloccipital-lh", "fusiform-lh"),
    "pMTG": ("middletemporal-lh", "bankssts-lh"),
    "ATL": ("temporalpole-lh",),
    "IFG": ("parsopercularis-lh", "parstriangularis-lh", "parsorbitalis-lh"),
}
NODE_POSITIONS = {
    "OT": (0.0, 0.0),
    "pMTG": (1.0, 0.55),
    "ATL": (1.0, -0.55),
    "IFG": (2.0, 0.0),
}


@dataclass
class ConnectivityInput:
    participant: str
    analysis: str
    kind: str
    epochs_file: Path
    window_start_seconds: float
    window_end_seconds: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process the two picture/rest datasets and estimate source-space "
            "state-space Granger connectivity using only Python/MNE."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Analysis output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=None,
        help=(
            "FreeSurfer subjects directory. If fsaverage is absent, MNE "
            "downloads it here (default: MNE's standard data location)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional dataset manifest accepted by run_dcm_pilot.py.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--permutations", type=int, default=4096)
    parser.add_argument(
        "--connectivity-bootstraps",
        type=int,
        default=200,
        help="Epoch bootstrap replicates for directed-edge intervals.",
    )
    parser.add_argument(
        "--gc-lags",
        type=int,
        default=10,
        help="Maximum autoregressive lag count for state-space GC.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--tmin", type=float, default=-0.2)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--reject-uv", type=float, default=500.0)
    parser.add_argument("--flat-uv", type=float, default=0.5)
    parser.add_argument("--rest-epoch-seconds", type=float, default=2.0)
    return parser.parse_args(argv)


def process_picture(
    dataset: dict[str, str], args: argparse.Namespace
) -> list[ConnectivityInput]:
    participant = dataset["participant"]
    subject_dir = args.output_dir / participant
    subject_dir.mkdir(parents=True, exist_ok=True)
    vhdr = Path(dataset["vhdr"])
    raw, montage_report = preprocess_raw(vhdr, args.l_freq, args.h_freq)
    audit = raw_audit(raw, vhdr, montage_report)
    audit["analysis_name"] = "directed connectivity (not DCM)"
    (subject_dir / "picture_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    trials = classify_trials(annotation_events(raw), raw.info["sfreq"], raw.first_samp)
    write_trial_table(subject_dir / "picture_trials.csv", participant, trials)

    analyses: list[ConnectivityInput] = []
    for variant, correct_only in (("picture_all", False), ("picture_correct", True)):
        endpoint_ms = dcm_pre_speech_end_ms(
            trials,
            raw.info["sfreq"],
            correct_only=correct_only,
            epoch_tmax=args.tmax,
        )
        LOGGER.info(
            "%s %s directed-connectivity window: 0–%.1f ms",
            participant,
            variant,
            endpoint_ms,
        )
        epochs = epoch_picture(
            raw,
            trials,
            correct_only=correct_only,
            tmin=args.tmin,
            tmax=args.tmax,
            reject_uv=args.reject_uv,
            flat_uv=args.flat_uv,
        )
        roi_channels = select_roi_channels(epochs)
        (subject_dir / f"{variant}_roi_channels.json").write_text(
            json.dumps(roi_channels, indent=2), encoding="utf-8"
        )
        cluster_rows, significant_masks = temporal_cluster_statistics(
            epochs,
            roi_channels,
            permutations=args.permutations,
            seed=args.seed,
        )
        write_csv(subject_dir / f"{variant}_erp_clusters.csv", cluster_rows)
        write_csv(
            subject_dir / f"{variant}_erp_descriptive_statistics.csv",
            descriptive_erp_statistics(epochs, roi_channels),
        )
        plot_erps(
            epochs,
            roi_channels,
            significant_masks,
            subject_dir / f"{variant}_erp.png",
            f"{participant}: {variant.replace('_', ' ')}",
        )
        epochs_file = subject_dir / f"{variant}_clean_epochs.fif"
        epochs.save(epochs_file, overwrite=True, verbose="ERROR")
        analyses.append(
            ConnectivityInput(
                participant=participant,
                analysis=variant,
                kind="picture",
                epochs_file=epochs_file,
                window_start_seconds=0.0,
                window_end_seconds=endpoint_ms / 1000.0,
            )
        )
    raw.close()
    return analyses


def process_rest(
    dataset: dict[str, str], args: argparse.Namespace
) -> ConnectivityInput:
    participant = dataset["participant"]
    subject_dir = args.output_dir / participant
    subject_dir.mkdir(parents=True, exist_ok=True)
    vhdr = Path(dataset["vhdr"])
    raw, montage_report = preprocess_raw(vhdr, 1.0, 45.0)
    audit = raw_audit(raw, vhdr, montage_report)
    audit["analysis_name"] = "directed connectivity (not DCM)"
    (subject_dir / "rest_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    epochs = epoch_rest(
        raw,
        args.rest_epoch_seconds,
        args.reject_uv,
        args.flat_uv,
    )
    epochs_file = subject_dir / "rest_clean_segments.fif"
    epochs.save(epochs_file, overwrite=True, verbose="ERROR")
    raw.close()
    return ConnectivityInput(
        participant=participant,
        analysis="rest",
        kind="rest",
        epochs_file=epochs_file,
        window_start_seconds=0.0,
        window_end_seconds=float(epochs.times[-1]),
    )


def fetch_template(subjects_dir: Path | None) -> tuple[Path, Path, Path]:
    """Fetch fsaverage and return its directory, source space, and BEM."""
    fsaverage_dir = Path(
        mne.datasets.fetch_fsaverage(
            subjects_dir=subjects_dir,
            verbose=True,
        )
    )
    src = fsaverage_dir / "bem" / "fsaverage-ico-5-src.fif"
    bem = fsaverage_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    for required in (src, bem):
        if not required.is_file():
            raise FileNotFoundError(required)
    return fsaverage_dir, src, bem


def make_language_labels(subjects_dir: Path) -> list[mne.Label]:
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc="aparc",
        hemi="lh",
        subjects_dir=subjects_dir,
        verbose="ERROR",
    )
    by_name = {label.name: label for label in labels}
    combined: list[mne.Label] = []
    for node in NODE_NAMES:
        missing = [name for name in LABEL_COMPONENTS[node] if name not in by_name]
        if missing:
            raise RuntimeError(f"fsaverage aparc labels are missing: {missing}")
        label = by_name[LABEL_COMPONENTS[node][0]]
        for component in LABEL_COMPONENTS[node][1:]:
            label = label + by_name[component]
        label.name = node
        combined.append(label)
    return combined


def build_forward(
    epochs: mne.Epochs,
    src_file: Path,
    bem_file: Path,
    output_file: Path,
    n_jobs: int,
) -> mne.Forward:
    forward = mne.make_forward_solution(
        epochs.info,
        trans="fsaverage",
        src=src_file,
        bem=bem_file,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=n_jobs,
        verbose="ERROR",
    )
    mne.write_forward_solution(output_file, forward, overwrite=True, verbose="ERROR")
    return forward


def source_time_courses(
    analysis: ConnectivityInput,
    labels: list[mne.Label],
    src_file: Path,
    bem_file: Path,
    args: argparse.Namespace,
    forward_cache: dict[tuple[str, ...], mne.Forward],
) -> tuple[np.ndarray, float]:
    """Project epochs to four fsaverage ROI time courses."""
    epochs = mne.read_epochs(analysis.epochs_file, preload=True, verbose="ERROR")
    epochs.set_eeg_reference("average", projection=True, verbose="ERROR")
    key = tuple(epochs.ch_names)
    analysis_dir = (
        args.output_dir / analysis.participant / "directed" / analysis.analysis
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if key not in forward_cache:
        forward_cache[key] = build_forward(
            epochs,
            src_file,
            bem_file,
            analysis_dir / "fsaverage-forward.fif",
            args.n_jobs,
        )
    forward = forward_cache[key]

    if analysis.kind == "picture":
        covariance = mne.compute_covariance(
            epochs,
            tmin=epochs.tmin,
            tmax=0.0,
            method="shrunk",
            rank="info",
            verbose="ERROR",
        )
    else:
        covariance = mne.make_ad_hoc_cov(epochs.info)

    inverse = make_inverse_operator(
        epochs.info,
        forward,
        covariance,
        loose=0.2,
        depth=0.8,
        rank="info",
        verbose="ERROR",
    )
    write_inverse_operator(
        analysis_dir / "fsaverage-inverse.fif",
        inverse,
        overwrite=True,
        verbose="ERROR",
    )
    stcs = apply_inverse_epochs(
        epochs,
        inverse,
        lambda2=1.0 / 9.0,
        method="dSPM",
        pick_ori="normal",
        return_generator=False,
        verbose="ERROR",
    )
    label_ts = np.asarray(
        mne.extract_label_time_course(
            stcs,
            labels,
            inverse["src"],
            mode="mean_flip",
            allow_empty=False,
            return_generator=False,
            verbose="ERROR",
        )
    )
    time_mask = (epochs.times >= analysis.window_start_seconds) & (
        epochs.times <= analysis.window_end_seconds
    )
    label_ts = label_ts[:, :, time_mask]
    if label_ts.shape[0] < 4:
        raise RuntimeError(
            f"{analysis.participant} {analysis.analysis} has only "
            f"{label_ts.shape[0]} retained epochs; at least 4 are required"
        )
    np.savez_compressed(
        analysis_dir / "fsaverage_roi_time_courses.npz",
        data=label_ts,
        times=epochs.times[time_mask],
        node_names=np.asarray(NODE_NAMES),
        sfreq=float(epochs.info["sfreq"]),
    )
    return label_ts, float(epochs.info["sfreq"])


def frequency_bands(
    duration_seconds: float, h_freq: float
) -> dict[str, tuple[float, float]]:
    """Use only bands with at least five cycles at their lower edge."""
    candidates = {
        "theta": (4.0, 7.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, min(30.0, h_freq)),
    }
    return {
        name: bounds
        for name, bounds in candidates.items()
        if bounds[1] > bounds[0] and duration_seconds * bounds[0] >= 5.0
    }


def ordered_pairs() -> tuple[tuple[str, str], ...]:
    """Include both directions needed for net time-reversal correction."""
    pairs: list[tuple[str, str]] = []
    for source, target in PLANNED_EDGES:
        pairs.extend(((source, target), (target, source)))
    return tuple(dict.fromkeys(pairs))


def estimate_gc(
    data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    gc_lags: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = ordered_pairs()
    seeds = np.asarray([[NODE_INDEX[source]] for source, _ in pairs])
    targets = np.asarray([[NODE_INDEX[target]] for _, target in pairs])
    gc_values = np.empty((len(pairs), len(bands)), dtype=float)
    gc_tr_values = np.empty_like(gc_values)
    # MNE-Connectivity currently does not support GC over multiple bands in
    # one call, so estimate every prespecified band separately.
    for band_index, (fmin, fmax) in enumerate(bands.values()):
        gc, gc_tr = spectral_connectivity_epochs(
            data,
            names=list(NODE_NAMES),
            method=["gc", "gc_tr"],
            indices=(seeds, targets),
            sfreq=sfreq,
            mode="multitaper",
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            gc_n_lags=gc_lags,
            n_jobs=n_jobs,
            verbose=False,
        )
        gc_values[:, band_index] = np.asarray(gc.get_data()).reshape(-1)
        gc_tr_values[:, band_index] = np.asarray(gc_tr.get_data()).reshape(-1)
    return gc_values, gc_tr_values


def net_time_reversed_gc(gc: np.ndarray, gc_tr: np.ndarray) -> np.ndarray:
    """Compute netTRGC: net original GC minus net time-reversed GC."""
    pairs = ordered_pairs()
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    values = np.empty((len(PLANNED_EDGES), gc.shape[1]), dtype=float)
    for edge_index, (source, target) in enumerate(PLANNED_EDGES):
        forward = pair_index[(source, target)]
        reverse = pair_index[(target, source)]
        values[edge_index] = (gc[forward] - gc[reverse]) - (
            gc_tr[forward] - gc_tr[reverse]
        )
    return values


def bootstrap_connectivity(
    data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed GC, GC-TR, and bootstrap netTRGC values."""
    lags = min(args.gc_lags, max(1, data.shape[-1] // 10))
    if lags < 2:
        raise RuntimeError("Connectivity window is too short for GC estimation")
    gc, gc_tr = estimate_gc(data, sfreq, bands, lags, args.n_jobs)
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(
        (args.connectivity_bootstraps, len(PLANNED_EDGES), len(bands)),
        dtype=float,
    )
    completed = 0
    attempts = 0
    maximum_attempts = args.connectivity_bootstraps * 2
    while completed < args.connectivity_bootstraps and attempts < maximum_attempts:
        attempts += 1
        selection = rng.integers(0, data.shape[0], size=data.shape[0])
        try:
            boot_gc, boot_gc_tr = estimate_gc(
                data[selection], sfreq, bands, lags, args.n_jobs
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            LOGGER.warning("Skipping failed GC bootstrap: %s", error)
            continue
        bootstrap[completed] = net_time_reversed_gc(boot_gc, boot_gc_tr)
        completed += 1
    if completed < max(20, int(args.connectivity_bootstraps * 0.8)):
        raise RuntimeError(
            f"Only {completed}/{args.connectivity_bootstraps} GC bootstraps succeeded"
        )
    return gc, gc_tr, bootstrap[:completed]


def connectivity_rows(
    gc: np.ndarray,
    gc_tr: np.ndarray,
    bootstrap: np.ndarray,
    bands: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    pairs = ordered_pairs()
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    observed_net = net_time_reversed_gc(gc, gc_tr)
    rows: list[dict[str, Any]] = []
    for edge_index, (source, target) in enumerate(PLANNED_EDGES):
        forward = pair_index[(source, target)]
        reverse = pair_index[(target, source)]
        for band_index, (band, (fmin, fmax)) in enumerate(bands.items()):
            values = bootstrap[:, edge_index, band_index]
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "band": band,
                    "fmin_hz": fmin,
                    "fmax_hz": fmax,
                    "gc_source_to_target": float(gc[forward, band_index]),
                    "gc_target_to_source": float(gc[reverse, band_index]),
                    "gc_tr_source_to_target": float(gc_tr[forward, band_index]),
                    "gc_tr_target_to_source": float(gc_tr[reverse, band_index]),
                    "net_trgc": float(observed_net[edge_index, band_index]),
                    "bootstrap_mean_net_trgc": float(values.mean()),
                    "bootstrap_ci95_low": float(ci_low),
                    "bootstrap_ci95_high": float(ci_high),
                    "ci95_excludes_zero_descriptive": bool(ci_low > 0 or ci_high < 0),
                    "n_bootstraps": bootstrap.shape[0],
                }
            )
    return rows


def hypothesis_rows(
    observed_net: np.ndarray,
    bootstrap: np.ndarray,
    bands: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    edge_index = {edge: index for index, edge in enumerate(PLANNED_EDGES)}
    rows: list[dict[str, Any]] = []
    for family, edges in FAMILY_EDGES.items():
        indices = [edge_index[edge] for edge in edges]
        for band_index, band in enumerate(bands):
            values = bootstrap[:, indices, band_index].mean(axis=1)
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "hypothesis_edge_set": family,
                    "band": band,
                    "n_edges": len(indices),
                    "mean_net_trgc": float(observed_net[indices, band_index].mean()),
                    "bootstrap_ci95_low": float(ci_low),
                    "bootstrap_ci95_high": float(ci_high),
                    "interpretation": (
                        "Descriptive edge-set score only; not model evidence "
                        "and not a winning-family probability"
                    ),
                }
            )
    return rows


def plot_band(
    analysis_dir: Path,
    analysis_name: str,
    band: str,
    rows: list[dict[str, Any]],
) -> None:
    band_rows = [row for row in rows if row["band"] == band]
    figure, (network_axis, bar_axis) = plt.subplots(
        1, 2, figsize=(15, 6), constrained_layout=True
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(NODE_NAMES)
    for row in band_rows:
        graph.add_edge(row["source"], row["target"], weight=row["net_trgc"])
    weights = np.asarray(
        [graph[source][target]["weight"] for source, target in graph.edges()]
    )
    maximum = max(float(np.max(np.abs(weights))), np.finfo(float).eps)
    colors = ["tab:red" if value > 0 else "tab:blue" for value in weights]
    widths = 0.8 + 4.0 * np.abs(weights) / maximum
    nx.draw_networkx_nodes(
        graph,
        NODE_POSITIONS,
        node_color="#d9e7f5",
        edgecolors="#24476b",
        node_size=2200,
        ax=network_axis,
    )
    nx.draw_networkx_labels(graph, NODE_POSITIONS, font_size=11, ax=network_axis)
    nx.draw_networkx_edges(
        graph,
        NODE_POSITIONS,
        edge_color=colors,
        width=widths,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.12",
        ax=network_axis,
    )
    edge_labels = {
        (row["source"], row["target"]): f"{row['net_trgc']:.3f}" for row in band_rows
    }
    nx.draw_networkx_edge_labels(
        graph,
        NODE_POSITIONS,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        ax=network_axis,
    )
    network_axis.set_title(
        "Net time-reversal-corrected Granger\nred=planned direction, blue=opposite"
    )
    network_axis.axis("off")

    labels = [f"{row['source']} → {row['target']}" for row in band_rows]
    means = np.asarray([row["bootstrap_mean_net_trgc"] for row in band_rows])
    low = means - np.asarray([row["bootstrap_ci95_low"] for row in band_rows])
    high = np.asarray([row["bootstrap_ci95_high"] for row in band_rows]) - means
    low = np.maximum(low, 0.0)
    high = np.maximum(high, 0.0)
    positions = np.arange(len(labels))
    bar_axis.barh(
        positions,
        means,
        xerr=np.vstack((low, high)),
        color=["tab:red" if value > 0 else "tab:blue" for value in means],
        alpha=0.75,
        capsize=3,
    )
    bar_axis.set_yticks(positions, labels)
    bar_axis.axvline(0, color="black", linewidth=0.8)
    bar_axis.set_xlabel("netTRGC (descriptive units)")
    bar_axis.set_title("Directed edges with epoch-bootstrap 95% intervals")
    bar_axis.grid(axis="x", alpha=0.25)
    figure.suptitle(f"{analysis_name.replace('_', ' ')} — {band}")
    figure.savefig(
        analysis_dir / f"{analysis_name}_{band}_directed_connectivity.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(figure)


def run_connectivity(
    analysis: ConnectivityInput,
    labels: list[mne.Label],
    src_file: Path,
    bem_file: Path,
    args: argparse.Namespace,
    forward_cache: dict[tuple[str, ...], mne.Forward],
) -> None:
    LOGGER.info("Source connectivity: %s %s", analysis.participant, analysis.analysis)
    data, sfreq = source_time_courses(
        analysis,
        labels,
        src_file,
        bem_file,
        args,
        forward_cache,
    )
    duration = data.shape[-1] / sfreq
    bands = frequency_bands(duration, args.h_freq if analysis.kind == "picture" else 45)
    if not bands:
        raise RuntimeError(
            f"No frequency band has at least five cycles in {duration:.3f} s"
        )
    gc, gc_tr, bootstrap = bootstrap_connectivity(data, sfreq, bands, args)
    rows = connectivity_rows(gc, gc_tr, bootstrap, bands)
    observed_net = net_time_reversed_gc(gc, gc_tr)
    family_rows = hypothesis_rows(observed_net, bootstrap, bands)
    analysis_dir = (
        args.output_dir / analysis.participant / "directed" / analysis.analysis
    )
    write_csv(analysis_dir / "directed_edges.csv", rows)
    write_csv(analysis_dir / "hypothesis_edge_set_scores.csv", family_rows)
    for band in bands:
        plot_band(analysis_dir, analysis.analysis, band, rows)
    metadata = {
        "participant": analysis.participant,
        "analysis": analysis.analysis,
        "method": ("MNE-Connectivity state-space spectral GC and time-reversed GC"),
        "metric": (
            "netTRGC=(GCxy-GCyx)-(GCtr_xy-GCtr_yx); positive supports the "
            "listed direction"
        ),
        "frequency_bands_hz": bands,
        "n_epochs": int(data.shape[0]),
        "n_samples": int(data.shape[-1]),
        "sfreq_hz": sfreq,
        "window_seconds": [
            analysis.window_start_seconds,
            analysis.window_end_seconds,
        ],
        "gc_lags_requested": args.gc_lags,
        "bootstrap_replicates": int(bootstrap.shape[0]),
        "warning": (
            "This is directed connectivity, not DCM. Bootstrap intervals are "
            "participant-level descriptive intervals and are not corrected "
            "population inference."
        ),
    }
    (analysis_dir / "connectivity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def write_method_notes(
    output_dir: Path, fsaverage_dir: Path, args: argparse.Namespace
) -> None:
    notes = f"""PYTHON DIRECTED-CONNECTIVITY PILOT

This analysis is not Dynamic Causal Modelling. It does not estimate DCM
neuronal parameters, variational free energy, model evidence, or winning
model-family probabilities.

Sensor/anatomy templates
------------------------
Sensor positions use the matched MNE GSN-HydroCel-128/129 or standard_1005
proxy documented in each audit. MRI anatomy, BEM, and source space use MNE's
open fsaverage template at:
  {fsaverage_dir}

Source ROIs (FreeSurfer aparc, left hemisphere)
------------------------------------------------
OT: lateraloccipital + fusiform
pMTG: middletemporal + bankssts
ATL: temporalpole
IFG: parsopercularis + parstriangularis + parsorbitalis

Connectivity
------------
MNE-Connectivity state-space spectral Granger causality (GC) and time-reversed
GC are computed on dSPM ROI time courses. The reported directional statistic:
  netTRGC = (GC X->Y - GC Y->X) - (GCtr X->Y - GCtr Y->X)

Positive values support the listed direction relative to its reverse.
Epoch-bootstrap 95% intervals quantify recording-level stability. They are
descriptive, uncorrected intervals—not N=2 population tests.

F1/F2/F3 outputs are means over prespecified edge sets. They are not fitted
generative models and cannot be called winning families. picture_correct is
primary; picture_all is a sensitivity analysis. Rest is analyzed separately.

Limitations
-----------
Template anatomy, proxy electrodes, inverse leakage, source mixing, GC model
order, short event windows, and unreviewed ICA all limit causal interpretation.
GC direction is predictive directed dependence, not proof of biological
causation. Individual MRI/digitization and a larger longitudinal sample are
required for the full study.

Software
--------
MNE-Python: {mne.__version__}
MNE-Connectivity: {mne_connectivity.__version__}
Bootstrap replicates requested: {args.connectivity_bootstraps}
Random seed: {args.seed}
"""
    (output_dir / "METHOD_NOTES.txt").write_text(notes, encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.tmin >= 0 or args.tmax <= 0 or args.tmin >= args.tmax:
        raise ValueError("Epoch must span zero with tmin < 0 < tmax")
    if not math.isfinite(args.reject_uv) or args.reject_uv <= 0 or args.flat_uv < 0:
        raise ValueError("Artifact thresholds must be finite and non-negative")
    if args.connectivity_bootstraps < 20:
        raise ValueError("--connectivity-bootstraps must be at least 20")
    if args.gc_lags < 2:
        raise ValueError("--gc-lags must be at least 2")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "pipeline.log"),
        ],
    )

    datasets = load_datasets(args.manifest)
    for dataset in datasets:
        if not Path(dataset["vhdr"]).is_file():
            raise FileNotFoundError(dataset["vhdr"])

    analyses: list[ConnectivityInput] = []
    for dataset in datasets:
        LOGGER.info("Processing %s %s", dataset["participant"], dataset["task"])
        if dataset["task"] == "picture":
            analyses.extend(process_picture(dataset, args))
        else:
            analyses.append(process_rest(dataset, args))

    fsaverage_dir, src_file, bem_file = fetch_template(args.subjects_dir)
    subjects_dir = fsaverage_dir.parent
    labels = make_language_labels(subjects_dir)
    (args.output_dir / "source_rois.json").write_text(
        json.dumps(
            {node: list(LABEL_COMPONENTS[node]) for node in NODE_NAMES},
            indent=2,
        ),
        encoding="utf-8",
    )
    forward_cache: dict[tuple[str, ...], mne.Forward] = {}
    for analysis in analyses:
        run_connectivity(
            analysis,
            labels,
            src_file,
            bem_file,
            args,
            forward_cache,
        )
    write_method_notes(args.output_dir, fsaverage_dir, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
