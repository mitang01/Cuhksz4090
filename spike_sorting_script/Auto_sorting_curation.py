#!/usr/bin/env python3
"""
Run automatic curation on existing SortingAnalyzer outputs.

This script processes existing sorting result folders (it does NOT run sorting):
  /share/home/mitan/spike_sorting/mountainsort4/sorting_results_*

For each session folder and region analyzer, it runs Bombcell labeling with a
project-specific two-metric threshold:
  - num_spikes > 100
  - isi_violations_ratio < 0.02

Important:
Merges are irreversible, so each method starts from its own physical copy of
the source analyzer folder before any merge operation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import spikeinterface.full as si
from spikeinterface.curation import (
    bombcell_label_units,
    compute_merge_unit_groups,
    remove_duplicated_spikes,
    remove_redundant_units,
)


DEFAULT_ANALYZER_ROOT = Path("/share/home/mitan/spike_sorting/mountainsort4")
DEFAULT_SESSION_GLOB = "sorting_results_*"
DEFAULT_REGIONS = ["ATL", "HG", "VMPFC", "Amygdala"]
DEFAULT_OUTPUT_SUBDIR = "auto_curation"
DEFAULT_GAIN_TO_UV = 0.195
# SpikeInterface's threshold API uses inclusive >= / <= comparisons. Use the
# next valid integer/float bounds to implement the requested strict > / < rules.
BOMBCELL_THRESHOLDS = {
    "noise": {},
    "mua": {
        "num_spikes": {"greater": 101, "less": None},
        "isi_violations_ratio": {
            "greater": None,
            "less": float(np.nextafter(0.02, -np.inf)),
        },
    },
    "non-somatic": {},
}
BOMBCELL_LABELING_RULES = {
    "num_spikes": "> 100",
    "isi_violations_ratio": "< 0.02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bombcell curation on existing SortingAnalyzer folders."
    )
    parser.add_argument(
        "--analyzer-root",
        type=Path,
        default=DEFAULT_ANALYZER_ROOT,
        help="Root containing sorting_results_* folders.",
    )
    parser.add_argument(
        "--session-folders",
        nargs="*",
        default=None,
        help=(
            "Optional session folder names to process. "
            "Examples: sorting_results_sub5_xxx_v2_rawmat"
        ),
    )
    parser.add_argument(
        "--session-glob",
        type=str,
        default=DEFAULT_SESSION_GLOB,
        help=(
            "Glob used for auto-discovery when --session-folders is omitted. "
            f"Default: {DEFAULT_SESSION_GLOB}"
        ),
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=DEFAULT_REGIONS,
        help="Regions to process. Default: ATL HG VMPFC Amygdala",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Per-region output subdirectory name (default: auto_curation).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing method outputs if they exist.",
    )
    parser.add_argument(
        "--duplicate-censored-ms",
        type=float,
        default=0.1,
        help="Censored window for duplicated spike removal (default: 0.1 ms).",
    )
    parser.add_argument(
        "--redundant-threshold",
        type=float,
        default=0.9,
        help="Duplicate threshold for remove_redundant_units (default: 0.9).",
    )
    parser.add_argument(
        "--redundant-remove-strategy",
        type=str,
        default="minimum_shift",
        help="remove_redundant_units strategy (default: minimum_shift).",
    )
    parser.add_argument(
        "--merge-preset",
        type=str,
        default="similarity_correlograms",
        help="Preset for compute_merge_unit_groups (default: similarity_correlograms).",
    )
    parser.add_argument(
        "--merge-censored-period-ms",
        type=float,
        default=0.5,
        help="Censored period passed to merge_units (default: 0.5 ms).",
    )
    parser.add_argument(
        "--merge-mode",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="merge_units merging_mode (default: soft).",
    )
    parser.add_argument(
        "--gain-to-uv",
        type=float,
        default=DEFAULT_GAIN_TO_UV,
        help=(
            "Microvolts per ADC count used to repair raw-MAT analyzers that were "
            f"saved without scaling metadata (default: {DEFAULT_GAIN_TO_UV})."
        ),
    )
    return parser.parse_args()


def discover_session_roots(analyzer_root: Path, args: argparse.Namespace) -> List[Path]:
    if args.session_folders:
        roots = [analyzer_root / name for name in args.session_folders]
    else:
        roots = sorted(p for p in analyzer_root.glob(args.session_glob) if p.is_dir())
    return [p for p in roots if p.is_dir()]


def resolve_region_analyzer_path(session_root: Path, region: str) -> Path:
    # Prefer current layout: <session>/<region>/analyzer
    candidate_new = session_root / region / "analyzer"
    if candidate_new.exists():
        return candidate_new
    # Fallback for older layout: <session>/<region>_analyzer
    candidate_old = session_root / f"{region}_analyzer"
    if candidate_old.exists():
        return candidate_old
    return candidate_new


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy_analyzer(source_analyzer, dst: Path, overwrite: bool):
    """Create an independent analyzer copy using SpikeInterface's save API.

    A plain shutil.copytree() breaks analyzers whose serialized recording paths
    are relative to the analyzer folder. save_as() rewrites the analyzer
    metadata for the new location and returns the copied analyzer directly.
    """
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst}")
        safe_rmtree(dst)
    return source_analyzer.save_as(format="binary_folder", folder=str(dst))


def ensure_uv_scaled_analyzer(analyzer, gain_to_uv: float):
    """Return an analyzer whose templates/traces are expressed in microvolts.

    Raw-MAT sorting outputs were built from int16 ADC counts without recording
    gain/offset properties, so SpikeInterface stored them with
    return_in_uV=False. Template metrics in SpikeInterface 0.104 request
    return_in_uV=True and reject such analyzers. Rebuilding after attaching the
    known Intan gain fixes the metadata and ensures newly computed templates
    are actually scaled rather than merely relabeled.
    """
    if bool(getattr(analyzer, "return_in_uV", False)):
        return analyzer

    if gain_to_uv <= 0:
        raise ValueError(f"--gain-to-uv must be positive, got {gain_to_uv}")

    try:
        recording = analyzer.recording
    except Exception as exc:
        raise RuntimeError(
            "Analyzer is unscaled and its recording could not be loaded; "
            "cannot rebuild templates in microvolts."
        ) from exc

    recording.set_channel_gains(gain_to_uv)
    recording.set_channel_offsets(0.0)
    print(
        "    [INFO] Rebuilding analyzer with microvolt scaling: "
        f"gain_to_uV={gain_to_uv}, offset_to_uV=0.0"
    )

    return si.create_sorting_analyzer(
        sorting=analyzer.sorting,
        recording=recording,
        format="memory",
        sparsity=analyzer.sparsity,
        return_in_uV=True,
    )


def compute_required_extensions(analyzer) -> None:
    # Compute extensions one-by-one to maximize compatibility across SI versions.
    # Missing/unsupported extensions are skipped with warning.
    steps = [
        ("random_spikes", dict(method="uniform", max_spikes_per_unit=500)),
        ("waveforms", dict(ms_before=1.0, ms_after=2.0)),
        ("templates", {}),
        ("noise_levels", {}),
        ("spike_amplitudes", {}),
        ("spike_locations", {}),
        ("unit_locations", {}),
        ("correlograms", {}),
        ("template_similarity", {}),
        ("principal_components", {}),
    ]
    for ext_name, kwargs in steps:
        try:
            has_ext = getattr(analyzer, "has_extension", None)
            if callable(has_ext) and analyzer.has_extension(ext_name):
                continue
            analyzer.compute(ext_name, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"    [WARN] Could not compute extension '{ext_name}': {exc}")


def compute_labeling_metrics(analyzer) -> None:
    """Compute quality/template metrics used by curation and merging.

    The sorting scripts computed only a small QC subset. Bombcell and
    auto-merging require additional waveform and template extensions, so merely
    seeing an existing quality_metrics extension is not sufficient.
    """
    compute_required_extensions(analyzer)

    analyzer.compute(
        "template_metrics",
        include_multi_channel_metrics=True,
        delete_existing_metrics=False,
    )
    analyzer.compute(
        "quality_metrics",
        metric_names=None,
        delete_existing_metrics=False,
    )


def persist_analyzer(analyzer, folder: Path, overwrite: bool):
    if folder.exists():
        if not overwrite:
            raise FileExistsError(f"Curated analyzer output exists: {folder}")
        safe_rmtree(folder)

    save_as = getattr(analyzer, "save_as", None)
    if callable(save_as):
        return analyzer.save_as(format="binary_folder", folder=str(folder))

    save = getattr(analyzer, "save", None)
    if callable(save):
        analyzer.save(folder=str(folder), overwrite=overwrite)
        return si.load_sorting_analyzer(str(folder))

    # Last-resort fallback: rebuild from sorting + recording.
    sorting = analyzer.sorting
    recording = analyzer.recording
    rebuilt = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=str(folder),
    )
    compute_required_extensions(rebuilt)
    return rebuilt


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def run_bombcell(
    source_analyzer_path: Path,
    method_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    method = "bombcell"
    method_root.mkdir(parents=True, exist_ok=True)
    input_copy = method_root / "analyzer_input_copy"
    source_analyzer = si.load_sorting_analyzer(str(source_analyzer_path))
    analyzer = copy_analyzer(source_analyzer, input_copy, overwrite=args.overwrite)
    analyzer = ensure_uv_scaled_analyzer(analyzer, gain_to_uv=args.gain_to_uv)
    compute_labeling_metrics(analyzer)

    # Optional cleaning step from curation tutorial.
    try:
        dedup_sorting = remove_duplicated_spikes(
            analyzer.sorting,
            censored_period_ms=args.duplicate_censored_ms,
        )
        dedup_analyzer = si.create_sorting_analyzer(
            sorting=dedup_sorting,
            recording=analyzer.recording,
            format="memory",
        )
        compute_labeling_metrics(dedup_analyzer)
        analyzer = dedup_analyzer
    except Exception as exc:  # pylint: disable=broad-except
        print(f"    [WARN] remove_duplicated_spikes step skipped: {exc}")

    # Remove redundant units using SortingAnalyzer-aware strategy.
    try:
        clean_sorting = remove_redundant_units(
            analyzer,
            duplicate_threshold=args.redundant_threshold,
            remove_strategy=args.redundant_remove_strategy,
        )
        nonredundant_analyzer = analyzer.select_units(clean_sorting.unit_ids)
        compute_labeling_metrics(nonredundant_analyzer)
        analyzer = nonredundant_analyzer
    except Exception as exc:  # pylint: disable=broad-except
        print(f"    [WARN] remove_redundant_units step skipped: {exc}")

    merge_groups: List[List[Any]] = []
    try:
        merge_groups = compute_merge_unit_groups(
            sorting_analyzer=analyzer,
            preset=args.merge_preset,
            resolve_graph=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"    [WARN] compute_merge_unit_groups failed: {exc}")

    if merge_groups:
        analyzer = analyzer.merge_units(
            merge_unit_groups=merge_groups,
            censored_period_ms=args.merge_censored_period_ms,
            merging_mode=args.merge_mode,
        )
        compute_labeling_metrics(analyzer)

    curated_analyzer_path = method_root / "analyzer_curated"
    analyzer = persist_analyzer(analyzer, curated_analyzer_path, overwrite=args.overwrite)

    labels_df = bombcell_label_units(
        sorting_analyzer=analyzer,
        thresholds=BOMBCELL_THRESHOLDS,
        label_non_somatic=False,
    )
    if "label" in labels_df.columns and "bombcell_label" not in labels_df.columns:
        labels_df = labels_df.rename(columns={"label": "bombcell_label"})
    label_column = "bombcell_label"

    labels_csv = method_root / f"{method}_labels.csv"
    write_dataframe(labels_df, labels_csv)

    merge_groups_json = method_root / "merge_groups.json"
    merge_groups_json.write_text(json.dumps(merge_groups, indent=2, default=str))

    label_counts = {}
    if label_column in labels_df.columns:
        vc = labels_df[label_column].astype(str).value_counts(dropna=False)
        label_counts = {str(k): int(v) for k, v in vc.to_dict().items()}

    summary = {
        "method": method,
        "source_analyzer": str(source_analyzer_path),
        "input_copy": str(input_copy),
        "curated_analyzer": str(curated_analyzer_path),
        "labels_csv": str(labels_csv),
        "merge_groups_json": str(merge_groups_json),
        "num_units_after_curation": int(len(analyzer.sorting.unit_ids)),
        "num_merge_groups": int(len(merge_groups)),
        "label_counts": label_counts,
    }
    (method_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def process_session_region(
    session_root: Path,
    region: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    analyzer_path = resolve_region_analyzer_path(session_root, region)
    result: Dict[str, Any] = {
        "session_folder": session_root.name,
        "region": region,
        "source_analyzer": str(analyzer_path),
        "status": "skipped",
        "methods": [],
        "error": None,
    }

    if not analyzer_path.exists():
        result["error"] = "analyzer path not found"
        return result

    output_root = session_root / region / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    method = "bombcell"
    method_root = output_root / method
    print(f"  [START] {session_root.name}/{region} :: {method}")
    try:
        method_summary = run_bombcell(
            source_analyzer_path=analyzer_path,
            method_root=method_root,
            args=args,
        )
        result["methods"].append(method_summary)
        result["status"] = "ok"
        print(f"  [OK] {session_root.name}/{region} :: {method}")
    except Exception as exc:  # pylint: disable=broad-except
        result["methods"].append(
            {
                "method": method,
                "status": "error",
                "error": repr(exc),
            }
        )
        result["status"] = "error"
        print(f"  [ERROR] {session_root.name}/{region} :: {method}: {exc}")
    return result


def main() -> None:
    args = parse_args()
    analyzer_root = args.analyzer_root
    if not analyzer_root.exists():
        raise SystemExit(f"[FATAL] analyzer root not found: {analyzer_root}")

    session_roots = discover_session_roots(analyzer_root, args)
    if not session_roots:
        raise SystemExit(
            f"[FATAL] No session folders found under {analyzer_root} "
            f"with pattern '{args.session_glob}'."
        )

    print(f"[INFO] analyzer_root: {analyzer_root}")
    print(f"[INFO] sessions found: {len(session_roots)}")
    print(f"[INFO] regions: {args.regions}")
    print("[INFO] method: bombcell")
    print(f"[INFO] Bombcell labeling rules: {BOMBCELL_LABELING_RULES}")
    print(f"[INFO] overwrite: {args.overwrite}")

    run_results = []
    for session_root in session_roots:
        print("=" * 88)
        print(f"[SESSION] {session_root.name}")
        print("=" * 88)
        for region in args.regions:
            run_results.append(process_session_region(session_root, region, args))

    run_log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analyzer_root": str(analyzer_root),
        "session_count": len(session_roots),
        "regions": args.regions,
        "methods": ["bombcell"],
        "bombcell_labeling_rules": BOMBCELL_LABELING_RULES,
        "bombcell_runtime_thresholds": BOMBCELL_THRESHOLDS,
        "results": run_results,
    }
    log_path = analyzer_root / f"auto_sorting_curation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(run_log, indent=2))
    print(f"[DONE] Run log: {log_path}")


if __name__ == "__main__":
    main()

