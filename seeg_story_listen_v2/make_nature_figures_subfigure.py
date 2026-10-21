#!/usr/bin/env python3
"""Render each publication panel as an independent PNG with an explanation.

The script consumes completed outputs from the four STRF/prosody entry points.
It reuses their saved statistics and never recomputes tests, samples rows, or
imputes missing conditions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_nature_figures as nature
import prosody_epoch_stats as epoch_stats


DEFAULT_ROOT = nature.DEFAULT_ROOT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render every main publication panel as a standalone PNG with a "
            "matching Markdown explanation."
        )
    )
    parser.add_argument("--strf-dir", type=Path, default=DEFAULT_ROOT / "strf")
    parser.add_argument(
        "--strf-group-dir",
        type=Path,
        default=DEFAULT_ROOT / "strf_group",
    )
    parser.add_argument(
        "--epochs-dir",
        type=Path,
        default=DEFAULT_ROOT / "prosody_epochs",
    )
    parser.add_argument(
        "--epochs-group-dir",
        type=Path,
        default=DEFAULT_ROOT / "prosody_epochs_group",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "nature_subfigures",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def prepare_output(args: argparse.Namespace) -> None:
    for name in ("strf_dir", "strf_group_dir", "epochs_dir", "epochs_group_dir"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_dir():
            raise FileNotFoundError(
                f"required result directory does not exist: {value}"
            )
        setattr(args, name, value)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir in {
        args.strf_dir,
        args.strf_group_dir,
        args.epochs_dir,
        args.epochs_group_dir,
    }:
        raise ValueError("standalone output must differ from every analysis input")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"output directory is not empty (use --overwrite): "
                f"{args.output_dir}"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def legend_audit(figure: plt.Figure) -> list[dict[str, object]]:
    """Return rendered legend/axes geometry and reject data-area overlap."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    records: list[dict[str, object]] = []
    for axis_index, axis in enumerate(figure.axes):
        legend = axis.get_legend()
        if legend is None or not legend.get_visible():
            continue
        legend_box = legend.get_window_extent(renderer)
        axes_box = axis.get_window_extent(renderer)
        overlap_width = max(
            0.0,
            min(legend_box.x1, axes_box.x1) - max(legend_box.x0, axes_box.x0),
        )
        overlap_height = max(
            0.0,
            min(legend_box.y1, axes_box.y1) - max(legend_box.y0, axes_box.y0),
        )
        overlaps_data_area = overlap_width > 0 and overlap_height > 0
        records.append(
            {
                "axis_index": axis_index,
                "legend_bbox_fraction": [
                    legend_box.x0 / width,
                    legend_box.y0 / height,
                    legend_box.x1 / width,
                    legend_box.y1 / height,
                ],
                "axes_bbox_fraction": [
                    axes_box.x0 / width,
                    axes_box.y0 / height,
                    axes_box.x1 / width,
                    axes_box.y1 / height,
                ],
                "overlaps_plotting_area": overlaps_data_area,
            }
        )
        if overlaps_data_area:
            raise AssertionError(
                f"legend overlaps plotting area for standalone axis {axis_index}"
            )
    return records


def save_standalone(
    output_dir: Path,
    *,
    slug: str,
    title: str,
    figure: plt.Figure,
    explanation: str,
    statistical_note: str,
    source_paths: Sequence[Path],
) -> dict[str, object]:
    audits = legend_audit(figure)
    png_path = output_dir / f"{slug}.png"
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    note_path = output_dir / f"{slug}.md"
    source_lines = "\n".join(f"- `{path}`" for path in source_paths)
    note_path.write_text(
        "\n".join(
            (
                f"# {title}",
                "",
                f"![{title}]({png_path.name})",
                "",
                "## What is shown",
                explanation,
                "",
                "## Statistical annotation",
                statistical_note,
                "",
                "## Source outputs",
                source_lines,
                "",
                "No observations were sampled or imputed for rendering.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return {
        "title": title,
        "png": png_path.name,
        "explanation": note_path.name,
        "source_outputs": [str(path) for path in source_paths],
        "legend_audit": audits,
        "legend_status": (
            "pass"
            if all(not row["overlaps_plotting_area"] for row in audits)
            else "review_required"
        ),
    }


def single_axis_figure(
    plotter: Callable[[plt.Axes], object],
    *,
    figsize: tuple[float, float] = (3.5, 3.0),
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
    plotter(axis)
    return figure


def run(args: argparse.Namespace) -> int:
    prepare_output(args)
    nature.validate_full_run_configs(
        (
            ("individual STRF", args.strf_dir / "analysis_config.json"),
            ("group STRF", args.strf_group_dir / "analysis_config.json"),
            ("individual epochs", args.epochs_dir / "analysis_config.json"),
            ("group epochs", args.epochs_group_dir / "analysis_config.json"),
        )
    )
    completion = nature.read_json_required(
        args.epochs_dir / epoch_stats.COMPLETION_FILENAME
    )
    if completion.get("status") != "complete":
        raise ValueError("individual epoch completion marker is not complete")

    group_recording = args.strf_group_dir / "recordings" / "GROUP"
    metrics = nature.read_csv_required(group_recording / "model_metrics.csv")
    models, folds = nature.validate_model_metrics(metrics)
    membership = nature.read_csv_required(
        args.strf_group_dir / "group_membership.csv"
    )
    feature_tests = nature.read_csv_required(
        args.strf_group_dir / "subject_feature_contributions.csv"
    )
    feature_points = nature.subject_feature_points(args.strf_dir, membership)
    group_stats = nature.read_csv_required(
        args.epochs_group_dir / "group_timepoint_statistics.csv"
    )
    epoch_paths = sorted(
        (args.epochs_dir / "recordings").glob("*/epoch_data.npz")
    )
    recording_dirs = sorted(
        path
        for path in (args.epochs_dir / "recordings").iterdir()
        if path.is_dir()
    )
    partial = [
        path for path in recording_dirs if not (path / "epoch_data.npz").is_file()
    ]
    if partial:
        raise ValueError(
            "partial individual epoch outputs lack epoch_data.npz: "
            f"{partial[:10]}"
        )
    times, waveform_rows, _, epoch_accounting = nature.load_subject_waveforms(
        epoch_paths,
        strf_dir=args.strf_dir,
    )
    strf_recordings = nature.individual_strf_recordings(args.strf_dir)
    epoch_recordings = {str(row["recording_id"]) for row in epoch_accounting}
    if strf_recordings != epoch_recordings:
        raise ValueError("individual STRF/epoch recording cohorts differ")
    if set(completion.get("recording_ids", [])) != epoch_recordings:
        raise ValueError("individual epoch completion marker differs from epoch files")
    group_epoch_membership = nature.read_csv_required(
        args.epochs_group_dir / "group_membership.csv"
    )
    if {row["recording_id"] for row in group_epoch_membership} != epoch_recordings:
        raise ValueError("group-epoch membership differs from individual epoch cohort")
    nature.validate_group_epoch_statistics(
        group_stats,
        expected_subjects=len(
            {nature.subject_id(recording) for recording in epoch_recordings}
        ),
    )

    (
        individual_strf_selection,
        individual_feature_rows,
        _,
        individual_coefficient_path,
    ) = nature.select_individual_strf_result(args.strf_dir)
    (
        individual_epoch_selection,
        individual_epoch_rows,
        individual_epoch_path,
    ) = nature.select_individual_epoch_result(args.epochs_dir)
    claims = nature.significant_claims(feature_tests, group_stats)
    nature.set_publication_style()

    entries: dict[str, dict[str, object]] = {}

    entries["01_pooled_encoding_accuracy"] = save_standalone(
        args.output_dir,
        slug="01_pooled_encoding_accuracy",
        title="Pooled encoding accuracy",
        figure=single_axis_figure(
            lambda axis: nature.plot_group_accuracy(
                axis, metrics, models, folds
            )
        ),
        explanation=(
            "Held-out Pearson correlation for every pooled model and story "
            "fold. Points are all held-out folds; bars are fold means."
        ),
        statistical_note=(
            "Error bars are SEM across held-out folds. No absolute-performance "
            "significance test is added."
        ),
        source_paths=(group_recording / "model_metrics.csv",),
    )

    entries["02_unique_feature_effects"] = save_standalone(
        args.output_dir,
        slug="02_unique_feature_effects",
        title="Unique feature effects",
        figure=single_axis_figure(
            lambda axis: nature.plot_subject_features(
                axis, feature_points, feature_tests
            )
        ),
        explanation=(
            "Unique held-out correlation contribution for every requested "
            "feature family. Every subject mean is displayed."
        ),
        statistical_note=(
            "Bars are means with 95% normal-approximation SEM intervals. "
            "Asterisks reproduce saved subject-level BH-FDR-adjusted tests."
        ),
        source_paths=(
            args.strf_group_dir / "subject_feature_contributions.csv",
            args.strf_dir / "recordings/*/feature_contributions.csv",
        ),
    )

    entries["03_observed_group_separability"] = save_standalone(
        args.output_dir,
        slug="03_observed_group_separability",
        title="Observed group ER-Hγ separability",
        figure=single_axis_figure(
            lambda axis: nature.plot_source_statistics(
                axis,
                group_stats,
                source="observed",
                title=claims[1],
            ),
            figsize=(4.5, 3.4),
        ),
        explanation=(
            "Observed group event-related high-gamma F statistics for boundary "
            "strength and structure depth across peri-event time."
        ),
        statistical_note=(
            "Light shading, horizontal lines, and asterisks reproduce intervals "
            "that passed the saved BH-FDR q<0.01 and four-sample duration rule."
        ),
        source_paths=(
            args.epochs_group_dir / "group_timepoint_statistics.csv",
        ),
    )

    group_coefficient_figure = plt.figure(
        figsize=(4.0, 4.2),
        layout="constrained",
    )
    nature.plot_coefficients(
        group_coefficient_figure,
        group_coefficient_figure.add_gridspec(1, 1)[0],
        group_recording / "model_coefficients.npz",
        title="Group M5 coefficients",
    )
    entries["04_group_m5_coefficients"] = save_standalone(
        args.output_dir,
        slug="04_group_m5_coefficients",
        title="Group M5 coefficients",
        figure=group_coefficient_figure,
        explanation=(
            "Mean full-model STRF coefficients across all pooled outer folds: "
            "mel coefficients as a heatmap and linguistic predictors as lines."
        ),
        statistical_note=(
            "Linguistic ribbons are descriptive ±1.96 SEM across dependent "
            "outer folds; no coefficient-wise inferential test is added."
        ),
        source_paths=(group_recording / "model_coefficients.npz",),
    )

    for order, predictor, title in (
        ("05", "struc_depth", "ER-Hγ by structure depth"),
        ("06", "boundary_strength", "ER-Hγ by boundary strength"),
    ):
        slug = (
            f"{order}_structure_depth_erhg"
            if predictor == "struc_depth"
            else f"{order}_boundary_strength_erhg"
        )
        entries[slug] = save_standalone(
            args.output_dir,
            slug=slug,
            title=title,
            figure=single_axis_figure(
                lambda axis, selected=predictor: nature.plot_subject_epochs(
                    axis,
                    times,
                    waveform_rows,
                    predictor=selected,
                ),
                figsize=(4.0, 3.4),
            ),
            explanation=(
                "All retained subject-level observed event-related high-gamma "
                "trajectories and their condition means."
            ),
            statistical_note=(
                "This is descriptive. Boundary strength uses within-recording "
                "equal-count quartiles only for visualization; missing cells "
                "are not imputed."
            ),
            source_paths=tuple(epoch_paths),
        )

    selected_strf_label = (
        f"{individual_strf_selection['recording_id']} / "
        f"{individual_strf_selection['channel']}"
    )
    entries["07_individual_feature_effects"] = save_standalone(
        args.output_dir,
        slug="07_individual_feature_effects",
        title="Selected individual feature effects",
        figure=single_axis_figure(
            lambda axis: nature.plot_individual_test_bars(
                axis,
                individual_feature_rows,
                key="feature",
                title=selected_strf_label,
            ),
            figsize=(4.0, 3.2),
        ),
        explanation=(
            "Feature contributions for the recording/channel with the minimum "
            "finite adjusted correlation p value across all individual STRF "
            "results."
        ),
        statistical_note=(
            f"Selected q="
            f"{individual_strf_selection['fdr_p_value_correlation']:.3g}. "
            "Asterisks reproduce saved BH-FDR-adjusted feature tests."
        ),
        source_paths=(
            args.strf_dir / "recordings/*/feature_contributions.csv",
        ),
    )

    individual_coefficient_figure = plt.figure(
        figsize=(4.0, 4.2),
        layout="constrained",
    )
    nature.plot_coefficients(
        individual_coefficient_figure,
        individual_coefficient_figure.add_gridspec(1, 1)[0],
        individual_coefficient_path,
        channel_name=str(individual_strf_selection["channel"]),
        title="Selected individual M5 coefficients",
    )
    entries["08_individual_m5_coefficients"] = save_standalone(
        args.output_dir,
        slug="08_individual_m5_coefficients",
        title="Selected individual M5 coefficients",
        figure=individual_coefficient_figure,
        explanation=(
            "Full-model coefficients for the deterministically selected "
            f"individual STRF result ({selected_strf_label})."
        ),
        statistical_note=(
            "Linguistic ribbons are descriptive ±1.96 SEM across dependent "
            "outer folds; no coefficient-wise inferential test is added."
        ),
        source_paths=(individual_coefficient_path,),
    )

    selected_epoch_label = (
        f"{individual_epoch_selection['recording_id']} / "
        f"{individual_epoch_selection['channel']}"
    )
    entries["09_individual_epoch_statistic"] = save_standalone(
        args.output_dir,
        slug="09_individual_epoch_statistic",
        title="Selected individual epoch statistic",
        figure=single_axis_figure(
            lambda axis: nature.plot_individual_epoch_statistic(
                axis,
                individual_epoch_rows,
                individual_epoch_selection,
            ),
            figsize=(4.0, 3.2),
        ),
        explanation=(
            "Time-resolved F statistic for the duration-qualified individual "
            "epoch result with the minimum adjusted p value when available."
        ),
        statistical_note=(
            f"Selected q={individual_epoch_selection['fdr_p_value']:.3g}; "
            "shading, line, and asterisk reproduce saved FDR-and-duration "
            "qualification."
        ),
        source_paths=(
            args.epochs_dir / "recordings/*/timepoint_statistics.csv",
        ),
    )

    entries["10_individual_event_response"] = save_standalone(
        args.output_dir,
        slug="10_individual_event_response",
        title="Selected individual event-related response",
        figure=single_axis_figure(
            lambda axis: nature.plot_individual_event_waveforms(
                axis,
                individual_epoch_path,
                individual_epoch_selection,
            ),
            figsize=(4.0, 3.4),
        ),
        explanation=(
            "All retained epochs for the source and predictor selected by the "
            f"individual epoch result ({selected_epoch_label})."
        ),
        statistical_note=(
            "Lines are condition means and ribbons are ±1.96 SEM across "
            "retained epochs. Boundary-strength quartiles are visualization-only."
        ),
        source_paths=(individual_epoch_path,),
    )

    manifest = {
        "renderer": Path(__file__).name,
        "no_composite_panels": True,
        "no_sampling_or_imputation": True,
        "selection_rules": {
            "individual_strf": individual_strf_selection,
            "individual_epoch": individual_epoch_selection,
        },
        "figures": entries,
        "status": (
            "pass"
            if len(entries) == 10
            and all(entry["legend_status"] == "pass" for entry in entries.values())
            else "review_required"
        ),
    }
    nature.write_json(args.output_dir / "figure_manifest.json", manifest)
    index_lines = [
        "# Standalone publication figures",
        "",
        "Each image is independent; no composite panel was generated.",
        "",
    ]
    for entry in entries.values():
        index_lines.extend(
            (
                f"- [{entry['title']}]({entry['explanation']})",
                f"  ![{entry['title']}]({entry['png']})",
            )
        )
    (args.output_dir / "README.md").write_text(
        "\n".join(index_lines) + "\n",
        encoding="utf-8",
    )
    print(
        f"OK standalone publication figures: {len(entries)} PNGs in "
        f"{args.output_dir}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (
        AssertionError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
        KeyError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
