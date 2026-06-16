import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLING_RATE = 30000.0
NUM_CHANNELS_TOTAL = 128
GAIN_TO_UV = 0.195
DEFAULT_OUTPUT_ROOT = Path("/share/home/mitan/spike_sorting")
MERGED_RECORDINGS = {
    "session3": Path("/share/workspace2/tangmi/bistable_sub4/bistable_sub4_session3"),
    "session5": Path("/share/workspace2/tangmi/bistable_sub4/bistable_sub4_session5"),
    "session2": Path("/share/workspace2/tangmi/bistable_sub4/bistable_sub4_session2"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot one exemplar spike-sorting fragment and save figure."
    )
    parser.add_argument("--session", default="session2", help="session name, e.g. session2")
    parser.add_argument(
        "--unit-key",
        default=None,
        help="global unit key like ATL_unit12; default picks a representative unit automatically",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=2.0,
        help="fragment duration in seconds",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root directory containing sorting_results_* folders",
    )
    return parser.parse_args()


def pick_unit(summary_df: pd.DataFrame, requested_key: str | None) -> str:
    if requested_key is not None:
        if requested_key not in set(summary_df["global_key"]):
            raise ValueError(f"Requested unit key not found: {requested_key}")
        return requested_key

    # Use a high-spike unit as a robust exemplar.
    best_row = summary_df.sort_values("n_spikes", ascending=False).iloc[0]
    return str(best_row["global_key"])


def load_binary_trace(recording_path: Path) -> np.memmap:
    raw = np.memmap(recording_path, dtype=np.int16, mode="r")
    n_frames = raw.size // NUM_CHANNELS_TOTAL
    if n_frames == 0:
        raise RuntimeError(f"Empty binary recording: {recording_path}")
    return raw[: n_frames * NUM_CHANNELS_TOTAL].reshape(n_frames, NUM_CHANNELS_TOTAL)


def main():
    args = parse_args()

    session_dir = args.output_root / f"sorting_results_{args.session}"
    summary_csv = session_dir / "all_regions_units_summary.csv"
    spikes_pkl = session_dir / "all_spike_times.pkl"
    recording_path = MERGED_RECORDINGS.get(args.session)

    if recording_path is None:
        raise ValueError(f"Unknown session '{args.session}'. Available: {sorted(MERGED_RECORDINGS)}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    if not spikes_pkl.exists():
        raise FileNotFoundError(f"Spike-times pickle not found: {spikes_pkl}")
    if not recording_path.exists():
        raise FileNotFoundError(f"Merged recording not found: {recording_path}")

    summary_df = pd.read_csv(summary_csv)
    with spikes_pkl.open("rb") as f:
        all_spike_times = pickle.load(f)

    unit_key = pick_unit(summary_df, args.unit_key)
    unit_spike_times_s = np.asarray(all_spike_times[unit_key], dtype=float)
    if unit_spike_times_s.size == 0:
        raise RuntimeError(f"No spikes for unit: {unit_key}")

    row = summary_df.loc[summary_df["global_key"] == unit_key].iloc[0]
    best_ch_1idx = int(row["best_channel_1idx"])
    if best_ch_1idx < 1 or best_ch_1idx > NUM_CHANNELS_TOTAL:
        raise RuntimeError(f"Invalid best channel for {unit_key}: {best_ch_1idx}")
    ch_idx = best_ch_1idx - 1

    trace = load_binary_trace(recording_path)
    duration_s = trace.shape[0] / SAMPLING_RATE

    center_s = float(np.median(unit_spike_times_s))
    half_window = args.window_s / 2.0
    start_s = max(0.0, center_s - half_window)
    end_s = min(duration_s, start_s + args.window_s)
    start_s = max(0.0, end_s - args.window_s)

    start_idx = int(start_s * SAMPLING_RATE)
    end_idx = int(end_s * SAMPLING_RATE)

    t = np.arange(start_idx, end_idx) / SAMPLING_RATE
    channel_uv = trace[start_idx:end_idx, ch_idx].astype(np.float32) * GAIN_TO_UV

    mask = (unit_spike_times_s >= start_s) & (unit_spike_times_s <= end_s)
    spikes_window_s = unit_spike_times_s[mask]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    axes[0].plot(t, channel_uv, lw=0.8, color="tab:blue")
    for s in spikes_window_s:
        axes[0].axvline(s, color="tab:red", alpha=0.4, lw=0.7)
    axes[0].set_ylabel("uV")
    axes[0].set_title(
        f"{args.session} | {unit_key} | channel {best_ch_1idx} | fragment {start_s:.3f}-{end_s:.3f}s"
    )
    axes[0].grid(alpha=0.2)

    axes[1].eventplot(spikes_window_s, lineoffsets=0, linelengths=0.8, colors="tab:red")
    axes[1].set_ylim(-1, 1)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Spikes")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    output_path = args.output_root / f"example_fragment_{args.session}_{unit_key}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[DONE] Saved figure: {output_path}")
    print(f"[INFO] Spikes in plotted fragment: {len(spikes_window_s)}")


if __name__ == "__main__":
    main()
