#!/usr/bin/env python3
"""Extract and sort each kept microwire with Combinato's own commands.

Reads the merged subject files written by merge_subject_mats.py.
For every kept channel it runs:

  css-extract
  css-mask-artifacts
  css-simple-clustering
  css-plot-sorted --label sort_pos_simple

The overview plot for that channel is written to:

  /share/home/mitan/combinato_result/word_boun/sub-009/A-001/overview/

This is the default positive-spike pass. The folder name sort_pos_simple
is the label Combinato stores.

There are 304 kept channels. The cluster still allows 64 CPUs and 64 GB
at once. The channels are split into 38 jobs with 8 channels each, and
each job is allowed 4 hours:

  python combinato_step1.py 1 38
  python combinato_step1.py 2 38

JOB is 1-based. Each job writes its own channel folders, so different
jobs can run at the same time. Results stay under combinato_result/word_boun.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

DATA_ROOT = Path("/share/workspace3/ieeg/micro/word_boun_perce_v2")
RESULT_ROOT = Path("/share/home/mitan/combinato_result/word_boun")
SUBJECTS = ["sub-007", "sub-008", "sub-009", "sub-010", "sub-011", "sub-012"]
SR = 30000.0
# Intan ADC counts to microvolts. The merged files are still raw integers.
SCALE_TO_UV = np.float32(0.195)
BLOCK = 200000
# 64+64+48+32+32+64 kept channels. 8 channels in each of 38 jobs.
EXPECTED_CHANNELS = 304
N_JOBS = 38
COMMANDS = (
    "css-extract",
    "css-mask-artifacts",
    "css-simple-clustering",
    "css-plot-sorted",
)

ENV = os.environ.copy()
ENV["MPLBACKEND"] = "Agg"
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["NUMEXPR_NUM_THREADS"] = "1"


def run(cmd, cwd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=ENV, check=True)


def time_axis(shape, n_channels):
    if shape[0] == n_channels:
        return 1
    if shape[1] == n_channels:
        return 0
    raise ValueError(f"channel count {n_channels} does not match shape {shape}")


def write_continuous(merged, index, axis, out_path):
    """One channel as HDF5 /data plus /sr, which css-extract --h5 reads."""
    with h5py.File(merged, "r") as src, h5py.File(out_path, "w") as out:
        amp = src["amplifier_data"]
        n = amp.shape[axis]
        scale = SCALE_TO_UV if np.issubdtype(amp.dtype, np.integer) else np.float32(1)
        out.create_dataset("sr", data=np.array([SR], dtype=np.float64))
        data = out.create_dataset("data", shape=(n,), dtype=np.float32)
        pos = 0
        while pos < n:
            take = min(BLOCK, n - pos)
            if axis == 1:
                piece = np.asarray(amp[index, pos:pos + take], dtype=np.float32)
            else:
                piece = np.asarray(amp[pos:pos + take, index], dtype=np.float32)
            data[pos:pos + take] = piece * scale
            pos += take


def channel_list(merged):
    with h5py.File(merged, "r") as f:
        if "channel_number" not in f or "amplifier_data" not in f:
            raise KeyError("merged file needs amplifier_data and channel_number")
        numbers = [int(x) for x in f["channel_number"][:]]
        if "brain_region" in f:
            regions = []
            for item in f["brain_region"][:]:
                if isinstance(item, bytes):
                    regions.append(item.decode())
                else:
                    regions.append(str(item))
        else:
            regions = [""] * len(numbers)
        axis = time_axis(tuple(int(x) for x in f["amplifier_data"].shape), len(numbers))
    return numbers, regions, axis


def sort_channel(subject, number, region, merged, axis, index, result_root, run_cmd=run):
    name = f"A-{number:03d}"
    subject_dir = result_root / subject
    ch_dir = subject_dir / name
    data_h5 = ch_dir / f"data_{name}.h5"
    overview = ch_dir / "overview"
    subject_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{subject} {name} {region}")
    if overview.is_dir() and any(overview.glob("*.png")):
        print(f"[SKIP] {next(overview.glob('*.png'))}")
        return True

    if not data_h5.is_file():
        continuous_dir = subject_dir / "continuous"
        continuous_dir.mkdir(parents=True, exist_ok=True)
        continuous = continuous_dir / f"{name}.h5"
        write_continuous(merged, index, axis, continuous)
        run_cmd(
            ["css-extract", "--h5", "--files", str(continuous), "--destination", str(subject_dir)],
            subject_dir,
        )
        continuous.unlink()
    if not data_h5.is_file():
        print(f"[STOP] {subject} {name}: css-extract did not write {data_h5}")
        return False

    run_cmd(["css-mask-artifacts", "--datafile", str(data_h5)], ch_dir)
    run_cmd(["css-simple-clustering", "--datafile", str(data_h5)], ch_dir)
    run_cmd(
        ["css-plot-sorted", "--datafile", str(data_h5), "--label", "sort_pos_simple"],
        ch_dir,
    )
    plots = sorted(overview.glob("*.png")) if overview.is_dir() else []
    if not plots:
        print(f"[WARN] {subject} {name}: no overview plot was written")
        return False
    print(f"[PLOT] {plots[0]}")
    return True


def collect_tasks(data_root=DATA_ROOT):
    tasks = []
    for subject in SUBJECTS:
        merged = data_root / subject / f"{subject}.mat"
        if not merged.is_file():
            print(f"[STOP] {subject}: merged file not found: {merged}")
            continue
        try:
            numbers, regions, axis = channel_list(merged)
        except Exception as exc:
            print(f"[STOP] {subject}: {exc}")
            continue
        for index, (number, region) in enumerate(zip(numbers, regions)):
            tasks.append((subject, number, region, merged, axis, index))
    return tasks


def parse_job(argv):
    """Return (job, n_jobs). No arguments means one job that runs every channel."""
    if len(argv) == 1:
        return 1, 1
    if len(argv) != 3:
        raise SystemExit("usage: python combinato_step1.py [JOB N_JOBS]")
    job = int(argv[1])
    n_jobs = int(argv[2])
    if n_jobs < 1 or job < 1 or job > n_jobs:
        raise SystemExit("JOB must be from 1 to N_JOBS")
    return job, n_jobs


def slice_tasks(tasks, job, n_jobs):
    n = len(tasks)
    start = (job - 1) * n // n_jobs
    stop = job * n // n_jobs
    return tasks[start:stop]


def main(argv=None, data_root=DATA_ROOT, result_root=RESULT_ROOT, run_cmd=run):
    if argv is None:
        argv = sys.argv
    job, n_jobs = parse_job(argv)
    tasks = collect_tasks(data_root)
    if n_jobs == N_JOBS and len(tasks) != EXPECTED_CHANNELS:
        print(f"[WARN] expected {EXPECTED_CHANNELS} channels for {N_JOBS} jobs, found {len(tasks)}")
    chosen = slice_tasks(tasks, job, n_jobs)
    print(f"[JOB] {job}/{n_jobs} channels={len(chosen)} of {len(tasks)}")
    if not chosen:
        if not tasks:
            raise SystemExit(1)
        return
    if run_cmd is run:
        missing = [cmd for cmd in COMMANDS if shutil.which(cmd) is None]
        if missing:
            raise SystemExit("Combinato command not on PATH: " + ", ".join(missing))
    result_root.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] {result_root}")
    ok = True
    for subject, number, region, merged, axis, index in chosen:
        if not sort_channel(subject, number, region, merged, axis, index, result_root, run_cmd):
            ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
