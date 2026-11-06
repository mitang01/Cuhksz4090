#!/usr/bin/env python3
"""Sort four sub-007 channels, one from each region.

The channels are the first microwire in each region:
  A-017 ATL
  A-033 VmPFC
  A-065 HG
  A-097 Amygdala

One channel per job, so four jobs can finish within an hour:

  python combinato_step1_test.py 1 4

Plots are written under /share/home/mitan/combinato_result/sub-007/.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import combinato_step1 as step

SUBJECT = "sub-007"
CHANNELS = [
    (17, "ATL"),
    (33, "VmPFC"),
    (65, "HG"),
    (97, "Amygdala"),
]
N_JOBS = 4


def collect_tasks(data_root=step.DATA_ROOT):
    merged = data_root / SUBJECT / f"{SUBJECT}.mat"
    if not merged.is_file():
        print(f"[STOP] {SUBJECT}: merged file not found: {merged}")
        return []
    try:
        numbers, regions, axis = step.channel_list(merged)
    except Exception as exc:
        print(f"[STOP] {SUBJECT}: {exc}")
        return []
    tasks = []
    for number, region in CHANNELS:
        if number not in numbers:
            print(f"[STOP] {SUBJECT}: channel {number} not in merged file")
            continue
        index = numbers.index(number)
        stored = regions[index]
        if stored and stored != region:
            print(f"[STOP] {SUBJECT} A-{number:03d}: file says {stored}, expected {region}")
            continue
        tasks.append((SUBJECT, number, region, merged, axis, index))
    return tasks


def main(argv=None, data_root=step.DATA_ROOT, result_root=step.RESULT_ROOT, run_cmd=step.run):
    if argv is None:
        argv = sys.argv
    job, n_jobs = step.parse_job(argv)
    tasks = collect_tasks(data_root)
    if n_jobs == N_JOBS and len(tasks) != N_JOBS:
        print(f"[WARN] expected {N_JOBS} channels, found {len(tasks)}")
    chosen = step.slice_tasks(tasks, job, n_jobs)
    print(f"[JOB] {job}/{n_jobs} channels={len(chosen)} of {len(tasks)}")
    if not chosen:
        if not tasks:
            raise SystemExit(1)
        return
    if run_cmd is step.run:
        missing = [cmd for cmd in step.COMMANDS if shutil.which(cmd) is None]
        if missing:
            raise SystemExit("Combinato command not on PATH: " + ", ".join(missing))
    result_root.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] {result_root}")
    ok = True
    for subject, number, region, merged, axis, index in chosen:
        if not step.sort_channel(subject, number, region, merged, axis, index, result_root, run_cmd):
            ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
