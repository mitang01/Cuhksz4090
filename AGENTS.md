# Cuhksz4090

Batch scripts for micro-electrode spike sorting (SpikeInterface + MountainSort4) plus
merging Intan MAT chunks, trigger alignment, and SortingView link generation. These are
standalone scripts run on a SLURM cluster; there is no web app, service, or test suite.

## Cursor Cloud specific instructions

- Runtime lives in the conda env `data-proc` (Python 3.10), built from
  `spike_sorting/environment.data-proc.yml` (the Linux-sanitized version of
  `environment.yml`). Miniconda is installed at `$HOME/miniconda3` and persists in the VM
  snapshot; the startup update script refreshes the env with `conda env update`.
- Activate before running anything:
  `source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate data-proc`.
- MountainSort4 sorts via dask multiprocessing that defaults to `spawn` here and crashes
  with `BrokenProcessPool` / "importing main". Force fork before sorting (the repo scripts
  already do this): `import multiprocessing as mp; mp.set_start_method("fork", force=True)`
  and `import dask; dask.config.set({"multiprocessing.context": "fork"})`.
- All input paths in the scripts are hardcoded to cluster locations
  (`/share/workspace*/...`, `/share/home/mitan/...`) that do NOT exist in the cloud VM, and
  the recording data is not in the repo. Full sessions cannot be run here without data;
  to exercise the pipeline, generate a synthetic recording with
  `spikeinterface.full.generate_ground_truth_recording(...)` and run the same steps
  (probe → `bandpass_filter` → `common_reference` → `run_sorter("mountainsort4", ...)` →
  `create_sorting_analyzer` → `compute("quality_metrics", ...)` → `export_to_phy`).
- `sbatch.sh` / `*.sbatch` are SLURM submission wrappers (there is no SLURM here); read them
  for the intended CLI invocations rather than submitting them.
- GUI tools (`phy template-gui`, `spikeinterface-gui`) are installed but need a display and
  are only for manual curation; they are not needed to run the sorting scripts.
