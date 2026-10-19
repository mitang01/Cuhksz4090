# Stage 4 manuscript handoff

Build the compact package after the deadline-scope jobs finish:

```bash
cd /share/home/mitan/Cuhksz4090/project

python3 scripts/build_stage4_handoff.py \
  --config configs/stage4_revision.yaml \
  --job-ids "${BENCH2_JOB},${PRIMARY_JOB},${CONTROL2_JOB},${NULL2_JOB}"

python3 scripts/build_stage4_handoff.py \
  --output outputs/stage4_revision/handoff \
  --verify
```

If those shell variables are no longer defined, pass the numeric Slurm job
IDs. Log filenames are also scanned for IDs. To rebuild, preserve the prior
package:

```bash
python3 scripts/build_stage4_handoff.py --archive-existing
```

The package is published atomically at
`outputs/stage4_revision/handoff/`. `HANDOFF.json` is `COMPLETE` only when all
expected units and inferential cells are available; otherwise it is `PARTIAL`
and `verification/missing_incomplete_units.csv` identifies every missing or
invalid unit. Manually deleted primary-fast-refit outputs are never replaced
with fold-level summaries.

The handoff contains:

- recording-level primary, split-sensitivity, selected-depth control, PCA,
  capacity, zero-lag, and structured-null tables;
- recording-bootstrap intervals, exact recording sign-flip tests, BH results,
  effect counts, and paired sensitivity comparisons where all 12 recordings
  exist;
- rich-feature inventory and copied provenance/alignment/temporal-context QC;
- final configs, resolved manifests, Stage 4 source snapshots, optional Git
  patch, test reports, Slurm accounting, runtime/memory, warnings, and errors;
- hashes and server paths for every prediction archive, plus one
  representative `predictions.npz`.

It excludes activation HDF5 stores, raw audio, feature NPZ trees, null
predictions, and all other prediction payloads. `SHA256SUMS` covers every
delivered file except itself.

For transfer:

```bash
tar -C outputs/stage4_revision -czf stage4_handoff.tar.gz handoff
rsync -avP stage4_handoff.tar.gz USER@DESTINATION:/path/
```
