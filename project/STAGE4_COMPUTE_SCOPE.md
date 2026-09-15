# Stage 4 deadline compute scope

This plan is additive. It never writes into legacy model fits or the completed
HuBERT Base original units. Invalid new units are renamed with a corruption
diagnostic before recomputation; integrity-valid units resume.

## Statistical roles

- Primary analysis: original five-lag predictors and the original five
  frame-weighted grouped outer folds. Scores are calculated separately for
  every held-out recording. Outer folds are scheduling/splitting structures,
  not inferential units.
- HuBERT Base: its completed grouped result is the primary result; its
  completed LORO result is sensitivity-only and is not rerun.
- Other eight checkpoints: full and five reduced models are refit with the
  exact saved alpha for the matching checkpoint/layer/fold/model. Inner CV is
  not repeated.
- Rich and capacity controls: input, deterministic middle, and final depths
  for the six configured control checkpoints.
- Zero lag: the same three depths for all nine checkpoints.
- Structured null: 20 shifts at the middle depth for four checkpoints and
  only prosodic, phonetic, and word models. These are explicitly
  fixed-hyperparameter null sensitivities.

Alpha reuse fails closed unless the existing `comparability_contract.json`
exactly matches the current recordings, feature matrices and grids, feature
schema, lag basis, target PCA specification, and grouped outer splits.

The middle depth is the encoder layer minimizing
`abs(2 * one_based_layer - encoder_depth)`; the lower layer wins an exact tie.
The input representation is excluded from encoder depth.

## Required pre-submission benchmark

```bash
cd /share/home/mitan/Cuhksz4090/project
mkdir -p outputs/stage4_revision/logs

PARTITION=EDIT_ME
ACCOUNT=EDIT_ME
SITE_ARGS=(--partition="$PARTITION" --account="$ACCOUNT")

BENCH_JOB=$(sbatch --parsable "${SITE_ARGS[@]}" \
  slurm/stage4_worker_benchmark.sbatch)
echo "$BENCH_JOB"
```

Wait for this one job. Do not submit the arrays yet. It atomically writes:

```text
outputs/stage4_revision/manifests/deadline_compute_scope/
  resolved_model_layers.json
  primary_fast_refits.tsv
  selected_depth_controls.tsv
  structured_nulls.tsv
  worker_benchmark.json
```

Review all five files. `worker_benchmark.json` contains measured sequential
and parallel two-layer runtimes, the selected worker count, fallback reason,
and scaled runtime estimates. The benchmark chooses two workers only if
`2 × 8` threads exits successfully and is faster than `1 × 8`; otherwise all
layer jobs use one worker with eight BLAS threads.

## Exact array submissions

After reviewing the resolved manifests and measured estimates, run the
preflight against the current activation stores. Do not submit any array if it
reports that the persisted model-specific layer selections are stale:

```bash
python3 scripts/validate_stage4_compute_scope.py \
  --config configs/stage4_revision.yaml \
  --manifest-dir outputs/stage4_revision/manifests/deadline_compute_scope
```

Only after that command reports `"state": "valid"`:

```bash
PRIMARY_JOB=$(sbatch --parsable "${SITE_ARGS[@]}" \
  slurm/stage4_primary_fast_refits.sbatch)

CONTROL_JOB=$(sbatch --parsable "${SITE_ARGS[@]}" \
  slurm/stage4_selected_depth_controls.sbatch)

NULL_JOB=$(sbatch --parsable "${SITE_ARGS[@]}" \
  --dependency="afterok:${PRIMARY_JOB}" \
  slurm/stage4_fixed_alpha_nulls.sbatch)

printf 'PRIMARY_JOB=%s\nCONTROL_JOB=%s\nNULL_JOB=%s\n' \
  "$PRIMARY_JOB" "$CONTROL_JOB" "$NULL_JOB"
```

The arrays are respectively `0-7%3`, `0-20%3`, and `0-79%3`. The null array
unit is exactly one model × shift × resolved middle layer. Restrict a retry to
failed task IDs with `sbatch --array=...%3`; completed units resume safely.

After primary completion, generate recording-level weighted summaries once:

```bash
python3 scripts/summarize_stage4_compute_scope.py \
  --config configs/stage4_revision.yaml
```

This writes separate recording-level results, equal-recording and
duration-weighted summaries, and the HuBERT Base LORO sensitivity table under
`outputs/stage4_revision/compute_scope_summaries/`.

Do not submit `stage4_dry_run.sbatch`, `stage4_full_fits.sbatch`, or
`stage4_nulls.sbatch` for this deadline scope.
