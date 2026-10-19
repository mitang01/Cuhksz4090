#!/bin/bash
#SBATCH --job-name=hubert-activations
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/project"
source .venv/bin/activate
python scripts/extract_activations.py \
  --config configs/models.yaml \
  --manifest outputs/manifest.csv \
  --validation-report outputs/validation_report.json

