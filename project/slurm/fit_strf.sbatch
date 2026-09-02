#!/bin/bash
#SBATCH --job-name=fit-speech-strf
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/project"
source .venv/bin/activate
python scripts/fit_strf.py --config configs/analysis.yaml --layer layer_00_input

