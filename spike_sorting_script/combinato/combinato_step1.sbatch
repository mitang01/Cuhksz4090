#!/bin/bash
# One Combinato channel per array task. 192 tasks cover every kept microwire.
# Submit from the Cuhksz4090 repo:
#   mkdir -p /share/home/mitan/combinato_result/logs
#   sbatch --partition=YOUR_CPU_PARTITION spike_sorting_script/combinato/combinato_step1.sbatch
#SBATCH --job-name=combinato-step1
#SBATCH --array=1-192
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/mitan/combinato_result/logs/%A_%a.out
#SBATCH --error=/share/home/mitan/combinato_result/logs/%A_%a.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate data-proc

cd "${SLURM_SUBMIT_DIR}"
python spike_sorting_script/combinato/combinato_step1.py "${SLURM_ARRAY_TASK_ID}" 192
