#!/bin/bash
# Four sub-007 channels, one job each. 4 x 8 CPUs = 32 CPUs and 32 GB.
# Submit from the Cuhksz4090 repo:
#   mkdir -p /share/home/mitan/combinato_result/logs
#   sbatch --partition=YOUR_CPU_PARTITION spike_sorting_script/combinato/combinato_step1_test.sbatch
#SBATCH --job-name=combinato-step1-test
#SBATCH --array=1-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/mitan/combinato_result/logs/test_%A_%a.out
#SBATCH --error=/share/home/mitan/combinato_result/logs/test_%A_%a.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate data-proc

cd "${SLURM_SUBMIT_DIR}"
python spike_sorting_script/combinato/combinato_step1_test.py "${SLURM_ARRAY_TASK_ID}" 4
