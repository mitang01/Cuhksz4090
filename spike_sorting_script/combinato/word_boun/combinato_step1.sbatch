#!/bin/bash
# 64 jobs, 3 channels each. At most 8 jobs run at once:
# 8 x 8 CPUs = 64 CPUs and 8 x 8 GB = 64 GB.
# Submit from the Cuhksz4090 repo:
#   mkdir -p /share/home/mitan/combinato_result/story_listen/logs
#   sbatch --partition=YOUR_CPU_PARTITION spike_sorting_script/combinato/story_listen/combinato_step1.sbatch
#SBATCH --job-name=combinato-step1
#SBATCH --array=1-64%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/share/home/mitan/combinato_result/story_listen/logs/%A_%a.out
#SBATCH --error=/share/home/mitan/combinato_result/story_listen/logs/%A_%a.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate data-proc

cd "${SLURM_SUBMIT_DIR}"
python spike_sorting_script/combinato/story_listen/combinato_step1.py "${SLURM_ARRAY_TASK_ID}" 64
