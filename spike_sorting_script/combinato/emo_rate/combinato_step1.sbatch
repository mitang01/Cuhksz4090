#!/bin/bash
# 8 jobs, 8 channels each. At most 8 jobs run at once:
# 8 x 8 CPUs = 64 CPUs and 8 x 8 GB = 64 GB.
# Submit from the Cuhksz4090 repo:
#   mkdir -p /share/home/mitan/combinato_result/emo_rate/logs
#   cd /share/home/mitan/Cuhksz4090/spike_sorting_script/combinato/emo_rate
#   sbatch combinato_step1.sbatch
#SBATCH --job-name=emo-rate-step1
#SBATCH --array=1-8%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=gpu3node
#SBATCH --nodelist=gpu05
#SBATCH --time=08:00:00
#SBATCH --output=/share/home/mitan/combinato_result/emo_rate/logs/%A_%a.out
#SBATCH --error=/share/home/mitan/combinato_result/emo_rate/logs/%A_%a.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate data-proc

cd "${SLURM_SUBMIT_DIR}"
python combinato_step1.py "${SLURM_ARRAY_TASK_ID}" 8
