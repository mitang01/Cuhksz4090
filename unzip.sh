#!/bin/bash
#SBATCH --job-name=unzip
#SBATCH --partition=gpu3node
#SBATCH --nodelist=gpu05
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

eval "$(conda shell.bash hook)"
conda activate data-proc

python move_data.py
