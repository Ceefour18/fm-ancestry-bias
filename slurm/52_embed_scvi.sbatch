#!/bin/bash
#SBATCH -J scvi
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
#SBATCH -o logs/scvi_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/fm-ancestry-bias
conda activate fm-scvi
python scripts/52_embed_scvi.py "$@" 2>&1 | grep -v -i "warning"
