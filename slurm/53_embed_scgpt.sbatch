#!/bin/bash
#SBATCH -J scgpt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
#SBATCH -o logs/scgpt_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/fm-ancestry-bias
echo "NODE: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
conda activate fm-scgpt
python scripts/53_embed_scgpt.py "$@" 2>&1 | grep -v -i "warning"