#!/bin/bash
#SBATCH -J sg_ft
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o logs/sg_ft_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/fm-ancestry-bias
echo "NODE: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
conda activate fm-scgpt
python scripts/92_finetune_scgpt.py "$@" 2>&1 | grep -v -i "warning\|flash_attn"