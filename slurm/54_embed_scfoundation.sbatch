#!/bin/bash
#SBATCH -J scf
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o logs/scf_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/fm-ancestry-bias
echo "NODE: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
conda activate fm-scf
python scripts/54_embed_scfoundation.py "$@" 2>&1 | grep -v -i "warning"