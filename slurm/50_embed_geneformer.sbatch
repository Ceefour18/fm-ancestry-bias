#!/bin/bash
#SBATCH -J geneformer
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o logs/geneformer_%j.out
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fm-geneformer
cd ~/fm-ancestry-bias
echo "NODE: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python scripts/50_embed_geneformer.py "$@" 2>&1 | grep -v -E "Warning|warn"
