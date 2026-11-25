#!/bin/bash
#SBATCH -J uce
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o logs/uce_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/fm-ancestry-bias
conda activate fm-uce
python scripts/51_embed_uce.py --stage run "$@" 2>&1 | grep -v -i "warning"
