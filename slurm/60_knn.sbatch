#!/bin/bash
#SBATCH -J knn
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o logs/knn_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/fm-ancestry-bias
echo "NODE: $(hostname)"
conda activate fm-core
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python scripts/60_knn_annotate.py "$@" 2>&1 | grep -v -i "warning"