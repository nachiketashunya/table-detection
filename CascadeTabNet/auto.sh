#!/bin/bash
#SBATCH --job-name=ctab_run
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 ##Define number of GPUs
#SBATCH --output=logs/ctab_%j.log

echo "JOb Submitted"

module load conda/conda
source /opt/ohpc/apps/conda/bin/activate
conda activate pyten

python train_cluster.py
# python nosplit.py

