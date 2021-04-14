#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 -p gpu --gres=gpu:a100:1 -t 15 --mem=96G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

module list
export LOCAL_RANK=$SLURM_LOCALID

set -x

date
hostname
nvidia-smi

python3 pytorch_synthetic_benchmark.py --num-iters=100 --batch-size=64
date
