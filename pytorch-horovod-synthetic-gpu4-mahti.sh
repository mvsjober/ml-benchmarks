#!/bin/bash
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=10 -p gpu --gres=gpu:a100:4 -t 15 --mem=96G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

module list

set -x

date
hostname
nvidia-smi

srun python3 pytorch_synthetic_horovod_benchmark.py --num-iters=100 --batch-size=64
date
