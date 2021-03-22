#!/bin/bash
#SBATCH --nodes=2 --ntasks=8 --cpus-per-task=10 -p gputest --gres=gpu:v100:4 -t 15 --mem=32G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

module list
export NCCL_DEBUG=INFO
export UCX_LOG_LEVEL=debug

set -x

date
hostname
nvidia-smi

srun python3 pytorch_synthetic_horovod_benchmark.py --num-iters=100 --batch-size=64
date
