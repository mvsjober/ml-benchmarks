#!/bin/bash
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=10 -p gputest --gres=gpu:v100:4 -t 15 --mem=32G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

MAIN_PY=horovod/examples/pytorch/pytorch_synthetic_benchmark.py

module list
#export NCCL_DEBUG=INFO

set -x

date
hostname
nvidia-smi

srun python3 $MAIN_PY --num-iters=100 --batch-size=64
date
