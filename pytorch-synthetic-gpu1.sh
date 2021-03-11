#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 -p gputest --gres=gpu:v100:1 -t 15 --mem=92G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

module list

set -x

date
hostname
nvidia-smi

python3 pytorch_synthetic_benchmark.py --num-iters=100 --batch-size=64
date
