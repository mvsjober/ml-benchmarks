#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 -p gputest --gres=gpu:v100:1 -t 15 --mem=92G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

PYTHON=python3
MAIN_PY=pytorch_synthetic_benchmark.py

module list

set -x

date
hostname
nvidia-smi

$PYTHON -c "import sys; import torch; print(sys.version, '\nPyTorch:', torch.__version__); print(*torch.__config__.show().split('\n'), sep='\n')"

srun $PYTHON $MAIN_PY --num-iters=100 --batch-size=64
date
