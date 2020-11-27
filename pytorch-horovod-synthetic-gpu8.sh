#!/bin/bash
#SBATCH --nodes=2 --ntasks=8 --cpus-per-task=10 -p gputest --gres=gpu:v100:4 -t 15 --mem=32G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

PYTHON=python3
if [ -n "$SING_IMAGE" ]; then
    PYTHON="singularity_wrapper exec python3"
    echo "Using Singularity image $SING_IMAGE"
fi

MAIN_PY=horovod/examples/pytorch/pytorch_synthetic_benchmark.py

module list
export NCCL_DEBUG=INFO

set -x

date
hostname
nvidia-smi

$PYTHON -c "import sys; import torch; print(sys.version, '\nPyTorch:', torch.__version__)"

srun $PYTHON $MAIN_PY --num-iters=100 --batch-size=64
date
