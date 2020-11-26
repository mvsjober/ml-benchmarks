#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task=40 -p gputest --gres=gpu:v100:4,nvme:100 -t 15 --mem=0
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

PYTHON=python3
if [ -n "$SING_IMAGE" ]; then
    PYTHON="singularity_wrapper exec python3"
    echo "Using Singularity image $SING_IMAGE"
fi

MAIN_PY=pytorch-benchmarks/main.py
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

module list
set -x

date
hostname
nvidia-smi

$PYTHON -V
$PYTHON -c "import torch; print(torch.__version__)"

tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH

date
srun $PYTHON $MAIN_PY -a resnet50 -p 10 -b 256 -j 40 --epoch 1 \
     --dist-url 'tcp://127.0.0.1:8842' --dist-backend 'nccl' \
     --multiprocessing-distributed \
     --world-size 1 --rank 0 \
     ${LOCAL_SCRATCH}/ilsvrc2012-torch

# srun $PYTHON -m torch.distributed.launch --nproc_per_node=4 $MAIN_PY \
#      -a resnet50 -p 10 -b 64 -j 40 --epoch 1 \
#      ${LOCAL_SCRATCH}/ilsvrc2012-torch
date
