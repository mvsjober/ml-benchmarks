#!/bin/bash
#SBATCH --nodes=2 --ntasks=8 --cpus-per-task=10 -p gputest --gres=gpu:v100:4 -t 15 --mem=32G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

PYTHON=python3
if [ -n "$SING_IMAGE" ]; then
    PYTHON="singularity_wrapper exec python3"
    echo "Using Singularity image $SING_IMAGE"
fi

MAIN_PY=horovod/examples/pytorch/pytorch_imagenet_resnet50.py
DATASET_TAR_ARCHIVE=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar
DATADIR=$LOCAL_SCRATCH

module list
export NCCL_DEBUG=INFO

set -x

date
hostname
nvidia-smi

$PYTHON -c "import sys; import torch; print(sys.version, '\nPyTorch:', torch.__version__)"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     tar xf $DATASET_TAR_ARCHIVE --strip 1 -C $LOCAL_SCRATCH/

srun $PYTHON $MAIN_PY --train-dir=${DATADIR}/train --val-dir=${DATADIR}/val
date
