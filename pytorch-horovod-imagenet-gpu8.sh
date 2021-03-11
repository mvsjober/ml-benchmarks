#!/bin/bash
#SBATCH --nodes=2 --ntasks=8 --cpus-per-task=10 -p gputest --gres=gpu:v100:4,nvme:50 -t 15 --mem=32G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

MAIN_PY=horovod/examples/pytorch/pytorch_imagenet_resnet50.py
DATASET_TAR_ARCHIVE=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar
DATADIR=$LOCAL_SCRATCH

module list
#export NCCL_DEBUG=INFO

set -x

date
hostname
nvidia-smi

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     tar xf $DATASET_TAR_ARCHIVE --strip 1 -C $LOCAL_SCRATCH/

srun python3 $MAIN_PY --train-dir=${DATADIR}/train --val-dir=${DATADIR}/val --epochs=2
date
