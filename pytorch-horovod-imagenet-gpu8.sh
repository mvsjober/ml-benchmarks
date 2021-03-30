#!/bin/bash
#SBATCH --nodes=2 --ntasks=8 --cpus-per-task=10 -p gputest --gres=gpu:v100:4,nvme:50 -t 15 --mem=0
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

DATASET_TAR_ARCHIVE=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

module list
#export NCCL_DEBUG=INFO

set -x

date
hostname
nvidia-smi

rm -f checkpoint-*.pth.tar

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     tar xf $DATASET_TAR_ARCHIVE --strip 1 -C $LOCAL_SCRATCH/

srun python3 pytorch_imagenet_resnet50_horovod_benchmark.py \
     --train-dir=${LOCAL_SCRATCH}/train \
     --val-dir=${LOCAL_SCRATCH}/val \
     --epochs=2
date
