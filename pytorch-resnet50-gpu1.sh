#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task=10 -p gputest --gres=gpu:v100:1,nvme:100 -t 15 --mem=64G
#SBATCH -A project_2001659 
#SBATCH --output=logs/slurm-%x-%j.out

MAIN_PY=pytorch-benchmarks/main.py
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar
IMAGENET_SQFS=/scratch/dac/data/ilsvrc2012-torch.sqfs

module list
set -x

date
hostname
nvidia-smi

# Download from Allas
# cd $LOCAL_SCRATCH
# time swift download mldata-auth ilsvrc2012-torch-resized-new.tar
# time tar -xf ilsvrc2012-torch-resized-new.tar

time tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH
# df -h $LOCAL_SCRATCH

# Use squashfs
# export SING_FLAGS="-B $IMAGENET_SQFS:/data:image-src=/ $SING_FLAGS"

date
#srun $PYTHON $MAIN_PY -a resnet50 -p 10 -b 64 -j 10 --epoch 1 /data
srun $PYTHON $MAIN_PY -a resnet50 -p 10 -b 64 -j 10 --epoch 1 ${LOCAL_SCRATCH}/ilsvrc2012-torch
date
