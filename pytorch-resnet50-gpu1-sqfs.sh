#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task=10 -p gputest --gres=gpu:v100:1 -t 15 --mem=64G
#SBATCH -A project_2001659 
#SBATCH --output=logs/slurm-%x-%j.out

IMAGENET_SQFS=/scratch/project_2001659/mvsjober/st-2/ilsvrc2012-torch.sqfs

module list
set -x

date
hostname
nvidia-smi

export SING_FLAGS="-B $IMAGENET_SQFS:/data:image-src=/ $SING_FLAGS"

date
srun python3 pytorch-benchmarks/main.py -a resnet50 -p 10 -b 64 -j 10 --epoch 1 /data
date
