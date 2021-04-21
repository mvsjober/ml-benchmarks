#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=15
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/slurm-%x-%j.out

module list

set -x

date
hostname
nvidia-smi

srun $*
date
