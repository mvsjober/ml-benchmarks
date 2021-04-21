#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4
#SBATCH --output=logs/slurm-%x-%j.out

module list

set -x

date
hostname
nvidia-smi

srun $*
date
