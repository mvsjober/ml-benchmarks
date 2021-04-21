#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=6
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=32
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
