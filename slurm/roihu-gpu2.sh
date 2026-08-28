#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:2
#SBATCH --output=logs/slurm-%x-%j.out

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
