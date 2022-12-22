#!/bin/bash
#SBATCH --account=project_462000007
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=63
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --output=logs/slurm-%x-%j.out

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
