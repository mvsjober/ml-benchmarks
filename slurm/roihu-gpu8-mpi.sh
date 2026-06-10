#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=480G
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:4
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --argos=no

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
