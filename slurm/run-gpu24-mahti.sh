#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --nodes=6
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=32
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4,nvme:200
#SBATCH --output=logs/slurm-%x-%j.out

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
