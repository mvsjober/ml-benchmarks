#!/bin/bash
#SBATCH --account=project_462000007
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=15
#SBATCH --output=logs/slurm-%x-%j.out

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
