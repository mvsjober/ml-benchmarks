#!/bin/bash
#SBATCH --account=project_462000007
#SBATCH --partition=eap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=15
#SBATCH --output=logs/slurm-%x-%j.out

cd $SLURM_SUBMIT_DIR
source slurm/common.sh