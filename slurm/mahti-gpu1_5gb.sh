#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=17500
#SBATCH --time=15
#SBATCH --gres=gpu:a100_1g.5gb:1,nvme:20
#SBATCH --output=logs/slurm-%x-%j.out

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
