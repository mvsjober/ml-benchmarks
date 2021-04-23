#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4,nvme:200
#SBATCH --output=logs/slurm-%x-%j.out

SCRIPT_DIR=$(dirname $(scontrol -o show job $SLURM_JOB_ID | sed -e 's/.*Command=//' | cut -d ' ' -f 1))
source $SCRIPT_DIR/common.sh

srun python3 $*
date
