#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --output=logs/slurm-%x-%j.out

export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
#export KMP_AFFINITY=granularity=fine,compact,1,0

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
