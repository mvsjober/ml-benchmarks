#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=0-1
#SBATCH --output=logs/slurm-%x-%j.out

# Recommended settings from
# https://software.intel.com/content/www/us/en/develop/articles/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html

export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128

export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="0-127"

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
