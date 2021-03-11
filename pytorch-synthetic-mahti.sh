#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --partition=test -t 0-1
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

module list

set -x

date
hostname

# Recommended settings from
# https://software.intel.com/content/www/us/en/develop/articles/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html

export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128

export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="0-127"

srun python3 pytorch_synthetic_benchmark.py --num-iters=10 --no-cuda $*
date
