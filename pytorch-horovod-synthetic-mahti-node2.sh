#!/bin/bash
#SBATCH --nodes=2 --ntasks-per-node=1 --cpus-per-task=128 --partition=test -t 0-1
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

module list

set -x

date
hostname

# Debugging
export UCX_LOG_LEVEL=debug

# Recommended settings from
# https://software.intel.com/content/www/us/en/develop/articles/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html

export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128

export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="0-127"

# bs=32,   nodes=2, 6.6 img/sec → 13.1 on 2 nodes
# bs=64,   nodes=2, 8.5 img/sec → 17.0
# bs=256,  nodes=2, 15.9 img/sec → 31.8
# bs=1024, nodes=2, job id 407174

srun python3 pytorch_synthetic_horovod_benchmark.py --num-iters=10 --no-cuda $*
date
