#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=40 -p test -t 15 --mem=0
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

module list

set -x

date
hostname
#nvidia-smi

# export SINGULARITYENV_PREPEND_PATH=/opt/intel/oneapi/intelpython/latest/envs/pytorch/bin/
# export SING_IMAGE=/appl/soft/ai/singularity/images/intel_aikit_192021.1.sif

export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
#export KMP_AFFINITY=granularity=fine,compact,1,0

# export OMP_SCHEDULE=STATIC
# export OMP_PROC_BIND=CLOSE
# export GOMP_CPU_AFFINITY="0-39"

singularity_wrapper exec python3 pytorch_synthetic_benchmark.py --num-iters=10 --batch-size=32 --no-cuda
date
