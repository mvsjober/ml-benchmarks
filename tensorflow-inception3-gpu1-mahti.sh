#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 -p gpu --gres=gpu:a100:1 -t 15 --mem=96G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

MAIN_PY=tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

module list

set -x

date
hostname

#export NCCL_DEBUG=WARN
export LOCAL_RANK=$SLURM_LOCALID

srun python3 $MAIN_PY --model inception3 --num_warmup_batches 10 --num_gpus 1 $*
date
