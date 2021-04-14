#!/bin/bash
#SBATCH --nodes=2 --ntasks=8 --cpus-per-task=10 -p gpu --gres=gpu:a100:4 -t 15 --mem=96G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

MAIN_PY=tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

module list

set -x

date
hostname

#export NCCL_DEBUG=WARN
#export LOCAL_RANK=$SLURM_LOCALID

srun python3 $MAIN_PY --use_fp16=true --model inception3 --variable_update horovod --horovod_device gpu --num_warmup_batches 10
date
