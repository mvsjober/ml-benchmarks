#!/bin/bash
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=10 -p gputest --gres=gpu:v100:4 -t 15 --mem=32G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

PYTHON=python3
if [ -n "$SING_IMAGE" ]; then
    PYTHON="singularity_wrapper exec python3"
    echo "Using Singularity image $SING_IMAGE"
fi

MAIN_PY=tensorflow-benchmarks-v2.1_compatible/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

module list

set -x

date
hostname

export NCCL_DEBUG=WARN

srun $PYTHON $MAIN_PY --use_fp16=true --model inception3 --variable_update horovod --horovod_device gpu --num_warmup_batches 10
date
