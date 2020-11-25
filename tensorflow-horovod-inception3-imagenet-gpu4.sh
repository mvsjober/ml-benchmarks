#!/bin/bash
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=10 -p gputest --gres=gpu:v100:4,nvme:200 -t 15 --mem=32G
#SBATCH -A project_2001659
#SBATCH --output=logs/slurm-%x-%j.out

PYTHON=python3
if [ -n "$SING_IMAGE" ]; then
    PYTHON="singularity_wrapper exec python3"
    echo "Using Singularity image $SING_IMAGE"
fi

MAIN_PY=tensorflow-benchmarks-v2.1_compatible/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

# Use Lustre
#DATADIR=/scratch/project_2001659/dac/data/ilsvrc2012/

# Use NVME
IMAGENET_TAR=/scratch/project_2001659/dac/data/ilsvrc2012-tf.tar
DATADIR=${LOCAL_SCRATCH}/ilsvrc2012/

export NCCL_DEBUG=WARN

module list

set -x

date
hostname

time tar xf $IMAGENET_TAR -C $LOCAL_SCRATCH
df -h $LOCAL_SCRATCH

srun $PYTHON $MAIN_PY --use_fp16=true --model inception3 --variable_update horovod --horovod_device gpu --num_warmup_batches 10 --data_dir $DATADIR --data_name imagenet
date
