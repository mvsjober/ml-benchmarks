#!/bin/bash
#SBATCH --account=project_462000007
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=63
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --output=logs/slurm-%x-%j.out

export MIOPEN_USER_DB_PATH=/tmp/miopen-userdb-$USER
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache-$USER

#export MIOPEN_DEBUG_DISABLE_FIND_DB=1
#unset MIOPEN_DISABLE_CACHE
export MIOPEN_DISABLE_CACHE=1
#export MIOPEN_FIND_ENFORCE=5 

#export NCCL_NET_GDR_LEVEL=3

cd $SLURM_SUBMIT_DIR
source slurm/common.sh
