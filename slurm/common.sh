SCRIPT=$1
shift
cd $SLURM_SUBMIT_DIR

module list

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

(set -x
#singularity --version
hostname
nvidia-smi -L
date
)

export NUM_GPUS=$(nvidia-smi -L | wc -l)

source $SCRIPT $*

(set -x
date
)
