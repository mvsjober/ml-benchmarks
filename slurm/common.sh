SCRIPT=$1
shift
cd $SLURM_SUBMIT_DIR

module list

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

set -x

date
hostname
nvidia-smi

source $SCRIPT $*

date
