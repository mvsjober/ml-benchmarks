SCRIPT=$1
shift
cd $SLURM_SUBMIT_DIR

module list

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

if which nvidia-smi > /dev/null 2>&1; then
    export SMI_CMD=nvidia-smi
else
    export SMI_CMD=rocm-smi
fi


(set -x
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 hostname
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 $SMI_CMD
which python3
date
)

if which nvidia-smi > /dev/null 2>&1; then
    export NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    export NUM_GPUS=$(rocm-smi -i | grep "GPU ID" | wc -l)
fi
echo "NUM_GPUS=$NUM_GPUS"

source $SCRIPT $*

(set -x
date
)
