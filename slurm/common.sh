SCRIPT=$1
shift
cd $SLURM_SUBMIT_DIR

module list

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export WANDB_DISABLED=true

if which nvidia-smi > /dev/null 2>&1; then
    export SMI_CMD=nvidia-smi
else
    export SMI_CMD=rocm-smi
fi


export PYTHON3="python3"
if [ -n "$SIF" ]; then
    PYTHON3="singularity exec --nv $SIF python3"
    if [ -x "$(command -v csc-common-bind)" ]; then
        PYTHON3="singularity exec --nv --bind=$(csc-common-bind) $SIF python3"
    fi
fi
echo "Launching python3 as \"$PYTHON3\""

(set -x
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 hostname
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 $SMI_CMD
date
)

if which nvidia-smi > /dev/null 2>&1; then
    export NUM_GPUS=$(nvidia-smi -L | grep "^GPU" |  wc -l)
elif which rocm-smi > /dev/null 2>&1; then
    export NUM_GPUS=$(rocm-smi -i --csv | grep "^card" | wc -l)
else
    export NUM_GPUS=0
fi
echo "NUM_GPUS=$NUM_GPUS"

LUMI_GPUENERGY=/appl/local/csc/soft/ai/bin/gpu-energy
if [ -x $LUMI_GPUENERGY ]; then
  srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 $LUMI_GPUENERGY --save
fi

source $SCRIPT $*

if [ -x $LUMI_GPUENERGY ]; then
  srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 $LUMI_GPUENERGY --diff
fi

(set -x
date
)
