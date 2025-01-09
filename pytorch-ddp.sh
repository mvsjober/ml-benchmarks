export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

PYTHON3="python3"

if [ -n "$SIF" ]; then
    PYTHON3="singularity exec $SIF python3"
    #c=fe
    #MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"
    #SRUN_OPTIONS="--cpu-bind=mask_cpu:$MYMASKS"
fi

echo "PYTHON3=$PYTHON3"

SCRIPT="benchmarks/pytorch_visionmodel_ddp.py"
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

DIST_OPTS="--standalone --master_port 0"

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))
#NUM_WORKERS=0

SCRIPT_OPTS="--warmup-steps 10 --workers=$NUM_WORKERS"

if [ "$SLURM_NTASKS" -ne "$SLURM_NNODES" ]; then
    echo "ERROR: this script needs to be run as one task per node."
    echo "SLURM_NNODES = $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
    exit 1
fi

if [ "$1" == "--data" ]; then
    shift

    if [ ! -f $IMAGENET_DATA ]; then   # LUMI
        IMAGENET_DATA=/flash/project_462000007/mvsjober/ilsvrc2012-torch-resized-new.tar
    fi

    if [ -z "$LOCAL_SCRATCH" ]; then
        LOCAL_SCRATCH=/tmp
        #LOCAL_SCRATCH=/flash/project_462000007/mvsjober/tmp
    fi
    
    (set -x
     srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
          tar xf $IMAGENET_DATA -C $LOCAL_SCRATCH
    )
    SCRIPT_OPTS="--datadir ${LOCAL_SCRATCH}/ilsvrc2012-torch $SCRIPT_OPTS"
fi


if [ "$SLURM_NNODES" -gt 1 ]; then
    export RDZV_HOST=$(hostname)
    export RDZV_PORT=29400
    DIST_OPTS="--rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT"
fi

(set -x
 srun $PYTHON3 -m torch.distributed.run $DIST_OPTS --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS $SCRIPT $SCRIPT_OPTS $*
)

rm -rf $LOCAL_SCRATCH/ilsvrc2012-torch
