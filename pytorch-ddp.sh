export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

SCRIPT="benchmarks/pytorch_visionmodel_ddp.py"
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

DIST_OPTS="--standalone"
SCRIPT_OPTS=""

if [ "$LMOD_FAMILY_PYTHON_ML_ENV" != "pytorch" ]
then
    echo "WARNING: no pytorch module loaded, loading default module"
    module load pytorch
fi

if [ "$SLURM_NTASKS" -ne "$SLURM_NNODES" ]; then
    echo "ERROR: this script needs to be run as one task per node."
    echo "SLURM_NNODES = $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
    exit 1
fi

if [ "$1" == "--data" ]; then
    shift
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
 srun python3 -m torch.distributed.run $DIST_OPTS --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS $SCRIPT $SCRIPT_OPTS $*
)
