export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

SCRIPT="benchmarks/pytorch_visionmodel_lightning.py"
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

SCRIPT_OPTS="--strategy=ddp"

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

    (set -x
     srun python3 -m torch.distributed.run \
          --nnodes=$SLURM_NNODES \
          --nproc_per_node=$NUM_GPUS \
          --rdzv_id=$SLURM_JOB_ID \
          --rdzv_backend=c10d \
          --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
          $SCRIPT --gpus=$NUM_GPUS --nodes=$SLURM_NNODES \
          $SCRIPT_OPTS $*
    )
else
    (set -x
     srun python3 $SCRIPT --gpus=$NUM_GPUS --nodes=1 $SCRIPT_OPTS $*
    )
fi

