export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

SCRIPT="benchmarks/pytorch_visionmodel_accelerate.py"
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))

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
    TMPDATADIR="${LOCAL_SCRATCH}/ilsvrc2012-torch"
    SCRIPT_OPTS="--datadir $TMPDATADIR $SCRIPT_OPTS"
fi


if [ "$SLURM_NNODES" -gt 1 ]; then
    export MASTER_PORT=29400
    #MASTER_IP=$(ip -4 -brief addr show | grep -E 'hsn0|ib0' | grep -oP '([\d]+.[\d.]+)')
    MASTER_IP=$(hostname -i)
    DIST_OPTS="--main_process_ip=$MASTER_IP --main_process_port=$MASTER_PORT"
fi


NUM_PROCESSES=$(( $NUM_GPUS * $SLURM_NNODES ))

export ACCELERATE_CPU_AFFINITY=1

# Note: --machine_rank must be evaluated on each node, hence the LAUNCH_CMD setup
export LAUNCH_CMD="
       accelerate launch \
              --multi_gpu \
              --num_processes=$NUM_PROCESSES \
              --num_machines=$SLURM_NNODES \
              --machine_rank=\$SLURM_NODEID \
              --mixed_precision=no \
              --num_cpu_threads_per_process=$NUM_WORKERS \
              --dynamo_backend=no \
              $DIST_OPTS \
           $SCRIPT $SCRIPT_OPTS"

(set -x
 srun bash -c "$LAUNCH_CMD"
)

if [ -d "$TMPDATADIR" ]; then
    (set -x
     rm -rf $TMPDATADIR
    )
fi
