ARGS="--batch-size=64 --num-iters=100"

if [ "$SLURM_NTASKS" -eq 1 ]; then
    SCRIPT="benchmarks/pytorch_synthetic_benchmark.py"
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        ARGS="$ARGS --multi-gpu"
    fi
else
    if [ $(( $NUM_GPUS * $SLURM_NNODES )) -ne $SLURM_NTASKS ]; then
        echo "ERROR: this script needs to be run as one task per GPU. Try using slurm/*-mpi.sh scripts."
        echo "NUM_GPUS * SLURM_NNODES = $NUM_GPUS * $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
        exit 1
    fi
    SCRIPT="benchmarks/pytorch_synthetic_horovod_benchmark.py"
fi

echo "SING_IMAGE=$SING_IMAGE"
(set -x
srun python3 $SCRIPT $ARGS $*
)
