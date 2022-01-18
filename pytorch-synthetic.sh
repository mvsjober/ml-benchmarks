ARGS="--batch-size=64 --num-iters=100"

if [ "$SLURM_NTASKS" -eq 1 ]; then
    SCRIPT="benchmarks/pytorch_synthetic_benchmark.py"
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        ARGS="$ARGS --multi-gpu"
    fi
else
    SCRIPT="benchmarks/pytorch_synthetic_horovod_benchmark.py"
fi

(set -x
srun python3 $SCRIPT $ARGS $*
)
