export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

env | grep NCCL
env | grep MIOPEN

SCRIPT="benchmarks/tensorflow_visionmodel_distributed.py"

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))
if (( NUM_WORKERS > 10 )); then
	NUM_WORKERS=10
fi

SCRIPT_OPTS="--warmup-steps 10 --workers=$NUM_WORKERS"

if [ "$SLURM_NTASKS" -ne "$SLURM_NNODES" ]; then
    echo "ERROR: this script needs to be run as one task per node."
    echo "SLURM_NNODES = $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
    exit 1
fi

(set -x
$PYTHON3 $SCRIPT $SCRIPT_OPTS
)
