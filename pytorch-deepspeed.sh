export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

SCRIPT="benchmarks/pytorch_visionmodel_deepspeed.py"
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

SCRIPT_OPTS="--deepspeed --deepspeed_config benchmarks/ds_config_benchmark.json"

if [ "$LMOD_FAMILY_PYTHON_ML_ENV" != "pytorch" ]
then
    echo "WARNING: no pytorch module loaded, loading default module"
    module load pytorch
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
    if [ $(( $NUM_GPUS * $SLURM_NNODES )) -ne $SLURM_NTASKS ]; then
        echo "ERROR: this script needs to be run as one task per GPU. Try using slurm/*-mpi.sh scripts."
        echo "NUM_GPUS * SLURM_NNODES = $NUM_GPUS * $SLURM_NNODES != SLURM_NTASKS = $SLURM_NTASKS"
        exit 1
    fi
    
    (set -x
     srun python3 $SCRIPT $SCRIPT_OPTS $*
    )
else
    if [ $SLURM_NTASKS -ne 1 ]; then
        echo "ERROR: single node runs need to be run as a single task"
        echo "SLURM_NTASKS = $SLURM_NTASKS != 1"
        exit 1
    fi
    (set -x
     srun singularity_wrapper exec deepspeed $SCRIPT $SCRIPT_OPTS $*
    )
fi

