export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

SCRIPT="benchmarks/pytorch_visionmodel_lightning.py"
IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

SCRIPT_OPTS="--strategy=horovod"

if [ "$1" == "--data" ]; then
    shift
    (set -x
     srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
          tar xf $IMAGENET_DATA -C $LOCAL_SCRATCH
    )
    SCRIPT_OPTS="--datadir ${LOCAL_SCRATCH}/ilsvrc2012-torch $SCRIPT_OPTS"
fi

(set -x
 srun python3 $SCRIPT $SCRIPT_OPTS $*
)
