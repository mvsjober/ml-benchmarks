if [ "$SLURM_NTASKS" -gt 1 ]; then
    echo "ERROR: this script does not support MPI tasks"
    exit 1
fi

SCRIPT="benchmarks/pytorch_imagenet.py"

if [ "$1" == "--amp" ]; then
    shift
    SCRIPT="benchmarks/pytorch_imagenet_amp.py"
fi

IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

(set -x
tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH
)

if [ "$NUM_GPUS" -gt 1 ]; then
    BATCH_SIZE=$(( NUM_GPUS * 32 ))
    NUM_CPUS=$(( NUM_GPUS * 10 ))
    (set -x
     srun python3 $SCRIPT -a resnet50 -p 10 -b $BATCH_SIZE -j $NUM_CPUS --epoch 2 \
          --dist-url 'tcp://127.0.0.1:8842' --dist-backend 'nccl' \
          --multiprocessing-distributed \
          --world-size 1 --rank 0 \
          ${LOCAL_SCRATCH}/ilsvrc2012-torch
    )
else
    (set -x
     srun python3 $SCRIPT -a resnet50 -p 10 -b 32 -j 10 --epoch 2 \
          ${LOCAL_SCRATCH}/ilsvrc2012-torch $*
     )
fi
