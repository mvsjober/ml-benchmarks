SCRIPT="benchmarks/tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py"
ARGS="--model inception3 --num_warmup_batches 10 --use_fp16=true"

if [ "$1" == "--data" ]; then
    DATASET_TAR_ARCHIVE=/scratch/dac/data/ilsvrc2012-tf.tar

    (set -x
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
         tar xf $DATASET_TAR_ARCHIVE -C $LOCAL_SCRATCH
    )
    ARGS="$ARGS --data_name imagenet --data_dir ${LOCAL_SCRATCH}/ilsvrc2012/"
fi

if [ "$SLURM_NTASKS" -gt 1 ]; then
    ARGS="$ARGS --variable_update horovod --horovod_device gpu"
fi

(set -x
srun python3 $SCRIPT $ARGS $*
)
