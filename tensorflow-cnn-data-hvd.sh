DATASET_TAR_ARCHIVE=/scratch/dac/data/ilsvrc2012-tf.tar

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     tar xf $DATASET_TAR_ARCHIVE -C $LOCAL_SCRATCH

srun python3 tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --use_fp16=true --model inception3 --variable_update horovod --horovod_device gpu --num_warmup_batches 10 --data_name imagenet --data_dir ${LOCAL_SCRATCH}/ilsvrc2012/ $*
