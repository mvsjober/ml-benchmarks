srun python3 tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --use_fp16=true --model inception3 --variable_update horovod --horovod_device gpu --num_warmup_batches 10 $*
