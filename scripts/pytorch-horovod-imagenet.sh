DATASET_TAR_ARCHIVE=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     tar xf $DATASET_TAR_ARCHIVE --strip 1 -C $LOCAL_SCRATCH/

srun python3 pytorch_imagenet_resnet50_horovod_benchmark.py --fp16-allreduce \
     --train-dir=${LOCAL_SCRATCH}/train \
     --val-dir=${LOCAL_SCRATCH}/val \
     --epochs=1 --batches-per-epoch 10000 $*
