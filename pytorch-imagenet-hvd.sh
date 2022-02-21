DATASET_TAR_ARCHIVE=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar
export OMP_NUM_THREADS=1

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     tar xf $DATASET_TAR_ARCHIVE --strip 1 -C $LOCAL_SCRATCH/

srun python3 benchmarks/pytorch_imagenet_resnet50_horovod_benchmark.py --fp16-allreduce \
     --train-dir=${LOCAL_SCRATCH}/train \
     --val-dir=${LOCAL_SCRATCH}/val \
     --epochs=1 $*

#     --epochs=1 --batches-per-epoch 10000 $*
