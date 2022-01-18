IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH

srun python3 pytorch_imagenet.py -a resnet50 -p 10 -b 256 -j 40 --epoch 2 \
     --dist-url 'tcp://127.0.0.1:8842' --dist-backend 'nccl' \
     --multiprocessing-distributed \
     --world-size 1 --rank 0 \
     ${LOCAL_SCRATCH}/ilsvrc2012-torch

