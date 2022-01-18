IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH

srun python3 pytorch_imagenet_amp.py -a resnet50 -p 10 -b 32 -j 10 --epoch 2 ${LOCAL_SCRATCH}/ilsvrc2012-torch $*
