# Machine Learning benchmarks

Mainly for use with Puhti

## TensorFlow Horovod

| script                                         | module               | partition | date       | result  |
| ---------------------------------------------- | -------------------- | --------- | ---------- |---------|
| tensorflow-horovod-inception3-gpu4.sh          | nvidia-20.07-tf2-py3 | gputest   | 2020-11-23 | 1928.56 |
| tensorflow-horovod-inception3-gpu8.sh          | -"-                  | -"-       | 2020-11-23 | 3597.21 |
| tensorflow-horovod-inception3-imagenet-gpu4.sh | -"-                  | -"-       | 2020-11-23 | 1837.63 |
| tensorflow-horovod-inception3-imagenet-gpu8.sh | -"-                  | -"-       | 2020-11-23 | 3504.88 |


## PyTorch ResNet

| script                                         | module               | partition | date       | result |
| ---------------------------------------------- | -------------------- | --------- | ---------- |--------|
| pytorch-resnet50-gpu1.sh                       | pytorch/1.6          |           | 2020-11-25 | 0.189  |
| pytorch-resnet50-gpu1-amp.sh                   |                      |           | 2020-11-25 | 0.105  |
| pytorch-resnet50-gpu4.sh                       |                      |           | 2020-11-25 | 0.202  |
| pytorch-resnet50-gpu4-amp.sh                   |                      |           | 2020-11-25 | 0.109  |
