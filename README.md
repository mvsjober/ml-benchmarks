# Machine Learning benchmarks

Mainly for use with Puhti

## Usage

```bash
module purge
module load <module name>
sbatch <script name>
```

## TensorFlow Horovod

| script                                         | module               | partition | date       | result    |
| ---------------------------------------------- | -------------------- | --------- | ---------- | --------- |
| tensorflow-horovod-inception3-gpu4.sh          | 2.2-hvd              | gputest   | 2020-11-26 | 1761.86   |
| tensorflow-horovod-inception3-gpu8.sh          |                      |           | 2020-11-26 | 3381.94   |
| tensorflow-horovod-inception3-imagenet-gpu4.sh |                      |           | 2020-11-26 | 1695.27   |
| tensorflow-horovod-inception3-imagenet-gpu8.sh |                      |           | 2020-11-26 | 3214.84   |
| ---------------------------------------------- | -------------------- | --------- | ---------- | --------- |
| tensorflow-horovod-inception3-gpu4.sh          | nvidia-20.07-tf2-py3 | gputest   | 2020-11-23 | 1928.56   |
| tensorflow-horovod-inception3-gpu8.sh          |                      |           | 2020-11-23 | 3597.21   |
| tensorflow-horovod-inception3-imagenet-gpu4.sh |                      |           | 2020-11-23 | 1837.63   |
| tensorflow-horovod-inception3-imagenet-gpu8.sh |                      |           | 2020-11-23 | 3504.88   |
  

## PyTorch ResNet

| script                                         | module               | partition | date       | result   |
| ---------------------------------------------- | -------------------- | --------- | ---------- | -------- |
| pytorch-resnet50-gpu1.sh                       | 1.6                  | gputest   | 2020-11-25 | 0.189    |
| pytorch-resnet50-gpu1-amp.sh                   |                      |           | 2020-11-25 | 0.105    |
| pytorch-resnet50-gpu4.sh                       |                      |           | 2020-11-25 | 0.202    |
| pytorch-resnet50-gpu4-amp.sh                   |                      |           | 2020-11-25 | 0.109    |
| ---------------------------------------------- | -------------------- | --------- | ---------- | -------- |
| pytorch-resnet50-gpu1.sh                       | nvidia-20.08-py3     | gputest   |            | ERROR    |
| pytorch-resnet50-gpu1-amp.sh                   |                      |           |            |          |
| pytorch-resnet50-gpu4.sh                       |                      |           |            |          |
| pytorch-resnet50-gpu4-amp.sh                   |                      |           |            |          |


## PyTorch Horovod

| script                                         | module               | partition | date       | result   |
| ---------------------------------------------- | -------------------- | --------- | ---------- | -------- |
| pytorch-horovod-synthetic-gpu4.sh              | nvidia-20.08-py3     | gputest   | 2020-11-25 | ERROR    |
| pytorch-horovod-synthetic-gpu8.sh              |                      |           |            |          |
| pytorch-horovod-imagenet-gpu4.sh               |                      |           |            |          |
| pytorch-horovod-imagenet-gpu8.sh               |                      |           |            |          |
