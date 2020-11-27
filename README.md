# Machine Learning benchmarks

Mainly for use with Puhti

## Setup

Clone this repository:

```bash
git clone --recursive https://gitlab.ci.csc.fi/msjoberg/ml-benchmarks.git
```

If you forget the `--recursive` flag you can always fetch the submodules manually: `git submodule init; git submodule update`.

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
| tensorflow-horovod-inception3-gpu4.sh          | nvidia-20.07-tf2-py3 | gputest   | 2020-11-23 | 1928.56   |
| tensorflow-horovod-inception3-gpu8.sh          |                      |           | 2020-11-23 | 3597.21   |
| tensorflow-horovod-inception3-imagenet-gpu4.sh |                      |           | 2020-11-23 | 1837.63   |
| tensorflow-horovod-inception3-imagenet-gpu8.sh |                      |           | 2020-11-23 | 3504.88   |
| tensorflow-horovod-inception3-imagenet-gpu8.sh |                      | gpu       | 2020-11-27 | 3419.12   |
  

## PyTorch ResNet

| script                                         | module               | partition | date       | result   |
| ---------------------------------------------- | -------------------- | --------- | ---------- | -------- |
| pytorch-resnet50-gpu1.sh                       | 1.6                  | gputest   | 2020-11-25 | 0.189    |
| pytorch-resnet50-gpu1-amp.sh                   |                      |           | 2020-11-25 | 0.105    |
| pytorch-resnet50-gpu4.sh                       |                      |           | 2020-11-25 | 0.202    |
| pytorch-resnet50-gpu4-amp.sh                   |                      |           | 2020-11-25 | 0.109    |
| pytorch-resnet50-gpu1.sh                       | nvidia-20.11-py3     | gputest   | 2020-11-26 | 0.186    |
| pytorch-resnet50-gpu1-amp.sh                   |                      |           | 2020-11-26 | 0.099    |
| pytorch-resnet50-gpu4.sh                       |                      |           | 2020-11-26 | 0.199    |
| pytorch-resnet50-gpu4-amp.sh                   |                      |           | 2020-11-26 | 0.105    |
| pytorch-resnet50-gpu1.sh                       | nvidia-20.08-py3     | gputest   | 2020-11-26 | 0.183    |
| pytorch-resnet50-gpu1-amp.sh                   |                      |           | 2020-11-26 | 0.096    |
| pytorch-resnet50-gpu4.sh                       |                      |           | 2020-11-26 | 0.196    |
| pytorch-resnet50-gpu4-amp.sh                   |                      |           | 2020-11-26 | 0.103    |
| pytorch-resnet50-gpu1.sh                       | nvidia-20.07-py3     | gputest   | 2020-11-27 | 0.181    |
| pytorch-resnet50-gpu1.sh                       | nvidia-20.03-py3     | gputest   | 2020-11-27 | 0.191    |
| pytorch-resnet50-gpu1.sh                       | nvidia-20.02-py3     | gputest   | 2020-11-27 | 0.190    |
| pytorch-resnet50-gpu1.sh                       | nvidia-19.11-py3     | gputest   | 2020-11-27 | 0.182    |

## PyTorch Horovod

| script                                         | module               | partition | date       | result        |
| ---------------------------------------------- | -------------------- | --------- | ---------- | --------      |
| pytorch-horovod-synthetic-gpu4.sh              | nvidia-20.11-py3     | gputest   | 2020-11-27 | 1032.5        |
| pytorch-horovod-synthetic-gpu8.sh              |                      |           | 2020-11-27 | **MPI error** |
| pytorch-horovod-imagenet-gpu4.sh               |                      |           | 2020-11-27 | 4.72it/s      |
| pytorch-horovod-imagenet-gpu8.sh               |                      |           | 2020-11-27 | **MPI error** |
| pytorch-horovod-synthetic-gpu4.sh              | nvidia-20.08-py3     | gputest   | 2020-11-26 | 1050.5        |
| pytorch-horovod-synthetic-gpu8.sh              |                      |           | 2020-11-26 | 2096.5        |
| pytorch-horovod-imagenet-gpu4.sh               |                      |           | 2020-11-27 | 5.05it/s      |
| pytorch-horovod-imagenet-gpu8.sh               |                      |           | 2020-11-27 | 5.05it/s      |
| pytorch-horovod-imagenet-gpu8.sh               |                      | gpu       | 2020-11-27 | 5.17it/s      |
