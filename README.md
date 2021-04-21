# Machine learning benchmarks

Mainly for use with CSC's supercomputers.

## Setup

Clone this repository:

```bash
git clone --recursive https://gitlab.ci.csc.fi/msjoberg/ml-benchmarks.git
```

If you forget the `--recursive` flag you can always fetch the submodules
manually: `git submodule init; git submodule update`.

## Slurm scripts

The [`slurm`](slurm) directory contains Slurm batch scripts for Puhti and Mahti.

Scripts are named `run-TYPE-SYSTEM.sh` where SYSTEM is `puhti` or `mahti` and
TYPE one of:

- `gpu1`: 1 GPU run with 1/4 of the node's cores and memory resources
- `gpu4`: 4 GPUs and the whole node's resources

For example to run one of the commands below on Puhti:

```bash
sbatch run-gpu1-puhti.sh python3 pytorch_synthetic_benchmark.py --num-iters=100 --batch-size=64
```

## PyTorch synthetic benchmark

Uses [`pytorch_synthetic_benchmark.py`](pytorch_synthetic_benchmark.py),
originally based on [Horovod's example script with the same name][1]. NOTE: the
original script used a single fixed random batch which was feed to the network
again and again. Some systems and setups are able to optimize this scenario
giving very unrealistic results. We have modified the script to generate a new
random batch each time.

[1]: https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_synthetic_benchmark.py

Run example:

```bash
python3 pytorch_synthetic_benchmark.py --num-iters=100 --batch-size=64
```



Scripts:

`pytorch-synthetic-gpu1.sh` - 1 GPU, Puhti
`pytorch-synthetic-gpu1-mahti.sh` - 1 GPU, Mahti
`pytorch-synthetic-cpu40.sh` - 40 CPU cores (whole node), Puhti
`pytorch-synthetic-cpu128-mahti.sh` - 128 CPU cores (whole node), Mahti

## PyTorch data

Uses [`pytorch-benchmarks/main.py`](pytorch-benchmarks/main.py) and
[`pytorch-benchmarks/main_amp.py`](pytorch-benchmarks/main_amp.py) for AMP.

`pytorch-resnet50-gpu1.sh`
`pytorch-resnet50-gpu4.sh`

`pytorch-resnet50-gpu1-amp-mahti.sh`
`pytorch-resnet50-gpu1-amp.sh`
`pytorch-resnet50-gpu1-mahti.sh`
`pytorch-resnet50-gpu1-sqfs.sh`
`pytorch-resnet50-gpu4-amp-mahti.sh`
`pytorch-resnet50-gpu4-amp.sh`

## PyTorch Horovod synthetic

`pytorch-horovod-synthetic-gpu16-mahti.sh`
`pytorch-horovod-synthetic-gpu4-mahti.sh`
`pytorch-horovod-synthetic-gpu4.sh`
`pytorch-horovod-synthetic-gpu64-mahti.sh`
`pytorch-horovod-synthetic-gpu8-mahti.sh`
`pytorch-horovod-synthetic-gpu8.sh`
`pytorch-horovod-synthetic-gpu96-mahti.sh`
`pytorch-horovod-synthetic-mahti-node2.sh`

## PyTorch Horovod data

`pytorch-horovod-imagenet-gpu4-mahti.sh`
`pytorch-horovod-imagenet-gpu4.sh`
`pytorch-horovod-imagenet-gpu64-mahti.sh`
`pytorch-horovod-imagenet-gpu8-mahti.sh`
`pytorch-horovod-imagenet-gpu8.sh`
`pytorch-horovod-imagenet-gpu96-mahti.sh`

## TensorFlow

`tensorflow-inception3-gpu1-mahti.sh`

`tensorflow-horovod-inception3-gpu4-mahti.sh`
`tensorflow-horovod-inception3-gpu4.sh`
`tensorflow-horovod-inception3-gpu8-mahti.sh`
`tensorflow-horovod-inception3-gpu8.sh`
`tensorflow-horovod-inception3-imagenet-gpu4-mahti.sh`
`tensorflow-horovod-inception3-imagenet-gpu4.sh`
`tensorflow-horovod-inception3-imagenet-gpu8-mahti.sh`
`tensorflow-horovod-inception3-imagenet-gpu8.sh`
