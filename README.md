# Machine learning benchmarks

Mainly for use with CSC's supercomputers.

## Setup

Clone this repository:

```bash
git clone --recursive https://gitlab.ci.csc.fi/msjoberg/ml-benchmarks.git
```

If you forget the `--recursive` flag you can always fetch the submodules
manually: `git submodule init; git submodule update`.

## Slurm job scripts

The job scripts have been split into two parts. The [`slurm`](slurm) directory
contains scripts with Slurm instructions for Puhti and Mahti for different
settings, while the [`scripts`](scripts) directory contains run scripts for
different benchmarks and configurations.

Slurm scripts are named `run-TYPE-SYSTEM.sh` where SYSTEM is `puhti` or `mahti`
and TYPE one of:

- `cpu40`: Using all CPUs on Puhti (no GPU)
- `cpu128`: Using all CPUs on Mahti (no GPU)
- `gpu1`: 1 GPU run with 1/4 of the node's cores and memory resources
- `gpu4`: 4 GPUs and the whole node's resources
- `gpu4-hvd`: Single node with 4 GPUs using one MPI task per GPU (used with Horovod)
- `gpu8`, `gpu16`, `gpu24`: 8, 16 or 24 GPUs (i.e, 2, 4 or 6 nodes) with one MPI
  task per GPU (for Horovod)

A batch job is then constructed by first selecting the appropriate Slurm script
(e.g., 4 GPUs on Mahti), and then the appropriate benchmark script (e.g. PyTorch
Horovod ImageNet benchmark):

```bash
sbatch slurm/run-gpu4-mahti.sh scripts/pytorch-horovod-imagenet.sh
```

## PyTorch synthetic benchmark

Uses [`pytorch_synthetic_benchmark.py`](pytorch_synthetic_benchmark.py),
originally based on [Horovod's example script with the same name][1]. Note that
the original script used a single fixed random batch which was feed to the
network again and again. Some systems and setups are able to optimize this
scenario giving very unrealistic results. We have modified the script to
generate a new random batch each time.

Runs with "resnet50" model by default, but also supports "inception_v3" and
other [models from torchvision.models][2].

[1]: https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_synthetic_benchmark.py
[2]: https://pytorch.org/vision/stable/models.html

Run example:

```bash
sbatch slurm/run-gpu1-mahti.sh scripts/pytorch-synthetic-benchmark.sh
```

Using Horovod:

```bash
sbatch slurm/run-gpu8-mahti.sh pytorch-synthetic-benchmark-hvd.sh
```

## PyTorch ImageNet benchmark

Uses [`pytorch_imagenet.py`](pytorch_imagenet.py) and
[`pytorch_imagenet_amp.py`](pytorch_imagenet_amp.py) for mixed precision.

Run example:

```
sbatch slurm/run-gpu1-mahti.sh scripts/pytorch-imagenet.sh
```

Run example with Multi-GPU and AMP:

```bash
sbatch slurm/run-gpu4-mahti.sh scripts/pytorch-imagenet-amp-multigpu.sh
```

## PyTorch ResNet50 Horovod benchmark

Uses [`pytorch_imagenet_resnet50_horovod_benchmark.py`](pytorch_imagenet_resnet50_horovod_benchmark.py),
based on [Horovod's example script][3].

[3]: https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py

Run example:

```bash
sbatch slurm/run-gpu8-mahti.sh scripts/pytorch-horovod-imagenet.sh
```


## TensorFlow CNN benchmark

Uses [`tf_cnn_benchmarks.py`][4] directly from TensorFlow's GitHub (as a git
submodule here).

[4]: tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

Run example:

```bash
sbatch slurm/run-gpu1-mahti.sh scripts/tensorflow-cnn-benchmarks.sh
```

Horovod with fp16:

```bash
sbatch slurm/run-gpu8-mahti.sh scripts/tensorflow-cnn-benchmarks-hvd.sh
```

With real data:

```bash
sbatch slurm/run-gpu1-mahti.sh scripts/tensorflow-cnn-benchmarks-data.sh
```

Horovod with real data:
```bash
sbatch slurm/run-gpu1-mahti.sh scripts/tensorflow-cnn-benchmarks-hvd-data.sh
```
