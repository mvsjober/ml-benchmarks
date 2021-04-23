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

- `cpu40`: Using all CPUs on Puhti (no GPU)
- `cpu128`: Using all CPUs on Mahti (no GPU)
- `gpu1`: 1 GPU run with 1/4 of the node's cores and memory resources
- `gpu4`: 4 GPUs and the whole node's resources
- `gpu4-hvd`: Single node with 4 GPUs using one MPI task per GPU (used with Horovod)
- `gpu8`, `gpu16`, `gpu24`: 8, 16 or 24 GPUs (i.e, 2, 4 or 6 nodes) with one MPI
  task per GPU (for Horovod)

For example to run a PyTorch script on a single GPU on Puhti:

```bash
module purge
module load pytorch/1.8
sbatch slurm/run-gpu1-puhti.sh python3 my_pytorch_script.py
```

The scripts also support extracting a tar-file to the NVME local drive by
setting the `$DATA_TAR` environment variable:

```bash
DATA_TAR=/scratch/dac/data/ilsvrc2012-tf.tar \
sbatch slurm/run-gpu8-puhti.sh my_pytorch_script.py --data_dir /run/nvme/*/data/
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
sbatch slurm/run-gpu1-mahti.sh pytorch_synthetic_benchmark.py --num-iters=100 --batch-size=64
```

Using Horovod:

```bash
sbatch slurm/run-gpu8-mahti.sh pytorch_synthetic_horovod_benchmark.py --num-iters=100 --batch-size=64
```

## TensorFlow CNN benchmark

Uses [`tf_cnn_benchmarks.py`][2] directly from TensorFlow's GitHub (as a git
submodule here).

[3]: tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

Run example:

```bash
sbatch slurm/run-gpu1-mahti.sh tf_cnn_benchmarks.py --model inception3 --num_warmup_batches 10 --num_gpus 1
```

Horovod with fp16:

```bash
sbatch slurm/run-gpu8-mahti.sh tf_cnn_benchmarks.py --use_fp16=true --model inception3 --variable_update horovod --horovod_device gpu --num_warmup_batches 10
```

With real data:

```bash
DATA_TAR=/scratch/dac/data/ilsvrc2012-tf.tar \
sbatch slurm/run-gpu1-mahti.sh python3 tf_cnn_benchmarks.py --use_fp16=true --model inception3 --num_warmup_batches 10 --data_name imagenet --data_dir /run/nvme/*/data/ilsvrc2012/
```

Horovod with real data:
```bash
DATA_TAR=/scratch/dac/data/ilsvrc2012-tf.tar \
sbatch slurm/run-gpu8-mahti.sh tf_cnn_benchmarks.py --use_fp16=true --model inception3 --variable_update horovod --horovod_device gpu --num_warmup_batches 10 --data_name imagenet --data_dir /run/nvme/*/data/ilsvrc2012/ 
```
