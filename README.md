# Machine learning benchmarks

Collection of various machine learning benchmarks together with Slurm scripts
for CSC's supercomputers.

The benchmarks themselves (Python code) can be found in the `benchmarks`
directory. Main run scripts are in the root directory as `*.sh` files. The Slurm
settings have been separated into their own scripts in the `slurm` directory.

Typical usage would be to first select a benchmark (e.g., PyTorch synthetic) and
then appropriate Slurm settings (e.g., Mahti with 4 GPUs on Mahti, single node,
no MPI). The command would then be:

```bash
sbatch slurm/mahti-gpu4.sh pytorch-synthetic.sh
```

## Available run scripts

Slurm run scripts can be found in the `slurm` directory, these are named as
`[puhti|mahti]-[cpu|gpu]N.sh` where `N` is the number of CPUs or GPUs reserved.

Scripts are all single-node, single MPI task unless it ends with `-mpi.sh`.
Tasks with the `-mpi.sh` ending launch a separate MPI task for each GPU,
assuming 4 GPUs per node. For example `mahti-gpu8-mpi.sh` reserves two nodes,
with 4 GPUs (and thus 4 MPI tasks) per node, giving a total of 8 GPUs (and 8 MPI
tasks).


## Available benchmarks

| Benchmark         | Script name                        | Data      | Multi-GPU | Horovod |
| ---------         | -----------                        | ----      | --------- | ---     |
| PyTorch synthetic | `pytorch-synthetic.sh`             | synthetic | X         | X       |
| PyTorch ImageNet  | `pytorch-imagenet.sh`              | ImageNet  | X         | -       |
| PyTorch Horovod   | `pytorch-imagenet-hvd.sh`          | ImageNet  | X         | X       |
| TensorFlow CNN    | `tensorflow-cnn.sh`                | synthetic | X         | -       |
|                   | `tensorflow-cnn-hvd.sh`            | synthetic | X         | X       |
|                   | `tensorflow-cnn-data.sh`           | ImageNet  | X         | -       |
|                   | `tensorflow-cnn-data-hvd.sh`       | ImageNet  | X         | X       |

An "X" in the Multi-GPU column in the table above means the script supports
multiple GPUs. An "X" in the MPI column this means the script support using MPI
(Horovod).

The different benchmarks are described below in more detail. 


### PyTorch synthetic

Originally based on [Horovod's example script with the same name][1]. Note that
the original script used a single fixed random batch which was feed to the
network again and again. Some systems and setups are able to optimize this
scenario giving very unrealistic results. We have modified the script to
generate a new random batch each time.

Runs with "resnet50" model by default, but also supports "inception_v3" and
other [models from torchvision.models][2].

[1]: https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_synthetic_benchmark.py
[2]: https://pytorch.org/vision/stable/models.html

Run example with single GPU:

```bash
sbatch slurm/mahti-gpu1.sh pytorch-synthetic.sh
```

Run example with 4 GPUs. Note that you can also add arguments to be given to
the Python script:

```bash
sbatch slurm/mahti-gpu4.sh pytorch-synthetic.sh --batch-size=32
```

Using 8 GPUs (i.e., 2 nodes) with Horovod and MPI:

```bash
sbatch slurm/mahti-gpu8-mpi.sh pytorch-synthetic.sh
```

## PyTorch ImageNet benchmark

Run example:

```
sbatch slurm/mahti-gpu1.sh pytorch-imagenet.sh
```

Run example with Multi-GPU and AMP:

```bash
sbatch slurm/mahti-gpu4.sh pytorch-imagenet.sh --amp
```

## PyTorch ResNet50 Horovod benchmark

Based on [Horovod's example script][3].

[3]: https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py

Run example:

```bash
sbatch slurm/mahti-gpu8-mpi.sh pytorch-imagenet-hvd.sh
```


## TensorFlow CNN benchmark

Uses [`tf_cnn_benchmarks.py`][4] directly from TensorFlow's GitHub (as a git
submodule here).

[4]: tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

Run example:

```bash
sbatch slurm/mahti-gpu1.sh tensorflow-cnn.sh
```

Horovod with fp16:

```bash
sbatch slurm/mahti-gpu8.sh tensorflow-cnn-hvd.sh
```

With real data:

```bash
sbatch slurm/mahti-gpu1.sh tensorflow-cnn-data.sh
```

Horovod with real data:
```bash
sbatch slurm/mahti-gpu8-mpi.sh tensorflow-cnn-hvd-data.sh
```
