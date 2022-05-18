| Benchmark                 | Framework            | Cluster | GPUs | Date       | Img/sec |
|---------------------------|----------------------|---------|------|------------|---------|
| DDP, synthetic            | PyTorch 1.11.0+cu115 | mahti   | 1    | 2022-05-18 | 758.90  |
| DDP, synthetic            | PyTorch 1.11.0+cu115 | mahti   | 4    | 2022-05-18 | 2836.54 |
| DDP, synthetic            | PyTorch 1.11.0+cu115 | mahti   | 8    | 2022-05-18 | 5502.49 |
| DDP, Imagenet data        | PyTorch 1.11.0+cu115 | mahti   | 1    | 2022-05-18 | 759.19  |
| DDP, Imagenet data        | PyTorch 1.11.0+cu115 | mahti   | 4    | 2022-05-18 | 2832.43 |
| DDP, Imagenet data        | PyTorch 1.11.0+cu115 | mahti   | 8    | 2022-05-18 | 5492.90 |
| DeepSpeed, synthetic data | PyTorch 1.11.0+cu115 | mahti   | 4    | 2022-05-18 | 2859.08 |
| DeepSpeed, synthetic data | PyTorch 1.11.0+cu115 | mahti   | 8    | 2022-05-18 | 5484.88 |
| Horovod, synthetic        | PyTorch 1.11.0+cu115 | mahti   | 8    | 2022-05-18 | 4967.57 |
| Horovod, Imagenet data    | PyTorch 1.11.0+cu115 | mahti   | 8    | 2022-05-18 | 4984.15 |
