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
| DDP, synthetic            | PyTorch 1.11.0+cu113 | puhti   | 1    | 2022-05-18 | 325.95  |
| DDP, synthetic            | PyTorch 1.11.0+cu113 | puhti   | 4    | 2022-05-18 | 1217.10 |
| DDP, synthetic            | PyTorch 1.11.0+cu113 | puhti   | 8    | 2022-05-18 | 2420.71 |
| DDP, Imagenet data        | PyTorch 1.11.0+cu113 | puhti   | 1    | 2022-05-18 | 328.21  |
| DDP, Imagenet data        | PyTorch 1.11.0+cu113 | puhti   | 4    | 2022-05-18 | 1217.30 |
| DDP, Imagenet data        | PyTorch 1.11.0+cu113 | puhti   | 8    | 2022-05-18 | 2431.23 |
| DeepSpeed, synthetic data | PyTorch 1.11.0+cu113 | puhti   | 4    | 2022-05-18 | 1255.24 |
| DeepSpeed, synthetic data | PyTorch 1.11.0+cu113 | puhti   | 8    | 2022-05-18 | 2420.89 |
| Horovod, synthetic        | PyTorch 1.11.0+cu113 | puhti   | 8    | 2022-05-18 | 2220.04 |
| Horovod, Imagenet data    | PyTorch 1.11.0+cu113 | puhti   | 8    | 2022-05-18 | ERROR   |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-05-08 | 333.44 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-05-08 | 1251.13 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-05-08 | 2289.89 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-05-08 | 332.51 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-05-08 | 1251.10 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-05-08 | 2304.76 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-05-08 | 1131.45 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-05-08 | 2141.84 |
| Horovod, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-05-08 | 1973.30 |
| Horovod, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-05-08 | 2189.90 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-05-23 | 796.17 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-05-23 | 3132.06 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-05-23 | 5950.39 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-05-23 | 793.94 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-05-23 | 3127.64 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-05-23 | 5954.15 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-05-23 | 3101.13 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-05-23 | 5782.77 |
| Horovod, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-05-23 | 5248.05 |
| Horovod, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-05-23 | 5214.18 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-06-29 | 507.38 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-06-29 | 3896.69 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-06-29 | 7641.90 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-06-29 | 509.67 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-06-29 | 3770.25 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-06-29 | 7214.12 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-09-15 | 511.34 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-09-15 | 3844.09 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-09-15 | 7599.91 |
| DDP, synthetic, fp16 | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-09-15 | 868.81 |
| DDP, synthetic, fp16 | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-09-15 | 6411.18 |
| DDP, synthetic, fp16 | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-09-15 | 12468.99 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-09-15 | 511.02 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-09-15 | 3734.80 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-09-15 | 7201.45 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-15 | 781.72 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-15 | 3054.03 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-15 | 5824.31 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-15 | 1128.61 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-15 | 4153.99 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-15 | 7849.03 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-15 | 798.08 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-15 | 3135.00 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-15 | 6010.30 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-15 | 783.28 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-15 | 3067.25 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-15 | 5777.12 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-15 | 3107.95 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-15 | 5813.62 |
| Horovod, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-15 | 5235.30 |
| Horovod, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-15 | 5230.77 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-16 | 331.39 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-16 | 1245.59 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-16 | 2473.86 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-16 | 674.17 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-16 | 2389.34 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-16 | 4644.40 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-16 | 331.98 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-16 | 1254.01 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-16 | 2488.49 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-16 | 329.76 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-16 | 1244.49 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-16 | 2470.56 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-16 | 1262.18 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-16 | 2429.24 |
| Horovod, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-16 | 2314.87 |
| Horovod, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-16 | 2313.93 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-09-28 | 503.67 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 2    | 2023-09-28 | 981.73 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-09-28 | 3892.65 |
| DDP, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-09-28 | 7556.66 |
| DDP, synthetic, fp16 | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-09-28 | 870.53 |
| DDP, synthetic, fp16 | PyTorch 2.0.1+rocm5.4.2 | lumi    | 2    | 2023-09-28 | 1630.93 |
| DDP, synthetic, fp16 | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-09-28 | 6421.78 |
| DDP, synthetic, fp16 | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-09-28 | 12388.28 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-09-28 | 510.99 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-09-28 | 3780.29 |
| DDP Lightning, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-09-28 | 7236.21 |
| run_clm, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 1    | 2023-09-28 | 22.5 |
| run_clm, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 2    | 2023-09-28 | 43.37 |
| run_clm, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 8    | 2023-09-28 | 145.513 |
| run_clm, synthetic | PyTorch 2.0.1+rocm5.4.2 | lumi    | 16    | 2023-09-28 | 231.133 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-28 | 329.27 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 2    | 2023-09-28 | 520.80 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-28 | 1249.31 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-28 | 2465.86 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-28 | 670.50 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | puhti    | 2    | 2023-09-28 | 1186.25 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-28 | 2392.44 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-28 | 4655.31 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-28 | 330.92 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-28 | 1252.09 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-28 | 2487.39 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-28 | 329.29 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-28 | 1244.50 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-28 | 2468.76 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-28 | 1263.66 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-28 | 2432.60 |
| Horovod, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-28 | 2310.85 |
| Horovod, Imagenet data | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-28 | 2310.54 |
| run_clm, synthetic | PyTorch 2.0.0+cu117 | puhti    | 1    | 2023-09-28 | 17.47 |
| run_clm, synthetic | PyTorch 2.0.0+cu117 | puhti    | 2    | 2023-09-28 | 33.563 |
| run_clm, synthetic | PyTorch 2.0.0+cu117 | puhti    | 4    | 2023-09-28 | 60.755 |
| run_clm, synthetic | PyTorch 2.0.0+cu117 | puhti    | 8    | 2023-09-28 | 105.145 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-28 | 784.38 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 2    | 2023-09-28 | 1495.75 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-28 | 3081.61 |
| DDP, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-28 | 5862.00 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-28 | 1123.50 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | mahti    | 2    | 2023-09-28 | 2125.94 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-28 | 4087.08 |
| DDP, synthetic, fp16 | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-28 | 7851.35 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-28 | 791.49 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-28 | 3137.78 |
| DDP Lightning, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-28 | 5987.94 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-28 | 781.59 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-28 | 3064.19 |
| DDP, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-28 | 5819.76 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-28 | 3099.41 |
| DeepSpeed, synthetic data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-28 | 5782.16 |
| Horovod, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-28 | 5252.11 |
| Horovod, Imagenet data | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-28 | 5254.03 |
| run_clm, synthetic | PyTorch 2.0.0+cu117 | mahti    | 1    | 2023-09-28 | 29.493 |
| run_clm, synthetic | PyTorch 2.0.0+cu117 | mahti    | 2    | 2023-09-28 | 57.184 |
| run_clm, synthetic | PyTorch 2.0.0+cu117 | mahti    | 4    | 2023-09-28 | 92.499 |
| run_clm, synthetic | PyTorch 2.0.0+cu117 | mahti    | 8    | 2023-09-28 | 139.883 |
| DDP, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 1    | 2023-11-17 | 542.91 |
| DDP, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 2    | 2023-11-17 | 1020.47 |
| DDP, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 8    | 2023-11-17 | 4034.92 |
| DDP, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 16    | 2023-11-17 | 7655.74 |
| DDP, synthetic, fp16 | PyTorch 2.1.1+rocm5.6 | lumi    | 1    | 2023-11-17 | 902.57 |
| DDP, synthetic, fp16 | PyTorch 2.1.1+rocm5.6 | lumi    | 2    | 2023-11-17 | 1694.43 |
| DDP, synthetic, fp16 | PyTorch 2.1.1+rocm5.6 | lumi    | 8    | 2023-11-17 | 6440.36 |
| DDP, synthetic, fp16 | PyTorch 2.1.1+rocm5.6 | lumi    | 16    | 2023-11-17 | 12066.58 |
| DDP Lightning, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 1    | 2023-11-17 | 522.54 |
| DDP Lightning, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 8    | 2023-11-17 | 3788.93 |
| DDP Lightning, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 16    | 2023-11-17 | 7139.44 |
| DDP, Imagenet data | PyTorch 2.1.1+rocm5.6 | lumi    | 1    | 2023-11-17 | 535.99 |
| DDP, Imagenet data | PyTorch 2.1.1+rocm5.6 | lumi    | 8    | 2023-11-17 | 3975.65 |
| DDP, Imagenet data | PyTorch 2.1.1+rocm5.6 | lumi    | 16    | 2023-11-17 | 7617.46 |
| DeepSpeed, synthetic data | PyTorch 2.1.1+rocm5.6 | lumi    | 8    | 2023-11-17 | 3638.44 |
| run_clm, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 1    | 2023-11-17 | 21.671 |
| run_clm, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 2    | 2023-11-17 | 41.527 |
| run_clm, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 8    | 2023-11-17 | 116.601 |
| run_clm, synthetic | PyTorch 2.1.1+rocm5.6 | lumi    | 16    | 2023-11-17 | 162.697 |
