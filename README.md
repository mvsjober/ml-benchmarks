# Machine Learning benchmarks

Mainly for use with Puhti

## TensorFlow Horovod

Typical results:

| script                                         | module               | partition | date       | result  |
| ---------------------------------------------- | -------------------- | --------- | ---------- |---------|
| tensorflow-horovod-inception3-gpu4.sh          | nvidia-20.07-tf2-py3 | gputest   | 2020-11-23 | 1928.56 |
| tensorflow-horovod-inception3-gpu8.sh          | -"-                  | -"-       | 2020-11-23 | 3597.21 |
| tensorflow-horovod-inception3-imagenet-gpu4.sh | -"-                  | -"-       | 2020-11-23 | 1837.63 |
| tensorflow-horovod-inception3-imagenet-gpu8.sh | -"-                  | -"-       | 2020-11-23 | 3504.88 |
