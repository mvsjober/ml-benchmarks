import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils import mkldnn as mkldnn_utils
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import timeit
import numpy as np


class RandomData(Dataset):
    def __init__(self, imsize, batch_size, batches_per_iter):
        self.imsize = imsize
        self.batch_size = batch_size
        self.batches_per_iter = batches_per_iter

    def __len__(self):
        return self.batch_size * self.batches_per_iter

    def __getitem__(self, index):
        data = torch.randn(3, self.imsize, self.imsize)
        target = torch.randint(0, 1000, ())
        return data, target


def log(s, nl=True):
    print(s, end='\n' if nl else '', flush=True)


def main(args):
    if args.ipex:
        import intel_pytorch_extension as ipex
        if args.fp16:
            ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.ipex:
        device = ipex.DEVICE
    else:
        device = torch.device('cpu')

    print('Using PyTorch version:', torch.__version__, 'Device:', device)
    print(torch.__config__.show())

    cudnn.benchmark = True

    # Set up standard model.
    log('Initializing %s model...' % args.model)
    model = getattr(models, args.model)()
    model = model.to(device)

    if args.mkldnn:
        model = mkldnn_utils.to_mkldnn(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    imsize = 224
    if args.model == 'inception_v3':
        imsize = 299
    dataset = RandomData(imsize, args.batch_size, args.num_batches_per_iter)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)

    def benchmark_step():
        data, target = next(iter(loader))

        if args.mkldnn:
            data = data.to_mkldnn()

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        if args.mkldnn:
            output = output.to_dense()
        if args.model == 'inception_v3':
            loss = F.cross_entropy(output.logits, target)
        else:
            loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    log('Model: %s' % args.model)
    log('Batch size: %d' % args.batch_size)

    # Warm-up
    log('Running warmup...')
    timeit.timeit(benchmark_step, number=args.num_warmup_batches)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec' % (x, img_sec))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Total img/sec %.1f +-%.1f' % (img_sec_mean, img_sec_conf))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_workers', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='enable mixed precision')

    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to benchmark')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size')

    parser.add_argument('--num-warmup-batches', type=int, default=10,
                        help='number of warm-up batches that don\'t count towards benchmark')
    parser.add_argument('--num-batches-per-iter', type=int, default=10,
                        help='number of batches per benchmark iteration')
    parser.add_argument('--num-iters', type=int, default=10,
                        help='number of benchmark iterations')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--ipex', action='store_true', default=False,
                        help='Enable Intel extension for PyTorch: '
                        'https://github.com/intel/intel-extension-for-pytorch')

    parser.add_argument('--mkldnn', action='store_true', default=False,
                        help='use tensor in _mkldnn layout')

    args = parser.parse_args()
    main(args)
