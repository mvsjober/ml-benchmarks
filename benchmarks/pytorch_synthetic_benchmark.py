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


def log(s, nl=True):
    print(s, end='\n' if nl else '', flush=True)


def main(args):
    if args.ipex:
        import intel_pytorch_extension as ipex
        if args.fp16:
            ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)

    use_amp = False
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        if args.fp16:
            use_amp = True
    elif args.ipex:
        device = ipex.DEVICE
    else:
        device = torch.device('cpu')

    log('Using PyTorch version: %s, Device: %s' % (torch.__version__, device))
    log(torch.__config__.show())

    cudnn.benchmark = True

    # Set up standard model.
    log('Initializing %s model...' % args.model)
    model = getattr(models, args.model)()
    model = model.to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        log('Using %d GPUs with torch.nn.DataParallel' % torch.cuda.device_count())

    if args.mkldnn:
        model = mkldnn_utils.to_mkldnn(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    imsize = 224
    if args.model == 'inception_v3':
        imsize = 299

    def benchmark_step():
        #data, target = next(iter(loader))
        data = torch.randn(args.batch_size, 3, imsize, imsize)
        target = torch.LongTensor(args.batch_size).random_() % 1000

        if args.mkldnn:
            data = data.to_mkldnn()

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(data)
            if args.mkldnn:
                output = output.to_dense()
            if args.model == 'inception_v3':
                loss = F.cross_entropy(output.logits, target)
            else:
                loss = F.cross_entropy(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='use multiple GPUs if available')

    parser.add_argument('--ipex', action='store_true', default=False,
                        help='Enable Intel extension for PyTorch: '
                        'https://github.com/intel/intel-extension-for-pytorch')

    parser.add_argument('--mkldnn', action='store_true', default=False,
                        help='use tensor in _mkldnn layout')

    args = parser.parse_args()
    main(args)
