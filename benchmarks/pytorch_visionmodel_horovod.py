from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import horovod.torch as hvd
from horovod import __version__ as hvd_version

from torchvision import models

from pytorch_visionmodel_ddp import dataset_from_datadir

torch.set_num_threads(1)


def train(args):
    hvd.init()

    print("Hello from local_rank {}/{}, rank {}/{}".format(
        hvd.local_rank(), hvd.local_size(), hvd.rank(), hvd.size()))

    verbose = hvd.rank() == 0

    if verbose:
        print('Using PyTorch version:', torch.__version__)
        print('Horovod version: {}, CUDA: {}, ROCM: {}, NCCL: {}, MPI: {}'.format(
            hvd_version,
            hvd.cuda_built(),
            hvd.rocm_built(),
            hvd.nccl_built(),
            hvd.mpi_built()))
        print(torch.__config__.show())

    cudnn.benchmark = True

    torch.cuda.set_device(hvd.local_rank())
    world_size = hvd.size()

    # Set up standard model.
    if verbose:
        print('Using {} model'.format(args.model))
    model = getattr(models, args.model)()
    model = model.cuda()

    # import torch.multiprocessing as mp
    # # # assert "forkserver" in mp.get_all_start_methods()
    # mp.set_start_method("forkserver")

    lr_scaler = hvd.size()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4 * lr_scaler)

    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters())
    train_dataset = dataset_from_datadir(args.datadir, verbose)
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=hvd.size(),
                                       rank=hvd.rank())
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=False, sampler=train_sampler,
                              multiprocessing_context='forkserver')

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    total_step = args.steps if args.steps is not None else len(train_loader)

    # For each block of printed steps
    last_start = datetime.now()
    last_images = 0

    # For final average
    avg_images = 0
    avg_start = None
    tot_steps = 0

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            li = len(images)
            last_images += li

            tot_steps += 1
            if tot_steps == args.warmup_steps:
                avg_start = datetime.now()
            elif tot_steps > args.warmup_steps:
                avg_images += li

            if (i + 1) % args.print_steps == 0 and verbose:
                now = datetime.now()
                last_secs = (now-last_start).total_seconds()

                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_step}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Images/sec: {last_images*world_size/last_secs:.2f} '
                      f'(last {args.print_steps} steps)')

                last_start = now
                last_images = 0

            if args.steps is not None and i >= args.steps:
                break
    if verbose:
        dur = datetime.now() - avg_start
        print(f"Training completed in: {dur}")
        print(f"Images/sec: {avg_images*world_size/dur.total_seconds():.2f} "
              f"(average, skipping {args.warmup_steps} warmup steps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to benchmark')
    parser.add_argument('--datadir', type=str, required=False,
                        help='Data directory')
    parser.add_argument('-b', '--batchsize', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-j', '--workers', type=int, default=10,
                        help='Number of data loader workers')
    parser.add_argument('--steps', type=int, required=False,
                        help='Maxium number of training steps')
    parser.add_argument('--print-steps', type=int, default=100)
    parser.add_argument('--warmup-steps', type=int, default=10,
                        help='Number of initial steps to ignore in average')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
