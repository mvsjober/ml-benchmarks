from datetime import datetime
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import horovod.torch as hvd
from horovod import __version__ as hvd_version

import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
        
    #torch.manual_seed(0)
    torch.cuda.set_device(hvd.local_rank())

    # Set up standard model.
    if verbose:
        print('Using {} model'.format(args.model))
    model = getattr(models, args.model)()
    model = model.cuda()

    import torch.multiprocessing as mp
    # # assert "forkserver" in mp.get_all_start_methods()
    mp.set_start_method("forkserver")

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

    start = datetime.now()
    num_images = 0
    if verbose == 0:
        print("Starting training at", start, flush=True)

    total_step = args.steps if args.steps is not None else len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_images += args.batchsize * hvd.size()

            if (i + 1) % 100 == 0 and verbose:
                tot_secs = (datetime.now()-start).total_seconds()

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Images/sec: {:.2f} [{}]'.
                      format(epoch + 1,
                             args.epochs,
                             i + 1,
                             total_step,
                             loss.item(),
                             num_images/tot_secs, datetime.now()), flush=True)
            if args.steps is not None and i >= args.steps:
                break
    if verbose:
        dur = datetime.now() - start
        ni = total_step * args.batchsize * hvd.size()
        print("Training completed in: " + str(dur))
        print("Images/sec: {:.4f}".format(ni/dur.total_seconds()))


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
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
