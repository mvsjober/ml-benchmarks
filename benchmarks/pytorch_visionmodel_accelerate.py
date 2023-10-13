from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from accelerate import Accelerator


class SyntheticData(torch.utils.data.Dataset):
    def __init__(self,
                 size=1281184,  # like ImageNet
                 image_size=(3, 224, 224),
                 num_classes=1000,
                 transform=None):
        super(SyntheticData, self).__init__()
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

    def __getitem__(self, index):
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]

        if self.transform is not None:
            img = self.transform(img)

        return img, target.item()

    def __len__(self):
        return self.size


def dataset_from_datadir(datadir, verbose=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if datadir is not None:
        traindir = os.path.join(datadir, 'train')
        if verbose:
            print('Reading training data from:', traindir)
        return datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if verbose:
        print('No --datadir argument given, using fake data for training...')
    return SyntheticData()


def train(args):
    accelerator = Accelerator()

    verbose = accelerator.is_main_process
    if verbose:
        print('Using PyTorch version:', torch.__version__)
        print(torch.__config__.show())

    world_size = torch.distributed.get_world_size()

    # Set up standard model.
    if verbose:
        print(f'Using {args.model} model')
    model = getattr(models, args.model)()

    criterion = nn.CrossEntropyLoss() #.cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    train_dataset = dataset_from_datadir(args.datadir, verbose=verbose)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True) #, sampler=train_sampler)

    total_step = args.steps if args.steps is not None else len(train_loader)//world_size

    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)

    # For each block of printed steps
    last_start = datetime.now()
    last_images = 0

    # For final average
    avg_images = 0
    avg_start = None
    tot_steps = 0

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if args.profiler:
                prof.step()

            li = len(images)
            last_images += li

            tot_steps += 1
            if tot_steps == args.warmup_steps:
                avg_start = datetime.now()
            elif tot_steps > args.warmup_steps:
                avg_images += li

            if (i + 1) % args.print_steps == 0 and accelerator.is_main_process:
                now = datetime.now()
                last_secs = (now-last_start).total_seconds()

                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_step}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Images/sec: {last_images*world_size/last_secs:.2f} '
                      f'(last {args.print_steps} steps)')

                last_start = now
                last_images = 0

            if args.steps is not None and tot_steps >= args.steps:
                break

    if args.profiler:
        if args.profiler_format == 'json' and verbose:
            trace_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            trace_fname = f"profiler-trace-{trace_datetime}.json"
            print(f'Writing profiler trace to {trace_fname}')
            prof.export_chrome_trace(trace_fname)

        prof.stop()

    if verbose:
        if avg_start is None:
            print("WARNING: stopped before warmup steps done, not printing stats.")
        else:
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
                        help='Maximum number of training steps')
    parser.add_argument('--profiler', action='store_true')
    parser.add_argument('--profiler-format', type=str,
                        choices=['tb', 'json'], default='tb')
    parser.add_argument('--print-steps', type=int, default=100)
    parser.add_argument('--warmup-steps', type=int, default=10,
                        help='Number of initial steps to ignore in average')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
