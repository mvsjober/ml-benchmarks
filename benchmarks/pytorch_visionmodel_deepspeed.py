import argparse
import deepspeed
import os
import torch
import torch.nn as nn
from datetime import datetime
from datetime import timedelta
from torchvision import models

from pytorch_visionmodel_ddp import dataset_from_datadir


def train(args):
    num_epochs = args.epochs
    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get('PMIX_RANK', -1))

    deepspeed.init_distributed(timeout=timedelta(minutes=5))
    world_size = int(os.environ['WORLD_SIZE'])

    torch.manual_seed(0)

    # Set up standard model.
    if local_rank == 0:
        print('Using {} model'.format(args.model))
    model = getattr(models, args.model)()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    train_dataset = dataset_from_datadir(args.datadir)

    model_engine, optimizer, train_loader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(),
        training_data=train_dataset)

    # For final average
    avg_images = 0
    avg_start = None
    tot_steps = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images = data[0].to(model_engine.local_rank)
            labels = data[1].to(model_engine.local_rank)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            li = len(images)
            # last_images += li

            tot_steps += 1
            if tot_steps == args.warmup_steps:
                avg_start = datetime.now()
            elif tot_steps > args.warmup_steps:
                avg_images += li

            if args.steps is not None and tot_steps >= args.steps:
                break

    if local_rank == 0:
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
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--steps', type=int, required=False,
                        help='Maxium number of training steps')
    parser.add_argument('--warmup-steps', type=int, default=10,
                        help='Number of initial steps to ignore in average')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
