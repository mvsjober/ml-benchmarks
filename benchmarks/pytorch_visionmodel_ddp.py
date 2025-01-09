# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

import multiprocessing
from datetime import datetime
import argparse
import os
import psutil

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models

def set_cpu_affinity(rank, local_rank):
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    cpu_list = LUMI_GPU_CPU_map[local_rank]
    print(f"Rank {rank} (local {local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)
    

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
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    verbose = rank == 0
    if verbose:
        print('Using PyTorch version:', torch.__version__)
        print(torch.__config__.show())

    if args.set_cpu_binds:
        set_cpu_affinity(rank, local_rank)

    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()

    torch.manual_seed(0)
    torch.cuda.set_device(local_rank)

    # Set up standard model.
    if verbose:
        print(f'Using {args.model} model')
    model = getattr(models, args.model)()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    model = DistributedDataParallel(model, device_ids=[local_rank])

    train_dataset = dataset_from_datadir(args.datadir, verbose=verbose)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True, sampler=train_sampler)

    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    if verbose and args.fp16:
        print(f"Using fp16 (PyTorch automatic mixed precision)")

    if args.profiler:
        th = None
        if args.profiler_format == 'tb':
            th = torch.profiler.tensorboard_trace_handler('./logs/profiler')
        prof = profile(
            schedule=torch.profiler.schedule(
                wait=1,     # number of steps steps not active
                warmup=1,   # warmup steps (tracing, but results discarded)
                active=10,  # tracing steps
                repeat=1),  # repeat procedure this many times
            on_trace_ready=th,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        )
        prof.start()

    total_steps = args.steps if args.steps is not None else len(train_loader)

    if args.mlflow and verbose:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow)

        experiment_name = os.path.basename(__file__)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name)
        else:
            exp_id = exp.experiment_id

        mlflow.start_run(run_name=os.getenv("SLURM_JOB_ID"), experiment_id=exp_id)

        print(f"MLflow tracking to {mlflow.get_tracking_uri()}")
        mlflow.log_params(vars(args))

    # For each block of printed steps
    last_start = datetime.now()
    last_images = 0

    # For final average
    avg_images = 0
    avg_start = None
    avg_stop = None
    steps_counter = 0

    real_start = datetime.now()

    if args.warmup_steps == 0:
        avg_start = datetime.now()

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with torch.amp.autocast('cuda', enabled=args.fp16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.profiler:
                prof.step()

            li = len(images)
            last_images += li

            steps_counter += 1

            # Start time counter for final average if we have processed all
            # warmup steps
            if steps_counter == args.warmup_steps:
                avg_start = datetime.now()
            # After warmup steps add image count to avg_images counter
            elif steps_counter > args.warmup_steps:
                # we also exclude the last batch as that has some additional
                # delays which affects the average
                if steps_counter < total_steps-1:
                    avg_images += li
                    avg_stop = datetime.now()

            if (i + 1) % args.print_steps == 0 and verbose:
                now = datetime.now()
                last_secs = (now-last_start).total_seconds()

                if args.mlflow:
                    mlflow.log_metrics({
                        "epoch": epoch+1,
                        "step": i+1,
                        "loss": loss.item(),
                        "images/sec": last_images*world_size/last_secs
                        })

                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_steps}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Images/sec: {last_images*world_size/last_secs:.2f} '
                      f'(last {args.print_steps} steps)')

                last_start = now
                last_images = 0

                # if args.mlflow:
                #     cp_fname = 'model_checkpoint.pt'
                #     torch.save({
                #         'epoch': epoch+1,
                #         'steps': i+1,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict()
                #         }, cp_fname)
                #     mlflow.log_artifact(cp_fname, artifact_path='checkpoints')

            if args.steps is not None and steps_counter >= args.steps:
                break

    dur = datetime.now() - real_start
    avg_dur = (avg_stop - avg_start).total_seconds()

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
            print(f"Training completed in: {dur}")
            print(f"Images/sec: {avg_images*world_size/avg_dur:.2f} "
                  f"(average, skipping {args.warmup_steps} warmup steps)")


def main():
    multiprocessing.set_start_method('spawn')

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
    parser.add_argument('--mlflow', nargs='?', type=str, const='./mlruns')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='enable mixed precision')
    parser.add_argument('--set-cpu-binds', default=False, action="store_true",
                        help='Bind the process to the CPU cores closest to the GPU used by the process (LUMI only).')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
