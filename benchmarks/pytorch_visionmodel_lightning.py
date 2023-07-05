from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import models
import argparse
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import Callback

from pytorch_visionmodel_ddp import dataset_from_datadir


class TorchvisionModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = getattr(models, model_name)()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 1e-4)
        return optimizer

class BenchmarkingCallback(Callback):
    def __init__(self, warmup_steps, batchsize, world_size):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.batchsize = batchsize
        self.world_size = world_size
        self.images = 0
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == self.warmup_steps:
            self.start = datetime.now()
        if batch_idx >= self.warmup_steps:
            self.images += self.batchsize

    def on_train_end(self, trainer, pl_module):
        dur = datetime.now() - self.start
        print(f"Training completed in: {dur}")
        print(f"Images/sec: {self.images*self.world_size/dur.total_seconds():.2f} "
              f"(average, skipping {self.warmup_steps} warmup steps)")

    

def train(args):
    print('Using PyTorch version:', torch.__version__)
    print(torch.__config__.show())

    model = TorchvisionModel(args.model)

    train_dataset = dataset_from_datadir(args.datadir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                              num_workers=args.workers, pin_memory=True)

    precision = 16 if args.fp16 else 32

    world_size = args.gpus*args.nodes
    
    if args.strategy == 'ddp':
        trainer = pl.Trainer(devices=args.gpus,
                             num_nodes=args.nodes,
                             max_epochs=args.epochs,
                             accelerator='gpu',
                             strategy='ddp',
                             precision=precision,
                             callbacks=[BenchmarkingCallback(args.warmup_steps,
                                                             args.batchsize,
                                                             world_size)])
    elif args.strategy == 'horovod':
        trainer = pl.Trainer(max_epochs=args.epochs,
                             gpus=1,
                             strategy="horovod") #,
                             #precision=precision)
    else:
        print("ERROR: Unsupported strategy '{}'".format(args.strategy))
        return

    trainer.fit(model, train_loader)
  
    # trainer.save_checkpoint("benchmark_lightning_model.ckpt")


def main():
    parser = argparse.ArgumentParser()

    # Lightning-specific params
    parser.add_argument('--gpus', default=1, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--strategy', default='ddp',
                        help='training strategy for Lightning, '
                        'currently supported values: ddp, horovod')
    parser.add_argument('--fp16', action='store_true', default=False)

    # Same as for pytorch_ddp.py
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='maximum number of epochs to run')
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
    parser.add_argument('--warmup-steps', type=int, default=10,
                        help='Number of initial steps to ignore in average')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
