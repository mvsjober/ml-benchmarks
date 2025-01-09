import argparse

import torchvision.datasets as datasets
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField


def main(args):
    ds = datasets.ImageFolder(args.datadir)

    writer = DatasetWriter(args.outdir, {
        'image': RGBImageField(write_mode='jpg'),
        'label': IntField(),
    }, num_workers=args.num_workers)

    writer.from_indexed_dataset(ds, chunksize=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str,
                        help='Data directory')
    parser.add_argument('outdir', type=str)
    parser.add_argument('-n', '--num_workers', type=int, default=-1)
    args = parser.parse_args()
    main(args)
