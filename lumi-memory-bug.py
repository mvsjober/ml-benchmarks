import torch
import torchvision
import sys

def main(bs):
    batch = torch.rand((bs, 3, 32, 32)).cuda()
    model = torchvision.models.resnet18().cuda()
    print(f'Feeding batch with size {bs} to model..')
    model(batch)
    print('Done.')
   
if __name__ == '__main__':
    bs = 256
    if len(sys.argv) > 1:
        bs = int(sys.argv[1])
    main(bs)

