# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import os
import sys

from torchvision import models, datasets, transforms
import torch
import torchvision
from PIL import Image, ImageOps, ImageFilter

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on CIFAR10')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('trained_classifier', type=Path, metavar='FILE',
                    help='path to trained classifier model')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--checkpoint-dir', default='checkpoint_sym_noise_1/lincls/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

def main():
    args = parser.parse_args()
    main_worker(0, args)

def main_worker(gpu, args):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)

    # load trained classifier model to measure test accuracy
    if (args.trained_classifier).is_file():
        ckpt = torch.load(args.trained_classifier, map_location='cpu')
        best_acc = ckpt['best_acc']
        model = models.resnet50().cuda(gpu)
        model.load_state_dict(ckpt['model'])

    # Data loading code
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])

    # loads original CIFAR10 testset without noise
    test_dataset = datasets.CIFAR10(root = args.data, 
                               train = False, 
                               download = True, 
                               transform = transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
    print(f'Loaded test data from {args.data}')
    print(f'Loaded test data from {args.data}', file=stats_file)

    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    # evaluate
    model.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    with torch.no_grad():
        for images, target in test_loader:
            output = model(images.cuda(gpu, non_blocking=True))
            acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
    best_acc.top1 = max(best_acc.top1, top1.avg)
    best_acc.top5 = max(best_acc.top5, top5.avg)
    stats = dict(epoch=-1, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)
    print(json.dumps(stats))
    print(json.dumps(stats), file=stats_file)

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# python evaluate.py <args.data> <args.trained_classifier> ...
# python evaluate_test.py data checkpoint_noisy_sym_1/lincls/checkpoint.pth --checkpoint-dir checkpoint_noisy_sym_1/lincls

if __name__ == '__main__':
    main()
