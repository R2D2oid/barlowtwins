#################################################
###                                           ###
###           SSL TRAIN Transforms            ###
###                                           ###
#################################################

import torchvision.transforms as transforms
from PIL import Image

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
my_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                saturation=0.2, hue=0.1)],
        p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(p=1.0),
    Solarization(p=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])

])

from dataloader import NoisyCIFAR10

train_dataset = NoisyCIFAR10(root='data', 
                   train=True, 
                   download=True, 
                   noise_type = 'sym', 
                   noise_rate = 0.1, 
                   transform = my_transform)

### !!!! writes the noisy dataset to disk
train_dataset.dump_(path_='checkpoint/cifar10_noisy.pkl')


#################################################
###                                           ###
### CLASSIFIER TRAIN and Evaluate Transforms  ###
###                                           ###
#################################################

# loads original CIFAR10 dataset without noise
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])

train_dataset = datasets.CIFAR10(root = 'data', 
                           train = True, 
                           download = True, 
                           transform = transforms.Compose([
                                transforms.Resize(32),
                                transforms.CenterCrop(32),
                                transforms.ToTensor(),
                                normalize,
                            ]))

# re-use previously generated noisy CIFAR10 targets (used for ssl training barlow twins)
### !!!! loads previously written noisy dataset targets from disk
noisy_targets = NoisyCIFAR10.load_('checkpoint/cifar10_noisy.pkl').targets
train_dataset.targets = noisy_targets
