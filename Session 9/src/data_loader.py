import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np


class ImageDataLoader(DataLoader):
    """
    Load Image datasets from torch
    Default category is MNIST
    Pass category as 'MNIST' or 'CIFAR10'
    """
    def __init__(self, transforms, data_dir, batch_size, shuffle, category='MNIST', num_workers=4, pin_memory=False, device='cpu'):
        self.data_dir = data_dir
        cuda = torch.cuda.is_available()

        if cuda:
            self.device = 'gpu'
            pin_memory = True
        else:
            self.device = device
        
        self.classes = None
        
        if category == 'MNIST':
            self.train_loader = datasets.MNIST(
                self.data_dir,
                train=True,
                download=True,
                # transform=transforms.build_transforms(train=True)
                transform=transforms
            )
            self.test_loader = datasets.MNIST(
                self.data_dir,
                train=False,
                download=True,
                # transform=transforms.build_transforms(train=False)
                transform=transforms
            )
        elif category == 'CIFAR10':
            self.train_loader = datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=transforms
                # transform=transforms.build_transforms(train=True)
            )
            self.test_loader = datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                # transform=transforms.build_transforms(train=False)
                transform=transforms
            )
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                            'horse', 'ship', 'truck')

        self.init_kwargs = {
                'batch_size': batch_size,
                'num_workers':num_workers,
                'pin_memory': pin_memory
            }
        # super().__init__(self.train_loader, shuffle=shuffle, **self.init_kwargs)
        self.train_loader = DataLoader(self.train_loader, shuffle=shuffle, ** self.init_kwargs)
        self.test_loader = DataLoader(self.test_loader, **self.init_kwargs)

    # def test_split(self):
    #     return DataLoader(self.test_loader, **self.init_kwargs)

    def show(self, type='train'):
        if type=='train':
            dataiter = iter(self.train_loader)
        else:
            dataiter = iter(self.test_loader)

        images, labels = dataiter.next()

        img = torchvision.utils.make_grid(images)

        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))




