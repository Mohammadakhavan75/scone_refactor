from torchvision import transforms
from torchvision import datasets 
import numpy as np
import torch
import PIL



cifar10_path = '../data/'
cifar100_path = '../data/'
svhn_path = '../data/'



mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


class np_loader(torch.utils.data.Dataset):
    def __init__(self, imgs_path, targets, transform):
        self.data = np.load(imgs_path)
        self.targets = np.load(targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]        
        img = PIL.Image.fromarray(img)
        img = self.transform(img)

        return img, self.targets[idx]


def load_CIFAR(dataset_name):
    if dataset_name in ['cifar10']:
        print('loading CIFAR-10...')
        train_data = datasets.CIFAR10(cifar10_path, train=True, transform=transform, download=True)
        test_data = datasets.CIFAR10(cifar10_path, train=False, transform=transform, download=True)

    elif dataset_name in ['cifar100']:
        print('loading CIFAR-100...')
        train_data = datasets.CIFAR100(cifar100_path, train=True, transform=transform, download=True)
        test_data = datasets.CIFAR100(cifar100_path, train=False, transform=transform, download=True)

    return train_data, test_data


def load_SVHN():
    print('loading SVHN...')

    train_data = datasets.SVHN(svhn_path, split='train', transform=transform)
    test_data = datasets.SVHN(svhn_path, split='test', transform=transform)
    return train_data, test_data


def load_np_dataset(dataset_name, train_img_path, train_target_path, test_img_path, test_traget_path):
    print(f'loading {dataset_name}...')

    if train_img_path is not None:
        train_data = np_loader(train_img_path, train_target_path, transform)
    else:
        train_data = None

    if test_img_path is not None:
        test_data = np_loader(test_img_path, test_traget_path, transform) 
    else:
        test_data = None

    return train_data, test_data

