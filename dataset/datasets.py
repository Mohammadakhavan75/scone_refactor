from torchvision import transforms
from torchvision import datasets 
from torch.utils.data import DataLoader
import numpy as np
import torch
import PIL


cifar10_path = '../data/'
cifar100_path = '../data/'
svhn_path = '../data/'


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


class np_dataset(torch.utils.data.Dataset):
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

    train_data = datasets.SVHN(svhn_path, split='train', transform=transform, download=True)
    test_data = datasets.SVHN(svhn_path, split='test', transform=transform, download=True)
    return train_data, test_data


def load_np_dataset(train_img_path, train_target_path, test_img_path, test_traget_path):

    if train_img_path is not None:
        train_data = np_dataset(train_img_path, train_target_path, transform)
    else:
        train_data = None

    if test_img_path is not None:
        test_data = np_dataset(test_img_path, test_traget_path, transform) 
    else:
        test_data = None

    return train_data, test_data


def loader(train_data, test_data, batch_size, num_worker):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    return train_dataloader, test_dataloader


def main(args):
    
    if args.in_dataset == 'cifar10':
        in_train_dataset, in_test_dataset = load_CIFAR('cifar10')
        
    
    if args.in_dataset == 'cifar100':
        in_train_dataset, in_test_dataset = load_CIFAR('cifar10')
    
    if args.aux_dataset == 'svhn':
        aux_train_dataset, aux_test_dataset = load_SVHN()
        
    
    if args.in_shift is not None:
        train_img_path = f'/storage/users/makhavan/CSI/exp05/scone_one_claas_train/data/CorCIFAR10_train/{args.in_shift}.npy'
        train_target_path = '/storage/users/makhavan/CSI/exp05/scone_one_claas_train/data/CorCIFAR10_train/labels.npy'
        test_img_path = f'/storage/users/makhavan/CSI/exp05/scone_one_claas_train/data/CorCIFAR10_test/{args.in_shift}.npy'
        test_traget_path = '/storage/users/makhavan/CSI/exp05/scone_one_claas_train/data/CorCIFAR10_test/labels.npy'

        in_shift_train_dataset, in_shift_test_dataset = load_np_dataset(train_img_path, train_target_path, test_img_path, test_traget_path)
        

    if args.ood_dataset == 'svhn':
        ood_train_dataset, ood_test_dataset = load_SVHN()
        
    rng = np.random.default_rng()

    max_length = np.min([len(in_train_dataset), len(in_shift_train_dataset), len(aux_train_dataset)])

    idx_ = np.array(range(len(in_train_dataset)))
    rng.shuffle(idx_)
    idx_ = idx_[:max_length]
    in_train_dataset = torch.utils.data.Subset(in_train_dataset, idx_)

    idx_ = np.array(range(len(in_shift_train_dataset)))
    rng.shuffle(idx_)
    idx_ = idx_[:max_length]
    in_shift_train_dataset = torch.utils.data.Subset(in_shift_train_dataset, idx_)

    idx_ = np.array(range(len(aux_train_dataset)))
    rng.shuffle(idx_)
    idx_ = idx_[:max_length]
    aux_train_dataset = torch.utils.data.Subset(aux_train_dataset, idx_)

    in_train_loader, in_test_loader = loader(in_train_dataset, in_test_dataset, args.batch_size, args.num_worker)
    aux_train_loader, aux_test_loader = loader(aux_train_dataset, aux_test_dataset, args.batch_size, args.num_worker)
    in_shift_train_loader, in_shift_test_loader = loader(in_shift_train_dataset, in_shift_test_dataset, args.batch_size, args.num_worker)
    ood_train_loader, ood_test_loader = loader(ood_train_dataset, ood_test_dataset, args.batch_size, args.num_worker)

    return in_train_loader, in_test_loader, in_shift_train_loader,in_shift_test_loader, aux_train_loader, aux_test_loader, ood_test_loader







    
    





