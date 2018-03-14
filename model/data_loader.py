import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters
    """

    #data transformers on train and dev('test') sets
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            # normalization values:
            # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            # normalization values:
            # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py

    # loader for development, no horizontal flip
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
        download=True, transform=train_transformer)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)

    testset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
        download=True, transform=dev_transformer)
    devloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD
    """
    train_transformer = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        # normalization values:
        # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py

    # loader for development, no horizontal flip
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
        download=True, transform=train_transformer)

    testset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
        download=True, transform=dev_transformer)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl




# def fetch_dataloader(types, data_dir, params):
#     """
#     Fetches the DataLoader object for each type in types from data_dir.

#     Args:
#         types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
#         data_dir: (string) directory containing the dataset
#         params: (Params) hyperparameters

#     Returns:
#         data: (dict) contains the DataLoader object for each type in types
#     """
#     dataloaders = {}

#     for split in ['train', 'dev', 'test']:
#         if split in types:
#             path = os.path.join(data_dir, "{}_signs".format(split))

#             # use the train_transformer if training data, else use eval_transformer without random flip
#             if split == 'train':
#                 dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
#                                         num_workers=params.num_workers,
#                                         pin_memory=params.cuda)
#             else:
#                 dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
#                                 num_workers=params.num_workers,
#                                 pin_memory=params.cuda)

#             dataloaders[split] = dl

#     return dataloaders
