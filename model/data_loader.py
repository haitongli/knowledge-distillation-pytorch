"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from dataset_processing import DatasetProcessing
import albumentations as A
from sampler import MultilabelBalancedRandomSampler
def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    train_dir = 'dataset/train/'
    valid_dir = 'dataset/valid/'
    TRAIN_IMG_FILE = train_dir + '2022_0706_6attr_sinhan_train.txt'
    TEST_IMG_FILE = valid_dir + '2022_0706_6attr_sinhan_valid.txt'

    TRAIN_IMG_DATA = ''
    TEST_IMG_DATA = ''
    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        aug_transfrom = A.Compose([
            A.RandomBrightnessContrast(),
            A.MotionBlur(),
            A.OpticalDistortion(),
            A.GaussNoise(),
            # A.Flip(),
            # A.Rotate(limit=90),
            # A.VerticalFlip(),
        ])

    # data augmentation can be turned off
    train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
    )

    # transformer for dev set
    dev_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
    )

    trainset = DatasetProcessing(
        TRAIN_IMG_DATA, TRAIN_IMG_FILE, train_transformer)

    devset = DatasetProcessing(
        TEST_IMG_DATA, TEST_IMG_FILE, dev_transformer)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl

def fetch_balance_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    train_dir = 'dataset/train/'
    valid_dir = 'dataset/valid/'
    TRAIN_IMG_FILE = train_dir + '2022_0706_6attr_sinhan_train.txt'
    TEST_IMG_FILE = valid_dir + '2022_0706_6attr_sinhan_valid.txt'

    TRAIN_IMG_DATA = ''
    TEST_IMG_DATA = ''
    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        aug_transfrom = A.Compose([
            A.RandomBrightnessContrast(),
            A.MotionBlur(),
            A.OpticalDistortion(),
            A.GaussNoise(),
            # A.Flip(),
            # A.Rotate(limit=90),
            # A.VerticalFlip(),
        ])

    # data augmentation can be turned off
    train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
    )

    # transformer for dev set
    dev_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
    )

    trainset = DatasetProcessing(
        TRAIN_IMG_DATA, TRAIN_IMG_FILE, train_transformer)

    devset = DatasetProcessing(
        TEST_IMG_DATA, TEST_IMG_FILE, dev_transformer)

    train_idx = list(range(0, len(trainset)))
    train_sampler = MultilabelBalancedRandomSampler(trainset.label, train_idx, class_choice="least_sampled")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """
    train_dir = 'dataset/train/'
    valid_dir = 'dataset/valid/'
    TRAIN_IMG_FILE = train_dir + '2022_0706_6attr_sinhan_train.txt'
    TEST_IMG_FILE = valid_dir + '2022_0706_6attr_sinhan_valid.txt'

    TRAIN_IMG_DATA = ''
    TEST_IMG_DATA = ''

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        aug_transfrom = A.Compose([
            A.RandomBrightnessContrast(),
            A.MotionBlur(),
            A.OpticalDistortion(),
            A.GaussNoise(),
            # A.Flip(),
            # A.Rotate(limit=90),
            # A.VerticalFlip(),
        ])

    # data augmentation can be turned off
    
    train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
    )

    # transformer for dev set
    dev_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
    )

    trainset = DatasetProcessing(
        TRAIN_IMG_DATA, TRAIN_IMG_FILE, train_transformer)

    devset = DatasetProcessing(
        TEST_IMG_DATA, TEST_IMG_FILE, dev_transformer)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl