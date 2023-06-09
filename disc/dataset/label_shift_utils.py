import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

from disc.models import model_attributes
from disc.dataset.dro_dataset import DRODataset


"""
This file is not used in DISC
"""
########################
### DATA PREPARATION ###
########################
def prepare_label_shift_data(args, train):
    settings = label_shift_settings[args.dataset]
    data = settings['load_fn'](args, train)
    n_classes = settings['n_classes']
    if train:
        train_data, val_data = data
        if args.fraction<1:
            train_data = subsample(train_data, args.fraction)
        train_data = apply_label_shift(train_data, n_classes, args.shift_type, args.minority_fraction, args.imbalance_ratio)
        data = [train_data, val_data]
    dro_data = [DRODataset(subset, process_item_fn=settings['process_fn'], n_groups=n_classes, 
                           n_classes=n_classes, group_str_fn=settings['group_str_fn']) \
                for subset in data]
    return dro_data


##############
### SHIFTS ###
##############

def apply_label_shift(dataset, n_classes, shift_type, minority_frac, imbalance_ratio):
    assert shift_type.startswith('label_shift')
    if shift_type=='label_shift_step':
        return step_shift(dataset, n_classes, minority_frac, imbalance_ratio) 


def step_shift(dataset, n_classes, minority_frac, imbalance_ratio):
    # get y info
    y_array = []
    for x,y in dataset:
        y_array.append(y)
    y_array = torch.LongTensor(y_array)
    y_counts = ((torch.arange(n_classes).unsqueeze(1)==y_array).sum(1)).float()
    # figure out sample size for each class
    is_major = (torch.arange(n_classes) < (1-minority_frac)*n_classes).float()
    major_count = int(torch.min(is_major*y_counts + (1-is_major)*y_counts*imbalance_ratio).item())
    minor_count = int(np.floor(major_count/imbalance_ratio))
    # subsample
    sampled_indices = []
    for y in np.arange(n_classes):
        indices,  = np.where(y_array==y)
        np.random.shuffle(indices)
        if is_major[y]:
            sample_size = major_count
        else:
            sample_size = minor_count
        sampled_indices.append(indices[:sample_size])
    sampled_indices = torch.from_numpy(np.concatenate(sampled_indices))
    return Subset(dataset, sampled_indices)


###################
### PROCESS FNS ###
###################

def xy_to_xyy(data):
    x,y = data
    return x,y,y


#####################
### GROUP STR FNS ###
#####################

def group_str_CIFAR10(group_idx):
    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return f'Y = {group_idx} ({classes[group_idx]})'


### CIFAR10 ###
def load_CIFAR10(args, train):
    transform = get_transform_CIFAR10(args, train)
    dataset = torchvision.datasets.CIFAR10(args.root_dir, train, transform=transform, download=True)
    if train:
        subsets = train_val_split(dataset, args.val_fraction)
    else:
        subsets = [dataset,]
    return subsets


def get_transform_CIFAR10(args, train):
    transform_list = []
    # resize if needed
    target_resolution = model_attributes[args.model]['target_resolution']
    if target_resolution is not None:
        transform_list.append(transforms.Resize(target_resolution))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    composed_transform = transforms.Compose(transform_list)
    return composed_transform


################
### SETTINGS ###
################

label_shift_settings = {
    'CIFAR10':{
        'load_fn': load_CIFAR10,
        'group_str_fn': group_str_CIFAR10,
        'process_fn': xy_to_xyy,
        'n_classes':10
    }
}

