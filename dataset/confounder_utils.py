import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from models import model_attributes
from dataset.cub_dataset import CUBDataset
from dataset.dro_dataset import DRODataset
from dataset.meta_dataset_cat_dog import MetaDatasetCatDog
from dataset.isic_dataset import ISICDataset


"""
Settings
"""
confounder_settings = {
    'CUB':{
        'constructor': CUBDataset
    },
    "MetaDatasetCatDog":{
        'constructor': MetaDatasetCatDog
    },
    "ISIC":{
        "constructor": ISICDataset
    }
}


"""
Data preparation
"""
def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](
        args=args,
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data,
        mix_up=args.lisa_mix_up)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets


def prepare_group_confounder_data(args, group_id, train_data=None, return_full_dataset=False):

    full_dataset = confounder_settings[args.dataset]['constructor'](
        args=args,
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data,
        mix_up=args.lisa_mix_up,
        group_id=group_id)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)

    splits = ['train']
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets[0]