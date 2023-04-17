import os
import os.path as osp
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

from dataset.load_data import dataset_attributes
from wilds.datasets.fmow_dataset import FMoWDataset
from dataset.transform import transform_dict


def prepare_fmow_data(args):
    """
    Prepare CIFAR10 data, where the corrupted testing data is from CIFAR-10-C 

    Args: 
        args: Arguments, see args.py

    Returns: 
        datasets (tuple([Dataset, Dataset, Dataset]))
        dataloaders (tuple([DataLoader, DataLoader, DataLoader]))

    """
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    
    transform_func = transform_dict['FMoW']
    transform_train = transform_func(model_type=args.model, train=True, augment_data=args.augment_data)
    transform_test = transform_func(model_type=args.model, train=False, augment_data=False)
    
    # load train/val data
    dataset = FMoWDataset(root_dir=args.root_dir, download=True)
    train_data = dataset.get_subset('train', transform=transform_train)
    val_data = dataset.get_subset('val', transform=transform_train)
    test_data = dataset.get_subset('test', transform=transform_test)

    # set group str
    metadata_str = dataset.metadata_map['region']
    meadata_idx = dataset.metadata_fields.index('region')
    metadata = dataset._metadata_array = dataset._metadata_array[:, meadata_idx]
    train_data.group_str = val_data.group_str = test_data.group_str = lambda id: metadata_str[id]

    # set n_groups
    n_groups = len(metadata_str)
    setattr(train_data, 'n_groups', n_groups)
    setattr(val_data, 'n_groups', n_groups)
    setattr(test_data, 'n_groups', n_groups)

    # set group counts
    split_dict = dataset.split_dict
    split_array = dataset.split_array
    y_array = dataset.y_array
    val_data.group_counts = lambda *args: torch.Tensor([torch.sum(metadata[split_array==split_dict['val']]==i).item() for i in range(n_groups)])
    train_data.group_counts = lambda *args: torch.Tensor([torch.sum(metadata[split_array==split_dict['train']]==i).item() for i in range(n_groups)])
    test_data.group_counts = lambda *args: torch.Tensor([torch.sum(metadata[split_array==split_dict['test']]==i).item() for i in range(n_groups)])
    
    # train_data
    train_data.input_size = lambda *args: (224, 224)
    train_data.get_loader = lambda train, batch_size, **kwargs: DataLoader(train_data, 
                                                                           batch_size=batch_size, 
                                                                           shuffle=train, num_workers=0)
    val_data.get_loader = lambda train, batch_size, **kwargs: DataLoader(val_data, 
                                                                         batch_size=batch_size, 
                                                                         shuffle=False, num_workers=0)
    test_data.get_loader = lambda train, batch_size, **kwargs: DataLoader(test_data, 
                                                                          batch_size=batch_size, 
                                                                          shuffle=False, num_workers=0)
    train_data.get_group_array = lambda *args, **kwargs: metadata[split_array == split_dict['train']]
    train_data.get_label_array = lambda *args, **kwargs: y_array[split_array == split_dict['train']].numpy()
    return train_data, val_data, test_data
