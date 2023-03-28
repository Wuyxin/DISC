import os
import os.path as osp
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

from dataset.confounder_utils import prepare_confounder_data, prepare_group_confounder_data
from dataset.load_data import dataset_attributes
from dataset.transform import transform_dict


def prepare_cifar10_data(args):
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

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    transform_func = transform_dict['CIFAR10']
    transform_train = transform_func(model_type=None, train=True, augment_data=args.augment_data)
    transform_test = transform_func(model_type=None, train=False, augment_data=False)
    # load train/val data
    train_data = datasets.CIFAR10(args.root_dir, train=True, download=True, transform=transform_train)
    val_data = datasets.CIFAR10(args.root_dir, train=True, download=True, transform=transform_test)
    id_test_data = datasets.CIFAR10(args.root_dir, train=False, download=True, transform=transform_test)
    
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # id_test_loader = DataLoader(id_test_data, batch_size=100,shuffle=False, num_workers=0)

    # load ood test data
    ood_data_groups = torch.LongTensor([])
    ood_test_data = torch.FloatTensor([])
    ood_data_targets = torch.from_numpy(np.load(osp.join(args.root_dir, '../CIFAR-10-C', 'labels.npy')))
    # more items can be added into the list
    corruption_lst = ['gaussian_noise', 'motion_blur', 'contrast', 'elastic_transform']
    for idx, corruption in enumerate(corruption_lst):
        dataset = torch.from_numpy(
            np.load(osp.join(args.root_dir, '../CIFAR-10-C', f'{corruption}.npy')).transpose((0,3,1,2))
            ) / 255.
        ood_test_data = torch.concat([ood_test_data, normalize(dataset)], dim=0)
        ood_data_groups = torch.concat([ood_data_groups, torch.ones(dataset.size(0)).long() * idx])

    ood_data_targets = ood_data_targets.repeat(idx + 1)
    ood_test_data = torch.utils.data.TensorDataset(ood_test_data, ood_data_targets, ood_data_groups)
    # set group str
    ood_test_data.group_str = lambda id: corruption_lst[id]
    train_data.group_str = val_data.group_str = id_test_data.group_str = lambda x: "whole"
    # set group counts
    val_data.group_counts = train_data.group_counts = lambda *args: torch.tensor([len(train_data)])
    id_test_data.group_counts = lambda *args: torch.tensor([len(id_test_data)])
    ood_test_data.group_counts = lambda *args: torch.ones(len(corruption_lst)).long() *  len(dataset)
    # set n_groups
    ood_test_data.n_groups = idx + 1
    id_test_data.n_groups = val_data.n_groups = train_data.n_groups = 1
    # set classes
    ood_test_data.n_classes = id_test_data.n_classes = val_data.n_classes = train_data.n_classes = 10
    # dummy functions of train_data
    train_data.input_size = lambda *args: (32, 32)
    train_data.get_loader = val_data.get_loader = ood_test_data.get_loader = \
        lambda train, batch_size, **kwargs: DataLoader(
                train_data, batch_size=args.batch_size, shuffle=train, num_workers=0)
    train_data.get_group_array = lambda *args, **kwargs: torch.zeros(len(train_data)).long()
    train_data.get_label_array = lambda *args, **kwargs: torch.Tensor(train_data.targets) 

    return train_data, val_data, ood_test_data
    
