import os
import os.path as osp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision

from train import train
from models import model_attributes
from dataset.cifar10_dataset import prepare_cifar10_data
from dataset.fmow_dataset import prepare_fmow_data
from dataset.dro_dataset import DRODataset
from dataset.folds import Subset, get_fold
from dataset.load_data import dataset_attributes, shift_types, prepare_data, log_data, log_meta_data
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model, check_args, set_log_dir
from args import parse_args



if __name__=='__main__':
    # Load args
    args = parse_args()
    set_log_dir(args)
    check_args(args)
    set_seed(args.seed)

    ## Initialize logs
    if not osp.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args.mode = 'a' if (osp.exists(osp.join(args.log_dir, 'last_model.pth')) and args.resume) else 'w'
    logger = Logger(osp.join(args.log_dir, f'log.txt'), args.mode)

    # Prepare data
    if args.dataset == 'CIFAR10':
        train_data, val_data, test_data = prepare_cifar10_data(args)
    elif args.dataset == 'FMoW':
        train_data, val_data, test_data = prepare_fmow_data(args)
    elif args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step': # not used
        train_data, val_data = prepare_data(args, train=True)        
    else:
        raise NotImplementedError

    # Record args
    args.n_groups = train_data.n_groups 
    args.n_classes = train_data.n_classes
    args.input_size = train_data.input_size()
    log_args(args, logger)

    # Prepare loaders
    args.loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 1, 'pin_memory': False}
    if args.fold:
        train_data, val_data = get_fold(
            train_data, args.fold,
            num_valid_per_point=args.num_sweeps,
            cross_validation_ratio=(1 / args.num_folds_per_sweep),
            seed=args.seed
        )
    if args.lisa_mix_up:
        for i in range(train_data.n_groups):
            idxes = np.where(train_data.get_group_array() == i)[0]
            if len(idxes) == 0: 
                continue
            subset = DRODataset(
                Subset(train_data, idxes), 
                process_item_fn=None, 
                n_groups=train_data.n_groups,
                n_classes=train_data.n_classes, 
                group_str_fn=train_data.group_str
                )
            train_loader[i] = subset.get_loader(
                train=True, reweight_groups=False, **args.loader_kwargs
                )
    else:
        train_loader = train_data.get_loader(reweight_groups=args.reweight_groups, 
                                             train=True, **args.loader_kwargs)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, **args.loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **args.loader_kwargs)

    # Gather all loaders and datasets
    data = {
        'train_data': train_data,
        'test_data': test_data,
        'val_data': val_data,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'val_loader': val_loader
        }

    ## Output logger to file
    if "Meta" in args.dataset:
        log_meta_data(data, logger)
    else:
        log_data(data, logger)

    ## Initialize model
    model = get_model(
        args, args.n_classes, 
        args.input_size[0], args.resume
        )
    model = model.cuda()
    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset == 'CUB' # Only supports binary
        def hinge_loss(yhat, y):
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Get resume information if needed
    if args.resume:
        df = pd.read_csv(osp.join(args.log_dir, f'test.csv'))
        epoch_offset = df.loc[len(df)-1,'epoch']+1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset=0

    # Set up CSV loggers
    csv_loggers = {}
    for split in ['train', 'test', 'val']:
        csv_loggers[split] = CSVBatchLogger(
            args, osp.join(args.log_dir, f'{split}.csv'), 
            data[f'{split}_data'].n_groups, mode=args.mode
            )
            
    # Training
    train(
        args, model, criterion, 
        data, logger, csv_loggers, 
        args.n_classes, epoch_offset=epoch_offset
        )
    for split in ['train', 'test', 'val']:
        csv_loggers[split].close()

    # Final results
    val_csv = pd.read_csv(osp.join(args.log_dir, f'val.csv'))
    test_csv = pd.read_csv(osp.join(args.log_dir, f'test.csv'))
    metric = 'roc_auc' if args.dataset == 'ISIC' else 'worst_group_acc'
    idx = np.argmax(val_csv[metric].values)
    logger.write(str(test_csv[[metric, 'mean_differences', "group_avg_acc", "avg_acc"]].iloc[idx]))