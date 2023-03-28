import sys
import os
import torch
import numpy as np
import csv
import argparse
import torch.nn as nn
import torch
import operator
from numbers import Number
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR
from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)
from models import model_attributes
import torchvision
from models import ResNet50


def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio
    if args.disc:
        assert args.erm_path is not None

def set_required_grad(model, required_grad=True):
    for p in model.parameters():
        p.requires_grad = required_grad
    return model

def get_optimizer(args, model):
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        raise ValueError(f"{args.optimizer} not recognized")
    return optimizer


def get_model(args, n_classes, d=None, resume=False):
    pretrained = not args.train_from_scratch
    if resume:
        model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
    elif args.dataset == 'CIFAR10':
        model = ResNet50()
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained and d
        # Load precomputed features
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'densenet121':
        model = torchvision.models.densenet121(pretrained=pretrained)
        d = model.classifier.in_features
        model.classifier = nn.Linear(d, n_classes)

    elif 'bert' in args.model:
        if args.is_featurizer:
            if args.model == 'bert':
                from bert.bert import BertFeaturizer
                featurizer = BertFeaturizer.from_pretrained("bert-base-uncased", **args.model_kwargs)
                classifier = nn.Linear(featurizer.d_out, 5 if args.dataset == "Amazon" else n_classes)
                model = torch.nn.Sequential(featurizer, classifier)
            elif args.model == 'distilbert':
                from bert.distilbert import DistilBertFeaturizer
                featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased", **args.model_kwargs)
                classifier = nn.Linear(featurizer.d_out, 5 if args.dataset == "Amazon" else n_classes)
                model = torch.nn.Sequential(featurizer, classifier)
            else:
                raise NotImplementedError

        else:
            from bert.bert import BertClassifier
            model = BertClassifier.from_pretrained(
                'bert-base-uncased',
                num_labels=512,
                **args.model_kwargs)

    else:
        raise ValueError('Model not recognized.')
    return model


def get_optimizer_weights(args, weights):
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            weights,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            weights,
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        raise ValueError(f"{args.optimizer} not recognized")
    return optimizer


def get_scheduler(args, optimizer, t_total):
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08)

    if args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=t_total)

    elif args.scheduler == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=t_total,
            num_warmup_steps=args.num_warmup_steps)

        step_every_batch = True
        use_metric = False

    elif args.scheduler == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=t_total,
            num_warmup_steps=args.num_warmup_steps)
        
    elif args.scheduler == 'StepLR':
        scheduler = StepLR(optimizer,
                            t_total,
                            gamma=args.step_gamma)

    else:
        scheduler = None
    return scheduler


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x


class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class CSVLogger:
    def __init__(self, args, csv_path, concept_names, mode='w'):
        self.concept_names = concept_names
        columns = ['epoch']
        for concept in concept_names:
            columns.append(concept)
        columns.append(f'average')

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, stats_dict):
        stats_dict['epoch'] = epoch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class CSVBatchLogger:
    def __init__(self, args, csv_path, n_groups, mode='w'):
        self.n_groups = n_groups
        columns = ['epoch', 'batch']
        for idx in range(n_groups):
            columns.append(f'avg_loss_group:{idx}')
            columns.append(f'exp_avg_loss_group:{idx}')
            columns.append(f'avg_acc_group:{idx}')
            columns.append(f'processed_data_count_group:{idx}')
            columns.append(f'update_data_count_group:{idx}')
            columns.append(f'update_batch_count_group:{idx}')
        columns.append('avg_actual_loss')
        columns.append('avg_per_sample_loss')
        columns.append('avg_acc')
        columns.append('model_norm_sq')
        columns.append('reg_loss')
        columns.append("worst_group_acc")
        columns.append("mean_differences")
        columns.append("group_avg_acc")
        if args.dataset == 'MetaDataset':
            columns.append("F1-score")
        if args.dataset == 'ISIC':
            columns.append("roc_auc")

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')



# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val
            
            
class ParamDict(OrderedDict):
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other).cpu() for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k].cpu(), other[k].cpu()) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)