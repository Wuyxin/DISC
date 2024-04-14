import os
import copy
import numpy as np

import torch
import torch.nn.functional as F
from disc.utils.loss import LossComputer
from disc.utils.mixup import mix_forward
from disc.utils.tools import set_required_grad, ParamDict


# For other baselines except for LISA
def run_epoch(
    epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
    is_training, show_progress=False, log_every=50, scheduler=None, n_classes=None
    ):

    if is_training:
        model.train()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader
    if args.jtt:
        erm_model = torch.load(args.erm_model_path).cuda()
    
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.cuda() for t in batch)
            x, y, g = batch[0], batch[1], batch[2]
            y_onehot = None
            if args.mix_up and (is_training):
                if args.dataset == 'CIFAR10':
                    y_onehot = torch.zeros(len(y), 10).cuda()
                    y_onehot = y_onehot.scatter_(1, y.unsqueeze(1), 1)
                else:
                    y_onehot = batch[3]

                idx = torch.randperm(x.size(0))
                if args.cut_mix:
                    x, y_onehot = cut_mix_up(args, x, x[idx], y_onehot, y_onehot[idx])
                    g = torch.zeros(len(x)).to(x.device)
                    y = torch.cat([y,y])
                else:
                    x, y_onehot = mix_up(args, x, x[idx], y_onehot, y_onehot[idx])

            if args.ibirm or args.coral:
                from models import NetBottom, NetTop
                features = NetBottom(args.model, model)(x)
                outputs = NetTop(model)(features)
            elif args.fish and is_training:
                model = model.cpu()
                model_inner = copy.deepcopy(model).cuda()
                set_required_grad(model_inner, True)
                optimizer_inner = get_optimizer(args, model_inner)
            else:
                outputs = model(x)
            
            if args.jtt:
                erm_model_outputs = erm_model(x)
                right_set = erm_model_outputs.argmax(dim=-1).view(-1) == y.view(-1)
            if args.irm or args.ibirm:
                dummy_w = torch.tensor(1.).to(x.device).requires_grad_()
                outputs = outputs * dummy_w
                    
            if args.fish and is_training:
                groups = torch.unique(g)
                loss_main = 0
                for group in groups:
                    if torch.sum(g==group) < 2: continue
                    outputs = model_inner(x[g==group])
                    loss = loss_computer.loss(
                        outputs, y[g==group], g[g==group], is_training, 
                        mix_up=args.mix_up, y_onehot=y_onehot
                        )
                    optimizer_inner.zero_grad()
                    loss_main += loss
                    loss.backward()
                    optimizer_inner.step()
                
                model.load_state_dict(
                    fish_step(meta_weights=model.state_dict(),
                              inner_weights=model_inner.state_dict(),
                              meta_lr=args.meta_lr)
                              )
                model_inner = model_inner.cpu()
                del model_inner
                torch.cuda.empty_cache()
            else:
                loss = loss_computer.loss(
                    outputs, y, g, is_training, mix_up=args.mix_up, y_onehot=y_onehot, 
                    return_group_loss=args.irm or args.rex or args.ibirm or args.jtt
                    )
            if args.irm or args.ibirm or args.rex or args.jtt:
                loss_main, group_losses = loss
                if args.jtt:
                    loss_main = group_losses[right_set].mean() + \
                        args.jtt_upweight * group_losses[~right_set].mean()
            elif not args.fish:
                loss_main = loss
            
            if (args.irm or args.ibirm) and is_training:
                unique_groups, group_indices, _ = split_into_groups(g)
                n_groups_per_batch = unique_groups.numel()
                penalty = 0.
                for i_group in range(n_groups_per_batch):
                    # pdb.set_trace()
                    penalty += irm_penalty(group_losses[group_indices[i_group]], dummy_w)
                if n_groups_per_batch > 1:
                    penalty /= n_groups_per_batch
                loss_main += penalty * args.irm_penalty
            
            if args.ibirm:
                var_loss = features.var(dim=0).mean()
                loss_main += args.ibirm_penalty * var_loss

            if args.rex and is_training:
                unique_groups, group_indices, _ = split_into_groups(g)
                n_groups_per_batch = unique_groups.numel()
                loss_list = []
                for i_group in range(n_groups_per_batch):
                    loss_list.append(group_losses[group_indices[i_group]].mean())
                loss_list = torch.stack(loss_list)
                loss_var = torch.var(loss_list)
                loss_main += loss_var * args.rex_penalty

            if args.coral and is_training:
                unique_groups, group_indices, _ = split_into_groups(g)
                n_groups_per_batch = unique_groups.numel()
                penalty = torch.zeros(1, device=outputs.device)
                for i_group in range(n_groups_per_batch):
                    for j_group in range(i_group + 1, n_groups_per_batch):
                        penalty += coral_penalty(features[group_indices[i_group]], features[group_indices[j_group]])
                if n_groups_per_batch > 1:
                    penalty /= (n_groups_per_batch * (n_groups_per_batch - 1) / 2)  # get the mean penalty

                loss_main += penalty[0] * 0.1

            if (not args.fish) and is_training:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args, is_training))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            model = model.cuda()
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args, is_training))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
    if scheduler is not None:                        
        scheduler.step()
                

def split_into_groups(g):
    """
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


def coral_penalty(x, y):
    if x.dim() > 2:
        # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
        # we flatten to Tensors of size (*, feature dimensionality)
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1 + 1e-5)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1 + 1e-5)
    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()
    return mean_diff + cova_diff


def irm_penalty(losses, scale):
    grad_1 = torch.autograd.grad(losses[0::2].mean(), [scale], create_graph=True)[0]
    grad_2 = torch.autograd.grad(losses[1::2].mean(), [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    return result


def fish_step(meta_weights, inner_weights, meta_lr):
    meta_weights, weights = ParamDict(meta_weights), ParamDict(inner_weights)
    meta_weights += meta_lr * sum([weights - meta_weights], 0 * meta_weights)
    return meta_weights


# LISA implementation
def run_epoch_mix_every_batch(
    epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
    is_training, show_progress=False, log_every=50, scheduler=None, count=0):
    assert is_training
    model.train()
    length = []
    data_iter = {}
    if "all" in loader: len_loader = len(loader) - 1
    else: len_loader = len(loader)

    if len_loader <= 10:
        for i in range(len_loader):
            length.append(len(loader[list(loader.keys())[i]]))
            data_iter[i] = iter(loader[list(loader.keys())[i]])
    else:
        for i in range(len_loader):
            length.append(len(loader[list(loader.keys())[i]]))

        group_loaders_idxes = np.random.permutation(len_loader)
        selected_data_iter_idxes = group_loaders_idxes[:4]
        count_group = 4
        for i, idx in enumerate(selected_data_iter_idxes):
            data_iter[i] = iter(loader[list(loader.keys())[idx]])

    len_dataloader = np.min(length)
    if show_progress:
        dataloader_iter = tqdm(range(len_dataloader))
    else:
        dataloader_iter = range(len_dataloader)

    for batch_idx, it in enumerate(dataloader_iter):
        x, y, g, y_onehot = [], [], [], []
        if len_loader <= 4:
            selected_data_iter_idxes = np.arange(len_loader)
        else:
            selected_data_iter_idxes = np.random.choice(np.arange(len_loader), 4, replace=False)
        for i in selected_data_iter_idxes:
            try:
                tmp_x, tmp_y, tmp_g, tmp_y_onehot, _ = data_iter[i].next()
            except:
                data_iter[i] = iter(loader[list(loader.keys())[i]])
                tmp_x, tmp_y, tmp_g, tmp_y_onehot, _ = data_iter[i].next()

            x.append(tmp_x)
            y.append(tmp_y)
            g.append(torch.ones(len(tmp_y)) * i)
            y_onehot.append(tmp_y_onehot)
        outputs, all_y, all_group, all_mix_y = mix_forward(args=args,
                                                           group_len=len(data_iter),
                                                           x=x, y=y, g=g, y_onehot=y_onehot, model=model)
        loss_main = loss_computer.loss(outputs, all_y.cuda(), all_group.cuda(), is_training,
                                       mix_up=args.lisa_mix_up, y_onehot=all_mix_y.cuda())
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()
        if (count+1) % log_every == 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            loss_computer.reset_stats()
        count+=1
    return count