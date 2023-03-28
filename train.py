import os
import copy
import numpy as np
from tqdm import tqdm
import datetime

import torch
import torch.nn.functional as F
from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import CSVLogger, set_required_grad, ParamDict, \
    get_optimizer, get_scheduler, get_model
from loss import LossComputer
from concept_utils.concept_dataset import ConceptDataset
from concept_utils.cluster import cluter_assignment
from concept_utils import load_concept, generate_top_concepts

from data.dro_dataset import DRODataset
from data.folds import Subset
from models import ResNetBottom, ResNetTop
from concept_utils.get_concept_scores import get_concept_bank
from concept_utils.grad import run_one_step_and_get_concept_score


device = torch.device("cuda")

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mix_up(args, x1, x2, y1, y2, n_classes=None):

    # y1, y2 should be one-hot label, which means the shape of y1 and y2 should be [bsz, n_classes]
    length = min(len(x1), len(x2))
    x1 = x1[:length]
    x2 = x2[:length]
    y1 = y1[:length]
    y2 = y2[:length]

    if n_classes is None:
        n_classes = y1.shape[1]
    else:
        n_classes = n_classes

    bsz = len(x1)
    l = np.random.beta(args.mix_alpha, args.mix_alpha, [bsz, 1])
    if len(x1.shape) == 4:
        l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
    else:
        l_x = np.tile(l, (1, *x1.shape[1:]))
    l_y = np.tile(l, [1, n_classes])

    # mixed_input = l * x + (1 - l) * x2
    mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
    mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2
    return mixed_x, mixed_y

def cut_mix_up(args, x1, x2, y1, y2):
    length = min(len(x1), len(x2))
    x1 = x1[:length]
    x2 = x2[:length]
    y1 = y1[:length]
    y2 = y2[:length]

    input = torch.cat([x1,x2])
    target = torch.cat([y1,y2])

    rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])

    lam = np.random.beta(args.alpha, args.alpha)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, lam*target_a + (1-lam)*target_b


def mix_forward(args, group_len, x, y, g, y_onehot, model):

    if len(x) == 4:
        # LISA for CUB, CMNIST, CelebA
        if np.random.rand() < args.mix_ratio:
            mix_type = 1
        else:
            mix_type = 2

        if mix_type == 1:
            # mix different A within the same feature Y
            mix_group_1 = [x[0], x[1], y_onehot[0], y_onehot[1]]
            mix_group_2 = [x[2], x[3], y_onehot[2], y_onehot[3]]
        elif mix_type == 2:
            # mix different Y within the same feature A
            mix_group_1 = [x[0], x[2], y_onehot[0], y_onehot[2]]
            mix_group_2 = [x[1], x[3], y_onehot[1], y_onehot[3]]

        if args.cut_mix:
            mixed_x_1, mixed_y_1 = cut_mix_up(args, mix_group_1[0], mix_group_1[1], mix_group_1[2],
                                              mix_group_1[3])
            mixed_x_2, mixed_y_2 = cut_mix_up(args, mix_group_2[0], mix_group_2[1], mix_group_2[2],
                                              mix_group_2[3])
        else:
            mixed_x_1, mixed_y_1 = mix_up(args, mix_group_1[0], mix_group_1[1], mix_group_1[2],
                                          mix_group_1[3])
            mixed_x_2, mixed_y_2 = mix_up(args, mix_group_2[0], mix_group_2[1], mix_group_2[2],
                                          mix_group_2[3])

        all_mix_x = [mixed_x_1, mixed_x_2]
        all_mix_y = [mixed_y_1, mixed_y_2]
        all_group = torch.ones(
            len(mixed_x_1) + len(mixed_x_2)) * 3  # all the mixed samples are set to be from group 3
        all_y = torch.ones(len(mixed_x_1) + len(mixed_x_2)).cuda()
        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)

    else:
        # MetaDataset group by label, the mixup should be performed within the label group.
        all_mix_x, all_mix_y, all_group, all_y = [], [], [], []
        for i in range(group_len):
            bsz = len(x[i])

            if args.cut_mix:
                mixed_x, mixed_y = cut_mix_up(args, x[i][: bsz // 2], x[i][bsz // 2:], y_onehot[i][:bsz // 2],
                                              y_onehot[i][bsz // 2:])
                all_group.append(g[i][:len(mixed_x)])
                all_y.append(y[i][:len(mixed_x)])
                assert len(mixed_x) == len(all_y[-1])
            else:
                mixed_x, mixed_y = mix_up(args, x[i][:bsz // 2], x[i][bsz // 2:],
                                          y_onehot[i][:bsz // 2], y_onehot[i][bsz // 2:])
                all_group.append(g[i][:len(mixed_x)])
                all_y.append(y[i][:len(mixed_x)])

            all_mix_x.append(mixed_x)
            all_mix_y.append(mixed_y)

        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)
        all_group = torch.cat(all_group)
        all_y = torch.cat(all_y)

    outputs = model(all_mix_x.cuda())
    return outputs, all_y, all_group, all_mix_y
 

# DISC training
def run_epoch_disc(
    epoch, model, optimizer, loader, stu_loaders, loss_computer, logger, csv_logger, sens_csv_logger, 
    args, is_training, concept_data, show_progress=False, log_every=50, scheduler=None, count=0,
    ):
    assert is_training
    
    model.train()
    C = model.fc.out_features
    all_concept_names = [item for _list in [concept_data[c].concept_names for c in range(C)] for item in _list]
    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader
        
    with torch.set_grad_enabled(is_training):
        ####################################
        #              Discover            #
        ####################################
        concept_bank = get_concept_bank(args, model, all_concept_names)
        top_concepts_res = []
        for i in range(args.n_students):      
            top_concepts = run_one_step_and_get_concept_score(
                args, model, stu_loaders[i], 
                loss_computer, concept_bank=concept_bank, is_training=True)
            top_concepts_res.append(top_concepts)
            
        # update concept_dataloader
        concept_dataloader, stats_dict = {}, {}
        for c in range(C):
            concept_names = concept_data[c].concept_names
            sensitivity = {name: [] for name in concept_names}
            for top_concepts in top_concepts_res:
                names, probs = top_concepts
                for i in range(len(names)):
                    if names[i] in concept_names:
                        # take the softmax output on class c
                        sensitivity[names[i]].append(probs[i][c])
            concept_probs = torch.Tensor([torch.std(torch.Tensor(sensitivity[name])) for name in concept_names])
            for idx, name in enumerate(concept_names):
                stats_dict[name] = concept_probs[idx].item()
            concept_dataloader[c] = concept_data[c].get_loader(concept_probs=concept_probs, **args.loader_kwargs)
            seq = concept_probs.argsort(descending=True)
            sorted_names = concept_names[seq]
            logger.write(f'\nEpoch {epoch}, Concepts: {sorted_names}, Prob: {concept_probs[seq]}\n')\
        stats_dict['average'] = np.array(list(stats_dict.values())).mean()
        sens_csv_logger.log(epoch, stats_dict)
        sens_csv_logger.flush()
        
        
        ####################################
        #                Cure              #
        ####################################
        model = model.to('cuda')
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        set_required_grad(model, True)
        model.train()
        concept_data_iter = {}
        for batch_idx, batch in enumerate(prog_bar_loader):    
            for c in range(C):
                concept_data_iter[c] = iter(concept_dataloader[c])
            batch = tuple(t.cuda() for t in batch)
            x, y = batch[0], batch[1]
            try: 
                g = batch[2]
            except: 
                g = torch.zeros(len(x)).long().cuda()
            y_onehot = None
            x_emb = backbone(x)
            outputs = model_top(x_emb)
            loss_main = loss_computer.loss(outputs, y, g, is_training, mix_up=False, y_onehot=y_onehot)
            for c in range(C):
                bool_c = ~(y == c)
                tmp_x = concept_data_iter[c].next()
                tmp_x = tmp_x.cuda()
                if bool_c.float().sum() < 2: # avoid shape error
                    continue
                mixed_x, mixed_y = mix_up(args, x1=x[bool_c], x2=tmp_x, y1=y[bool_c], y2=y[bool_c], n_classes=1)
                length = min(len(x[bool_c]), len(tmp_x))
                mixed_g = g[bool_c][:length]
                mixed_y = y[bool_c][:length].long()

                x_mix_emb_c = backbone(mixed_x.cuda())
                outputs = model_top(x_mix_emb_c)
                loss_main += loss_computer.loss(outputs, mixed_y, mixed_g, 
                                                is_training=is_training, 
                                                mix_up=False, y_onehot=y_onehot)
                    
            if is_training:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx + 1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
            
        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
                

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


# For other baselines
def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None, n_classes=None):
    """
    scheduler is only used inside this function if model is bert.
    """

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

            x = batch[0]
            y = batch[1]
            try: g = batch[2]
            except: g = torch.zeros(len(x)).long().cuda()
            y_onehot = None
            if 'bert' in args.model:
                outputs = model(x)
            else:
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
                    from models import ResNetBottom, ResNetTop
                    features = ResNetBottom(model)(x)
                    outputs = ResNetTop(model)(features)
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
                    loss = loss_computer.loss(outputs, y[g==group], g[g==group], is_training, mix_up=args.mix_up, y_onehot=y_onehot)
                    optimizer_inner.zero_grad()
                    loss_main += loss
                    loss.backward()
                    optimizer_inner.step()
                
                model.load_state_dict(fish_step(meta_weights=model.state_dict(),
                                      inner_weights=model_inner.state_dict(),
                                      meta_lr=args.meta_lr))
                model_inner = model_inner.cpu()
                del model_inner
                torch.cuda.empty_cache()
            else:
                loss = loss_computer.loss(outputs, y, g, is_training, mix_up=args.mix_up, y_onehot=y_onehot, return_group_loss=args.irm or args.rex or args.ibirm or args.jtt)

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
                if 'bert' in args.model:
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                # pdb.set_trace()
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            model = model.cuda()
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
    if scheduler is not None:                        
        scheduler.step()
                

# LISA implementation
def run_epoch_mix_every_batch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
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


#train
def train(args, model, criterion, dataset,
          logger, csv_loggers, n_classes, 
          epoch_offset, csv_name=None, exp_string=None):
          
    C = dataset['train_data'].n_classes
    if args.disc:
        # 1. Filter top concepts to reduce computational cost
        concept_bank = load_concept(args.concept_bank_path, 'cuda')
        normalized_concepts = concept_bank.normalized_concepts
        single_datum_loader = dataset['train_data'].get_loader(
            train=False, 
            batch_size=1, 
            reweight_groups=None
            )
        erm_model = torch.load(args.erm_path)
        top_concepts = generate_top_concepts(
            erm_model, single_datum_loader, 
            concept_bank, args.log_dir, 
            resume_path=args.owl_resume_path
            )
        # .. and construct dataset containing images of top concepts
        concept_data = {}
        for c in range(C):
            concept_names, concept_probs = top_concepts[c]
            concept_data[c] = ConceptDataset(
                args, root_dir=args.concept_img_folder,
                concept_names=concept_names[:args.topk], 
                concept_probs=concept_probs[:args.topk],
                split='positives',
                model_type=args.model,
                augment_data=args.augment_data
                )
        all_concept_names = [item for _list in [concept_data[c].concept_names for c in range(C)] for item in _list]
        # 2. Conduct clutering
        cluster_dict = cluter_assignment(args, dataset['train_data'], erm_model, logger)
        sens_csv_logger = CSVLogger(args, os.path.join(args.log_dir, f'{exp_string}_concept_sens.csv'), all_concept_names)
        
    # Process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    # Set up loss computer
    train_loss_computer = LossComputer(
        args,
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight
        )

    # Set up optimizer & scheduler
    t_total = args.n_epochs
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, t_total)
    set_required_grad(model, True)

    # Start training
    best_val_perf, count = 0, 1
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        t1 = datetime.datetime.now()
        if args.lisa_mix_up:
            count = run_epoch_mix_every_batch(
                            epoch, model, optimizer,
                            dataset['train_loader'],
                            train_loss_computer,
                            logger, csv_loggers['train'], args,
                            is_training=True,
                            show_progress=args.show_progress,
                            log_every=args.log_every,
                            scheduler=scheduler,
                            count=count
                            )
                            
        elif args.disc:
            # Permute clusters and update dataloaders
            if epoch % 2 == 0:
                stu_loaders = []
                perm = [np.random.permutation(args.n_students) for _ in range(C)]
                perm = np.stack(perm, axis=0)
                # Here each row of `perm` is a permutation
                # E.g., when `n_classes` = 2 and `n_students` = 3
                # perm = np.array([[0, 1, 2], 
                #                  [2, 1, 0]])
                for i in range(args.n_students):
                    # Each student use each column in perm to construct the data
                    # E.g., if i = 0, subset_idx = the indices of the instances in
                    # 0-th cluster of class 0 and 2-nd cluster of class 1
                    subset_idx = [item \
                                for _list in [
                                    cluster_dict[c][l] for (c, l) in zip(range(C), perm[:, i])] \
                                for item in _list]
                    # Construct subset loader based on clustering 
                    stu_data = DRODataset(
                        Subset(dataset['train_data'], subset_idx), 
                        process_item_fn=None, n_groups=args.n_groups,
                        n_classes=dataset['train_data'].n_classes, 
                        group_str_fn=dataset['train_data'].group_str
                        )
                    stu_loader = stu_data.get_loader(train=True, reweight_groups=False, **args.loader_kwargs)
                    stu_loaders.append(stu_loader)

            run_epoch_disc(
                epoch, model, optimizer,
                dataset['train_loader'],
                stu_loaders,
                train_loss_computer,
                logger, csv_loggers['train'], 
                sens_csv_logger, args,
                is_training=True,
                show_progress=args.show_progress,
                log_every=args.log_every,
                scheduler=scheduler,
                concept_data=concept_data
                )

        else:
            run_epoch(
                epoch, model, optimizer,
                dataset['train_loader'],
                train_loss_computer,
                logger, csv_loggers['train'], args,
                is_training=True,
                show_progress=args.show_progress,
                log_every=args.log_every,
                scheduler=scheduler
                )

        # Evaluate on validation set
        logger.write(f'\n Epoch {epoch}, Validation:\n')
        val_loss_computer = LossComputer(
            args,
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha,
            is_val=True
            )
        run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, csv_loggers['val'], args,
            is_training=False
            )

        # Evaluate on test set
        if dataset['test_data'] is not None:
            logger.write(f'\nEpoch {epoch}, Testing:\n')
            test_loss_computer = LossComputer(
                args,
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha
                )
            run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                logger, csv_loggers['test'], args,
                is_training=False
                )
        
        # Inspect learning rates
        t2 = datetime.datetime.now()
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)
                logger.write(f'Time Cost: {t2 - t1}\n')

        # Update learning rate
        if args.scheduler:
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            try:
                scheduler.step()
            except:
                scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        # Save models
        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))
        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))
        if args.save_best:
            if args.dataset == 'ISIC':
                curr_val_perf = val_loss_computer.roc_auc
                logger.write(f'Current validation ROCAUC: {curr_val_perf}\n')
            else:
                curr_val_perf = val_loss_computer.worst_group_acc
                logger.write(f'Current validation accuracy: {curr_val_perf}\n')
            if curr_val_perf > best_val_perf:
                best_val_perf = curr_val_perf
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')
        
        # Automatic adjust training loss computer
        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')
