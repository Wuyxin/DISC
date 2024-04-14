import os
import torch
import torch.nn.functional as F
import itertools
import numpy as np
import sklearn


class LossComputer:
    def __init__(
        self, args, criterion, is_robust, dataset, 
        alpha=None, gamma=0.1, adj=None, min_var_weight=0, 
        step_size=0.01, normalize_loss=False, btl=False, is_val=False
        ):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.args = args
        self.dataset = dataset
        self.n_groups = dataset.n_groups
        self.is_val = is_val
        print("Loss n groups:", self.n_groups)

        self.group_counts = dataset.group_counts().cuda()
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False, 
             mix_up=False, y_onehot=None, return_group_loss=False
             ):
        # compute per-sample and per-group losses
        if mix_up and is_training:
            per_sample_losses = - F.log_softmax(yhat, dim=-1) * y_onehot
            per_sample_losses = per_sample_losses.sum(-1)
            group_acc = 0.0
        else:
            per_sample_losses = self.criterion(yhat, y)
            group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)
        
        if self.args.dataset == 'ISIC':
            prob = F.softmax(yhat, dim=-1)[:,1].view(-1)
            self.probs.append(prob.detach().cpu())
            self.ys.append(y.detach().cpu())
        
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        # update historical losses
        if not mix_up:
            self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        if return_group_loss:
            return actual_loss, per_sample_losses
        else:
            return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialinpzed = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.
        self.roc_auc = 0.
        self.probs, self.ys = [], []

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None, is_training=True):
        stats_dict = {}
        accs = []
        
        if self.args.dataset == 'ISIC':
            self.probs = np.concatenate(self.probs)
            self.ys = np.concatenate(self.ys)
            self.roc_auc = sklearn.metrics.roc_auc_score(self.ys, self.probs)
            stats_dict['roc_auc'] = self.roc_auc
            
        for idx in range(self.n_groups):
            group_str = self.group_str(idx, is_training) if 'Meta' in self.args.dataset else self.group_str(idx)
            n = int(self.processed_data_counts[idx])
            if n == 0 or (self.args.dataset == 'FMoW' and "Other" in group_str):
                continue

            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()
            accs.append(self.avg_group_acc[idx].item())

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()
        stats_dict['worst_group_acc'] = np.min(accs) if len(accs) > 0 else 0
        self.worst_group_acc = stats_dict['worst_group_acc']
        differences = []
        for i, j in itertools.combinations(accs, 2):
            differences.append(abs(i - j))
        stats_dict['mean_differences'] = np.mean(differences)
        stats_dict['group_avg_acc'] = np.mean(accs)
        if args is not None and args.dataset == 'MetaDataset':
            stats_dict['F1-score'] = 1 / (1 / (stats_dict['avg_acc_group:0'] + 1e-5) + 1 / (stats_dict['avg_acc_group:1'] + 1e-5))

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return
        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.4f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.4f}  \n')
        if is_training and self.args.lisa_mix_up:
            return
            
        logger.write(f'Average acc: {self.avg_acc.item():.4f}  \n')
        for group_idx in range(self.n_groups):
            group_str = self.group_str(group_idx, is_training) if 'Meta' in self.args.dataset else self.group_str(group_idx)
            n = int(self.processed_data_counts[group_idx])
            # follow wilds implementation to skip `Other`
            if n == 0 or (self.args.dataset == 'FMoW' and "Other" in group_str):
                continue
            
            logger.write(
                f'  {group_str}  '
                f'[n = {n}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.4f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.4f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.4f}  '
                f'adv prob = {self.adv_probs[group_idx]:4f}   '
                f'acc = {self.avg_group_acc[group_idx]:.4f}\n')

        if self.args.dataset == 'ISIC':
            logger.write(f'roc auc = {self.roc_auc:4f}\n')
        logger.flush()
