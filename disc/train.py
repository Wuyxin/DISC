import os
import os.path as osp
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from disc.utils.loss import LossComputer
from disc.utils.cluster import cluter_assignment
from disc.utils.tools import CSVLogger, set_required_grad, get_optimizer, get_scheduler

from disc.models import NetBottom, NetTop
from disc.run_epoch_disc import run_epoch_disc
from disc.run_epoch_baselines import run_epoch, run_epoch_mix_every_batch

from disc.dataset.folds import Subset
from disc.dataset.dro_dataset import DRODataset
from disc.concept_utils.concept_dataset import ConceptDataset
from disc.concept_utils.filter_concepts import filter_relevant_concepts


def train(
    args, model, criterion, dataset, logger, 
    csv_loggers, n_classes, epoch_offset
    ):      
    if args.disc:
        erm_model = torch.load(args.erm_path)
        # Filter relevent concepts to reduce computational cost
        relevant_concepts = filter_relevant_concepts(args, erm_model, dataset['train_data'])
        # Construct datasets of concept images
        concept_data = {}
        print('Processing concept images in each class..')
        for c in tqdm(range(args.n_classes)):
            concept_names, concept_probs = relevant_concepts[c]
            concept_data[c] = ConceptDataset(
                args, root_dir=args.concept_img_folder,
                concept_names=concept_names, 
                concept_probs=concept_probs,
                model_type=args.model,
                augment_data=args.augment_data
                )
        print('Done!')
        all_concept_names = [item for _list in \
            [concept_data[c].concept_names for c in range(args.n_classes)] for item in _list]
        # Clustering
        cluster_dict = cluter_assignment(args, dataset['train_data'], erm_model, logger)
        sens_csv_logger = CSVLogger(args, osp.join(args.log_dir, f'concept_sens.csv'), all_concept_names)
        
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
    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
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
                env_loaders = []
                perm = [np.random.permutation(args.n_clusters) for _ in range(args.n_classes)]
                perm = np.stack(perm, axis=0)
                # Here each row of `perm` is a permutation
                # E.g., when `n_classes` = 2 and `n_clusters` = 3
                # perm = np.array([[0, 1, 2], 
                #                  [2, 1, 0]])
                for i in range(args.n_clusters):
                    # Each student use each column in perm to construct the data
                    # E.g., if i = 0, subset_idx = the indices of the instances in
                    # 0-th cluster of class 0 and 2-nd cluster of class 1
                    subset_idx = [item \
                                for _list in [
                                    cluster_dict[c][l] for (c, l) in zip(range(args.n_classes), perm[:, i])] \
                                for item in _list]
                    # Construct subset loader based on clustering 
                    env_data = DRODataset(
                        Subset(dataset['train_data'], subset_idx), 
                        process_item_fn=None, n_groups=args.n_groups,
                        n_classes=dataset['train_data'].n_classes, 
                        group_str_fn=dataset['train_data'].group_str
                        )
                    env_loader = env_data.get_loader(train=True, reweight_groups=False, **args.loader_kwargs)
                    env_loaders.append(env_loader)

            run_epoch_disc(
                epoch, model, optimizer,
                dataset['train_loader'],
                env_loaders,
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

         # Scheduler step to update lr at the end of epoch
        if args.scheduler:
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            try:
                scheduler.step()
            except:
                scheduler.step(val_loss)

        # Save models
        if epoch % args.save_step == 0:
            torch.save(model, osp.join(args.log_dir, '%d_model.pth' % epoch))
        if args.save_last:
            torch.save(model, osp.join(args.log_dir, 'last_model.pth'))
        if args.save_best:
            if args.dataset == 'ISIC':
                curr_val_perf = val_loss_computer.roc_auc
                logger.write(f'Current validation ROCAUC: {curr_val_perf}\n')
            else:
                curr_val_perf = val_loss_computer.worst_group_acc
                logger.write(f'Current validation accuracy: {curr_val_perf}\n')
            if curr_val_perf > best_val_perf:
                best_val_perf = curr_val_perf
                torch.save(model, osp.join(args.log_dir, 'best_model.pth'))
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
