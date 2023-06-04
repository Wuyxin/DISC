import numpy as np
from tqdm import tqdm
import torch

from disc.concept_utils.concept_bank import learn_concept_bank
from disc.concept_utils.get_cts import run_one_step_and_get_cts
from disc.utils.mixup import mix_up
from disc.utils.tools import set_required_grad
from disc.models import NetBottom, NetTop


# DISC training
def run_epoch_disc(
    epoch, model, optimizer, loader, env_loaders, 
    loss_computer, logger, csv_logger, sens_csv_logger, 
    args, is_training, concept_data, 
    show_progress=False, log_every=50, scheduler=None
    ):
    
    assert is_training
    all_concept_names = [item for _list in \
        [concept_data[c].concept_names for c in range(args.n_classes)] for item in _list]
    
    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    model.train()
    with torch.set_grad_enabled(is_training):
        ####################################
        ##             Discover           ##
        ####################################
        # Compute concept tendency score (CTS)
        concept_bank = learn_concept_bank(
            args, model, 
            concept_names=all_concept_names, 
            processed_data=concept_data
            )
        top_concepts_res = []
        for i in range(args.n_clusters):      
            top_concepts = run_one_step_and_get_cts(
                args, model, 
                env_loaders[i], 
                loss_computer, 
                concept_bank=concept_bank
                )
            top_concepts_res.append(top_concepts)
            
        concept_dataloader, stats_dict = {}, {}
        logger.write('>' * 100 + '\n')

        for c in range(args.n_classes):
            concept_names = concept_data[c].concept_names
            sensitivity = {name: [] for name in concept_names}
            for top_concepts in top_concepts_res:
                names, probs = top_concepts
                for i in range(len(names)):
                    if names[i] in concept_names:
                        # Take the softmax output on class c
                        sensitivity[names[i]].append(probs[i][c])

            # Compute concept sensitivity
            concept_probs = torch.Tensor([torch.std(torch.Tensor(sensitivity[name])) for name in concept_names])
            concept_probs = concept_probs / concept_probs.sum()
            for idx, name in enumerate(concept_names):
                stats_dict[name] = concept_probs[idx].item()
                
            concept_dataloader[c] = concept_data[c].get_loader(concept_probs=concept_probs, **args.loader_kwargs)
            seq = concept_probs.argsort(descending=True)
            sorted_names = concept_names[seq]
            logger.write(f'Epoch={epoch} Class={c}: \n  concepts={sorted_names}\n  prob={concept_probs[seq]}\n')
        
        logger.write('<' * 100 + '\n')
        stats_dict['average'] = np.array(list(stats_dict.values())).mean()
        sens_csv_logger.log(epoch, stats_dict)
        sens_csv_logger.flush()
        
        ####################################
        ##               Cure             ##
        ####################################
        model = model.to('cuda')
        backbone, model_top = NetBottom(args.model, model), NetTop(model)
        set_required_grad(model, True)

        concept_data_iter = {}
        for batch_idx, batch in enumerate(prog_bar_loader):    
            if args.dataset == 'FMoW':
                class_enum = np.random.choice(args.n_classes, 3)
            else:
                class_enum = range(args.n_classes)
            for c in class_enum:
                concept_data_iter[c] = iter(concept_dataloader[c])

            batch = tuple(t.cuda() for t in batch)
            x, y, g = batch[0], batch[1], batch[2]
            y_onehot = None

            x_emb = backbone(x)
            outputs = model_top(x_emb)
            loss_main = loss_computer.loss(outputs, y, g, is_training, mix_up=False, y_onehot=y_onehot)
            
            for c in class_enum:
                bool_c = ~(y == c)
                tmp_x = concept_data_iter[c].next()
                tmp_x = tmp_x.cuda()
                if bool_c.float().sum() < 2: 
                    continue # avoid shape error
                mixed_x, mixed_y = mix_up(
                    args, x1=x[bool_c], x2=tmp_x, 
                    y1=y[bool_c], y2=y[bool_c], n_classes=1
                    )
                length = min(len(x[bool_c]), len(tmp_x))
                mixed_g = g[bool_c][:length]
                mixed_y = y[bool_c][:length].long()

                x_mix_emb_c = backbone(mixed_x.cuda())
                outputs = model_top(x_mix_emb_c)
                loss_main += loss_computer.loss(
                    outputs, mixed_y, mixed_g, 
                    is_training=is_training, 
                    mix_up=False, y_onehot=y_onehot
                    )
                    
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