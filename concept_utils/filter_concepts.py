import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from datetime import datetime
from models import NetBottom, NetTop
from concept_utils.concept_bank import learn_concept_bank, ConceptBank
from concept_utils.cce_utils import conceptual_counterfactual


def filter_relevant_concepts(
    args, model, dataset, temperature=1.0
    ):
    '''
    Filtering relevant concepts in a dataset
    Args: 
        args : Arguments, see args.py
        model (nn.Module): The model in the current training epoch. 
        dataset (Dataset): The training dataset
        temperature (float, optional): A hyperparameter used to control the selection
    Return:
        concept_relevance (dict, {class_0: (concept_names, concept_probs), ...)}),
                          where concept_probs indicate the concept importance
    
    Motivation: 
        (1) A subset of concepts in a category may not be relevant in the dataset.
        (2) Large concept bank requires high computational cost. 
        See the discussion section in the paper.

    '''
    print('Filtering relevant concepts:')
    # Save the filtered results to avoid recomputation on the dataset
    prefix = f'{args.concept_categories}-n_concept_imgs={args.n_concept_imgs}'
    if args.dataset == 'ISIC':
        prefix += f'-trapset_id={args.seed}'
    filtered_dir = osp.join(args.log_dir, '..', 'filtered', prefix)
    os.makedirs(filtered_dir, exist_ok=True)
    erm_bank_path = osp.join(filtered_dir, f'concept_bank.pt')
    relevance_path = osp.join(filtered_dir, f'concept_relevance.pt')
    log_file_path = osp.join(filtered_dir, f'info.txt')

    # Learn a concept bank based on ERM
    if not osp.exists(erm_bank_path):
        concept_bank = learn_concept_bank(
            args, model, concept_categories=args.concept_categories
            )
        concept_bank.save(erm_bank_path)
    concept_bank = ConceptBank.load(erm_bank_path)
    
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    concepts = np.array(concept_bank.concept_info.concept_names)
    backbone, model_top = NetBottom(model), NetTop(model)

    cnt = 0
    t1 = datetime.now()
    # --------------------------------------------------------------------
    # Roughly filter concepts by emsembling counterfactual explanations
    # For an instance (x_i, y_i), 
    # --------------------------------------------------------------------
    # If model prediction y == y_i (correct)
    # then we generate x' = x + a^T * c so that the prediction y' != y_i, 
    # and a^T will be a positive concept score array on class y'
    # and a negative concept score array on class y_i;
    # --------------------------------------------------------------------
    # If model prediction y != y_i (mistaken)
    # then we generate x' = x + a^T * c so that the prediction y' = y_i, 
    # and a^T will be a positive concept score array on class y_i
    # --------------------------------------------------------------------
    if not osp.exists(relevance_path):
        cf_tensors = {label: {'pos': [], 'neg': []} for label in range(args.n_classes)}
        loader = dataset.get_loader(train=False, batch_size=1, reweight_groups=None)
        for batch in tqdm(loader):
            x, y = batch[0].cuda(), batch[1].cuda().item()
            embedding = backbone(x)
            prediction = model_top(embedding)
            if prediction.argmax(-1).item() == y:
                for cf_label in range(args.n_classes):
                    if cf_label == y: 
                        continue
                    y_tensor = cf_label * torch.ones(1, dtype=torch.long)
                    cf = conceptual_counterfactual(
                        embedding, y_tensor.to('cuda'), 
                        concept_bank, model_top
                        )
                    if cf.success:
                        cnt += 1
                        cf_tensor = torch.Tensor([[cf.concept_scores[c] for c in concepts]])
                        cf_tensors[cf_label]['pos'].append(cf_tensor.detach().cpu())
                        cf_tensors[y]['neg'].append(cf_tensor.detach().cpu())
            else:
                y_tensor = y * torch.ones(1, dtype=torch.long)
                cf = conceptual_counterfactual(
                    embedding, y_tensor.to('cuda'), 
                    concept_bank, model_top
                    )
                if cf.success:
                    cnt += 1
                    cf_tensor = torch.Tensor([[cf.concept_scores[c] for c in concepts]])
                    cf_tensors[y]['pos'].append(cf_tensor.detach().cpu())
            if cnt > 100 * args.n_classes and \
               ((datetime.now() - t1).seconds > 300): 
                break

        concept_relevance = {}
        for y in range(args.n_classes):
            assert len(cf_tensors[y]['pos']) + len(cf_tensors[y]['neg']) > 0
            try: sim_pos = torch.stack(cf_tensors[y]['pos'], dim=0).mean(dim=0).squeeze(0)
            except: sim_pos = torch.ones(len(concepts))
            try: sim_neg = torch.stack(cf_tensors[y]['neg'], dim=0).mean(dim=0).squeeze(0)
            except: sim_neg = torch.zeros(len(concepts))
            sim = sim_pos - sim_neg
            sim = torch.exp(sim / temperature)
            sim = sim / sim.sum()
            seq = sim.argsort(descending=True)
            sorted_sim = sim[seq]
            concept_relevance[y] = (concepts[seq], sorted_sim)
            with open(log_file_path, 'a') as f:
                f.write(f'Class({y}): \n')
                f.write('# Pos: {}, # Neg: {} \n'.format(len(cf_tensors[y]['pos']), len(cf_tensors[y]['neg'])))
                f.write(f'{[(concepts[seq][idx], round(sorted_sim[idx].item(), 4)) for idx in range(len(concepts))]}\n')
            torch.save(concept_relevance, relevance_path)

    concept_relevance = torch.load(relevance_path)
    for y in range(args.n_classes):
        names, prob = concept_relevance[y]
        # Take the concepts that contribute 50% in total
        topk = min(20, torch.arange(len(prob))[torch.cumsum(prob, dim=0) > 0.5][0])
        concept_relevance[y] = (names[:topk], prob[:topk])
    return concept_relevance