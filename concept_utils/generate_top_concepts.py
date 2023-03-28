import torch
import os.path as osp
import numpy as np
from tqdm import tqdm
from datetime import datetime
from models import NetBottom, NetTop
from concept_utils.cce_utils import conceptual_counterfactual


'''
Given a concept bank containing concept actiave vectors, 
filter important concepts
'''
def generate_top_concepts(
    model, loader, concept_bank, log_dir, 
    resume_path=None, save=True, temperature=1.0
    ):
    print('Generating top concepts')
    cf_tensors_path = osp.join(log_dir, 'cf_tensors.pt') if resume_path is None else resume_path
    log_file_path = osp.join(log_dir, 'concept_log')
    concepts = np.array(concept_bank.concept_info.concept_names)
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    backbone, model_top = NetBottom(model), NetTop(model)
    C = model_top.out_features 
    cnt = 0
    if not osp.exists(cf_tensors_path):
        t1 = datetime.now()
        cf_tensors = {label: {'pos': [], 'neg': []} for label in range(C)}
        for batch in loader:
            x, y = batch[0].cuda(), batch[1].cuda().item()
            embedding = backbone(x)
            prediction = model_top(embedding)
            if prediction.argmax(-1).item() == y:
                for cf_label in range(C):
                    if cf_label == y: continue
                    y_tensor = cf_label * torch.ones(1, dtype=torch.long)
                    explanation = conceptual_counterfactual(embedding, y_tensor.to('cuda'), 
                                                            concept_bank, model_top)
                    if explanation.success:
                        cnt += 1
                        explanation_tensor = torch.Tensor([[explanation.concept_scores[c] for c in concepts]])
                        cf_tensors[cf_label]['pos'].append(explanation_tensor.detach().cpu())
                        cf_tensors[y]['neg'].append(explanation_tensor.detach().cpu())
            else:
                y_tensor = y * torch.ones(1, dtype=torch.long)
                explanation = conceptual_counterfactual(embedding, y_tensor.to('cuda'), 
                                                        concept_bank, model_top)
                if explanation.success:
                    cnt += 1
                    explanation_tensor = torch.Tensor([[explanation.concept_scores[c] for c in concepts]])
                    cf_tensors[y]['pos'].append(explanation_tensor.detach().cpu())
            t2 = datetime.now()
            if cnt > 200 and ((t2-t1).seconds > 600): break
        if save:
            torch.save(cf_tensors, cf_tensors_path)
    else:
        cf_tensors = torch.load(cf_tensors_path)

    sorted_concepts = {}
    for y in range(C):
        try: sim_pos = torch.stack(cf_tensors[y]['pos'], dim=0).mean(dim=0).squeeze(0)
        except: sim_pos = torch.ones(len(concepts))
        
        try: sim_neg = torch.stack(cf_tensors[y]['neg'], dim=0).mean(dim=0).squeeze(0)
        except: sim_neg = torch.zeros(len(concepts))

        sim = sim_pos - sim_neg
        sim = torch.exp(sim / temperature)
        sim = sim / sim.sum()
        
        seq = sim.argsort(descending=True)
        sorted_sim = sim[seq]
        sorted_concepts[y] = (concepts[seq], sorted_sim)
        
        with open(log_file_path, 'a') as f:
            f.write(f'class({y}): \n')
            f.write('#pos: {}, #neg: {} \n'.format(len(cf_tensors[y]['pos']), len(cf_tensors[y]['neg'])))
            f.write(f'{[(concepts[seq[idx]], round(sorted_sim[idx].item(), 4)) for idx in range(len(seq))]}\n')
    print(f'Done! Sved to {log_file_path}')
    return sorted_concepts