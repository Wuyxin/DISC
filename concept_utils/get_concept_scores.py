import torch
import random

import os
import os.path as osp
from tqdm import tqdm
from glob import glob

from concept_utils import ConceptBank
from models import ResNetBottom, ResNetTop
from dataset.transform import transform_dict
from concept_utils.generate_top_concepts import generate_top_concepts
from concept_utils import learn_concept_bank, ListDataset


'''
Given specific concept names, compute their scores
'''
def get_concept_scores(args, model, loader, concept_names):
    concept_bank = get_concept_bank(args, model, concept_names)
    log_dir = osp.join(args.log_dir, 'student_concept')
    os.makedirs(log_dir, exist_ok=True)
    return generate_top_concepts(model, loader, concept_bank, 
                                 log_dir, save=False)
    
'''
Compute concept bank 
'''
def get_concept_bank(args, model, concept_names):
    model.eval()
    model = model.to('cuda')
    backbone, top = ResNetBottom(model), ResNetTop(model)
    backbone = backbone.eval()
    
    transform = transform_dict(args.dataset) 
    preprocess = transform(args.model, train=False, augment_data=False)
    
    concept_lib = {}
    for concept in concept_names:
        print(concept)
        pos_imgs = glob(osp.join(args.concept_img_folder, concept, "positives", "*"))
        pos_imgs = pos_imgs[:args.n_concept_imgs]
        neg_imgs = []
        M = int(max(len(pos_imgs) // len(concept_names), 3))
        for neg_concept in concept_names:
            if neg_concept == concept: continue
            ims = glob(osp.join(args.concept_img_folder, neg_concept, "positives", "*"))
            random.shuffle(ims)
            neg_imgs = neg_imgs + ims[:M]
        
        pos_set = ListDataset(pos_imgs, preprocess=preprocess)
        neg_set = ListDataset(neg_imgs, preprocess=preprocess)
        pos_loader = torch.utils.data.DataLoader(pos_set, batch_size=args.batch_size, shuffle=False)
        neg_loader = torch.utils.data.DataLoader(neg_set, batch_size=args.batch_size, shuffle=False)
        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, 
                                      n_samples=int(0.8 * max(len(pos_set), len(neg_set))), 
                                      C=[args.c_svm], device='cuda')
        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        print(f"{concept} with C={args.c_svm}: \
                Training Accuracy: {cav_info[args.c_svm][1]:.2f}, \
                Validation Accuracy: {cav_info[args.c_svm][2]:.2f}")
        concept_lib[concept] = cav_info[args.c_svm]
    concept_bank = ConceptBank(concept_lib, device='cuda')
    return concept_bank