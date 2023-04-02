import torch
import random
import pickle

import os
import json
import os.path as osp
from tqdm import tqdm
from glob import glob
import numpy as np


from concept_utils.cav_utils import ListDataset, get_cavs
from models import NetBottom, NetTop
from dataset.transform import transform_dict
from itertools import product
from collections import defaultdict
from torch.multiprocessing import Manager, Pool, cpu_count

    
class ConceptBank:
    def __init__(self, concept_dict, device):
        self.concept_dict = concept_dict
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    try:
                        all_margin_info[key].append(np.array(value).reshape(1, 1))
                    except:
                        all_margin_info[key].append(value)
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(
                val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        self.concept_info.normalized_concepts = self.concept_info.margin_info.max * \
            self.concept_info.vectors / self.concept_info.norms

    def __getattr__(self, item):
        return self.concept_info[item]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.concept_dict, f)

    @staticmethod
    def load(path):
        return ConceptBank(pickle.load(open(path, "rb")), device='cuda')


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def process_func(args):
    transform, img_paths = args
    return ListDataset(img_paths, preprocess=transform)


def learn_concept_bank(
    args, model, concept_names=None, 
    concept_categories=None, processed_data=None
    ):
    '''
    Learn a concept bank with concepts either defined in `concept_names` or `concept_categories`
    Args: 
        args : Arguments, see args.py
        model (nn.Module): The model in the current training epoch. 
        concept_names (list([str,str,...])): The concept names in the concept bank
        concept_categories (list([str,str,...])): The concept_categories in the concept bank
        processed_data ({class_0: ConceptDataset, ...}, optional): 
            The processed concept data in each class, if already being processed
    Return:
        concept_bank (ConceptBank): A bank containing CAVs
    '''
    assert concept_names is not None or concept_categories is not None
    model.eval()
    model = model.to('cuda')
    backbone, top = NetBottom(args.model, model), NetTop(model)

    # Map concept categories to concept names
    if concept_categories is not None:
        if "everything" in concept_categories:
            concept_names = os.listdir(args.concept_img_folder)
            concept_names.remove('metadata.json')
        else:
            metadata = json.load(open(osp.join(args.concept_img_folder, 'metadata.json')))
            concept_names = []
            for key in concept_categories.split('-'):
                concept_names.extend(metadata[key])
            concept_names = list(set(concept_names))
    
    if processed_data is None:
        print(f'  Preprocess {len(concept_names)} concepts...')
        concept_names = tqdm(concept_names)

    # Get pos images
    concept_imgs = {}
    for concept in tqdm(concept_names):
        concept_imgs[concept] = {'pos': [], 'neg': []}
        if processed_data is None:
            pos_imgs = glob(osp.join(args.concept_img_folder, concept, "positives", "*"))
            pos_imgs = pos_imgs[:args.n_concept_imgs]
            if processed_data is None:
                n_cpu = 4
                n_pos_imgs = len(pos_imgs)
                n_imgs_per_pool = n_pos_imgs // n_cpu
                split_imgs = [pos_imgs[i * n_imgs_per_pool: (i + 1) * n_imgs_per_pool] for i in range(n_cpu)]
                if n_cpu * n_imgs_per_pool < n_pos_imgs:
                    split_imgs.append(pos_imgs[n_cpu * n_imgs_per_pool:])
                # Process positive images by multiprocessing, which is
                # about 3~5x more efficient than the sequential processing
                pool = Pool(processes=n_cpu)
                transform = transform_dict[args.dataset]
                transform = transform(args.model, train=False, augment_data=False)
                l = pool.map(process_func, list(product([transform], split_imgs)))
                concept_imgs[concept]['pos'] = [img for sublist in l for img in sublist]
                assert len(concept_imgs[concept]['pos']) == n_pos_imgs
        else:
            for c in range(args.n_classes):
                concept_imgs[concept]['pos'] = processed_data[c].get_concept_data(concept) 
                if len(concept_imgs[concept]['pos']) > 0: break

    # Construct negative images by sampling
    for concept in concept_names:
        M = int(max(len(concept_imgs[concept]['pos']) // len(concept_names), 0)) + 1
        for neg_concept in concept_names:
            if neg_concept == concept: continue
            indices = np.random.choice(len(concept_imgs[neg_concept]['pos']), M)
            neg_samples = [concept_imgs[neg_concept]['pos'][index] for index in indices]
            concept_imgs[concept]['neg'].extend(neg_samples)

    concept_dict = {}
    print('  Learning CAVs...')
    for concept in concept_names:
        pos_loader = torch.utils.data.DataLoader(concept_imgs[concept]['pos'], batch_size=100, shuffle=False)
        neg_loader = torch.utils.data.DataLoader(concept_imgs[concept]['neg'], batch_size=100, shuffle=False)
        n_train = int(0.8 * args.n_concept_imgs)
        cav_info = get_cavs(
            pos_loader, neg_loader, backbone, 
            n_train, c=args.c_svm, device='cuda'
            )
        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        print(f"  {concept}: Training Accuracy: {cav_info[args.c_svm][1]:.2f} "
              f" Validation Accuracy: {cav_info[args.c_svm][2]:.2f}")
        concept_dict[concept] = cav_info[args.c_svm]
    concept_bank = ConceptBank(concept_dict, device='cuda')
    return concept_bank
