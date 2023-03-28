import torch
import random

import os
import os.path as osp
from tqdm import tqdm
from glob import glob

from models import NetBottom, NetTop
from dataset.transform import transform_dict
from concept_utils.generate_top_concepts import generate_top_concepts


def learn_concept_bank(args, model, concept_names):
    '''
    Learn a concept bank 
    Args: 
        args : Arguments, see args.py
        model (nn.Module): The model in the current training epoch. 
        concept_names (list([str,str,...])): The concepts in the concept bank
    '''
    model.eval()
    model = model.to('cuda')
    backbone, top = NetBottom(model), NetTop(model)
    backbone = backbone.eval()
    
    transform = transform_dict(args.dataset) 
    preprocess = transform(args.model, train=False, augment_data=False)
    
    concept_dict = {}
    for concept in concept_names:
        print(concept)
        neg_imgs = []
        pos_imgs = glob(osp.join(args.concept_img_folder, concept, "positives", "*"))
        pos_imgs = pos_imgs[:args.n_concept_imgs]
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
        cav_info = get_cavs(pos_loader, neg_loader, backbone, 
                            n_samples=int(0.8 * max(len(pos_set), len(neg_set))), 
                            C=[args.c_svm], device='cuda')
        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        print(f"{concept} with C={args.c_svm}: \
                Training Accuracy: {cav_info[args.c_svm][1]:.2f}, \
                Validation Accuracy: {cav_info[args.c_svm][2]:.2f}")
        concept_dict[concept] = cav_info[args.c_svm]
    concept_bank = ConceptBank(concept_dict, device='cuda')
    return concept_bank


def load_concept_bank(concept_path, device):
    concept_bank = ConceptBank(pickle.load(open(concept_path, "rb")), device=device)
    concepts = concept_bank.concept_info.concept_names
    print("Number of concepts: ", len(concepts))
    return concept_bank


class ConceptBank:
    def __init__(self, concept_dict, device):
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



class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
