import os
import pickle
import argparse
import torch
import numpy as np
import os.path as osp

from glob import glob

import json
from dataset.transform import transform_dict
from models import NetBottom, NetTop
from concept_utils.concept_bank import learn_concept_bank, ListDataset
from config import n_clusters

import random
import warnings
warnings.filterwarnings("ignore")

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_dir", type=str, default=f'broden_concepts',
                    help="Directory containing concept images. See below for a detailed description.")

    parser.add_argument("--dataset", type=str, help="The dataset to use")
    parser.add_argument("--model_path", type=str, help="Model path.")
    parser.add_argument("--model_type", default='resnet50', type=str, help="Model type")
    parser.add_argument("--out_dir", default='examples/', type=str, help="Where to save the concept bank.")
    parser.add_argument("--resize_dim", default=224, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--C", nargs="+", default=[0.1], type=float, 
                        help="Regularization parameter for SVMs. Can specify multiple values.")
    
    parser.add_argument("--n_samples", default=75, type=int, 
                        help="Number of pairs of positive/negative samples used to train SVMs.")
    parser.add_argument("--concept_sets", type=str, default='everything', 
                        help="Specify concept categories and use '-' as delimiter or use 'everything' to include all."
                             "Example: Color-Texture-Nature or everything")
    return parser.parse_args()


def main(args):
    args.concept_sets = args.concept_sets.split('-')
    np.random.seed(args.seed)
    if args.out_dir == 'none':
        args.out_dir = osp.dirname(args.model_path)
    model_name = args.model_path.split('/')[-1].split('.')[0]

    # Concept images are expected in the following format:
    # args.concept_dir/concept_name/positives/1.jpg, args.concept_dir/concept_name/positives/2.jpg, ...
    # args.concept_dir/concept_name/negatives/1.jpg, args.concept_dir/concept_name/negatives/2.jpg, ...
    preprocess = transform_dict[args.dataset]('resnet50', train=False, augment_data=False)
    if "everything" in args.concept_sets:
        concept_names = os.listdir(args.concept_dir)
        concept_names.remove('metadata.json')
    else:
        metadata = json.load(open(osp.join(args.concept_dir, 'metadata.json')))
        concept_names = []
        for key in args.concept_sets:
            concept_names.extend(metadata[key])
        concept_names = list(set(concept_names))
    # Get the backbone
    model = torch.load(args.model_path)
    if isinstance(model, dict):
        print('Reading dict ', list(model.keys()))
        model = model['net'].module
    model = model.to(args.device)
    backbone, top = NetBottom(model), NetTop(model)
    backbone = backbone.eval()

    print(f"Attempting to learn {len(concept_names)} concepts.")
    concept_lib = {C: {} for C in args.C}
    for concept in concept_names:
        pos_ims = glob(os.path.join(args.concept_dir, concept, "positives", "*"))
        pos_ims = pos_ims[:args.n_samples]
        neg_ims = []
        M = int(max(len(pos_ims) / len(concept_names), 3))
        for neg_concept in concept_names:
            if neg_concept == concept: continue
            ims = glob(os.path.join(args.concept_dir, neg_concept, "positives", "*"))
            random.shuffle(ims)
            neg_ims = neg_ims + ims[:M]
        
        pos_dataset = ListDataset(pos_ims, preprocess=preprocess)
        neg_dataset = ListDataset(neg_ims, preprocess=preprocess)
        pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, 
                                    n_samples=args.n_samples, 
                                    C=args.C, device=args.device)
        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        for C in args.C:
            concept_lib[C][concept] = cav_info[C]
            print(f"{concept} with C={C}: Training Accuracy: {cav_info[C][1]:.2f}, Validation Accuracy: {cav_info[C][2]:.2f}")
    
    # Save CAV results 
    os.makedirs(args.out_dir, exist_ok=True)
    for C in concept_lib.keys():
        lib_path = os.path.join(args.out_dir, f"{model_name}_{C}_{args.n_samples}.pkl")
        with open(lib_path, "wb") as f:
            pickle.dump(concept_lib[C], f)
        print(f"Saved to: {lib_path}")        
    

if __name__ == "__main__":
    args = config()
    main(args)
