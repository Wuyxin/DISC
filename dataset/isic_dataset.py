import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms

from models import model_attributes
from dataset.confounder_dataset import ConfounderDataset
from dataset.transform import get_transform_ISIC


class ISICDataset(ConfounderDataset):
    """
    ISIC dataset

    Args: 
        args : Arguments, see run_expt.py
        root_dir (str): Arguments, see run_expt.py
        target_name (str): Data label
        confounder_names (list): A list of confounders
        model_type (str, optional): Type of model on the dataset, see models.py
        augment_data (bool, optional): Whether to use data augmentation, e.g., RandomCrop
        mix_up (bool, optional): Whether to use mixup 
        group_id (int, optional): Select a subset of dataset with the group id
        id_val (bool, optional): Whether to use in-distribution validation data
        
    """
    def __init__(self, args, 
                 root_dir,
                 target_name, 
                 confounder_names=['hair'],
                 model_type=None,
                 augment_data=False,
                 mix_up=False,
                 group_id=None,
                 id_val=True):
        self.args = args
        self.augment_data = augment_data
        self.group_id = group_id
        self.mix_up = mix_up
        self.model_type = model_type
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.split_dir = osp.join(root_dir, 'trap-sets')
        self.data_dir = osp.join(root_dir, 'ISIC2018_Task1-2_Training_Input')
        
        metadata = {}
        metadata['train'] = pd.read_csv(osp.join(self.split_dir, f'isic_annotated_train{args.seed}.csv'))
        if id_val:
            test_val_data = pd.read_csv(osp.join(self.split_dir, f'isic_annotated_test{args.seed}.csv'))
            idx_val, idx_test = train_test_split(np.arange(len(test_val_data)), 
                                                test_size=0.8, random_state=0)
            metadata['test'] = test_val_data.iloc[idx_test]
            metadata['val'] = test_val_data.iloc[idx_val]
        else:
            metadata['test'] = pd.read_csv(osp.join(self.split_dir, f'isic_annotated_test{args.seed}.csv'))
            metadata['val'] = pd.read_csv(osp.join(self.split_dir, f'isic_annotated_val{args.seed}.csv'))
            # subtracting two dataframes 
            metadata_new = metadata['train'].merge(metadata['val'], how='left', indicator=True)
            metadata_new = metadata_new[metadata_new['_merge'] == 'left_only']
            metadata['train'] = metadata_new.drop(columns=['_merge'])
        
        self.train_transform = get_transform_ISIC(None, train=True, augment_data=augment_data)
        self.eval_transform = get_transform_ISIC(None, train=False, augment_data=augment_data)
        
        self.precomputed = True
        self.pretransformed = True
        self.n_classes = 2
        self.n_confounders = 1
        confounder = confounder_names[0]
        
        self.filename_array = np.concatenate([metadata[split]['image'] for split in ['train', 'val', 'test']])
        self.group_array = np.concatenate([metadata[split][confounder] for split in ['train', 'val', 'test']])
        self.y_array = np.concatenate([metadata[split][target_name] for split in ['train', 'val', 'test']])
        self.train_split_array = np.zeros(len(metadata['train']))
        self.test_split_array = 2 * np.ones(len(metadata['test']))
        self.val_split_array = np.ones(len(metadata['val']))
        
        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(1, torch.tensor(self.y_array).unsqueeze(1), 1).numpy()
        self.mix_array = [False] * len(self.y_array)
        
        self.split_array = np.concatenate([self.train_split_array, self.test_split_array, self.val_split_array])
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        if osp.exists(osp.join(root_dir, 'features', "feature_mat.npy")):
            self.features_mat = torch.from_numpy(np.load(
                osp.join(root_dir, 'features', 'feature_mat.npy'), allow_pickle=True)).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = []
            for idx in tqdm(range(len(self.y_array))):
                img_filename = osp.join(
                    self.data_dir, self.filename_array[idx][:-4] + '.jpg'
                    )
                img = Image.open(img_filename).convert('RGB')
                # Figure out split and transform accordingly
                if self.split_array[idx] == self.split_dict['train']:
                    img = self.train_transform(img)
                elif self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']]:
                    img = self.eval_transform(img)
                # Flatten if needed
                if model_attributes[self.model_type]['flatten']:
                    assert img.dim()==3
                    img = img.view(-1)
                self.features_mat.append(img)
                
            self.features_mat = torch.cat([x.unsqueeze(0) for x in self.features_mat], dim=0)
            os.makedirs(osp.join(root_dir, "features"))
            np.save(osp.join(root_dir, 'features', "feature_mat.npy"), self.features_mat.numpy())
        self.n_groups = 2

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

    def group_str(self, group_idx, train=False):
        return f'{self.confounder_names[0]}={group_idx}'