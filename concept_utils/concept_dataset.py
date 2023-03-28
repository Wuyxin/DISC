import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from dataset.transform import transform_dict


class ConceptDataset(Dataset):
    """
    Concept dataset
    Args: 
        args : Arguments, see run_expt.py
        root_dir (str): Arguments, see run_expt.py
        concept_names (list([str, str, ...])): Concepts names in the concept dataset
        concept_probs (list([float, float, ...])): Concept probabilities used for sampling
        model_type (str, optional): Type of model on the dataset, see models.py
        augment_data (bool, optional): Whether to use data augmentation, e.g., RandomCrop
        mix_up (bool, optional): Whether to use mixup 
        
    """

    def __init__(self, args, root_dir,
                 concept_names, 
                 concept_probs=None,
                 model_type=None,
                 augment_data=False):
        
        self.args = args
        self.root_dir = root_dir
        self.concept_names = concept_names
        self.model_type = model_type
        self.augment_data = augment_data

        if not os.path.exists(self.root_dir):
            raise ValueError(
                f'Missing concepts in {self.root_dir}')
        
        self.img_files = {}
        transform = transform_dict[args.dataset]
        self.train_transform = transform(model_type=self.model_type, 
                                         train=True,
                                         augment_data=augment_data)
        self.data = []
        self.concept_label = []
        for concept_id, concept in enumerate(concept_names):
            concept_dir = os.path.join(self.root_dir, concept, 'positives')
            for file_path in os.listdir(concept_dir)[:args.n_concept_imgs]:
                img_filename = os.path.join(
                    concept_dir, file_path)
                img = Image.open(img_filename).convert('RGB')
                img = self.train_transform(img)
                
                self.data.append(img)
                self.concept_label.append(concept_id)
                
        if not concept_probs == None:
            self.concept_label = np.array(self.concept_label)
            self.update_concept_prob(concept_probs)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

    def update_concept_prob(self, concept_probs):
        self.concept_probs = np.array(concept_probs)
        self.concept_probs = concept_probs / concept_probs.sum()
    
    def get_loader(self, concept_probs=None, **kwargs):

        if not concept_probs is None:
            self.update_concept_prob(concept_probs)
            
        weights = self.concept_probs[self.concept_label]
        assert not np.isnan(weights).any()
        sampler = WeightedRandomSampler(weights, len(self), replacement=True)
        loader = DataLoader(
            self,
            shuffle=False,
            sampler=sampler,
            **kwargs)
        return loader
