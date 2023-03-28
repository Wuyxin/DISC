import torch
from collections import defaultdict
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm
from PIL import Image

"""
From https://github.com/mertyg/debug-mistakes-cce
"""
class ListDataset:
    def __init__(self, images, preprocess=None):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(img_path)
        if self.preprocess:
            image = self.preprocess(image)
        return image


def learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, C, device="cuda"):
    """Learning CAVs and related margin stats.
    Args:
        pos_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding positive samples for each concept
        neg_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding negative samples for each concept
        model_bottom (nn.Module): Mode
        n_samples (int): Number of positive samples to use while learning the concept.
        C (float): Regularization parameter for the SVM. Possibly multiple options.
        device (str, optional): Device to use while extracting activations. Defaults to "cuda".

    Returns:
        dict: Concept information, including the CAV and margin stats.
    """
    print("Extracting Embeddings: ")
    pos_act = get_embeddings(pos_loader, backbone, device=device)
    neg_act = get_embeddings(neg_loader, backbone, device=device)
    X_train = np.concatenate([pos_act[:n_samples], neg_act[:n_samples]], axis=0)
    X_val = np.concatenate([pos_act[n_samples:], neg_act[n_samples:]], axis=0)
    y_train = np.concatenate([np.ones(pos_act[:n_samples].shape[0]), np.zeros(neg_act[:n_samples].shape[0])], axis=0)
    y_val = np.concatenate([np.ones(pos_act[n_samples:].shape[0]), np.zeros(neg_act[n_samples:].shape[0])], axis=0)
    concept_info = {}
    for c in C:
        concept_info[c] = get_cavs(X_train, y_train, X_val, y_val, c)
    return concept_info


def get_cavs(X_train, y_train, X_val, y_val, C):
    """Extract the concept activation vectors and the corresponding stats

    Args:
        X_train, y_train, X_val, y_val: activations (numpy arrays) to learn the concepts with.
        C: Regularizer for the SVM. 
    """
    svm = SVC(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_val, y_val)
    train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
    margin_info = {"max": np.max(train_margin),
                   "min": np.min(train_margin),
                   "pos_mean":  np.nanmean(train_margin[train_margin > 0]),
                   "pos_std": np.nanstd(train_margin[train_margin > 0]),
                   "neg_mean": np.nanmean(train_margin[train_margin < 0]),
                   "neg_std": np.nanstd(train_margin[train_margin < 0]),
                   "q_90": np.quantile(train_margin, 0.9),
                   "q_10": np.quantile(train_margin, 0.1),
                   "pos_count": y_train.sum(),
                   "neg_count": (1-y_train).sum(),
                   "prototype": np.mean(X_train[y_train==1], axis=0)
                   }
    concept_info = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)
    return concept_info


@torch.no_grad()
def get_embeddings(loader, model, device="cuda"):
    """
    Args:
        loader ([torch.utils.data.DataLoader]): Data loader returning only the images
        model ([nn.Module]): Backbone
        n_samples (int, optional): Number of samples to extract the activations
        device (str, optional): Device to use. Defaults to "cuda".

    Returns:
        np.array: Activations as a numpy array.
    """
    activations = None
    for image in tqdm(loader):
        image = image.to(device)
        batch_act = model(image).squeeze().detach().cpu().numpy()
        if activations is None:
            activations = batch_act
        else:
            if len(batch_act.shape) == 1:
                activations = np.concatenate([activations, batch_act.reshape(1,-1)], axis=0)
            else:
                activations = np.concatenate([activations, batch_act], axis=0)
    return activations
