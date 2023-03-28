import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import warnings
from numba.core.errors import NumbaWarning
from umap import UMAP

import torch
import os
import os.path as osp
from tqdm import tqdm
from models import NetBottom, NetTop
from concept_utils.cav_utils import get_embeddings


def cluter_assignment(args, train_data, model, logger):
    """
    Conduct class-wise group assignment

    Args:
        args : Arguments, see run_expt.py
        train_data (torch.utils.data.Dataset): training dataset
        model (nn.Module): The pretrained ERM model. 
        logger (Logger): logger that wirtes to output file

    Returns:
        cluster_dict: Dictionary of the clustering results
            
    """
    save_path = osp.join(args.log_dir, 'cluster')
    os.makedirs(save_path, exist_ok=True)
    assignments_path = osp.join(save_path, 'assignments.pt')
    
    backbone = NetBottom(model).cuda()
    model_top = NetTop(model).cuda()
    cluster_dict = {l : {} for l in range(model_top.out_features)}
    loader = train_data.get_loader(train=False, batch_size=args.batch_size)

    # If clustering results already exist, then use the saved one
    if not osp.exists(assignments_path):
        # get instance embeddings and accuracies
        embeddings = np.array([])
        labels, ids, acc, gs = [], [], [], []
        for batch in tqdm(loader):
            x, y = batch[0].cuda(), batch[1].cuda()
            try:
                g = batch[2].detach().cpu().numpy()
            except:
                g = np.zeros(len(x), dtype=np.int32)
            embs = backbone(x)
            outputs = model_top(embs)
            embs = embs.detach().cpu().numpy()
            embeddings = np.concatenate([embeddings, embs], axis=0)
            gs.append(g)
            labels.append(y.detach().cpu().numpy())
            acc.append((outputs.argmax(dim=1) == y).float().detach().cpu().numpy())
        gs = np.concatenate(gs)
        labels = np.concatenate(labels)
        acc = np.concatenate(acc).reshape(-1, 1)
        ids = np.arange(len(labels))
        
        # conduct clustering using kmeans/gmm
        silhouette_score = []
        reducer = UMAPReducer()
        for l in np.unique(labels):
            reps = np.concatenate([embeddings[labels==l], acc[labels==l]], axis=1)
            reducer.fit(reps)
            reps = reducer.transform(reps)
            if args.cluster == 'kmeans':
                kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(reps)
                cluters_ = kmeans.labels_
            elif args.cluster == 'gmm':
                gm = GaussianMixture(n_components=args.n_clusters).fit(reps)
                cluters_ = gm.predict(reps)
            silhouette_score.append(sklearn.metrics.silhouette_score(reps, cluters_))
            unique_clusters, counts = np.unique(cluters_, return_counts=True)
            for c in range(len(unique_clusters)):
                cluster_dict[l][c] = ids[labels==l][cluters_==c]
            logger.write(f'\nClass {l}; Cluter Size {counts}\n')
            # for visualization
            torch.save(reps, osp.join(info_path, f'cluster_rep_class={l}.pt'))
            torch.save(cluters_, osp.join(info_path, f'cluster_idx_class={l}.pt'))
            torch.save(gs[labels==l], osp.join(info_path, f'group_idx_class={l}.pt'))
        logger.write(f'Silhouette Scores: {silhouette_score}\n  Mean: {np.mean(silhouette_score)}\n')
        torch.save(cluster_dict, assignments_path)
    logger.flush()
    return torch.load(assignments_path)


class UMAPReducer:
    
    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        kwargs = {**{'n_neighbors': 10, 'min_dist': 0.}, **kwargs}
        self.model = UMAP(n_components=n_components, **kwargs)

    def fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NumbaWarning)
            self.model.fit(X)
        return self

    def transform(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NumbaWarning)
            result = self.model.transform(X)
        return result
    