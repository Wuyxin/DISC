import argparse
from disc.dataset.load_data import dataset_attributes, shift_types
from disc.models import model_attributes
from disc.utils.tools import ParseKwargs


def parse_args():
    parser = argparse.ArgumentParser()
    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), default="MetaDatasetCatDog")
    parser.add_argument('-s', '--shift_type', choices=shift_types, default='confounder')
    # Confounders
    parser.add_argument('-t', '--target_name', default='waterbird_complete95')
    parser.add_argument('-c', '--confounder_names', nargs='+', default=['forest2water2'])
    # Resume
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--root_dir', default=None) 
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    # Model
    parser.add_argument('--model', choices=model_attributes.keys(), default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization. Example: key1=value1')
    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    parser.add_argument('--lisa_mix_up', action='store_true', default=False)
    parser.add_argument('--mix_ratio', default=0.5, type=float)
    parser.add_argument('--mix_alpha', default=2, type=float)
    parser.add_argument('--cut_mix', default=False, action='store_true')
    parser.add_argument('--num_warmup_steps', default=0, type=int)
    # baseline
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument("--coral", action='store_true', default=False)
    parser.add_argument("--mix_up", action='store_true', default=False)
    parser.add_argument("--rex", action='store_true', default=False)
    parser.add_argument("--irm", action='store_true', default=False)
    parser.add_argument("--fish", action='store_true', default=False)
    parser.add_argument("--ibirm", action='store_true', default=False)
    parser.add_argument("--jtt", action='store_true', default=False)
    parser.add_argument("--jtt_upweight", default=10, type=float)
    parser.add_argument("--irm_penalty", default=1.0, type=float)
    parser.add_argument("--rex_penalty", default=10, type=float)
    parser.add_argument("--ibirm_penalty", default=10, type=float)
    parser.add_argument("--meta_lr", default=1e-4, type=float)
    # DISC: required
    parser.add_argument('--disc', action='store_true', default=False)
    parser.add_argument('--erm_path', type=str, help="Model path of ERM")
    parser.add_argument('--concept_img_folder', type=str, default=f'synthetic_concepts')
    parser.add_argument("--concept_categories", type=str, help="Specify concept categories. Use '-' as delimiter "
                        "or use 'everything' to include all. Example: color-texture-nature / everything")
    # DISC: optional
    parser.add_argument('--n_clusters', type=int, default=3, 
                        help="Number of clusters for constructing environments")
    parser.add_argument('--n_concept_imgs', default=200, type=int)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--cluster', choices=['kmeans', 'gmm'], default='gmm')
    parser.add_argument("--c_svm", default=0.1, type=float, help="Regularization for SVMs")
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=50)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)
    parser.add_argument('--fold', default=None)
    parser.add_argument('--num_folds_per_sweep', type=int, default=5)
    parser.add_argument('--num_sweeps', type=int, default=4)
    parser.add_argument('--is_featurizer', type=int, default=True)
    parser.add_argument('--step_gamma', type=float, default=0.96)
    parser.add_argument('--group_by_label', action='store_true', default=False)

    return parser.parse_args()

