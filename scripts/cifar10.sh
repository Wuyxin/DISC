# DISC
SEED=0
N_CLUSTERS=3
ROOT=./DISC
python run_expt.py -s confounder -d CIFAR10 --lr 0.005 --batch_size 64 --weight_decay 5e-4 --model resnet50 --n_epochs 20 --erm_path $ROOT/output/CIFAR10/ERM/reweight_groups=0-augment=0-lr=0.01-batch_size=128-n_epochs=200-seed=0/best_model.pth --root_dir $ROOT/data/cifar10 --log_dir $ROOT/output/ --concept_img_folder $ROOT/synthetic_concepts --concept_categories everything --n_clusters $N_CLUSTERS --augment_data --scheduler CosineAnnealingLR --save_best --save_last --seed $SEED --disc


# ERM
ROOT=./DISC
python run_expt.py -s confounder -d CIFAR10 --lr 0.01 --batch_size 128 --weight_decay 5e-4 --model resnet50 --n_epochs 200 --root_dir $ROOT/data/cifar10 --log_dir $ROOT/output/ --scheduler CosineAnnealingLR --save_best --save_last


# ERM + aug
ROOT=./DISC
python run_expt.py -s confounder -d CIFAR10 --lr 0.01 --batch_size 128 --weight_decay 5e-4 --model resnet50 --n_epochs 200 --root_dir $ROOT/data/cifar10 --log_dir $ROOT/output/ --scheduler CosineAnnealingLR --save_best --save_last --augment_data
