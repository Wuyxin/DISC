# DISC
SEED=0
N_CLUSTERS=3
NUM=200
ROOT=/oak/stanford/groups/jamesz/shirwu/
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 200GB --time=7-00:00:00 python run_expt.py -s confounder -d CIFAR10 --lr 0.005 --batch_size 64 --weight_decay 5e-4 --model resnet50 --n_epochs 20 --erm_path $ROOT/DISC/output/CIFAR10/ERM/reweight_groups=0-augment=0-lr=0.01-batch_size=128-n_epochs=200-seed=0/best_model.pth --root_dir $ROOT/DISC/data/cifar10 --log_dir $ROOT/DISC/output/ --concept_img_folder $ROOT/synthetic_concepts --concept_categories everything --n_concept_imgs $NUM --n_clusters $N_CLUSTERS --augment_data --scheduler CosineAnnealingLR --save_best --save_last --seed $SEED --disc


# ERM
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CIFAR10 --lr 0.01 --batch_size 128 --weight_decay 5e-4 --model resnet50 --n_epochs 200 --root_dir $ROOT/data/cifar10 --log_dir $ROOT/output/ --scheduler CosineAnnealingLR --save_best --save_last


# ERM + aug
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CIFAR10 --lr 0.01 --batch_size 128 --weight_decay 5e-4 --model resnet50 --n_epochs 200 --root_dir $ROOT/data/cifar10 --log_dir $ROOT/output/ --scheduler CosineAnnealingLR --save_best --save_last --augment_data
