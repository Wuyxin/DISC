# DISC
SEED=3
N_CLUSTERS=3
NUM=200
ROOT=/oak/stanford/groups/jamesz/shirwu/
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 1e-4 --batch_size 32 --weight_decay 1e-4 --model resnet50 --n_epochs 10 --erm_path $ROOT/DISC/output/CUB/ERM/reweight_groups=0-lr=0.001-batch_size=32-n_epochs=300-seed=0/best_model.pth --root_dir $ROOT/DISC/data/cub --log_dir $ROOT/DISC/output/ --concept_img_folder $ROOT/synthetic_concepts --concept_categories color-texture-nature --n_concept_imgs $NUM --n_clusters $N_CLUSTERS --reweight_groups --seed $SEED --augment_data --save_best --save_last --disc

# ERM
SEED=0
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --root_dir $ROOT/data/cub --log_dir $ROOT/output/ --save_best --save_last --seed 1

# ERM + aug
SEED=0
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --root_dir $ROOT/data/cub --log_dir $ROOT/output/waterbirds/ERM_aug --augment_data --save_best --save_last

