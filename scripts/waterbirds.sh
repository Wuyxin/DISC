# DISC
SEED=0
N_CLUSTERS=3
conda activate torch
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 10 --gamma 0.1 --generalization_adjustment 0 --disc --root_dir $ROOT/data/cub --concept_bank_path $ROOT/output/waterbirds/ERM/syn-concept_bank_nc/last_model_0.1_400.pkl --log_dir $ROOT/output/waterbirds/DISC/ --concept_img_folder $ROOT/synthetic_concepts --lr 0.0005 --n_clusters $N_CLUSTERS --topk 10 --pretrained_model_path $ROOT/output/waterbirds/ERM/last_model.pth --reweight_groups --seed $SEED  --save_best --save_last

# ERM
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/cub --log_dir $ROOT/output/ --save_best --save_last

# ERM + aug
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/cub  --log_dir $ROOT/output/waterbirds/ERM_aug --augment_data  --save_best --save_last

