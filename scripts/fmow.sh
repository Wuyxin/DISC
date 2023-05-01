# DISC
SEED=0
N_CLUSTERS=3
ROOT=./DISC
python run_expt.py -s confounder -d FMoW --lr 1e-4 --batch_size 16 --weight_decay 0 --model densenet121 --optimizer Adam --scheduler StepLR --n_epochs 30 --erm_path $ROOT/output/FMoW/ERM/reweight_groups=0-augment=0-lr=0.0001-batch_size=32-n_epochs=60-seed=0/best_model.pth --log_dir $ROOT/output --root_dir $ROOT/data/fmow --concept_img_folder $ROOT/synthetic_concepts --concept_categories color-texture-nature-city --n_clusters $N_CLUSTERS --augment_data --save_best --save_last --seed $SEED --disc


# ERM
ROOT=./DISC
python run_expt.py -s confounder -d FMoW --lr 1e-4 --batch_size 32 --weight_decay 0 --model densenet121 --optimizer Adam --scheduler StepLR --n_epochs 60 --log_dir $ROOT/output --root_dir $ROOT/data/fmow --save_best --save_last 


# ERM + aug
ROOT=./DISC
python run_expt.py -s confounder -d FMoW --lr 1e-4 --batch_size 32 --weight_decay 0 --model densenet121 --optimizer Adam --scheduler StepLR --n_epochs 60 --log_dir $ROOT/output --root_dir $ROOT/data/fmow --save_best --save_last --augment_data