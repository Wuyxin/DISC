# DISC
SEED=0
N_CLUSTERS=3
NUM=200
ROOT=/oak/stanford/groups/jamesz/shirwu/
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=7-00:00:00 --mem 200GB python run_expt.py -s confounder -d FMoW --lr 1e-4 --batch_size 16 --weight_decay 0 --model densenet121 --optimizer Adam --scheduler StepLR --n_epochs 30 --erm_path $ROOT/DISC/output/FMoW/ERM/reweight_groups=0-augment=0-lr=0.0001-batch_size=32-n_epochs=60-seed=0/best_model.pth --root_dir $ROOT/DISC/data/fmow --log_dir $ROOT/DISC/output --concept_img_folder $ROOT/synthetic_concepts --concept_categories color-texture-nature-city --n_concept_imgs $NUM --n_clusters $N_CLUSTERS --augment_data --save_best --save_last --seed $SEED --disc


# ERM
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=7-00:00:00 --mem 100GB python run_expt.py -s confounder -d FMoW --lr 1e-4 --batch_size 32 --weight_decay 0 --model densenet121 --optimizer Adam --scheduler StepLR --n_epochs 60 --root_dir $ROOT/data/fmow --log_dir $ROOT/output --save_best --save_last 

# ERM + aug
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=7-00:00:00 --mem 100GB python run_expt.py -s confounder -d FMoW --lr 1e-4 --batch_size 32 --weight_decay 0 --model densenet121 --optimizer Adam --scheduler StepLR --n_epochs 60 --root_dir $ROOT/data/fmow --log_dir $ROOT/output --save_best --save_last --augment_data