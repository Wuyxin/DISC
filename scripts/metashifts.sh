# DISC
SEED=0
N_CLUSTERS=2
NUM=200
ROOT=/oak/stanford/groups/jamesz/shirwu/
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 200GB --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.0005 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --erm_path $ROOT/DISC/output/MetaDatasetCatDog/ERM/reweight_groups=0-augment=0-lr=0.001-batch_size=16-n_epochs=100-seed=0/best_model.pth --root_dir $ROOT/DISC/data/metashifts/MetaDatasetCatDog  --log_dir $ROOT/DISC/output/ --concept_img_folder $ROOT/synthetic_concepts --concept_categories everything --n_concept_imgs $NUM --n_clusters $N_CLUSTERS --augment_data --save_last --save_best --seed $SEED --disc 

# ERM
SEED=0
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --save_last  --save_best --log_dir $ROOT/output/ --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --seed $SEED --save_best --save_last


# ERM + aug
SEED=0
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --save_last  --save_best --log_dir $ROOT/output/ --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --seed $SEED --save_best --save_last --augment_data