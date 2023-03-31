# DISC
# Note here GROUP \in [1, 2, 3, 4, 5] 
# which indicates the id of the trap-sets
GROUP=5
NUM=200
N_CLUSTERS=3
ROOT=/oak/stanford/groups/jamesz/shirwu/
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 --mem 50GB python run_expt.py -s confounder -d ISIC -t label -c hair --lr 5e-4 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100 --root_dir $ROOT/DISC/data/isic/ --log_dir $ROOT/DISC/output_isic/ --erm_path $ROOT/DISC/output_isic/ISIC/ERM/reweight_groups=0-augment=0-lr=0.001-batch_size=16-n_epochs=100-trapset_id=$GROUP/best_model.pth --concept_img_folder $ROOT/synthetic_concepts --concept_categories color-texture --n_concept_imgs $NUM --n_clusters $N_CLUSTERS --save_best --save_last --seed $GROUP --disc

# ERM
GROUP=5
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100 --log_dir $ROOT/output_isic/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $GROUP

# ERM + aug
GROUP=5
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100 --log_dir $ROOT/output_isic/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $GROUP --augment_data
