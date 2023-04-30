# DISC
# Note here GROUP \in [1, 2, 3, 4, 5] 
# which indicates the id of the trap-sets
GROUP=5
N_CLUSTERS=3
ROOT=shirwu/DISC
python run_expt.py -s confounder -d ISIC -t label -c hair --lr 5e-4 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100 --root_dir $ROOT/data/isic/ --log_dir $ROOT/output/ --erm_path $ROOT/output/ISIC/ERM/reweight_groups=0-augment=0-lr=0.001-batch_size=16-n_epochs=100-trapset_id=$GROUP/best_model.pth --concept_img_folder $ROOT/synthetic_concepts --concept_categories color-texture --n_clusters $N_CLUSTERS --save_best --save_last --seed $GROUP --disc


# ERM
GROUP=5
ROOT=shirwu/DISC
python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100 --log_dir $ROOT/output/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $GROUP


# ERM + aug
GROUP=5
ROOT=shirwu/DISC
python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100 --log_dir $ROOT/output/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $GROUP --augment_data
