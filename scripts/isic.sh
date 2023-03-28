# ERM
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=2-00:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/ERM/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED

# ERM+aug
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=2-00:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/ERM/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED --augment_data

# UW
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=2-00:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5  --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/UW/ --root_dir $ROOT/data/isic/ --save_best --save_last  --reweight_groups --group_by_label --seed $SEED

# Fish
conda activate torch
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 128GB --time=2:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/Fish_repeat/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED --fish --meta_lr 0.001

# GroupDRO
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/GroupDRO/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED --robust

# IRM
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/IRM-penalty=0.99/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED --irm --irm_penalty 0.99

# IBIRM
SEED=1
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/IBIRM-ibirm_penalty=0.99_irm_penalty=1.0/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED --ibirm --irm_penalty 1.0 --ibirm_penalty 0.99

# REx
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/REx-penalty=1e-1/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED --rex --rex_penalty 1e-1

# CORAL
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/CORAL/ --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED --coral

# LISA
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=2-00:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 1e-3 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/lisa/ --root_dir $ROOT/data/isic/ --save_best --save_last  --reweight_groups  --group_by_label --lisa_mix_up  --seed $SEED

# JTT
SEED=1
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3:00:00 python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.0001 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/isic/group_$SEED/jtt/upweight=4.999  --root_dir $ROOT/data/isic/ --save_best --save_last --seed $SEED  --pretrained_model_path $ROOT/logs/isic/group_$SEED/ERM/last_model.pth --jtt --jtt_upweight 4.999 

# Owl ood
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=2-00:00:00 --mem 128GB  python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.0005 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/isic/ --intervene_grad_reset --concept_bank_path $ROOT/logs/isic/ERM_ood_val/syn-concept_bank_500_t/last_model_0.1_400.pkl --log_dir $ROOT/logs/isic/Owl_ood_val/final-C=0.1-n_students=3-topk=20_500_t --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --n_students 3 --topk 20 --mix_ent --syn --pretrained_model_path $ROOT/logs/isic/ERM_ood_val/last_model.pth --save_best --save_last

# Owl id
# ink
SEED=1
STUDENTS=3
conda activate torch
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3-00:00:00 --mem 50GB python run_expt.py -s confounder -d ISIC -t label -c ink --lr 0.0005 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/isic/ --intervene_grad_reset --concept_bank_path $ROOT/logs/isic/group_$SEED/ERM/syn-concept_bank_500_ct/last_model_0.1_400.pkl --log_dir $ROOT/logs/isic/group_$SEED/Owl/final-C=0.1-n_students=$STUDENTS-topk=10_umap_500_ct_cluster_correct_ink --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --n_students $STUDENTS --topk 10 --mix_ent --syn --pretrained_model_path $ROOT/logs/isic/group_$SEED/ERM/last_model.pth --save_best --save_last --seed $SEED --num_concept_img 500

# ablation
SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=2-00:00:00 --mem 100GB  python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.0005 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/isic/ --rand_intervene --concept_bank_path $ROOT/logs/isic/group_$SEED/ERM/syn-concept_bank_500_ct/last_model_0.1_400.pkl --log_dir $ROOT/logs/isic/group_$SEED/Owl/rand_intervene-C=0.1-topk=50_500_ct_seed=$SEED --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --mix_ent --syn --pretrained_model_path $ROOT/logs/isic/group_$SEED/ERM/last_model.pth --save_best --save_last --topk 50 --seed $SEED


SEED=5
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3-00:00:00 --mem 50GB  python run_expt.py -s confounder -d ISIC -t label -c hair --lr 0.0005 --batch_size 16 --weight_decay 1e-5 --model resnet50 --n_epochs 100  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/isic/ --upweight_intervene --concept_bank_path $ROOT/logs/isic/group_$SEED/ERM/syn-concept_bank_500_ct/last_model_0.1_400.pkl --log_dir $ROOT/logs/isic/group_$SEED/Owl/upweight_intervene-C=0.1-topk=10_500_ct_seed=$SEED --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --mix_ent --syn --pretrained_model_path $ROOT/logs/isic/group_$SEED/ERM/last_model.pth --save_best --save_last --topk 10 --seed $SEED