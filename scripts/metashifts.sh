# ERM
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/ERM/seed=2 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog

--seed 2

# ERM+aug
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/ERM-aug-seed=9 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --seed 9 --augment_data

# UW
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/seed=0 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --reweight_groups --seed 0

# Fish
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 128GB --time=2:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/Fish/seed=3 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog  --seed 3 --fish --meta_lr 0.01

# GroupDRO
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/GroupDRO/gamma=0.1-seed=7 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --robust --seed 7

# IRM
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/IRM/irm_penalty=1e-2-seed=4 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --irm --irm_penalty 1e-2 --seed 4

# IBIRM
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/IBIRM/ibirm_penalty=1-irm_penalty=0.01-seed=4 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --ibirm --irm_penalty 0.01 --ibirm_penalty 1 --seed 4

# REx
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/REx/rex_penalty=1e-3-seed=4 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --rex --rex_penalty 1e-3 --seed 4

# Coral
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/CORAL/gamma=0.1-seed=3 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --coral  --seed 3


# ERM
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/ERM --root_dir $ROOT/data/metashifts/MetaDatasetCatDog

#LISA
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --lisa_mix_up --mix_alpha 2 --cut_mix  --save_last  --save_best --log_dir $ROOT/logs/data/metashifts/group_$GROUP/lisa/final-seed=3 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --seed 3

# JTT 
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=3:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.0001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/jtt/weight=5-seed=2 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --pretrained_model_path $ROOT/logs/metashifts/ERM/train/last_model.pth --jtt --jtt_upweight 5 --seed 2

# Owl: w/o variance
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/mixup-topk=5-lr=5e-4 --root_dir $ROOT/metashifts/MetaDatasetCatDog --intervenev2 --concept_bank_path $ROOT/logs/metashifts/ERM/concept_bank/last_model_0.1_75.pkl --topk 5 --lr 0.0005 --save_best --save_last --owl_resume_path $ROOT/logs/metashifts/Owl/train/mixup-topk=2-lr=5e-4-mixent/cf_tensors.pt --mix_ent

# Owl: w/ variance
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP  --save_last  --save_best --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/mixup-topk=2-variance-c=5-lr=5e-4 --root_dir $ROOT/metashifts/MetaDatasetCatDog --intervenev2 --concept_bank_path $ROOT/logs/metashifts/ERM/concept_bank/last_model_0.1_75.pkl --topk 2 --lr 0.0005 --save_best --save_last --variance  --mix_ent --c_variance 5 --pretrained_model_path $ROOT/logs/metashifts/ERM/train/last_model.pth

--owl_resume_path $ROOT/logs/metashifts/Owl/train/mixup-topk=2-lr=5e-4-mixent/cf_tensors.pt

# syn
# Owl: w/ variance
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1  --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/syn-mixup-topk=2-variance-c=5-lr=5e-4 --root_dir $ROOT/metashifts/MetaDatasetCatDog --intervenev2 --concept_bank_path $ROOT/logs/metashifts/ERM/syn-concept_bank/last_model_0.1_75.pkl --topk 2 --lr 0.0005 --save_best --save_last --variance --c_variance 5 --mix_ent  --pretrained_model_path $ROOT/logs/metashifts/ERM/train/last_model.pth

--owl_resume_path $ROOT/logs/metashifts/Owl/syn-cf_tensors.pt 

# pro
--variance --c_variance 5 

GROUP=2
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/pro-xlr-n_students=3-topk=5-seed=1 --root_dir $ROOT/metashifts/MetaDatasetCatDog --intervene_pro --n_students 3 --concept_bank_path $ROOT/logs/metashifts/ERM/syn-concept_bank/last_model_0.1_75.pkl --topk 5 --lr 0.0005 --save_best --save_last --mix_ent --syn --pretrained_model_path $ROOT/logs/metashifts/ERM/train/last_model.pth --seed 1


-C GPU_BRD:TESLA 
#pro
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --mem 64GB --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/pro-n_students=5-topk=10-variance-c=5-lr=5e-4 --root_dir $ROOT/metashifts/MetaDatasetCatDog --intervene_pro --n_students 5 --concept_bank_path $ROOT/logs/metashifts/group_$GROUP/ERM/syn-concept_bank/last_model_0.1_75.pkl --topk 10 --lr 0.0005 --save_best --save_last --variance --c_variance 5 --mix_ent --syn --pretrained_model_path $ROOT/logs/metashifts/group_$GROUP/ERM/train/last_model.pth

#grad, no variance
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --mem 64GB --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/default-xlr-std-grad_pro-n_students=3-topk=10 --root_dir $ROOT/metashifts/MetaDatasetCatDog --intervene_grad_pro --n_students 3 --concept_bank_path $ROOT/logs/metashifts/group_$GROUP/ERM/syn-concept_bank/last_model_0.1_75.pkl --topk 10 --lr 0.0005 --save_best --save_last --syn --mix_ent --pretrained_model_path $ROOT/logs/metashifts/group_$GROUP/ERM/last_model.pth 

#grad updates, no variance
default: no variance; with adaptive

-C GPU_BRD:TESLA 

GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --mem 64GB --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/default-grad_updates-n_students=3-topk=10 --root_dir $ROOT/metashifts/MetaDatasetCatDog --intervene_grad_updates --n_students 3 --concept_bank_path $ROOT/logs/metashifts/group_$GROUP/ERM/syn-concept_bank/last_model_0.1_75.pkl --topk 10 --lr 0.0005 --save_best --save_last --syn --mix_ent --pretrained_model_path $ROOT/logs/metashifts/group_$GROUP/ERM/last_model.pth 

#absent
conda activate torch
SEED=2
GROUP=3
NUM=100
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --mem 32GB --cpus-per-gpu=2 --time=3-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/final-absent-n_students=3-topk=10_umap_${NUM}_everything-seed=$SEED --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --intervene_grad_reset --n_students 3 --concept_bank_path $ROOT/logs/metashifts/group_$GROUP/ERM/syn-concept_bank_100_absent/last_model_0.1_100.pkl --topk 10 --lr 0.0005 --save_best --save_last --syn --mix_ent --pretrained_model_path $ROOT/logs/metashifts/group_$GROUP/ERM/last_model.pth --seed $SEED --num_concept_img $NUM 

#reset
conda activate torch
SEED=0
GROUP=3
NUM=500
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --mem 32GB --cpus-per-gpu=2 --time=3-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/final-n_students=3-topk=10_umap_${NUM}_everything-seed=$SEED --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --intervene_grad_reset --n_students 3 --concept_bank_path $ROOT/logs/metashifts/group_$GROUP/ERM/syn-concept_bank_$NUM/last_model_0.1_400.pkl --topk 10 --lr 0.0005 --save_best --save_last --syn --mix_ent --pretrained_model_path $ROOT/logs/metashifts/group_$GROUP/ERM/last_model.pth --seed $SEED --num_concept_img $NUM 

#ablation
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --mem 128GB --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/rand_intervene-topk=100_500_everything_seed=4 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --rand_intervene --concept_bank_path $ROOT/logs/metashifts/group_$GROUP/ERM/syn-concept_bank_500/last_model_0.1_400.pkl --topk 100 --lr 0.0005 --save_best --save_last --syn --mix_ent --pretrained_model_path $ROOT/logs/metashifts/group_$GROUP/ERM/last_model.pth --reweight_groups --seed 4

#upweight
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --mem 128GB --cpus-per-gpu=2 --time=2:00:00 python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group $GROUP --log_dir $ROOT/logs/metashifts/group_$GROUP/Owl/upweight_intervene_500_everything_seed=3 --root_dir $ROOT/data/metashifts/MetaDatasetCatDog --upweight_intervene --concept_bank_path $ROOT/logs/metashifts/group_$GROUP/ERM/syn-concept_bank_500/last_model_0.1_400.pkl --topk 10 --lr 0.001 --save_best --save_last --syn --mix_ent --pretrained_model_path $ROOT/logs/metashifts/group_$GROUP/ERM/last_model.pth --reweight_groups --seed 3