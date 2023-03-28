#ERM
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/cub  --log_dir $ROOT/logs/waterbirds/ERM-time


#ERM+aug
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/cub  --log_dir $ROOT/logs/waterbirds/ERM-aug --augment_data

ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --intervenev2 --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/concept_bank/last_model_0.1_75.pkl --log_dir $ROOT/logs/waterbirds/Owl/mixup-topk=5-variance-c=5-lr=5e-4  --concept_img_folder /oak/stanford/groups/jamesz/shirwu/broden_concepts  --lr 0.0005 --topk 5 --c_variance 5 --mix_ent --save_best --save_last --reweight_groups --variance --owl_resume_path $ROOT/logs/waterbirds/Owl/cf_tensors.pt

#syn
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --intervenev2 --root_dir $ROOT/data/cub  --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank/last_model_0.1_75.pkl --log_dir $ROOT/logs/waterbirds/Owl/syn-mixup-topk=5-variance-c=5-lr=5e-4 --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts  --lr 0.0005 --topk 5 --c_variance 5 --mix_ent --save_best --save_last --reweight_groups --variance --owl_resume_path $ROOT/logs/waterbirds/Owl/cf_tensors.pt


#pro
--c_variance 5 --variance 
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --intervene_pro --root_dir $ROOT/data/cub  --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank/last_model_0.1_75.pkl --log_dir $ROOT/logs/waterbirds/Owl/pro-time-n_students=3-topk=5 --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts  --lr 0.0005 --n_students 3 --topk 5  --mix_ent --save_best --save_last --reweight_groups --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth


#grad
-C GPU_BRD:TESLA  --variance --c_variance 5

ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=2-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 200  --gamma 0.1 --generalization_adjustment 0 --intervene_grad_pro --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank/last_model_0.1_75.pkl --log_dir $ROOT/logs/waterbirds/Owl/default-xlr-std-grad_pro-n_students=3-topk=20 --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts  --lr 0.0005 --n_students 3 --topk 20 --mix_ent --save_best --save_last --reweight_groups --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth

#updates
--c_variance 5 --variance 
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --intervene_grad_updates --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank/last_model_0.1_75.pkl --log_dir $ROOT/logs/waterbirds/Owl/updates-grad_updates-n_students=3-topk=20 --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts  --lr 0.0005 --n_students 3 --topk 20  --mix_ent --save_best --save_last --reweight_groups --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth


--owl_resume_path $ROOT/logs/waterbirds/Owl/cf_tensors.pt

#reset
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 10  --gamma 0.1 --generalization_adjustment 0 --intervene_grad_reset --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank/last_model_0.1_400.pkl --log_dir $ROOT/logs/waterbirds/Owl/final-C=0.1-n_students=3-topk=20_500 --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --lr 0.0005 --n_students 3 --topk 20 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth  --reweight_groups 


#reset
SEED=20
conda activate torch
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=2:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 1e-5 --model resnet50 --n_epochs 10 --gamma 0.1 --generalization_adjustment 0 --intervene_grad_reset --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank_nc/last_model_0.1_400.pkl --log_dir $ROOT/logs/waterbirds/Owl/final-C=0.1-lr=5e-4-n_students=5-topk=10_umap_nc_seed=$SEED --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts  --lr 0.0005 --n_students 5 --topk 10 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth --reweight_groups --seed $SEED 

#absent
SEED=0
conda activate torch
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=2:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 1e-5 --model resnet50 --n_epochs 10 --gamma 0.1 --generalization_adjustment 0 --intervene_grad_reset --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank_nc_absent/last_model_0.1_400.pkl --log_dir $ROOT/logs/waterbirds/Owl/final-absent-C=0.1-lr=5e-4-n_students=3-topk=10_umap_nc_seed=$SEED --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts  --lr 0.0005 --n_students 3 --topk 10 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth --reweight_groups --seed $SEED 


# ablation
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 10  --gamma 0.1 --generalization_adjustment 0 --rand_intervene --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank_nature/last_model_0.1_400.pkl --log_dir $ROOT/logs/waterbirds/Owl/rand_intervene-topk=5-C=0.1_500_nature_seed=4 --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --lr 0.0005 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth  --reweight_groups --topk 5 --seed 4

#upweight
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 10  --gamma 0.1 --generalization_adjustment 0 --upweight_intervene --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank_nature/last_model_0.1_400.pkl --log_dir $ROOT/logs/waterbirds/Owl/upweight_intervene_norm-topk=5_500_nature_seed=3 --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --lr 0.001 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth  --reweight_groups --topk 5 --seed 3

# gradcam
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt_gradcam.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 1 --weight_decay 0.0001 --model resnet50 --n_epochs 10  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/cub --concept_bank_path $ROOT/logs/waterbirds/ERM/syn-concept_bank_nature/last_model_0.1_400.pkl --log_dir $ROOT/logs/waterbirds/Owl/gradcam --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --lr 0.001 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/waterbirds/ERM/last_model.pth  --reweight_groups --topk 5 --seed 3
