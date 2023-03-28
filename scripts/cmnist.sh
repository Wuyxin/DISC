# cmnist
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --log_dir $ROOT/logs/cmnist/ERM/ --root_dir $ROOT/cmnist/ --save_best --save_last 


ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --log_dir $ROOT/logs/cmnist/LISA-mixup/ --root_dir $ROOT/cmnist/ --save_best --save_last 


ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --lisa_mix_up --log_dir $ROOT/logs/cmnist/LISA-cutmix/ --root_dir $ROOT/cmnist/ --cut_mix --save_best --save_last 

#real
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --intervenev2 --root_dir $ROOT/cmnist --concept_bank_path $ROOT/logs/cmnist/ERM/concept_bank/last_model_0.1_75.pkl --log_dir $ROOT/logs/cmnist/Owl/mixup-topk=2-variance-c=5-pro --concept_bank_folder /oak/stanford/groups/jamesz/shirwu/broden_concepts  --topk 2 --c_variance 5 --mix_ent --save_best --save_last --reweight_groups --variance

#syn
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --intervenev2 --root_dir $ROOT/cmnist --concept_bank_path $ROOT/logs/cmnist/ERM/syn-concept_bank/last_model_0.1_75.pkl --log_dir $ROOT/logs/cmnist/Owl/syn-mixup-topk=2-variance-c=5 --concept_bank_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts  --topk 2 --c_variance 5 --mix_ent --save_best --save_last --reweight_groups --variance

 --owl_resume_path $ROOT/logs/cmnist/Owl/cf_tensors.pt


#reset
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0  --intervene_grad_reset --root_dir $ROOT/data/cmnist --concept_bank_path $ROOT/logs/cmnist/ERM/syn-concept_bank_500_color/last_model_0.1_400.pkl --log_dir $ROOT/logs/cmnist/Owl/final-n_students=3-topk=2_500_color --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --lr 0.0005 --n_students 3 --topk 2 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/cmnist/ERM/last_model.pth  --reweight_groups