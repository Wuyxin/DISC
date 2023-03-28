ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 -C GPU_BRD:TESLA --mem 64GB --time=1-00:00:00 python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 50 --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/celeba --log_dir $ROOT/logs/celeba/ERM_larger_mem --save_best --save_last 

#real
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=3-00:00:00 python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --root_dir $ROOT/celeba --intervenev2 --concept_bank_path $ROOT/logs/celeba/ERM/concept_bank/last_model_0.1_75.pkl   --log_dir $ROOT/logs/celeba/Owl/mixup-topk=2-variance-c=5-lr=5e-5 --topk 2 --lr 0.00005 --save_best --save_last --variance --c_variance 5 --mix_ent --pretrained_model_path /oak/stanford/groups/jamesz/shirwu/Owl/logs/celeba/ERM/last_model.pth

--owl_resume_path $ROOT/logs/celeba/Owl/mixup-topk=2-variance-c=5-lr=5e-5/cf_tensors.pt

# syn
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=3-00:00:00 python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --root_dir $ROOT/celeba --intervenev2 --concept_bank_path $ROOT/logs/celeba/ERM/syn-concept_bank/last_model_0.1_75.pkl   --log_dir $ROOT/logs/celeba/Owl/syn-mixup-topk=2-variance-c=5-lr=5e-5 --topk 2 --lr 0.00005 --save_best --save_last --variance --c_variance 5 --mix_ent --concept_bank_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --pretrained_model_path /oak/stanford/groups/jamesz/shirwu/Owl/logs/celeba/ERM/last_model.pth

--owl_resume_path $ROOT/logs/celeba/Owl/ syn-mixup-topk=2-variance-c=5-lr=5e-5_face/cf_tensors.pt

# syn face
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=3-00:00:00 python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --root_dir $ROOT/celeba --intervenev2 --concept_bank_path $ROOT/logs/celeba/ERM/syn-concept_bank_face/last_model_0.1_75.pkl   --log_dir $ROOT/logs/celeba/Owl/syn-mixup-topk=2-variance-c=5-lr=5e-5_face --topk 2 --lr 0.00005 --save_best --save_last --variance --c_variance 5 --mix_ent --concept_bank_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --pretrained_model_path /oak/stanford/groups/jamesz/shirwu/Owl/logs/celeba/ERM/last_model.pth


#pro
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=3-00:00:00 python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --root_dir $ROOT/data/celeba --intervene_pro --concept_bank_path $ROOT/logs/celeba/ERM/syn-concept_bank_face/last_model_0.1_75.pkl   --log_dir $ROOT/logs/celeba/Owl/pro-n_students=5-topk=5 --n_students 5 --topk 5 --lr 0.00005 --save_best --save_last --variance --c_variance 5 --mix_ent --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts --pretrained_model_path /oak/stanford/groups/jamesz/shirwu/Owl/logs/celeba/ERM/last_model.pth --syn


--owl_resume_path $ROOT/logs/celeba/Owl/ syn-mixup-topk=2-variance-c=5-lr=5e-5_face/cf_tensors.pt 

#reset
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=3-00:00:00 python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --batch_size 32 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --lr 0.00005 --intervene_grad_reset --root_dir $ROOT/data/celeba --concept_bank_path $ROOT/logs/celeba/ERM/syn-concept_bank/last_model_0.1_150.pkl --log_dir $ROOT/logs/celeba/Owl/final-n_students=3-topk=3 --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts_200 --n_students 3 --topk 3 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/celeba/ERM/last_model.pth  --reweight_groups 