#ERM
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CIFAR10 --lr 0.01 --batch_size 128 --weight_decay 5e-4 --model resnet50 --n_epochs 200  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/cifar10  --log_dir $ROOT/logs/cifar10/ERM_ood_test_no_aug --scheduler CosineAnnealingLR --save_best --save_last

#Owl
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=1-00:00:00 --mem 64GB python run_expt.py -s confounder -d CIFAR10 --lr 0.01 --batch_size 128 --weight_decay 5e-4 --model resnet50 --n_epochs 200  --gamma 0.1 --generalization_adjustment 0 --root_dir $ROOT/data/cifar10  --log_dir $ROOT/logs/cifar10/ERM_ood_test --augment_data --scheduler CosineAnnealingLR --save_best --save_last

SEED=1
conda activate torch
ROOT=/oak/stanford/groups/jamesz/shirwu/Owl
srun -p jamesz -G 1 --cpus-per-gpu=2 --mem 64GB --time=3-00:00:00 python run_expt.py -s confounder -d CIFAR10 --batch_size 128 --weight_decay 5e-4 --model resnet50 --n_epochs 200 --gamma 0.1 --generalization_adjustment 0 --intervene_grad_reset --root_dir $ROOT/data/cifar10 --concept_bank_path $ROOT/logs/cifar10/ERM/syn-concept_bank_500_ctnc/150_model_0.1_400.pkl --log_dir $ROOT/logs/cifar10/Owl/final-C=0.1-lr=5e-3-n_students=3-topk=5_umap_ctnc_seed=$SEED --concept_img_folder /oak/stanford/groups/jamesz/shirwu/synthetic_concepts  --lr 0.005 --n_students 3 --topk 5 --mix_ent --save_best --save_last --syn --pretrained_model_path $ROOT/logs/cifar10/ERM/150_model.pth --seed $SEED 
