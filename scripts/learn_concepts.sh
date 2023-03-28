# real
DATASET=cmnist
C_DATASET=CMNIST
ROOT=/oak/stanford/groups/jamesz/shirwu
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python learn_concepts.py --dataset $C_DATASET --concept-dir $ROOT/synthetic_concepts --model-path $ROOT/Owl/logs/$DATASET/ERM/last_model.pth --out-dir $ROOT/Owl/logs/$DATASET/ERM/syn-concept_bank_500_color --concept_sets color --syn

# syn
DATASET=waterbirds
C_DATASET=CUB
ROOT=/oak/stanford/groups/jamesz/shirwu
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python learn_concepts.py --dataset $C_DATASET --concept-dir $ROOT/synthetic_concepts --model-path $ROOT/Owl/logs/$DATASET/ERM/last_model.pth --out-dir $ROOT/Owl/logs/$DATASET/ERM/syn-concept_bank_nature_absent --C 0.1 --syn --n-samples 400 --concept_sets nature-color

# syn
DATASET=celeba
C_DATASET=CelebA
ROOT=/oak/stanford/groups/jamesz/shirwu
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python learn_concepts.py --dataset $C_DATASET --concept-dir $ROOT/synthetic_concepts_200 --model-path $ROOT/Owl/logs/$DATASET/ERM/last_model.pth --out-dir $ROOT/Owl/logs/$DATASET/ERM/syn-concept_bank --C 0.1 --syn

# syn
DATASET=metashifts
C_DATASET=MetaDatasetCatDog
GROUP=3
ROOT=/oak/stanford/groups/jamesz/shirwu
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python learn_concepts.py --dataset $C_DATASET --concept-dir $ROOT/synthetic_concepts --model-path $ROOT/Owl/logs/$DATASET/group_$GROUP/ERM/last_model.pth --out-dir $ROOT/Owl/logs/$DATASET/group_$GROUP/ERM/syn-concept_bank_400_absent --C 0.1 --syn --n-samples 400 --concept_sets everything --syn

# syn
SEED=5
DATASET=isic
C_DATASET=ISIC
ROOT=/oak/stanford/groups/jamesz/shirwu
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python learn_concepts.py --dataset $C_DATASET --concept-dir $ROOT/synthetic_concepts --model-path $ROOT/Owl/logs/$DATASET/group_$SEED/ERM/last_model.pth --out-dir $ROOT/Owl/logs/$DATASET/group_$SEED/ERM/syn-concept_bank_500_ct --C 0.1 --syn --n-samples 400 --concept_sets color-texture


# syn
DATASET=rxrx
C_DATASET=rxrx
ROOT=/oak/stanford/groups/jamesz/shirwu
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python learn_concepts.py --dataset $C_DATASET --concept-dir $ROOT/synthetic_concepts --model-path $ROOT/Owl/logs/$DATASET/ERM/last_model.pth --out-dir $ROOT/Owl/logs/$DATASET/ERM/syn-concept_bank_500_texture --C 0.1 --syn  --n-samples 400 --concept_sets texture



# syn
DATASET=cifar10
C_DATASET=CIFAR10
ROOT=/oak/stanford/groups/jamesz/shirwu
srun -p jamesz -G 1 --cpus-per-gpu=2 --time=1-00:00:00 python learn_concepts.py --dataset $C_DATASET --concept-dir $ROOT/synthetic_concepts --model-path $ROOT/Owl/logs/$DATASET/ERM/150_model.pth --out-dir $ROOT/Owl/logs/$DATASET/ERM/syn-concept_bank_500_ctnc --C 0.1 --syn  --n-samples 100 --concept_sets color-texture-nature-city
