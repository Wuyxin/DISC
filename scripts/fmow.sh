
# ERM
ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=7-00:00:00 --mem 64GB python run_expt.py -s confounder -d FMoW --lr 1e-4 --batch_size 32 --weight_decay 0 --model densenet121 --optimizer Adam --scheduler StepLR --n_epochs 60 --root_dir $ROOT/data/fmow --log_dir $ROOT/output --save_best --save_last 

ROOT=/oak/stanford/groups/jamesz/shirwu/DISC
srun -p jamesz -G 1 --cpus-per-gpu=2  --time=7-00:00:00 --mem 64GB python run_expt.py -s confounder -d FMoW --lr 1e-4 --batch_size 32 --weight_decay 0 --model densenet121 --optimizer Adam --scheduler StepLR --n_epochs 60 --root_dir $ROOT/data/fmow --log_dir $ROOT/output --save_best --save_last --augment_data