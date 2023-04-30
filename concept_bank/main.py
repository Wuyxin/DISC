import os
import shlex, subprocess


for N in range(10):
    command_line = f'srun -p jamesz -G 1 --time=1-00:00:00 python generate_concept_bank.py --concept {dir_name} --n_samples 200'
    args = shlex.split(command_line)
    p = subprocess.Popen(args)