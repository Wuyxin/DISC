# make sure you're logged in with `huggingface-cli login`
import os
import os.path as osp
import argparse
import numpy as np
import json
import random
from torch import autocast
from diffusers import StableDiffusionPipeline
from utils import pluralize, synonym_extractor
from tqdm import tqdm


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--root", default='../', type=str) # root to store the generated concept bank
    return parser.parse_args()


args = config()

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)

metadata = json.load(open('../synthetic_concepts/metadata.json'))

concepts = [item for lst in list(metadata.values()) for item in lst]
random.shuffle(concepts)

for concept in tqdm(concepts):
    pos_root = os.path.join(args.root, f"synthetic_concepts/{concept}/positives")

    if os.path.exists(f'{pos_root}/0.png'):
        print(f'Skip {concept} due to existing images')
        continue 
    os.makedirs(pos_root, exist_ok=True)
    if concept[-2:] == '_s':
        prompt = concept[:-2]
    else:
        prompt = concept

    prompt = prompt.replace("_", " ")

    for i in range(args.n_samples):
        input_prompt = np.random.choice([
            prompt, pluralize(prompt)], p=[0.8, 0.2])
        with autocast("cuda"):
            image = pipe(prompt=input_prompt).images[0]
        image.save(f"{pos_root}/{i}.png")
