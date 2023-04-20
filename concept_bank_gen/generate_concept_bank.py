# make sure you're logged in with `huggingface-cli login`
import os
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
    parser.add_argument("--n_samples", default=200, type=int)
    parser.add_argument("--root", default='../', type=str) # root to store the generated concept bank
    return parser.parse_args()


strength = 0.8
args = config()

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)

metadata = json.load(open(f'metadata.json'))

concepts = [item for lst in list(metadata.values()) for item in lst]
random.shuffle(concepts)

for cnocept in tqdm(concepts):
    pos_root = osp.join(root, f"synthetic_concepts/{concept}/positives")
    os.makedirs(pos_root, exist_ok=True)

    if os.path.exists(f'{pos_root}/0.png'):
        continue 

    if concept[-2:] == '_s':
        prompt = concept[:-2]
    else:
        prompt = concept

    prompt = prompt.replace("_", " ")

    for i in range(args.n_samples):
        input_prompt = np.random.choice([
            prompt, pluralize(prompt)], p=[0.8, 0.2])
        with autocast("cuda"):
            image = pipe(prompt=input_prompt, strength=strength).images[0]
        image.save(f"{pos_root}/{i}.png")

with open(osp.join(root, "synthetic_concepts/metadata.json"), "w") as outfile:
    outfile.write(json.dumps(metadata, indent=4))
