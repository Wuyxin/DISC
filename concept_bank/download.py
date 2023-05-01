import os
import os.path as osp
import gdown
import zipfile
import shutil

# We store the concept bank in two .zip files due to its large size
# Set your ROOT here!
ROOT = '/oak/stanford/groups/jamesz/shirwu/test' 

concept_root = osp.join(ROOT, 'synthetic_concepts')
os.makedirs(concept_root, exist_ok=True)
tmp_dir = osp.join(ROOT, 'tmp')

# Download texture concepts
output = 'synthetic_concepts_texture.zip'
url = 'https://drive.google.com/uc?id=1EX0MP39xvUaYprCIReuTQWVZPj7xsops'
if not osp.exists(osp.join(tmp_dir, 'synthetic_concepts_texture.zip')):
    os.makedirs(tmp_dir, exist_ok=True)
    gdown.download(url, osp.join(tmp_dir, output), quiet=False)
    with zipfile.ZipFile(osp.join(tmp_dir, output), 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    # gather all files
    allfiles = os.listdir(osp.join(tmp_dir, 'texture'))
    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = osp.join(tmp_dir, 'texture', f)
        dst_path = os.path.join(concept_root, f)
        shutil.move(src_path, dst_path)


# Download all the concepts except for texture concepts
output = 'synthetic_concepts_exclude_texture.zip'
url = 'https://drive.google.com/uc?id=1h-9uS7kDjDZSbbZc90da3lzMjPpwLj_g'
if not osp.exists(osp.join(tmp_dir, 'synthetic_concepts_exclude_texture.zip')):
    os.makedirs(tmp_dir, exist_ok=True)
    gdown.download(url, osp.join(tmp_dir, output), quiet=False)
    with zipfile.ZipFile(osp.join(tmp_dir, output), 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    # gather all files
    allfiles = os.listdir(osp.join(tmp_dir, 'non_texture'))
    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = osp.join(tmp_dir, 'non_texture', f)
        dst_path = os.path.join(concept_root, f)
        shutil.move(src_path, dst_path)

os.remove(tmp_dir)

