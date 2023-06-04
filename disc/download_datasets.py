import os
import os.path as osp
import gdown
import zipfile
import tarfile

# Set your ROOT
ROOT = '..' 

data_root = osp.join(ROOT, 'data')
os.makedirs(data_root, exist_ok=True)

# Download metashift
output = 'metashifts.zip'
url = 'https://drive.google.com/uc?id=1WySOxBRkxAUlSokgZrC-0JaWZwcG5UMT'
if not osp.exists(osp.join(data_root, 'metashifts')):
    gdown.download(url, osp.join(data_root, output), quiet=False)
    with zipfile.ZipFile(osp.join(data_root, output), 'r') as zip_ref:
        zip_ref.extractall(data_root)

# Download waterbirds
if not osp.exists(osp.join(data_root, 'cub')):
    os.system(f'wget https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz --directory-prefix={data_root}')
    file = tarfile.open(osp.join(data_root, 'waterbird_complete95_forest2water2.tar.gz'))
    file.extractall(osp.join(data_root, 'cub'))
    file.close()

# Download ISIC
output = 'isic.zip'
if not osp.exists(osp.join(data_root, 'isic')):
    url = 'https://drive.google.com/uc?id=1Os34EapIAJM34DrwZMw2rRRJij3HAUDV'
    gdown.download(url, osp.join(data_root, output), quiet=False)
    with zipfile.ZipFile(osp.join(data_root, output), 'r') as zip_ref:
        zip_ref.extractall(data_root)

# Skip downloading FMoW

# Download CIFAR-10-C
# Can try axel instead if wget is too slow
print('Downloading CIFAR-10-C...\nIt might take a while.')
if not osp.exists(osp.join(data_root, 'CIFAR-10-C')):
    os.system(f'wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar --directory-prefix={data_root}')
    file = tarfile.open(osp.join(data_root, 'CIFAR-10-C.tar'))
    file.extractall(data_root)
    file.close()