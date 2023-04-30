<p align="center">
    <br>
    <img src="scripts/disc.png" width="450"/>
    <br>    

# Discover and Cure: Concept-aware Mitigation of Spurious Correlation (ICML 2023)

Paper is coming soon!

## **Overview**
**DISC** is an adaptive framework that discovers and mitigates spurious correlations during model training. 

## Installation

```shell
conda create -n disc python=3.9
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install scikit-learn transformers wilds umap-learn diffusers nltk
pip install tarfile zipfile gdown # Used for data download
```

## **Data download**

(Recommended) Download all the datasets via the commands below!

```
cd dataset
python download_all.py
```

Manual download (If auto download fails) :

- **MetaShift**: Download the dataset from [here](https://drive.google.com/drive/folders/1Ll3-4TNU_ZRKR2VoptUTnOGTDnGDixNt?usp=sharing). Unzipping this should result in a folder `metashifts`, which should be moved as `$ROOT/data/metashifts` depending on your root directory.
- **Waterbirds**: Download the dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). Unzipping this should result in a folder `waterbird_complete95_forest2water2`. Place this folder under `$ROOT/data/cub/`.
-  **FMoW**: Dataset download is automatic and can be found in `$ROOT/data/fmow/fmow_v1.1`. We recommend following the setup instructions provided by the official WILDS [website](https://wilds.stanford.edu/get_started/).
-  **ISIC**: Download the dataset from [here](https://drive.google.com/drive/folders/1Ll3-4TNU_ZRKR2VoptUTnOGTDnGDixNt?usp=sharing). Unzipping this should result in a folder `isic`, which should be moved as `$ROOT/data/isic` depending on your root directory.

## **Prepare Concept Bank**

(Recommended) Download the concept bank we have already generated via the commands below!

```
cd concept_bank
python download.py
```

Manual generation (Can be used for customizing your own concept bank!):

- Define the concept bank in `concept_bank/metadata.json`
- Run the generation using Stable Diffusion v1-4:
```
cd concept_bank
python generate_concept_bank.py --n_samples 200 
```


## Run DISC

We provide scripts in `scripts`.



## Reference

```
@inproceedings{
    wu2023disc,
    title={Discover and Cure: Concept-aware Mitigation of Spurious Correlation},
    author={Shirley Wu and Mert Yuksekgonul and Linjun Zhang and James Zou},
    booktitle={ICML},
    year={2023},
}
```
