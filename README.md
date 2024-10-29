# MoPaDi - Morphing Histopathology Diffusion

MoPaDi combines [Diffusion Autoencoders](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html) with multiple instance learning (MIL) for explainability of deep learning classifiers in histopathology. This repository contains the supplementary material for the following publication:

> *Insert future preprint/publication here*

## Getting started

Create a virtual environment, e.g. with conda or mamba, clone the repository, and install required packages:

```
mamba create -n mopadi python=3.8 -c conda-forge
pip install -r requirements.txt
```

## Pretrained Models

TBA

## Datasets

- Diagnostic WSI from The Cancer Genome Atlas [(TCGA)](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)
- Histology images from uniform tumor regions in TCGA Whole Slide Images [(Komura & Ishikawa, 2021)](https://zenodo.org/records/5889558)
- 100,000 histological images of human colorectal cancer and healthy tissue [(Kather et al., 2018)](https://zenodo.org/records/1214456)
