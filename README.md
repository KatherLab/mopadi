# MoPaDi - Morphing Histopathology Diffusion

MoPaDi combines [Diffusion Autoencoders](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html) with multiple instance learning (MIL) for explainability of deep learning classifiers in histopathology. This repository contains the supplementary material for the following [preprint](https://www.biorxiv.org/content/10.1101/2024.10.29.620913v1):

> Žigutytė, L., Lenz, T., Han, T., Hewitt, K. J., Reitsam, N. G., Foersch, S., ... & Kather, J. N. (2024). Counterfactual Diffusion Models for Mechanistic Explainability of Artificial Intelligence Models in Pathology. bioRxiv, 2024.

![image info](./images/fig1_paper.png)

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

## Acknowledgements
This project was built upon a [DiffAE](https://github.com/phizaz/diffae) (MIT license) repository. We thank the developers for making their code open source.

## Reference
If you find our work useful for your research or if you use parts of the code please consider citing our [preprint](https://www.biorxiv.org/content/10.1101/2024.10.29.620913v1):

```
Žigutytė, L., Lenz, T., Han, T., Hewitt, K. J., Reitsam, N. G., Foersch, S., ... & Kather, J. N. (2024). Counterfactual Diffusion Models for Mechanistic Explainability of Artificial Intelligence Models in Pathology. bioRxiv, 2024.
```

```
@misc{zigutyte2024mopadi,
      title={ounterfactual Diffusion Models for Mechanistic Explainability of Artificial Intelligence Models in Pathology}, 
      author={Laura Žigutytė and Tim Lenz and Tianyu Han and Katherine Jane Hewitt and Nic Gabriel Reitsam and Sebastian Foersch and Zunamys I Carrero and Michaela Unger and Alexander T Pearson and Daniel Truhn and Jakob Nikolas Kather},
      year={2024},
      eprint={2024.10.29.620913},
      archivePrefix={bioRxiv},
      url={https://www.biorxiv.org/content/10.1101/2024.10.29.620913v1}, 
}
```


