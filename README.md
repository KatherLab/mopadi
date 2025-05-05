# MoPaDi - Morphing Histopathology Diffusion

MoPaDi combines [Diffusion Autoencoders](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html) with multiple instance learning (MIL) for explainability of deep learning classifiers in histopathology. 

> [!IMPORTANT] 
> This repository contains an updated version of the codebase. For the experiments described in the [preprint](https://www.biorxiv.org/content/10.1101/2024.10.29.620913v1), please refer to [version 0.0.1 of MoPaDi](https://github.com/KatherLab/mopadi/tree/v0.0.1).

For segmentation of 6 cell types to quantify changes in original and counterfactual images, [DeepCMorph](https://github.com/aiff22/DeepCMorph) pretrained models were used.

For preprocessing of whole slide images (WSIs), please refer to KatherLab's [STAMP protocol](https://github.com/KatherLab/STAMP).

---
**Table of Contents:**

- [Getting started](#getting-started)
- [Training the Models from Scratch](#training-the-models-from-scratch)
- [Pretrained Models](#pretrained-models)
- [Datasets](#datasets)
- [Acknowledgements](#acknowledgements)
- [Reference](#reference)

![image info](./images/fig1_paper.png)

## Getting started

Clone the repository and create a virtual environment to install required packages, e.g. with uv (instructions below), conda or mamba.

1. **Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies**. In the root of this repository, run:
   ```bash
   uv sync
   ```
   This creates a virtual environment at `.venv/` and installs all necessary Python dependencies.

3. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

You can obtain access to pretrained models on [Hugging Face](https://huggingface.co/KatherLab/MoPaDi).
Once the environment is set up and access to models has been granted, you can run the example notebooks (all the necessary data for these examples has been provided) or train your own models.

## Training the Models from Scratch

To train the models from scratch, follow these steps:

1. **Prepare the Environment**: Ensure you have set up the virtual environment and installed the required packages.

2. **Download Datasets**: Obtain the [Datasets](#Datasets) used in the preprint or use your own.

3. **Preprocess the Data**: If the dataset consists of WSIs and not tiles, use the [STAMP protocol](https://github.com/KatherLab/STAMP) for preprocessing WSIs as needed. The starting point for MoPaDi is folders of tiles (color normalized or not). Multiple cohorts can be used, all tiles do not need to be in the same folder. Resizing, if needed, can be done automatically during the training.  Accepted image formats: JPEG, TIFF and PNG.

> [!TIP] 
> ZIP files containing tiles for each patient ([STAMP's](https://github.com/KatherLab/STAMP) output) are also accepted and do not need to be extracted beforehand.

4. **Configure Training**: Modify [`conf.yaml`](https://github.com/KatherLab/mopadi/blob/main/conf.yaml) file to match your dataset, define output path and desired training parameters.

5. **Run Trainings**: Execute the training scripts for the desired models.

 - **Diffusion autoencoder**: the core component of MoPaDi, encodes and decodes the images. Training this model is the longest step in the pipeline. Depending on the data and hardware, training time may vary between a couple of days to a few weeks.
   ```
   mopadi autoenc --config conf.yaml
   ```
You can **evaluate the trained autoencoder** by adapting `src/mopadi/utils/reconstruct_1k_images.py` for your data to reconstruct images from the test set and compute corresponding metrics: SSIM, MS-SSIM, MSE.

 - **Latent DPM** (optional, not required for counterfactuals generation): for unconditional synthetic image generation. Enables sampling feature vectors from the latent space of semantic encoder, which are then decoded to synthetic histopathology tiles.
   ```
   mopadi latent --config conf.yaml
   ```
 - **Linear classifier**: the simplest classifier for linearly separable classes, based on the original [DiffAE](https://github.com/phizaz/diffae) method. Ground truth labels are needed for each tile. Enables counterfactual image generation.
   ```
   mopadi linear_classifier --config conf.yaml
   ```
 - **MIL classifier**: more complex approach to guide counterfactual image generation, when a label is given on a patient level and not for each tile, introduced in our [preprint](https://www.biorxiv.org/content/10.1101/2024.10.29.620913v1).
   ```
   mopadi mil --config conf.yaml --mode crossval
   mopadi mil --config conf.yaml --mode train
   mopadi mil --config conf.yaml --mode manipulate
   ```

## Pretrained Models

Pretrained models can be found on [Hugging Face](https://huggingface.co/KatherLab/MoPaDi). If you have already obtained access to models in that repository, automatic download is set up in example notebooks. Included models are:

<ol type="a">
  <li><b>Tissue classes </b>autoencoding diffusion model (trained on 224 x 224 px tiles from NCT-CRC-HE-100K dataset <a href="https://zenodo.org/records/1214456">(Kather et al., 2018)</a>) + linear 9 classes classifier (Adipose [ADI], background [BACK], debris [DEB], lymphocytes [LYM], mucus [MUC], smooth muscle [MUS], normal colon mucosa [NORM], cancer-associated stroma [STR], colorectal adenocarcinoma epithelium [TUM]);</li>
  <li><b>Colorectal (CRC) cancer</b> autoencoding diffusion model (trained on 512 x 512 px tiles (0.5 microns per px, MPP) from tumor regions from TCGA CRC cohort) + microsatellite instability (MSI) status MIL classifier (MSI high [MSIH] vs. nonMSIH);</li>
  <li><b>Breast cancer (BRCA)</b> autoencoding diffusion model (trained on 512 x 512 px tiles (0.5 MPP) from tumor regions from TCGA BRCA cohort) + breast cancer type (invasive lobular carcinoma [ILC] vs. invasive ductal carcinoma [IDC]) and E2 center MIL classifiers;</li>
  <li><b>Pancancer </b>autoencoding diffusion model (trained on 256 x 256 px tiles (varying MPP) from histology images from uniform tumor regions in TCGA WSI <a href="https://zenodo.org/records/5889558">(Komura & Ishikawa, 2021)</a>) + liver cancer types (hepatocellular carcinoma [HCC] vs. cholangiocarcinoma [CCA]) MIL & linear classifiers and lung cancer types (lung adenocarcinoma [LUAD] vs. lung squamous cell carcinoma [LUSC]) MIL & linear classifiers.</li>
</ol>

If you want to process multiple images/folders, you can use [`conf.yaml`](https://github.com/KatherLab/mopadi/blob/main/conf.yaml), with `use_pretrained` parameter set to `True`, which will trigger automatic download of the selected model, and the following command.
```
mopadi mil --config conf.yaml --mode manipulate
```

For this to work, make sure you are logged in to your Hugging Face account:
```
huggingface-cli login
```

You can check if you are already logged in by running:
```
huggingface-cli whoami
```

Examples of counterfactual images generated with corresponding models (please refer to the [preprint](https://www.biorxiv.org/content/10.1101/2024.10.29.620913v1) for more examples):
![image info](./images/models.png)

## Datasets

- Diagnostic WSI from The Cancer Genome Atlas [(TCGA)](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)
- Histology images from uniform tumor regions in TCGA Whole Slide Images [(Komura & Ishikawa, 2021)](https://zenodo.org/records/5889558)
- 100,000 histological images of human colorectal cancer and healthy tissue [(Kather et al., 2018)](https://zenodo.org/records/1214456)

## Acknowledgements
This project was built upon a [DiffAE](https://github.com/phizaz/diffae) (MIT license) repository. We thank the developers for making their code open source.

## Reference
If you find our work useful for your research or if you use parts of the code please consider citing our [preprint](https://www.biorxiv.org/content/10.1101/2024.10.29.620913v1):

> Žigutytė, L., Lenz, T., Han, T., Hewitt, K. J., Reitsam, N. G., Foersch, S., Carrero, Z. I., Unger, M., Pearson, T. A., Truhn, D. & Kather, J. N. (2024). Counterfactual Diffusion Models for Mechanistic Explainability of Artificial Intelligence Models in Pathology. bioRxiv, 2024.

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
