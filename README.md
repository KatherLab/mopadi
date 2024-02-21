# MoPaDi - Morphing Pathology Diffusion model

This repository contains the code for the adaptation of Diffusion Autoencoders for histopathology and contains the supplementary material for the following publication:

*Insert future preprint/publication here*

This repository is based on the official implementation of Diffusion Autoencoders (DiffAE) for natural images:
A CVPR 2022 (ORAL) paper ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html)).

Scripts in this repository are provided under the MIT license, with original authorship as specified within each file or folder. In cases where authorship is not explicitly mentioned, the work is attributed to the contributors of this repository and the copyright belongs to KatherLab.

## Getting started

Create a virtual environment, e.g. with conda or mamba:

```
mamba create -n mopadi python=3.8 -c conda-forge
```

After cloning the repository, required packages can be installed by running the following command:

```
pip install -r requirements.txt
```

Create *.env* file and add *WORKSPACE_PATH* environment variable containing the path of the workspace where MoPaDi was cloned.

## Pretrained Models

ToDo

## Datasets

ToDo

## Configurations

ToDo

## Configuration for latent classifier (for manipulation):

ToDo: adjust instructions for MIL

* Edit paths in the following file and run to create the file containing ground truth labels:

```
python3 data_prep/create_attr_file_from_clini_table.py
```

* Then either create a custom class in 'dataset.py' and add the paths ('data_paths' line 19) in 'configs/config.py'

* Create a configuration in 'configs/templates_cls.py'

* Add a new mode in ManipulateMode class in 'configs/choices.py'

* Edit ClsModel class in 'exp_linear_cls.py' (load_dataset function and lines 71-79) 

* Edit 'load_dataset' function in 'exp_linear_cls.py' (lines 136-148)

## Training

First, diffusion autoencoder needs to be trained. It requires 8 (40Gb) or 4 (80Gb) x A100.
Submit a batch job on the cluster:

```
sbatch run_hpc.py
```

Then run the following script to train latent DPM and latent classifier:

```
python run_exp01.py
```
