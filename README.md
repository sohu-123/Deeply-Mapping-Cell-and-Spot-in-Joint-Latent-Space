# Deeply Mapping Cell and Spot in Joint Latent Space

This repository provides the implementation for **deeply mapping single‑cell RNA‑seq and spatial transcriptomics data into a joint latent space**. The model learns a shared representation that aligns cellular and spatial measurements, enabling cross‑modality integration, imputation, and downstream analysis.

## 📌 Overview

  ![Method: Figure1](./Figure1-final-fit-nolegend-2.png)

Single‑cell RNA‑seq (scRNA‑seq) captures gene expression at high resolution but loses spatial context, while spatial transcriptomics (ST) retains tissue architecture but at lower cellular resolution. This project bridges the two modalities by learning a **joint embedding** where cells from scRNA‑seq and spots from ST are mapped to a common latent space. The learned representations can be used for:

- Aligning cell types with spatial locations  
- Predicting unmeasured gene expression in spatial data  
- Integrating multiple datasets across technologies




## 🚀 Getting Started

### Prerequisites
conda env create -f environment.yml

- Python 3.8 or later  
- PyTorch (version ≥1.10)  
- CUDA‑capable GPU (recommended)  

### Installation

Clone the repository and install the required packages:
conda env create -f environment.yml

import os
import scanpy as sc
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import softmax, cosine_similarity
import logging
import numpy as np
## Project Summary

This repository implements a method for mapping single-cell RNA-seq (scRNA-seq) and spatial transcriptomics (ST) data into a joint latent space. The approach learns a shared representation that aligns cells and spatial spots, enabling cross-modal integration, spatial expression prediction, and downstream analyses such as cell-to-spot mapping.

Key capabilities:
- Align scRNA-seq cells with spatial spots in a shared embedding
- Predict spatial gene expression from scRNA-seq-derived cell profiles
- Produce a cell–spot assignment matrix for spatial deconvolution

## Repository Structure

- `JointEmbedding4.py`: Main experiment script containing training, evaluation, and optimization flows.
- `DrawPicture2.py`: Visualization utilities for spatial plots and embeddings.
- `environment.yml`: Conda environment specification for reproducibility.
- `README.md`: Original README; this file is an enhanced English version.
- `20k_markers.npy`: Marker gene list used by the scripts.
- `harmony_embedding.txt`: PCA/embedding guidance used during training.
- `spot_loc_with_counts_r_f.csv`: Spot metadata used for adjacency and smoothing.
- Additional folders: `compare/`, `dataGithub/`, `method/` contain comparisons, auxiliary data, and notebooks.

## Dependencies and Environment

We recommend using Conda. The provided `environment.yml` lists the packages required.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate GPU
```

Note: `environment.yml` contains GPU-specific PyTorch versions. If you do not have a CUDA-enabled GPU, replace `torch` with a CPU build compatible with your platform.

## Preparing Input Data

The main script expects these files in the repository root (or update the paths in the scripts accordingly):

- `scRNA_subsampled_20k.h5ad`
- `Visium_FAD.h5ad`
- `20k_markers.npy`
- `harmony_embedding.txt`
- `spot_loc_with_counts_r_f.csv`
- `S3_GT.txt`

Some datasets are too large to include, you can find them here: https://drive.google.com/drive/folders/1Vf8iVi29hQqXOYWpDYSgmuAbWvS5l6XL?usp=sharing, place them in a `data/` directory and modify the paths in `JointEmbedding4.py`.

## How to Run
Run the primary experiment with:

```bash
python JointEmbedding4.py
```

This script performs two main stages:
1. Train encoder/decoder (Stage 1) and save model checkpoints to `SpatialVG_improved_NMF/models/`.
2. Optimize assignment matrix `A` (Stage 2) and produce final predictions and evaluation.

A visualization helper:

```bash
python DrawPicture2.py
```

## Expected Outputs

After a successful run, the script generates files such as:

- `SpatialVG_improved_NMF/models/fix_enc_pca1_top5000_kl_soft_harm-best_result0.5-withmarker2testforzhou.pth`
- `SpatialVG_improved_NMF/models/fix_enc_pca1_top5000_kl_soft_harm_getA2forzhou.pth`
- `fix_enc_pca1_top5000_kl_soft_harm_fval.npy`
- `fix_enc_pca1_top5000_kl_soft_harm_z_cell.npy`
- `pcc_list_oursbest1.csv`

## Notes and Recommendations

- If your machine lacks a GPU, set the device to CPU in `JointEmbedding4.py` (change `cuda:5` to `cpu` or `cuda:0`).
- Ensure `Visium_FAD.h5ad` includes `adata.obsm['spatial']` or adapt the script to use coordinates from another column.
- For quick testing, use smaller h5ad files or subsets of the data before running full-scale experiments.

## Extending the Project

Possible directions:
- Explore additional spatial smoothing regularizers
- Expand the marker gene set for more robust predictions
- Adapt the pipeline for other ST data formats (Slide-seq, MERFISH, etc.)

## Contact

For questions or collaboration, please open an issue or reach out to the repository owner.
