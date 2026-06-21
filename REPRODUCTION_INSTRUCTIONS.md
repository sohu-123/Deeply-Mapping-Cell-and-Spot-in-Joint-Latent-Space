# Reproduction Instructions

## Purpose

This document provides the information required to reproduce the experiments in this repository and satisfies NC requirements for a test dataset and replication instructions.

## Test Dataset

The experiment relies on the following test files (place them in the project root or update paths accordingly):

- `scRNA_subsampled_20k.h5ad` — single-cell RNA-seq AnnData file (with `.var_names`, `.obs`).
- `Visium_FAD.h5ad` — spatial transcriptomics AnnData file with `.obsm['spatial']` coordinates.
- `20k_markers.npy` — marker gene list (NumPy array of gene names).
- `harmony_embedding.txt` — PCA/embedding guidance used during Stage 1.
- `spot_loc_with_counts_r_f.csv` — spot metadata with `imagerow`, `imagecol`, and `n_cells` columns.
- `S3_GT.txt` — ground-truth spot-to-cell mapping file for evaluation.

Some datasets are too large to include, you can find them here: https://drive.google.com/drive/folders/1Vf8iVi29hQqXOYWpDYSgmuAbWvS5l6XL?usp=sharing, Now, we include a short script to download the files.

## Environment Setup

Create the Conda environment using the provided specification:

```bash
conda env create -f environment.yml
conda activate GPU
```

If GPU is unavailable, install a CPU-compatible `torch` and adjust device selection in the scripts.

## Reproducing the Experiment

1. Place the required test files in the repository root (or `data/` and update paths).

2. Run the primary script:

```bash
python JointEmbedding4.py
```

3. Monitor console output for `Stage 1` and `Stage 2` completion messages. The two stages are:
- Stage 1: Train encoder and decoder; save checkpoint.
- Stage 2: Optimize assignment matrix `A` using the saved encoder outputs.

## Outputs and Evaluation

Result files:

- Model checkpoints: `SpatialVG_improved_NMF/models/*.pth`
- Predicted cell expressions and embeddings: `*.npy`
- PCC list (evaluation): `pcc_list_oursbest1.csv`

To validate reproduction, compare the `pcc_list_oursbest1.csv` values against expected benchmarks (if provided) or inspect the Pearson correlations computed in the script.

## Optional - Download Script Example

If you choose to host data publicly, include a small script `download_data.sh` like the following and call it before running the main script.

```bash
#!/usr/bin/env bash
# Example: download data from Google Drive or another host
# Replace URLs with your publicly hosted file locations
wget -O scRNA_subsampled_20k.h5ad "https://drive.google.com/file/d/1uk9A_Xgyy8UyxLzt0DLW6g2pf9U9dZtK/view?usp=sharing"
wget -O Visium_FAD.h5ad "https://drive.google.com/file/d/1a-xKHn2xXAcTODFKbtrlbVZK76Ik43Mu/view?usp=sharing"
wget -O 20k_markers.npy "https://drive.google.com/file/d/1taE4DiNtQBCg4SjHCZvKBBoP4ShOinMN/view?usp=sharing"
```

## Troubleshooting

- Missing `.obsm['spatial']`: confirm `Visium_FAD.h5ad` contains spatial coordinates, otherwise adapt `JointEmbedding4.py` to read coordinates from `adata.obs`.
- Device errors: edit `setup_device()` or change `cuda:5` to an available GPU device index or `cpu`.
- Dependency issues: check `environment.yml` and install missing packages via `pip` or `conda`.

## Notes for Submission

To meet NC submission requirements, include:

- The code repository with `REPRODUCTION_INSTRUCTIONS.md`.
- The test data files or a public link and a download script.
- A short note indicating any deviations (e.g., CPU-only runs).
