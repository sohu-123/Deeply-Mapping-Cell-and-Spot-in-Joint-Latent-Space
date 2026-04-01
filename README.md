# Deeply-Mapping-Cell-and-Spot-in-Joint-Latent-Space

For input data, you can download here: https://drive.google.com/drive/folders/1Vf8iVi29hQqXOYWpDYSgmuAbWvS5l6XL?usp=sharing

Dependent Python Packages:
import os
import scanpy as sc
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import softmax, cosine_similarity
import logging
import numpy as np

You can Run example in JointEmbedding.ipynb
