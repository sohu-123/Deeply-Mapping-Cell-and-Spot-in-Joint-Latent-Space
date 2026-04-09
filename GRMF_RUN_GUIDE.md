# GRMF Eval-Only Run Guide

This guide covers the corrected GRMF pipeline where `G` is used only for final evaluation.

## Required files

Place the following files under `data/`:

- `adata_sc_obs.csv`
- `S3_GT.txt`
- `scRNA_subsampled_20k.h5ad`
- `spot_loc_with_counts_r_f.csv`
- `Visium_FAD.h5ad`

Optional marker file:

- `Tangram/20k_markers.npy`

## Main script

Run the corrected pipeline:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python run_grmf_assignment_evalonly.py --svd-init --output-dir .\grmf_assignment_evalonly_run
```

Outputs:

- `grmf_assignment_evalonly_run/metrics.json`
- `grmf_assignment_evalonly_run/G_pred_direct_row.csv`
- `grmf_assignment_evalonly_run/G_pred_row_assignment.csv`
- `grmf_assignment_evalonly_run/G_pred_column_assignment.csv`

## Recommended configuration

The best legal configuration found so far is:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python run_grmf_assignment_evalonly.py `
  --svd-init `
  --output-dir .\grmf_assignment_evalonly_final\final_k16_no_pred_graph `
  --rank 32 `
  --epochs-grmf 30 `
  --epochs-A 20 `
  --lr-grmf 0.05 `
  --lr-A 0.05 `
  --lambda-u-graph 0.01 `
  --lambda-pred-graph 0 `
  --lambda-sc 0.01 `
  --lambda-l2 1e-4 `
  --lambda-x-assign 1.0 `
  --lambda-dist 0.05 `
  --lambda-assign-pred-graph 0 `
  --lambda-entropy 0 `
  --lambda-count 0.1 `
  --k-graph 15 `
  --k-assign 16 `
  --assign-metric cosine
```

## Hyperparameter search

Run the short tuning sweep:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python tune_grmf_assignment_evalonly.py
```

Main summary:

- `tuning_grmf_assignment_evalonly/summary.json`

## Interpretation

- Model selection is based on non-`G` criteria only.
- `G` is used only in the final metrics written to `metrics.json`.
- In the current results, `direct_row_assignment` under the best GRMF embedding gives the strongest final `G` PCC.
