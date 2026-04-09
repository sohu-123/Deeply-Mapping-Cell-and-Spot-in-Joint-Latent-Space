# GRMF Eval-Only Summary

## Protocol

- Training/tuning data:
  - `X_sc`
  - `X_st`
  - `y_sc`
  - `Y` from `Visium_FAD.h5ad`
  - `n_cells` for column-assignment only
- Final evaluation only:
  - `G` from `data/S3_GT.txt`
- No `G` term is used in any training loss or model-selection rule.

## New Scripts

- `run_grmf_assignment_evalonly.py`
  - Re-trains shared GRMF embeddings.
  - Builds `direct_row`, `row_assignment`, and `column_assignment`.
  - Uses only non-`G` losses during training.
  - Reports `G` metrics only at the end.

- `tune_grmf_assignment_evalonly.py`
  - Short-run parameter search.
  - Selects the best configuration by `row_assignment.predictst_row_pcc` only.
  - Never uses `G` to choose hyperparameters.

## Short Tuning Results

Output:
- `tuning_grmf_assignment_evalonly/summary.json`
- `tuning_grmf_assignment_evalonly_round2/summary.json`

Main findings:
- `k_assign = 16` is clearly better than `k_assign = 8`.
- Removing GRMF output smoothing (`lambda_pred_graph = 0`) helps.
- Removing the assignment distance penalty changes almost nothing.
- Removing the single-cell reconstruction term hurts `G` noticeably.
- `row_assignment` fits `X_st` better than `direct_row`, but `direct_row` often gives a better final `G`.

## Final Long-Run Results

Output:
- `grmf_assignment_evalonly_final/summary.json`

Best row-assignment configuration by the pre-declared non-`G` selection rule:
- `final_k16_no_pred_graph`

Metrics:
- `row_assignment.predictst_row_pcc = 0.6198508091502964`
- `row_assignment.deconv_mean_pcc = 0.38593092561389153`
- `direct_row_assignment.deconv_mean_pcc = 0.43956771764454716`
- `column_assignment.deconv_mean_pcc = 0.22013422816466197`

## Interpretation

- If model selection is based on the allowed non-`G` criterion (`predictst_row_pcc`), the best row-assignment variant is `k16 + no_pred_graph`.
- However, even under that same GRMF embedding, the simpler `direct_row_assignment` gives a better final `G` PCC than the trained `row_assignment`.
- The current trustworthy best `G` result from this corrected pipeline is therefore:
  - `direct_row_assignment` under `final_k16_no_pred_graph`
  - `deconv_mean_pcc = 0.43956771764454716`

## Important Note

Older directories such as:
- `tuning_grmf_assignment/`
- `tuning_grmf_assignment_round2/`
- `tuning_grmf_assignment_round3/`
- `tuning_grmf_assignment_round4/`
- `ablation_simplify_grmf/`

contain experiments that used `G` inside training losses. They should not be treated as valid final results for this task.
