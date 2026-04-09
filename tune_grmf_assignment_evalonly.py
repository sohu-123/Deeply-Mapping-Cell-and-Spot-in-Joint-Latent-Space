import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_one(cfg: dict, output_root: Path) -> dict:
    run_name = cfg["name"]
    out_dir = output_root / run_name
    cmd = [
        sys.executable,
        "run_grmf_assignment_evalonly.py",
        "--output-dir",
        str(out_dir),
        "--rank",
        str(cfg["rank"]),
        "--epochs-grmf",
        str(cfg["epochs_grmf"]),
        "--epochs-A",
        str(cfg["epochs_A"]),
        "--lr-grmf",
        str(cfg["lr_grmf"]),
        "--lr-A",
        str(cfg["lr_A"]),
        "--lambda-u-graph",
        str(cfg["lambda_u_graph"]),
        "--lambda-pred-graph",
        str(cfg["lambda_pred_graph"]),
        "--lambda-sc",
        str(cfg["lambda_sc"]),
        "--lambda-l2",
        str(cfg["lambda_l2"]),
        "--lambda-x-assign",
        str(cfg["lambda_x_assign"]),
        "--lambda-dist",
        str(cfg["lambda_dist"]),
        "--lambda-assign-pred-graph",
        str(cfg["lambda_assign_pred_graph"]),
        "--lambda-entropy",
        str(cfg["lambda_entropy"]),
        "--lambda-count",
        str(cfg["lambda_count"]),
        "--k-assign",
        str(cfg["k_assign"]),
        "--k-graph",
        str(cfg["k_graph"]),
        "--assign-metric",
        cfg["assign_metric"],
        "--seed",
        str(cfg["seed"]),
        "--log-every",
        str(cfg["log_every"]),
        "--marker-path",
        cfg["marker_path"],
    ]
    if cfg.get("svd_init", False):
        cmd.append("--svd-init")
    subprocess.run(cmd, check=True)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    row = metrics["row_assignment"]
    grmf = metrics["grmf"]
    return {
        "name": run_name,
        "selection_metric": row["selection_metric"],
        "row_predictst_row_pcc": row["predictst_row_pcc"],
        "row_predictst_mse": row["predictst_mse"],
        "row_deconv_mean_pcc": row["deconv_mean_pcc"],
        "row_deconv_mean_mae": row["deconv_mean_mae"],
        "row_deconv_mean_rmse": row["deconv_mean_rmse"],
        "direct_row_deconv_mean_pcc": metrics["direct_row_assignment"]["deconv_mean_pcc"],
        "column_deconv_mean_pcc": metrics["column_assignment"]["deconv_mean_pcc"],
        "grmf_predictst_row_pcc": grmf["predictst_row_pcc_grmf"],
        "grmf_predictsc_row_pcc": grmf["predictsc_row_pcc_grmf"],
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune GRMF+assignment without using G for model selection.")
    parser.add_argument("--output-root", default=r".\tuning_grmf_assignment_evalonly")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base = {
        "rank": 32,
        "epochs_grmf": 8,
        "epochs_A": 6,
        "lr_grmf": 5e-2,
        "lr_A": 5e-2,
        "lambda_u_graph": 1e-2,
        "lambda_pred_graph": 1e-2,
        "lambda_sc": 1e-2,
        "lambda_l2": 1e-4,
        "lambda_x_assign": 1.0,
        "lambda_dist": 5e-2,
        "lambda_assign_pred_graph": 0.0,
        "lambda_entropy": 0.0,
        "lambda_count": 0.1,
        "k_assign": 8,
        "k_graph": 15,
        "assign_metric": "cosine",
        "seed": 7,
        "log_every": 4,
        "marker_path": r".\Tangram\20k_markers.npy",
        "svd_init": True,
    }

    grid = [
        {"name": "base"},
        {"name": "k16", "k_assign": 16},
        {"name": "no_u_graph", "lambda_u_graph": 0.0},
        {"name": "no_pred_graph", "lambda_pred_graph": 0.0},
        {"name": "no_sc", "lambda_sc": 0.0},
        {"name": "no_dist", "lambda_dist": 0.0},
        {"name": "assign_graph", "lambda_assign_pred_graph": 1e-2},
        {"name": "rank16", "rank": 16},
        {"name": "euclidean", "assign_metric": "euclidean"},
    ]

    rows = []
    for delta in grid:
        cfg = dict(base)
        cfg.update(delta)
        print(f"\n=== Running {cfg['name']} ===")
        rows.append(run_one(cfg, output_root))

    rows.sort(key=lambda x: x["selection_metric"], reverse=True)
    summary = {
        "selection_rule": "Choose best config by row_assignment.predictst_row_pcc only; G-based metrics are reported after training and never used for selection.",
        "results": rows,
        "best_by_selection_metric": rows[0]["name"] if rows else None,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
