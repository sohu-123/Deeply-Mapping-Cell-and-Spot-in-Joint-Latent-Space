import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_similarity_graph(y: np.ndarray, k: int = 15) -> np.ndarray:
    n_spots = y.shape[0]
    k = min(k, n_spots - 1)
    knn = NearestNeighbors(n_neighbors=k + 1).fit(y)
    distances, indices = knn.kneighbors(y)
    s = np.zeros((n_spots, n_spots), dtype=np.float32)
    for i in range(n_spots):
        sigma = max(np.median(distances[i, 1:]), 1e-8)
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                s[i, j] = np.exp(-(distances[i, j_idx] ** 2) / (2 * sigma ** 2))
    if np.sum(s) == 0:
        s += np.random.random(s.shape).astype(np.float32) * 0.01
    return s


def build_laplacian_from_similarity(s: np.ndarray) -> np.ndarray:
    deg = np.diag(s.sum(axis=1))
    return (deg - s).astype(np.float32)


def load_current_data(data_dir: Path, marker_path: Path | None = None):
    y_sc = pd.read_csv(data_dir / "adata_sc_obs.csv", usecols=[0, 1])
    y_sc.set_index(y_sc.columns[0], inplace=True)
    G = pd.read_csv(data_dir / "S3_GT.txt", sep=None, engine="python", index_col=0)
    adata_sc = ad.read_h5ad(data_dir / "scRNA_subsampled_20k.h5ad")
    loc_temp = pd.read_csv(data_dir / "spot_loc_with_counts_r_f.csv", index_col=0)
    n_cells = loc_temp.iloc[:, -1].to_frame()
    adata_st = ad.read_h5ad(data_dir / "Visium_FAD.h5ad")

    common_genes = adata_sc.var_names.intersection(adata_st.var_names)
    if marker_path is not None and marker_path.exists():
        markers = np.load(marker_path, allow_pickle=True).tolist()
        common_genes = pd.Index(common_genes).intersection(pd.Index(markers))

    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()

    common_barcodes = G.index.intersection(n_cells.index).intersection(adata_st.obs_names)
    G = G.loc[common_barcodes].copy()
    n_cells = n_cells.loc[common_barcodes].copy()
    adata_st = adata_st[common_barcodes, :].copy()

    X_sc = adata_sc.X.toarray() if hasattr(adata_sc.X, "toarray") else np.asarray(adata_sc.X)
    X_st = adata_st.X.toarray() if hasattr(adata_st.X, "toarray") else np.asarray(adata_st.X)
    cell_type_labels = y_sc.loc[adata_sc.obs_names].iloc[:, 0].astype(str).values
    Y = np.asarray(adata_st.obsm["spatial"], dtype=np.float32)

    return {
        "y_sc": y_sc,
        "G": G,
        "n_cells": n_cells,
        "adata_sc": adata_sc,
        "adata_st": adata_st,
        "X_sc": np.asarray(X_sc, dtype=np.float32),
        "X_st": np.asarray(X_st, dtype=np.float32),
        "Y": Y,
        "cell_type_labels": cell_type_labels,
        "common_genes": list(common_genes),
    }


def svd_warm_init(u: torch.Tensor, v: torch.Tensor, x: np.ndarray):
    try:
        from sklearn.utils.extmath import randomized_svd

        U, S, Vt = randomized_svd(x, n_components=u.shape[1], random_state=42)
    except Exception:
        U, S, Vt = np.linalg.svd(x, full_matrices=False)
        U, S, Vt = U[:, : u.shape[1]], S[: u.shape[1]], Vt[: u.shape[1], :]

    scale = np.sqrt(S).astype(np.float32)
    u_init = (U * scale).astype(np.float32)
    v_init = (Vt.T * scale).astype(np.float32)
    with torch.no_grad():
        u.copy_(torch.tensor(u_init, device=u.device))
        v.copy_(torch.tensor(v_init, device=v.device))


class SharedEmbeddingGRMF(nn.Module):
    def __init__(self, n_spots: int, n_cells: int, n_genes: int, rank: int):
        super().__init__()
        self.u_st = nn.Parameter(torch.randn(n_spots, rank) * 0.01)
        self.u_sc = nn.Parameter(torch.randn(n_cells, rank) * 0.01)
        self.v = nn.Parameter(torch.randn(n_genes, rank) * 0.01)
        self.st_row_bias = nn.Parameter(torch.zeros(n_spots, 1))
        self.sc_row_bias = nn.Parameter(torch.zeros(n_cells, 1))
        self.col_bias_st = nn.Parameter(torch.zeros(1, n_genes))
        self.col_bias_sc = nn.Parameter(torch.zeros(1, n_genes))
        self.global_bias = nn.Parameter(torch.zeros(1))

    def predict_st(self):
        return self.u_st @ self.v.t() + self.st_row_bias + self.col_bias_st + self.global_bias

    def predict_sc(self):
        return self.u_sc @ self.v.t() + self.sc_row_bias + self.col_bias_sc + self.global_bias


def row_mean_pcc(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    vals = []
    for i in range(x_true.shape[0]):
        t = x_true[i]
        p = x_pred[i]
        if np.std(t) > 1e-8 and np.std(p) > 1e-8:
            vals.append(float(pearsonr(t, p)[0]))
        else:
            vals.append(0.0)
    return float(np.mean(vals))


def composition_metrics(g_true: pd.DataFrame, g_pred: pd.DataFrame):
    g_true_np = g_true.to_numpy(dtype=np.float32)
    g_pred_np = g_pred.loc[g_true.index, g_true.columns].to_numpy(dtype=np.float32)
    pcc_list = []
    for i in range(g_true_np.shape[0]):
        gt_i = g_true_np[i]
        pred_i = g_pred_np[i]
        if np.std(gt_i) < 1e-8 or np.std(pred_i) < 1e-8:
            pcc_list.append(0.0)
        else:
            r = float(pearsonr(gt_i, pred_i)[0])
            pcc_list.append(r if np.isfinite(r) else 0.0)
    return {
        "deconv_mean_pcc": float(np.mean(pcc_list)),
        "deconv_mean_mae": float(np.mean(np.abs(g_true_np - g_pred_np))),
        "deconv_mean_rmse": float(np.sqrt(np.mean((g_true_np - g_pred_np) ** 2))),
        "pcc_all": pcc_list,
    }


def assign_cells_to_spots(u_st: np.ndarray, u_sc: np.ndarray, spot_names, cell_type_labels, g_columns, metric: str):
    nn_model = NearestNeighbors(n_neighbors=1, metric=metric)
    nn_model.fit(u_st)
    nearest = nn_model.kneighbors(u_sc, return_distance=False).reshape(-1)

    le = LabelEncoder()
    labels = le.fit_transform(cell_type_labels)
    pred_order = le.classes_.tolist()
    g_columns = list(g_columns)
    reorder_idx = [pred_order.index(ct) for ct in g_columns]
    one_hot = np.eye(len(pred_order), dtype=np.float32)[labels][:, reorder_idx]

    counts = np.zeros((u_st.shape[0], len(g_columns)), dtype=np.float32)
    np.add.at(counts, nearest, one_hot)
    row_sums = counts.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.squeeze() == 0)
    row_sums[row_sums == 0] = 1.0
    proportions = counts / row_sums
    g_pred = pd.DataFrame(proportions, index=spot_names, columns=g_columns)
    return g_pred, {
        "n_zero_assigned_spots": int(np.sum(zero_rows)),
        "mean_assigned_cells_per_spot": float(np.mean(counts.sum(axis=1))),
        "max_assigned_cells_per_spot": float(np.max(counts.sum(axis=1))),
    }


def run(args):
    set_seed(args.seed)
    device = choose_device()
    data = load_current_data(Path(args.data_dir), Path(args.marker_path) if args.marker_path else None)

    X_sc = np.log1p(data["X_sc"]).astype(np.float32)
    X_st = np.log1p(data["X_st"]).astype(np.float32)
    S = create_similarity_graph(data["Y"], k=args.k_graph)
    L = build_laplacian_from_similarity(S)

    model = SharedEmbeddingGRMF(
        n_spots=X_st.shape[0],
        n_cells=X_sc.shape[0],
        n_genes=X_st.shape[1],
        rank=args.rank,
    ).to(device)

    if args.svd_init:
        svd_warm_init(model.u_st, model.v, X_st)

    x_st_t = torch.tensor(X_st, dtype=torch.float32, device=device)
    x_sc_t = torch.tensor(X_sc, dtype=torch.float32, device=device)
    lap_t = torch.tensor(L, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed)
    sc_batch_size = min(args.sc_batch_size, X_sc.shape[0])
    losses = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        pred_st = model.predict_st()
        loss = F.mse_loss(pred_st, x_st_t)
        loss = loss + args.lambda_l2 * (
            model.u_st.pow(2).mean() + model.u_sc.pow(2).mean() + model.v.pow(2).mean()
        )
        if args.lambda_u_graph > 0:
            loss = loss + args.lambda_u_graph * (
                torch.trace(model.u_st.t() @ lap_t @ model.u_st) / model.u_st.shape[0]
            )
        if args.lambda_pred_graph > 0:
            loss = loss + args.lambda_pred_graph * (
                torch.sum(pred_st * (lap_t @ pred_st)) / pred_st.numel()
            )
        if args.lambda_sc > 0:
            idx = rng.choice(X_sc.shape[0], size=sc_batch_size, replace=False)
            idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            pred_sc = model.u_sc[idx_t] @ model.v.t() + model.sc_row_bias[idx_t] + model.col_bias_sc + model.global_bias
            loss = loss + args.lambda_sc * F.mse_loss(pred_sc, x_sc_t[idx_t])

        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % max(args.log_every, 1) == 0 or epoch == 0:
            print(f"epoch {epoch+1:04d}/{args.epochs} total={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_st = model.predict_st().cpu().numpy()
        pred_sc_full = model.predict_sc().cpu().numpy()
        u_st = model.u_st.detach().cpu().numpy()
        u_sc = model.u_sc.detach().cpu().numpy()

    g_pred_euclid, assign_stats_e = assign_cells_to_spots(
        u_st=u_st,
        u_sc=u_sc,
        spot_names=data["G"].index,
        cell_type_labels=data["cell_type_labels"],
        g_columns=data["G"].columns,
        metric="euclidean",
    )
    g_pred_cosine, assign_stats_c = assign_cells_to_spots(
        u_st=u_st,
        u_sc=u_sc,
        spot_names=data["G"].index,
        cell_type_labels=data["cell_type_labels"],
        g_columns=data["G"].columns,
        metric="cosine",
    )

    result = {
        "config": vars(args),
        "device": str(device),
        "shapes": {
            "G": list(data["G"].shape),
            "X_sc": list(X_sc.shape),
            "X_st": list(X_st.shape),
            "n_common_genes_used": int(X_st.shape[1]),
        },
        "spot_reconstruction": {
            "row_pcc": row_mean_pcc(X_st, pred_st),
            "mae": float(np.mean(np.abs(X_st - pred_st))),
            "rmse": float(np.sqrt(np.mean((X_st - pred_st) ** 2))),
        },
        "cell_reconstruction": {
            "row_pcc": row_mean_pcc(X_sc, pred_sc_full),
            "mae": float(np.mean(np.abs(X_sc - pred_sc_full))),
            "rmse": float(np.sqrt(np.mean((X_sc - pred_sc_full) ** 2))),
        },
        "euclidean_assignment": {
            **composition_metrics(data["G"], g_pred_euclid),
            **assign_stats_e,
        },
        "cosine_assignment": {
            **composition_metrics(data["G"], g_pred_cosine),
            **assign_stats_c,
        },
        "final_loss": losses[-1] if losses else None,
    }

    return result, g_pred_euclid, g_pred_cosine, losses


def main():
    parser = argparse.ArgumentParser(description="Stack X_st and X_sc into a shared-embedding GRMF, then assign cells to nearest spots to predict G.")
    parser.add_argument("--data-dir", default=r".\data")
    parser.add_argument("--marker-path", default=r".\Tangram\20k_markers.npy")
    parser.add_argument("--output-dir", default=r".\stacked_grmf_to_G")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--lambda-u-graph", type=float, default=1e-2)
    parser.add_argument("--lambda-pred-graph", type=float, default=1e-2)
    parser.add_argument("--lambda-sc", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=1e-4)
    parser.add_argument("--k-graph", type=int, default=15)
    parser.add_argument("--sc-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--svd-init", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result, g_pred_euclid, g_pred_cosine, losses = run(args)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    g_pred_euclid.to_csv(out_dir / "G_pred_euclidean.csv")
    g_pred_cosine.to_csv(out_dir / "G_pred_cosine.csv")
    pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(out_dir / "training_losses.csv", index=False)
    print(json.dumps(result, indent=2))
    print(f"saved metrics to {(out_dir / 'metrics.json').resolve()}")


if __name__ == "__main__":
    main()
