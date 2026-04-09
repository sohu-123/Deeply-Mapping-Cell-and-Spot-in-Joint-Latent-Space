import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from run_stacked_grmf_assign_G import (
    SharedEmbeddingGRMF,
    build_laplacian_from_similarity,
    choose_device,
    composition_metrics,
    create_similarity_graph,
    load_current_data,
    set_seed,
    svd_warm_init,
)


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import pearsonr

    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    r = float(pearsonr(x, y)[0])
    return r if np.isfinite(r) else 0.0


def row_mean_pcc(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    vals = [pearson_safe(x_true[i], x_pred[i]) for i in range(x_true.shape[0])]
    return float(np.mean(vals))


def prepare_label_matrix(cell_type_labels: np.ndarray, g_columns) -> tuple[np.ndarray, list[str]]:
    le = LabelEncoder()
    labels = le.fit_transform(cell_type_labels)
    pred_order = le.classes_.tolist()
    g_columns = list(g_columns)
    reorder_idx = [pred_order.index(ct) for ct in g_columns]
    one_hot = np.eye(len(pred_order), dtype=np.float32)[labels][:, reorder_idx]
    return one_hot, g_columns


def train_shared_grmf_embeddings(
    X_sc: np.ndarray,
    X_st: np.ndarray,
    Y: np.ndarray,
    args,
    device: torch.device,
):
    S = create_similarity_graph(Y, k=args.k_graph)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_grmf)
    rng = np.random.default_rng(args.seed)
    sc_batch_size = min(args.sc_batch_size, X_sc.shape[0])

    history = []
    for epoch in range(args.epochs_grmf):
        model.train()
        optimizer.zero_grad()
        pred_st = model.predict_st()
        loss_st = F.mse_loss(pred_st, x_st_t)
        loss = loss_st
        loss_l2 = args.lambda_l2 * (
            model.u_st.pow(2).mean() + model.u_sc.pow(2).mean() + model.v.pow(2).mean()
        )
        loss = loss + loss_l2

        loss_u_graph = torch.tensor(0.0, device=device)
        if args.lambda_u_graph > 0:
            loss_u_graph = args.lambda_u_graph * (
                torch.trace(model.u_st.t() @ lap_t @ model.u_st) / model.u_st.shape[0]
            )
            loss = loss + loss_u_graph

        loss_pred_graph = torch.tensor(0.0, device=device)
        if args.lambda_pred_graph > 0:
            loss_pred_graph = args.lambda_pred_graph * (
                torch.sum(pred_st * (lap_t @ pred_st)) / pred_st.numel()
            )
            loss = loss + loss_pred_graph

        loss_sc = torch.tensor(0.0, device=device)
        if args.lambda_sc > 0:
            idx = rng.choice(X_sc.shape[0], size=sc_batch_size, replace=False)
            idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            pred_sc = (
                model.u_sc[idx_t] @ model.v.t()
                + model.sc_row_bias[idx_t]
                + model.col_bias_sc
                + model.global_bias
            )
            loss_sc = args.lambda_sc * F.mse_loss(pred_sc, x_sc_t[idx_t])
            loss = loss + loss_sc

        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch + 1,
                "total": float(loss.item()),
                "st": float(loss_st.item()),
                "sc": float(loss_sc.item()),
                "u_graph": float(loss_u_graph.item()),
                "pred_graph": float(loss_pred_graph.item()),
            }
        )
        if (epoch + 1) % max(args.log_every, 1) == 0 or epoch == 0:
            print(
                f"GRMF epoch {epoch+1:04d}/{args.epochs_grmf} "
                f"total={loss.item():.4f} st={loss_st.item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        pred_st = model.predict_st().detach().cpu().numpy()
        pred_sc = model.predict_sc().detach().cpu().numpy()
        u_st = model.u_st.detach().cpu().numpy()
        u_sc = model.u_sc.detach().cpu().numpy()
    metrics = {
        "predictst_row_pcc_grmf": row_mean_pcc(X_st, pred_st),
        "predictsc_row_pcc_grmf": row_mean_pcc(X_sc, pred_sc),
        "predictst_mse_grmf": float(np.mean((X_st - pred_st) ** 2)),
        "predictsc_mse_grmf": float(np.mean((X_sc - pred_sc) ** 2)),
    }
    return u_st, u_sc, pred_st, pred_sc, history, metrics


def build_row_candidates(u_st: np.ndarray, u_sc: np.ndarray, k: int, metric: str):
    knn = NearestNeighbors(n_neighbors=min(k, u_sc.shape[0]), metric=metric).fit(u_sc)
    distances, indices = knn.kneighbors(u_st)
    scale = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-8)
    init_scores = -distances / scale
    return indices.astype(np.int64), init_scores.astype(np.float32), distances.astype(np.float32)


def build_column_candidates(u_st: np.ndarray, u_sc: np.ndarray, k: int, metric: str):
    knn = NearestNeighbors(n_neighbors=min(k, u_st.shape[0]), metric=metric).fit(u_st)
    distances, indices = knn.kneighbors(u_sc)
    scale = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-8)
    init_scores = -distances / scale
    return indices.astype(np.int64), init_scores.astype(np.float32), distances.astype(np.float32)


class RowAssignment(nn.Module):
    def __init__(self, candidate_cells: np.ndarray, init_scores: np.ndarray):
        super().__init__()
        self.register_buffer("candidate_cells", torch.tensor(candidate_cells, dtype=torch.long))
        self.logits = nn.Parameter(torch.tensor(init_scores, dtype=torch.float32))

    def weights(self):
        return F.softmax(self.logits, dim=1)


class ColumnAssignment(nn.Module):
    def __init__(self, candidate_spots: np.ndarray, init_scores: np.ndarray):
        super().__init__()
        self.register_buffer("candidate_spots", torch.tensor(candidate_spots, dtype=torch.long))
        self.logits = nn.Parameter(torch.tensor(init_scores, dtype=torch.float32))

    def weights(self):
        return F.softmax(self.logits, dim=1)


def evaluate_row_assignment(
    weights: np.ndarray,
    candidate_cells: np.ndarray,
    X_sc: np.ndarray,
    X_st: np.ndarray,
    one_hot: np.ndarray,
    G_true: pd.DataFrame,
):
    pred_x = np.sum(weights[:, :, None] * X_sc[candidate_cells], axis=1)
    pred_g = np.sum(weights[:, :, None] * one_hot[candidate_cells], axis=1)
    pred_g = pred_g / np.clip(pred_g.sum(axis=1, keepdims=True), 1e-8, None)
    G_pred = pd.DataFrame(pred_g, index=G_true.index, columns=G_true.columns)
    return {
        "G_pred": G_pred,
        "predictst_row_pcc": row_mean_pcc(X_st, pred_x),
        "predictst_mse": float(np.mean((X_st - pred_x) ** 2)),
        **composition_metrics(G_true, G_pred),
    }


def evaluate_column_assignment(
    weights: np.ndarray,
    candidate_spots: np.ndarray,
    X_sc: np.ndarray,
    X_st: np.ndarray,
    one_hot: np.ndarray,
    G_true: pd.DataFrame,
):
    n_spots = X_st.shape[0]
    n_types = one_hot.shape[1]
    pred_x = np.zeros((n_spots, X_st.shape[1]), dtype=np.float32)
    pred_counts = np.zeros((n_spots, n_types), dtype=np.float32)
    for r in range(candidate_spots.shape[1]):
        spot_idx = candidate_spots[:, r]
        weight_r = weights[:, r]
        np.add.at(pred_x, spot_idx, weight_r[:, None] * X_sc)
        np.add.at(pred_counts, spot_idx, weight_r[:, None] * one_hot)
    pred_g = pred_counts / np.clip(pred_counts.sum(axis=1, keepdims=True), 1e-8, None)
    G_pred = pd.DataFrame(pred_g, index=G_true.index, columns=G_true.columns)
    return {
        "G_pred": G_pred,
        "predictst_row_pcc": row_mean_pcc(X_st, pred_x),
        "predictst_mse": float(np.mean((X_st - pred_x) ** 2)),
        "n_zero_assigned_spots": int(np.sum(pred_counts.sum(axis=1) == 0)),
        **composition_metrics(G_true, G_pred),
    }


def train_row_assignment_unsupervised(
    X_sc: np.ndarray,
    X_st: np.ndarray,
    one_hot: np.ndarray,
    G_true: pd.DataFrame,
    candidate_cells: np.ndarray,
    init_scores: np.ndarray,
    distances: np.ndarray,
    laplacian: np.ndarray,
    args,
    device: torch.device,
):
    model = RowAssignment(candidate_cells, init_scores).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_A)
    x_sc_t = torch.tensor(X_sc, dtype=torch.float32, device=device)
    x_st_t = torch.tensor(X_st, dtype=torch.float32, device=device)
    dist_t = torch.tensor(distances, dtype=torch.float32, device=device)
    lap_t = torch.tensor(laplacian, dtype=torch.float32, device=device)

    history = []
    for epoch in range(args.epochs_A):
        model.train()
        optimizer.zero_grad()
        w = model.weights()
        cells = model.candidate_cells
        pred_x = torch.sum(w.unsqueeze(-1) * x_sc_t[cells], dim=1)
        loss_x = F.mse_loss(pred_x, x_st_t)
        loss = args.lambda_x_assign * loss_x

        loss_dist = torch.tensor(0.0, device=device)
        if args.lambda_dist > 0:
            loss_dist = args.lambda_dist * torch.mean(w * dist_t)
            loss = loss + loss_dist

        loss_pred_graph = torch.tensor(0.0, device=device)
        if args.lambda_assign_pred_graph > 0:
            loss_pred_graph = args.lambda_assign_pred_graph * (
                torch.sum(pred_x * (lap_t @ pred_x)) / pred_x.numel()
            )
            loss = loss + loss_pred_graph

        loss_entropy = torch.tensor(0.0, device=device)
        if args.lambda_entropy > 0:
            entropy = -(w * torch.log(torch.clamp(w, min=1e-8))).sum(dim=1).mean()
            loss_entropy = args.lambda_entropy * entropy
            loss = loss + loss_entropy

        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch + 1,
                "total": float(loss.item()),
                "x": float(loss_x.item()),
                "dist": float(loss_dist.item()),
                "pred_graph": float(loss_pred_graph.item()),
                "entropy": float(loss_entropy.item()),
            }
        )
        if (epoch + 1) % max(args.log_every, 1) == 0 or epoch == 0:
            print(
                f"ROW-A epoch {epoch+1:04d}/{args.epochs_A} "
                f"total={loss.item():.4f} x={loss_x.item():.4f}"
            )

    with torch.no_grad():
        weights = model.weights().cpu().numpy()
    result = evaluate_row_assignment(weights, candidate_cells, X_sc, X_st, one_hot, G_true)
    result["train_history"] = history
    result["selection_metric"] = result["predictst_row_pcc"]
    return result


def train_column_assignment_unsupervised(
    X_sc: np.ndarray,
    X_st: np.ndarray,
    one_hot: np.ndarray,
    G_true: pd.DataFrame,
    n_cells_df: pd.DataFrame,
    candidate_spots: np.ndarray,
    init_scores: np.ndarray,
    distances: np.ndarray,
    args,
    device: torch.device,
):
    model = ColumnAssignment(candidate_spots, init_scores).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_A)
    x_sc_t = torch.tensor(X_sc, dtype=torch.float32, device=device)
    x_st_t = torch.tensor(X_st, dtype=torch.float32, device=device)
    one_hot_t = torch.tensor(one_hot, dtype=torch.float32, device=device)
    num_cell = torch.tensor(n_cells_df.iloc[:, 0].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    num_cell = num_cell / torch.clamp(num_cell.sum(), min=1e-8)
    dist_t = torch.tensor(distances, dtype=torch.float32, device=device)

    n_spots = X_st.shape[0]
    n_types = one_hot.shape[1]
    history = []
    for epoch in range(args.epochs_A):
        model.train()
        optimizer.zero_grad()
        w = model.weights()
        spots = model.candidate_spots

        pred_x = torch.zeros((n_spots, X_st.shape[1]), dtype=torch.float32, device=device)
        pred_counts = torch.zeros((n_spots, n_types), dtype=torch.float32, device=device)
        spot_mass = torch.zeros((n_spots,), dtype=torch.float32, device=device)
        for r in range(spots.shape[1]):
            spot_idx = spots[:, r]
            weight_r = w[:, r]
            pred_x.index_add_(0, spot_idx, weight_r.unsqueeze(1) * x_sc_t)
            pred_counts.index_add_(0, spot_idx, weight_r.unsqueeze(1) * one_hot_t)
            spot_mass.index_add_(0, spot_idx, weight_r)

        loss_x = F.mse_loss(pred_x, x_st_t)
        loss = args.lambda_x_assign * loss_x

        loss_dist = torch.tensor(0.0, device=device)
        if args.lambda_dist > 0:
            loss_dist = args.lambda_dist * torch.mean(w * dist_t)
            loss = loss + loss_dist

        loss_count = torch.tensor(0.0, device=device)
        if args.lambda_count > 0:
            spot_mass_n = spot_mass / torch.clamp(spot_mass.sum(), min=1e-8)
            loss_count = args.lambda_count * F.mse_loss(spot_mass_n, num_cell)
            loss = loss + loss_count

        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch + 1,
                "total": float(loss.item()),
                "x": float(loss_x.item()),
                "dist": float(loss_dist.item()),
                "count": float(loss_count.item()),
            }
        )
        if (epoch + 1) % max(args.log_every, 1) == 0 or epoch == 0:
            print(
                f"COL-A epoch {epoch+1:04d}/{args.epochs_A} "
                f"total={loss.item():.4f} x={loss_x.item():.4f}"
            )

    with torch.no_grad():
        weights = model.weights().cpu().numpy()
    result = evaluate_column_assignment(weights, candidate_spots, X_sc, X_st, one_hot, G_true)
    result["train_history"] = history
    result["selection_metric"] = result["predictst_row_pcc"]
    return result


def direct_row_assignment(
    X_sc: np.ndarray,
    X_st: np.ndarray,
    one_hot: np.ndarray,
    G_true: pd.DataFrame,
    candidate_cells: np.ndarray,
    init_scores: np.ndarray,
):
    logits = init_scores - np.max(init_scores, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    result = evaluate_row_assignment(weights, candidate_cells, X_sc, X_st, one_hot, G_true)
    result["selection_metric"] = result["predictst_row_pcc"]
    return result


def run(args):
    set_seed(args.seed)
    device = choose_device()
    data = load_current_data(Path(args.data_dir), Path(args.marker_path) if args.marker_path else None)
    X_sc = np.log1p(data["X_sc"]).astype(np.float32)
    X_st = np.log1p(data["X_st"]).astype(np.float32)
    one_hot, g_columns = prepare_label_matrix(data["cell_type_labels"], data["G"].columns)

    u_st, u_sc, pred_st_grmf, pred_sc_grmf, grmf_history, grmf_metrics = train_shared_grmf_embeddings(
        X_sc, X_st, data["Y"], args, device
    )
    S = create_similarity_graph(data["Y"], k=args.k_graph)
    L = build_laplacian_from_similarity(S)

    row_cells, row_init, row_dist = build_row_candidates(u_st, u_sc, args.k_assign, args.assign_metric)
    col_spots, col_init, col_dist = build_column_candidates(u_st, u_sc, args.k_assign, args.assign_metric)

    direct_row = direct_row_assignment(
        X_sc=X_sc,
        X_st=X_st,
        one_hot=one_hot,
        G_true=data["G"][g_columns],
        candidate_cells=row_cells,
        init_scores=row_init,
    )
    row_result = train_row_assignment_unsupervised(
        X_sc=X_sc,
        X_st=X_st,
        one_hot=one_hot,
        G_true=data["G"][g_columns],
        candidate_cells=row_cells,
        init_scores=row_init,
        distances=row_dist,
        laplacian=L,
        args=args,
        device=device,
    )
    col_result = train_column_assignment_unsupervised(
        X_sc=X_sc,
        X_st=X_st,
        one_hot=one_hot,
        G_true=data["G"][g_columns],
        n_cells_df=data["n_cells"],
        candidate_spots=col_spots,
        init_scores=col_init,
        distances=col_dist,
        args=args,
        device=device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_st_grmf, index=data["G"].index, columns=data["common_genes"]).to_csv(out_dir / "Xst_pred_grmf.csv")
    direct_row["G_pred"].to_csv(out_dir / "G_pred_direct_row.csv")
    row_result["G_pred"].to_csv(out_dir / "G_pred_row_assignment.csv")
    col_result["G_pred"].to_csv(out_dir / "G_pred_column_assignment.csv")

    metrics = {
        "config": vars(args),
        "device": str(device),
        "shapes": {
            "G": list(data["G"].shape),
            "X_sc": list(X_sc.shape),
            "X_st": list(X_st.shape),
            "n_common_genes_used": int(X_st.shape[1]),
        },
        "grmf": grmf_metrics,
        "direct_row_assignment": {k: v for k, v in direct_row.items() if k != "G_pred"},
        "row_assignment": {k: v for k, v in row_result.items() if k not in {"G_pred", "train_history"}},
        "column_assignment": {k: v for k, v in col_result.items() if k not in {"G_pred", "train_history"}},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "grmf_history.json").write_text(json.dumps(grmf_history, indent=2), encoding="utf-8")
    (out_dir / "row_assignment_history.json").write_text(json.dumps(row_result["train_history"], indent=2), encoding="utf-8")
    (out_dir / "column_assignment_history.json").write_text(json.dumps(col_result["train_history"], indent=2), encoding="utf-8")
    compact = {
        "grmf": grmf_metrics,
        "direct_row_assignment": {k: v for k, v in direct_row.items() if k not in {"G_pred", "pcc_all"}},
        "row_assignment": {k: v for k, v in row_result.items() if k not in {"G_pred", "train_history", "pcc_all"}},
        "column_assignment": {k: v for k, v in col_result.items() if k not in {"G_pred", "train_history", "pcc_all"}},
    }
    print(json.dumps(compact, indent=2))
    print(f"saved metrics to {(out_dir / 'metrics.json').resolve()}")


def main():
    parser = argparse.ArgumentParser(description="GRMF + assignment with G used only for final evaluation.")
    parser.add_argument("--data-dir", default=r".\data")
    parser.add_argument("--marker-path", default=r".\Tangram\20k_markers.npy")
    parser.add_argument("--output-dir", default=r".\grmf_assignment_evalonly")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs-grmf", type=int, default=30)
    parser.add_argument("--epochs-A", type=int, default=20)
    parser.add_argument("--lr-grmf", type=float, default=5e-2)
    parser.add_argument("--lr-A", type=float, default=5e-2)
    parser.add_argument("--lambda-u-graph", type=float, default=1e-2)
    parser.add_argument("--lambda-pred-graph", type=float, default=1e-2)
    parser.add_argument("--lambda-sc", type=float, default=1e-2)
    parser.add_argument("--lambda-l2", type=float, default=1e-4)
    parser.add_argument("--lambda-x-assign", type=float, default=1.0)
    parser.add_argument("--lambda-dist", type=float, default=5e-2)
    parser.add_argument("--lambda-assign-pred-graph", type=float, default=0.0)
    parser.add_argument("--lambda-entropy", type=float, default=0.0)
    parser.add_argument("--lambda-count", type=float, default=0.1)
    parser.add_argument("--k-graph", type=int, default=15)
    parser.add_argument("--k-assign", type=int, default=8)
    parser.add_argument("--assign-metric", choices=["euclidean", "cosine"], default="cosine")
    parser.add_argument("--sc-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--svd-init", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
