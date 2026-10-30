"""Train/evaluate compact gene-level graph and hybrid neural models.

The split is gene-level and transductive: graph structure and non-label node
features are visible for every mapped candidate, while only development labels
enter fitting/tuning. Unknown target labels are always masked.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from graph_models import NodeClassifier, normalized_sparse_adjacency

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = max((p for p in (ROOT / "data/processed").glob("integrated_model_inputs_*") if (p / "integrated_input_summary.json").exists()), key=lambda p: p.stat().st_mtime)
SEED = 42
MODEL_NAMES = [
    "expression_residual", "expression_transformer",
    "graph_gcn", "graph_sage", "graph_gat",
    "expression_gat_concat", "expression_sage_concat",
    "hmgat_et_gated", "hmgat_et_concat",
]
CONFIG = {"seed": SEED, "folds": 3, "test_fraction": 0.20, "max_epochs": 70,
    "patience": 12, "learning_rate": 0.008, "weight_decay": 0.001,
    "hidden": 64, "dropout": 0.30, "class_weight": "training-fold negative/positive ratio",
    "graph_protocol": "transductive; held-out labels masked, graph and non-label node features visible",
    "edge_weights": "STRING combined_score / 1000, threshold >=700"}
MAX_RSS_GIB = 0.0


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def metric_pack(y, p, threshold=0.5):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    klist = {}
    for k in (10, 25, 50):
        k = min(k, len(y))
        chosen = np.argsort(-p)[:k]
        relevant = int(y[chosen].sum())
        klist[f"precision_at_{k}"] = relevant / k if k else 0.0
        klist[f"recall_at_{k}"] = relevant / max(int(y.sum()), 1)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "threshold": float(threshold),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        **klist,
    }


def make_graph_features(feature_frame, edge_index, edge_weight, n):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    src, dst = edge_index
    for a, b, w in zip(src, dst, edge_weight):
        if a < b:
            graph.add_edge(int(a), int(b), weight=float(w))
    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))
    clustering = nx.clustering(graph, weight="weight")
    return np.asarray([[math.log1p(degree[i]), math.log1p(strength[i]), clustering[i]] for i in range(n)], dtype=np.float32), graph


def prepare_data():
    meta = json.loads((INPUT_DIR / "integrated_input_summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(INPUT_DIR / "gene_features_and_labels.csv")
    x_expr = np.load(INPUT_DIR / "expression_features_raw.npy", allow_pickle=False)
    target = np.load(INPUT_DIR / "target_values.npy", allow_pickle=False)
    mask = np.load(INPUT_DIR / "target_mask.npy", allow_pickle=False)
    edge_npz = np.load(INPUT_DIR / "graph_edge_index.npz")
    edge_index = edge_npz["edge_index"].astype(np.int64)
    edge_weight = edge_npz["edge_weight"].astype(np.float32)
    if not (len(frame) == len(x_expr) == len(target) == len(mask) == meta["gene_count"]):
        raise ValueError("Integrated input row counts do not match")
    if frame.gene_index.tolist() != list(range(len(frame))):
        raise ValueError("Gene-index table is not in contiguous canonical order")
    topology, graph = make_graph_features(frame, edge_index[:, :edge_index.shape[1] // 2], edge_weight[:edge_weight.shape[0] // 2], len(frame))
    known = np.flatnonzero(mask)
    y = target[known].astype(np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Supervised target must contain both explicit classes")
    return frame, x_expr, topology, edge_index, edge_weight, known, y, graph


def scaler_transform(raw, train_nodes):
    scaler = StandardScaler()
    scaler.fit(raw[train_nodes])
    return scaler.transform(raw).astype(np.float32), scaler


def device_tensors(xexpr, xtopo, edge_index, edge_weight, n):
    device = torch.device("cpu")
    ei = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    ew = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)
    adj = normalized_sparse_adjacency(ei, ew, n)
    return (torch.as_tensor(xexpr, dtype=torch.float32, device=device),
        torch.as_tensor(xtopo, dtype=torch.float32, device=device), ei, ew, adj)


def forward(model, tensors):
    xe, xt, ei, ew, adj = tensors
    return model(xe, xt, adj, ei, ew)


def fit_epoch(model, tensors, train_idx, y_all, optimizer, pos_weight):
    global MAX_RSS_GIB
    MAX_RSS_GIB = max(MAX_RSS_GIB, psutil.Process().memory_info().rss / 1024**3)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = forward(model, tensors)
    target = torch.as_tensor(y_all[train_idx], dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[train_idx], target, pos_weight=torch.tensor(pos_weight))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def predict(model, tensors):
    model.eval()
    return torch.sigmoid(forward(model, tensors)).cpu().numpy()


def make_scaled_tensors(x_expr, x_topo, edge_index, edge_weight, n, train_nodes):
    xe, expr_scaler = scaler_transform(x_expr, train_nodes)
    xt, topo_scaler = scaler_transform(x_topo, train_nodes)
    return device_tensors(xe, xt, edge_index, edge_weight, n), expr_scaler, topo_scaler


def select_threshold(y, p):
    best = (float("-inf"), 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (p >= threshold).astype(np.int8)
        score = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else -1.0
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def smoke_test(n=80):
    set_seed(SEED)
    x_expr = torch.randn(n, 3)
    x_topo = torch.randn(n, 3)
    src = torch.randint(0, n, (500,))
    dst = torch.randint(0, n, (500,))
    loops = torch.arange(n)
    ei = torch.stack([torch.cat([src, dst, loops]), torch.cat([dst, src, loops])])
    ew = torch.cat([torch.rand(1000) * 0.3 + 0.7, torch.ones(n)])
    adj = normalized_sparse_adjacency(ei, ew, n)
    tensors = (x_expr, x_topo, ei, ew, adj)
    results = {}
    for name in MODEL_NAMES:
        model = NodeClassifier(name, dropout=0.2)
        logits = forward(model, tensors)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:40], torch.randint(0, 2, (40,)).float())
        loss.backward()
        finite = bool(torch.isfinite(logits).all() and torch.isfinite(loss))
        grads = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        results[name] = {"output_shape": list(logits.shape), "loss_finite": finite, "gradients_finite": bool(grads)}
        if not finite or not grads:
            raise RuntimeError(f"Smoke test failed for {name}: {results[name]}")
    print(json.dumps({"smoke_test": "passed", "models": results}, indent=2))


def train_cv_model(name, fold, train_nodes, val_nodes, y_by_node, x_expr, x_topo, edge_index, edge_weight, n, run_dir):
    set_seed(SEED + fold * 101 + MODEL_NAMES.index(name))
    tensors, expr_scaler, topo_scaler = make_scaled_tensors(x_expr, x_topo, edge_index, edge_weight, n, train_nodes)
    model = NodeClassifier(name, dropout=CONFIG["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    npos = int(y_by_node[train_nodes].sum()); nneg = len(train_nodes) - npos
    pos_weight = nneg / max(npos, 1)
    best_ap, best_epoch, stale = -1.0, 0, 0
    log_rows = []
    best_path = run_dir / "checkpoints" / name / f"fold_{fold}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / name).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        loss = fit_epoch(model, tensors, train_nodes, y_by_node, optimizer, pos_weight)
        val_prob = predict(model, tensors)[val_nodes]
        y_val = y_by_node[val_nodes]
        ap = float(average_precision_score(y_val, val_prob))
        roc = float(roc_auc_score(y_val, val_prob))
        log_rows.append({"epoch": epoch, "train_bce": loss, "validation_ap": ap, "validation_roc_auc": roc})
        if ap > best_ap + 1e-6:
            best_ap, best_epoch, stale = ap, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "name": name,
                "seed": SEED + fold * 101 + MODEL_NAMES.index(name), "pos_weight": pos_weight}, best_path)
        else:
            stale += 1
        if stale >= CONFIG["patience"]:
            break
    pd.DataFrame(log_rows).to_csv(run_dir / "logs" / name / f"fold_{fold}_epochs.csv", index=False)
    state = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    probabilities = predict(model, tensors)
    return model, probabilities, best_epoch, best_ap, expr_scaler, topo_scaler


def train_final(name, train_nodes, y_by_node, x_expr, x_topo, edge_index, edge_weight, n, epochs, run_dir):
    set_seed(SEED + 999 + MODEL_NAMES.index(name))
    tensors, expr_scaler, topo_scaler = make_scaled_tensors(x_expr, x_topo, edge_index, edge_weight, n, train_nodes)
    model = NodeClassifier(name, dropout=CONFIG["dropout"])
    (run_dir / "logs" / name).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    npos = int(y_by_node[train_nodes].sum()); nneg = len(train_nodes) - npos
    pos_weight = nneg / max(npos, 1)
    rows = []
    for epoch in range(1, max(1, epochs) + 1):
        loss = fit_epoch(model, tensors, train_nodes, y_by_node, optimizer, pos_weight)
        rows.append({"epoch": epoch, "train_bce": loss})
        if epoch % 10 == 0:
            checkpoint = run_dir / "checkpoints" / name / "final_periodic.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "name": name, "seed": SEED + 999 + MODEL_NAMES.index(name)}, checkpoint)
    pd.DataFrame(rows).to_csv(run_dir / "logs" / name / "final_train_epochs.csv", index=False)
    checkpoint = run_dir / "checkpoints" / name / "final.pt"
    torch.save({"model": model.state_dict(), "epoch": epochs, "name": name,
        "expression_scaler_mean": expr_scaler.mean_.tolist(), "expression_scaler_scale": expr_scaler.scale_.tolist(),
        "topology_scaler_mean": topo_scaler.mean_.tolist(), "topology_scaler_scale": topo_scaler.scale_.tolist()}, checkpoint)
    return model, checkpoint, expr_scaler, topo_scaler


def component_summary(graph, known_nodes, y_by_node):
    components = list(nx.connected_components(graph))
    counts = []
    for i, comp in enumerate(components):
        labeled = [n for n in comp if n in known_nodes]
        pos = sum(int(y_by_node[n]) for n in labeled)
        counts.append({"component": i, "nodes": len(comp), "labeled": len(labeled),
            "positive": pos, "explicit_zero": len(labeled) - pos})
    giant = max(counts, key=lambda x: x["nodes"])
    other_label_counts = {"positive": sum(x["positive"] for x in counts if x["component"] != giant["component"]),
        "explicit_zero": sum(x["explicit_zero"] for x in counts if x["component"] != giant["component"])}
    return {"component_count": len(components), "largest_component": giant,
        "all_other_components_labeled_support": other_label_counts,
        "component_holdout_feasible": min(other_label_counts.values()) >= 10,
        "interpretation": "A class-balanced component-held-out split is considered infeasible when the non-giant components do not contain at least 10 examples per class."}


def main():
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=CONFIG["max_epochs"])
    args = parser.parse_args()
    CONFIG["max_epochs"] = args.epochs
    torch.set_num_threads(2)
    if args.smoke_only:
        smoke_test()
        return
    set_seed(SEED)
    frame, x_expr, x_topo, edge_index, edge_weight, known, y, graph = prepare_data()
    n = len(frame)
    y_by_node = np.zeros(n, dtype=np.int8)
    y_by_node[known] = y
    split = StratifiedShuffleSplit(n_splits=1, test_size=CONFIG["test_fraction"], random_state=SEED)
    dev_local, test_local = next(split.split(np.zeros(len(known)), y))
    dev_nodes, test_nodes = known[dev_local], known[test_local]
    cv = StratifiedKFold(n_splits=CONFIG["folds"], shuffle=True, random_state=SEED + 1)
    run_dir = ROOT / "outputs" / "experiments" / ("integrated_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for folder in ("checkpoints", "logs", "predictions", "plots", "configs", "metrics"):
        (run_dir / folder).mkdir(parents=True, exist_ok=False)
    (run_dir / "configs" / "experiment.json").write_text(json.dumps({**CONFIG, "started_at": started_at, "input_dir": str(INPUT_DIR.relative_to(ROOT)),
        "gene_count": n, "known_label_count": len(known), "positive": int(y.sum()), "explicit_zero": int(len(y) - y.sum()),
        "torch": torch.__version__, "device": "cpu", "cpu_threads": torch.get_num_threads(),
        "ram_available_at_start_gib": round(psutil.virtual_memory().available / 1024**3, 3)}, indent=2), encoding="utf-8")
    y_by_node[dev_nodes] = y_by_node[dev_nodes]
    y_by_node[test_nodes] = y_by_node[test_nodes]
    split_role = np.full(n, "unlabeled", dtype=object)
    split_role[dev_nodes] = "development"
    split_role[test_nodes] = "held_out_test"
    split_frame = frame[["gene_index", "entrez", "symbol", "ensembl_gene_id_versioned", "oncogene_status", "oncogene_label"]].copy()
    split_frame["partition"] = split_role
    component_by_node = {}
    for ci, comp in enumerate(nx.connected_components(graph)):
        for node in comp:
            component_by_node[node] = ci
    split_frame["ppi_component"] = [component_by_node.get(i, -1) for i in range(n)]
    split_frame.to_csv(run_dir / "predictions" / "split_manifest.csv", index=False)
    component_eval = component_summary(graph, set(known), y_by_node)
    (run_dir / "metrics" / "component_holdout_feasibility.json").write_text(json.dumps(component_eval, indent=2), encoding="utf-8")

    results = []
    cv_predictions = {}
    all_logs = {}
    cv_fold_map = {}
    for name in MODEL_NAMES:
        print(f"Starting {name}", flush=True)
        oof = np.full(n, np.nan, dtype=np.float32)
        fold_records, best_epochs, fold_logs = [], [], []
        for fold, (tr_local, va_local) in enumerate(cv.split(dev_nodes, y_by_node[dev_nodes]), start=1):
            tr_nodes, va_nodes = dev_nodes[tr_local], dev_nodes[va_local]
            model, probabilities, best_epoch, best_ap, _, _ = train_cv_model(name, fold, tr_nodes, va_nodes, y_by_node,
                x_expr, x_topo, edge_index, edge_weight, n, run_dir)
            p_val = probabilities[va_nodes]
            oof[va_nodes] = p_val
            best_epochs.append(best_epoch)
            fold_records.append({"model": name, "fold": fold, "validation_average_precision": best_ap,
                "validation_roc_auc": float(roc_auc_score(y_by_node[va_nodes], p_val)), "best_epoch": best_epoch,
                "train_n": len(tr_nodes), "validation_n": len(va_nodes),
                "validation_positive": int(y_by_node[va_nodes].sum()), "validation_explicit_zero": int(len(va_nodes) - y_by_node[va_nodes].sum())})
            fold_logs.append(pd.read_csv(run_dir / "logs" / name / f"fold_{fold}_epochs.csv").assign(fold=fold))
            cv_fold_map.update({int(i): fold for i in va_nodes})
        cv_predictions[name] = oof
        all_logs[name] = pd.concat(fold_logs, ignore_index=True)
        pd.DataFrame(fold_records).to_csv(run_dir / "metrics" / f"{name}_cv_folds.csv", index=False)
        median_epoch = max(1, int(np.median(best_epochs)))
        final_model, checkpoint, expr_scaler, topo_scaler = train_final(name, dev_nodes, y_by_node, x_expr, x_topo,
            edge_index, edge_weight, n, median_epoch, run_dir)
        # The held-out test is evaluated once after the model's CV-selected epoch is fixed.
        tensors = device_tensors(expr_scaler.transform(x_expr).astype(np.float32),
            topo_scaler.transform(x_topo).astype(np.float32), torch.as_tensor(edge_index), torch.as_tensor(edge_weight), n)
        p_all = predict(final_model, tensors)
        p_test = p_all[test_nodes]
        threshold = select_threshold(y_by_node[dev_nodes], oof[dev_nodes])
        cv_metrics = pd.DataFrame(fold_records)
        test_metric = metric_pack(y_by_node[test_nodes], p_test, threshold)
        cv_record = {"model": name, "cv_average_precision_mean": float(cv_metrics.validation_average_precision.mean()),
            "cv_average_precision_sd": float(cv_metrics.validation_average_precision.std(ddof=1)),
            "cv_roc_auc_mean": float(cv_metrics.validation_roc_auc.mean()),
            "cv_roc_auc_sd": float(cv_metrics.validation_roc_auc.std(ddof=1)),
            "cv_selected_epochs_median": median_epoch, "validation_selected_threshold_mcc": threshold,
            "checkpoint": str(checkpoint.relative_to(ROOT)), "test_n": len(test_nodes),
            **{"test_" + k: v for k, v in test_metric.items()}}
        results.append(cv_record)
        pd.DataFrame({"gene_index": np.flatnonzero(np.isfinite(oof)), "oof_probability": oof[np.isfinite(oof)],
            "target": y_by_node[np.isfinite(oof)], "fold": [cv_fold_map[i] for i in np.flatnonzero(np.isfinite(oof))]}).to_csv(
                run_dir / "predictions" / f"{name}_development_oof.csv", index=False)
        pd.DataFrame({"gene_index": test_nodes, "entrez": frame.loc[test_nodes, "entrez"].to_numpy(),
            "target": y_by_node[test_nodes], "probability": p_test}).to_csv(run_dir / "predictions" / f"{name}_heldout_test.csv", index=False)
        pd.DataFrame({"gene_index": np.arange(n), "entrez": frame.entrez, "symbol": frame.symbol,
            "ensembl_gene_id_versioned": frame.ensembl_gene_id_versioned, "oncogene_status": frame.oncogene_status,
            "oncogene_label": frame.oncogene_label, "probability": p_all}).sort_values("probability", ascending=False).to_csv(
                run_dir / "predictions" / f"{name}_all_genes.csv", index=False)
        all_logs[name].to_csv(run_dir / "logs" / name / "cv_epoch_logs.csv", index=False)
        print(f"Finished {name}: CV AP {cv_record['cv_average_precision_mean']:.4f} ± {cv_record['cv_average_precision_sd']:.4f}; test AP {test_metric['average_precision']:.4f}", flush=True)

    # Regularized reference baseline on expression + topology, with the same splits.
    baseline_name = "logistic_expression_topology"
    X = np.concatenate([x_expr, x_topo], axis=1)
    oof = np.full(n, np.nan, dtype=np.float32)
    for tr_local, va_local in cv.split(dev_nodes, y_by_node[dev_nodes]):
        tr_nodes, va_nodes = dev_nodes[tr_local], dev_nodes[va_local]
        pipe = make_pipeline(SimpleImputer(strategy="median"), SkStandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED))
        pipe.fit(X[tr_nodes], y_by_node[tr_nodes])
        oof[va_nodes] = pipe.predict_proba(X[va_nodes])[:, 1]
    threshold = select_threshold(y_by_node[dev_nodes], oof[dev_nodes])
    pipe = make_pipeline(SimpleImputer(strategy="median"), SkStandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED))
    pipe.fit(X[dev_nodes], y_by_node[dev_nodes])
    p_all = pipe.predict_proba(X)[:, 1]
    p_test = p_all[test_nodes]
    fold_ap = [average_precision_score(y_by_node[dev_nodes[va]], oof[dev_nodes[va]]) for _, va in cv.split(dev_nodes, y_by_node[dev_nodes])]
    fold_roc = [roc_auc_score(y_by_node[dev_nodes[va]], oof[dev_nodes[va]]) for _, va in cv.split(dev_nodes, y_by_node[dev_nodes])]
    test_metric = metric_pack(y_by_node[test_nodes], p_test, threshold)
    baseline_record = {"model": baseline_name, "cv_average_precision_mean": float(np.mean(fold_ap)),
        "cv_average_precision_sd": float(np.std(fold_ap, ddof=1)), "cv_roc_auc_mean": float(np.mean(fold_roc)),
        "cv_roc_auc_sd": float(np.std(fold_roc, ddof=1)), "cv_selected_epochs_median": None,
        "validation_selected_threshold_mcc": threshold, "checkpoint": None, "test_n": len(test_nodes),
        **{"test_" + k: v for k, v in test_metric.items()}}
    results.append(baseline_record)
    pd.DataFrame({"gene_index": dev_nodes, "oof_probability": oof[dev_nodes], "target": y_by_node[dev_nodes]}).to_csv(
        run_dir / "predictions" / f"{baseline_name}_development_oof.csv", index=False)
    pd.DataFrame({"gene_index": test_nodes, "entrez": frame.loc[test_nodes, "entrez"].to_numpy(),
        "target": y_by_node[test_nodes], "probability": p_test}).to_csv(run_dir / "predictions" / f"{baseline_name}_heldout_test.csv", index=False)
    pd.DataFrame({"gene_index": np.arange(n), "entrez": frame.entrez, "symbol": frame.symbol,
        "ensembl_gene_id_versioned": frame.ensembl_gene_id_versioned, "oncogene_status": frame.oncogene_status,
        "oncogene_label": frame.oncogene_label, "probability": p_all}).sort_values("probability", ascending=False).to_csv(
            run_dir / "predictions" / f"{baseline_name}_all_genes.csv", index=False)
    results_df = pd.DataFrame(results).sort_values("cv_average_precision_mean", ascending=False)
    results_df.to_csv(run_dir / "metrics" / "model_comparison.csv", index=False)
    results_df.to_json(run_dir / "metrics" / "model_comparison.json", orient="records", indent=2)
    plot_training_curves(all_logs, run_dir)

    # Cross-validation-selected neural model is selected using development CV only.
    neural = results_df[results_df.model.isin(MODEL_NAMES)]
    best = neural.iloc[0].model
    best_prob = pd.read_csv(run_dir / "predictions" / f"{best}_all_genes.csv")
    top_unknown = best_prob[best_prob.oncogene_status == "unknown"].head(100).copy()
    graph_frame = pd.read_csv(INPUT_DIR / "gene_features_and_labels.csv")
    idx_to_entrez = graph_frame.entrez.astype(str).tolist()
    adjacency = {i: [] for i in range(n)}
    for a, b, score in zip(torch.as_tensor(edge_index)[0].tolist(), torch.as_tensor(edge_index)[1].tolist(), torch.as_tensor(edge_weight).tolist()):
        if a < b:
            adjacency[a].append((b, score)); adjacency[b].append((a, score))
    label_map = dict(zip(frame.entrez.astype(str), frame.oncogene_status))
    score_map = dict(zip(frame.entrez.astype(str), best_prob.probability))
    nbr_rows = []
    for r in top_unknown.itertuples(index=False):
        gene = str(r.entrez); idx = int(r.gene_index)
        neighbors = sorted(adjacency[idx], key=lambda z: z[1], reverse=True)[:20]
        nbr_rows.append({"entrez": gene, "symbol": r.symbol, "prediction_score": r.probability,
            "neighbors": json.dumps([{"entrez": idx_to_entrez[j], "symbol": frame.loc[j, "symbol"],
                "edge_score": float(w), "target_status": label_map.get(idx_to_entrez[j]),
                "model_score": float(score_map.get(idx_to_entrez[j], 0.0))} for j, w in neighbors])})
    pd.DataFrame(nbr_rows).to_csv(run_dir / "predictions" / "top_candidate_ppi_neighborhoods.csv", index=False)

    plot_comparison(results_df, run_dir)
    best_result = next(r for r in results if r["model"] == best)
    plot_curves(run_dir, best, y_by_node[test_nodes], pd.read_csv(run_dir / "predictions" / f"{best}_heldout_test.csv").probability.to_numpy(), best_result["validation_selected_threshold_mcc"])
    (run_dir / "metrics" / "experiment_summary.json").write_text(json.dumps({
        "run_status": "completed", "best_validation_model": best,
        "selection_metric": "mean development 3-fold cross-validated average precision",
        "test_evaluated_once_after_model_selection": True,
        "cv_variability": "fold mean and sample SD reported",
        "component_dependence": component_eval,
        "not_attempted_due_to_mapping_or_representation": {
            "cosmic_target_or_features": "excluded to prevent leakage",
            "transformer_over_patient_samples": "not applicable; features are cohort summaries per gene",
            "cross_attention": "not used; modalities are single aligned gene vectors without token sequences",
            "multi_scale_graph_branch": "not used; no predeclared independent graph scales",
        },
        "models_attempted": [*MODEL_NAMES, baseline_name],
        "best_model_test_metrics": next(r for r in results if r["model"] == best),
        "output_directory": str(run_dir.relative_to(ROOT)),
        "ram_available_at_end_gib": round(psutil.virtual_memory().available / 1024**3, 3),
        "elapsed_note": "See experiment.json and per-fold logs.",
        "runtime_seconds": round(time.time() - start_time, 2),
        "peak_process_rss_gib_observed_during_training": round(MAX_RSS_GIB, 3),
        "device": "CPU",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"run_directory": str(run_dir.relative_to(ROOT)), "best_validation_model": best,
        "comparison": results_df[["model", "cv_average_precision_mean", "cv_average_precision_sd", "test_average_precision", "test_roc_auc", "test_mcc"]].to_dict(orient="records")}, indent=2), flush=True)


def plot_comparison(df, run_dir):
    ordered = df.sort_values("cv_average_precision_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered.model, ordered.cv_average_precision_mean, xerr=ordered.cv_average_precision_sd, label="Development CV AP")
    ax.scatter(ordered.test_average_precision, ordered.model, color="crimson", marker="D", label="Held-out test AP")
    ax.set_xlabel("Average precision"); ax.set_title("Gene-level driver annotation model comparison")
    ax.legend(); fig.tight_layout(); fig.savefig(run_dir / "plots" / "model_comparison_average_precision.png", dpi=180); plt.close(fig)
    metrics = ["test_macro_f1", "test_balanced_accuracy", "test_mcc"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ordered.set_index("model")[metrics].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score"); ax.set_title("Held-out test classification metrics"); ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(run_dir / "plots" / "heldout_metrics.png", dpi=180); plt.close(fig)


def plot_training_curves(logs, run_dir):
    for name, frame in logs.items():
        stats = frame.groupby("epoch").agg(loss_mean=("train_bce", "mean"), loss_sd=("train_bce", "std"),
            ap_mean=("validation_ap", "mean"), ap_sd=("validation_ap", "std")).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(stats.epoch, stats.loss_mean, label="Training BCE")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
        ax2 = ax.twinx(); ax2.plot(stats.epoch, stats.ap_mean, color="darkorange", label="Validation AP")
        ax2.set_ylabel("Validation average precision")
        ax.set_title(f"{name}: development CV training curves")
        lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout(); fig.savefig(run_dir / "plots" / f"{name}_cv_training_curves.png", dpi=160); plt.close(fig)


def plot_curves(run_dir, model, y, p, threshold):
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].set_title(f"{model}: test ROC")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[1].plot(recall, precision); axes[1].set_title(f"{model}: test PR")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    fig.tight_layout(); fig.savefig(run_dir / "plots" / "best_model_test_roc_pr.png", dpi=180); plt.close(fig)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues"); ax.set_xticks([0, 1], ["explicit zero", "positive"]); ax.set_yticks([0, 1], ["explicit zero", "positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    for i in range(2):
        for j in range(2): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title(f"{model}: test confusion matrix at validation threshold {threshold:.2f}")
    fig.tight_layout(); fig.savefig(run_dir / "plots" / "best_model_test_confusion.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
