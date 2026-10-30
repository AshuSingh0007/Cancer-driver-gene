"""Regenerate presentation figures from saved NCG project artifacts."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs" / "experiments" / "oncogene_logreg_20261014T101544Z"
PRED = pd.read_csv(RUN / "holdout_predictions.csv")
METRICS = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
MERGED = pd.read_csv(ROOT / "outputs" / "NCG_preprocessed_merged.csv", low_memory=False)
EDGES = pd.read_csv(ROOT / "outputs" / "NCG_STRING_PPI_edges.csv")
OUT = ROOT / "outputs" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 15, "axes.labelsize": 12, "figure.dpi": 120,
    "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN = "#2878B5", "#E07A27", "#3B8C6E"


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    y = PRED["y_true"].astype(int).to_numpy()
    score = PRED["logistic_probability"].astype(float).to_numpy()
    pred = PRED["logistic_prediction"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 5.7))
    disp = ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]),
        display_labels=["Other / negative", "Oncogene"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Oncogene baseline: holdout confusion matrix", pad=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    save(fig, "oncogene_confusion_matrix.png")

    precision, recall, _ = precision_recall_curve(y, score)
    ap = average_precision_score(y, score)
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot(recall, precision, color=BLUE, lw=2.5, label=f"Logistic regression (AP = {ap:.3f})")
    ax.axhline(y.mean(), color="gray", ls="--", lw=1.4, label=f"Positive prevalence = {y.mean():.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
        title="Precision–recall curve (oncogene holdout)")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=.22)
    save(fig, "oncogene_precision_recall_curve.png")

    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot(fpr, tpr, color=GREEN, lw=2.5, label=f"Logistic regression (ROC-AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.4, label="Chance (AUC = 0.500)")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
        title="ROC curve (oncogene holdout)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=.22)
    save(fig, "oncogene_roc_curve.png")

    order = ["DummyClassifier(most_frequent)", "LogisticRegression(class_weight=balanced)"]
    models = {m["model"]: m for m in METRICS["models"]}
    labels = ["DummyClassifier", "Logistic regression"]
    measures = [("accuracy", "Accuracy"), ("f1_positive", "Positive-class F1"),
        ("average_precision", "Average precision"), ("roc_auc", "ROC-AUC")]
    x = np.arange(len(measures)); width = .34
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for idx, (model, label, color) in enumerate(zip(order, labels, ["#9299A1", BLUE])):
        vals = [models[model][key] for key, _ in measures]
        bars = ax.bar(x + (idx - .5) * width, vals, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, [title for _, title in measures])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the same oncogene holdout")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.22)
    ax.text(.99, .02, "One stratified 80/20 split (seed 42)", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color="#555")
    save(fig, "oncogene_model_comparison.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    categories = ["Oncogene", "Tumour suppressor"]
    targets = ["NCG_oncogene", "NCG_tsg"]
    pos, neg, unlabeled = [], [], []
    for col in targets:
        vals = MERGED[col]
        pos.append(int((vals == 1).sum()))
        neg.append(int((vals == 0).sum()))
        unlabeled.append(int(vals.isna().sum()))
    x = np.arange(2)
    ax.bar(x, pos, label="Labeled positive", color=ORANGE)
    ax.bar(x, neg, bottom=pos, label="Labeled negative", color=BLUE)
    ax.bar(x, unlabeled, bottom=np.array(pos) + np.array(neg), label="Unlabeled", color="#C7CCD1")
    for i in range(2):
        bottoms = [0, pos[i], pos[i] + neg[i]]
        for count, bottom in zip([pos[i], neg[i], unlabeled[i]], bottoms):
            ax.text(i, bottom + count / 2, f"{count:,}", ha="center", va="center",
                fontsize=10, color="white" if bottom < pos[i] + neg[i] else "#333", fontweight="bold")
    ax.set_xticks(x, categories)
    ax.set_ylabel("Genes (n)")
    ax.set_title("NCG label coverage and class counts")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, len(MERGED) * 1.08)
    ax.grid(axis="y", alpha=.18)
    save(fig, "oncogene_tsg_label_distribution.png")

    # Reproducible induced subgraph: top 35 endpoint genes by degree, ties by ID.
    # Label only the 12 highest-degree nodes to keep the figure legible.
    graph = nx.from_pandas_edgelist(EDGES, "source", "target", edge_attr="combined_score")
    ranked = sorted(graph.degree, key=lambda item: (-item[1], int(item[0])))
    selected = [node for node, _ in ranked[:35]]
    subgraph = graph.subgraph(selected).copy()
    pos_layout = nx.spring_layout(subgraph, seed=42, weight="combined_score", k=0.8, iterations=150)
    node_rank = sorted(subgraph.degree, key=lambda item: (-item[1], int(item[0])))
    labeled_nodes = [node for node, _ in node_rank[:12]]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=.86)
    ax.set_title("Provisional STRING PPI subgraph: top-degree NCG genes", pad=18)
    nx.draw_networkx_edges(subgraph, pos_layout, ax=ax, edge_color="#AAB4BE", alpha=.55,
        width=[.5 + 1.5 * subgraph[u][v].get("combined_score", 0) / 1000 for u, v in subgraph.edges])
    deg = dict(subgraph.degree)
    nodes = nx.draw_networkx_nodes(subgraph, pos_layout, ax=ax, node_size=[90 + 28 * deg[n] for n in subgraph],
        node_color=[deg[n] for n in subgraph], cmap="viridis", edgecolors="white", linewidths=.6)
    nx.draw_networkx_labels(subgraph, pos_layout,
        labels={n: str(n) for n in labeled_nodes}, ax=ax, font_size=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78, "pad": 1.3})
    fig.colorbar(nodes, ax=ax, shrink=.72, label="Degree within displayed subgraph")
    ax.text(.5, 1.005, "35 highest-degree nodes from the existing edge list; labels show Entrez IDs for 12 highest-degree nodes",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#4F5963")
    ax.text(.5, -.045, "PROVISIONAL: source aliases are v12.0 while STRING links are v12.5; validate ID mapping before interpretation.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#9B3A32", fontweight="bold")
    ax.axis("off")
    save(fig, "ppi_provisional_top_degree_subgraph.png")

    print(f"Saved 6 PNG figures to: {OUT}")
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
