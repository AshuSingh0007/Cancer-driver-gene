"""Reproducible gene-level classical baseline for the NCG annotations.

Run from the project root with: .\.venv\Scripts\python.exe train_baseline.py
The default experiment predicts NCG_oncogene from systems-level measurements.
"""
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "NCG_preprocessed_merged.csv"
SEED = 42
TARGET = "NCG_oncogene"
# Source-defined numeric systems properties only. IDs, symbols, annotation
# evidence, labels, and merge metadata are deliberately excluded.
FEATURES = [
    "duplicability", "essential_celllines_percentage", "ppin_degree",
    "ppin_betweenness", "ppin_clustering", "complexes", "mirna",
    "expressed_tissues_mRNA_GTEx", "expressed_tissues_mRNA_protatlas",
    "expressed_tissues_protein_protatlas", "expressed_celllines_CLP",
    "expressed_celllines_GNE", "expressed_celllines_CCLE", "germline_SNVs",
    "germline_SVs", "LOEUF_score",
]


def score_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    return model, {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_positive": float(precision_score(y_test, pred, zero_division=0)),
        "recall_positive": float(recall_score(y_test, pred, zero_division=0)),
        "f1_positive": float(f1_score(y_test, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, pred, labels=[0, 1],
            target_names=["negative", "positive"], output_dict=True, zero_division=0),
    }, pred, proba


def main():
    df = pd.read_csv(DATA, low_memory=False)
    missing_features = sorted(set(FEATURES) - set(df.columns))
    if missing_features:
        raise ValueError(f"Required feature columns missing: {missing_features}")
    labeled = df[df[TARGET].isin([0, 1])].copy()
    X = labeled[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = labeled[TARGET].astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Need at least two observed examples in both classes")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.20,
        random_state=SEED, stratify=y)
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy, dummy_metrics, dummy_pred, dummy_proba = score_model(
        "DummyClassifier(most_frequent)", dummy, x_train, x_test, y_train, y_test)
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED))
    lr, lr_metrics, lr_pred, lr_proba = score_model(
        "LogisticRegression(class_weight=balanced)", lr, x_train, x_test, y_train, y_test)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "outputs" / "experiments" / f"oncogene_logreg_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    metrics = {
        "experiment": "oncogene prediction, labeled rows only",
        "target": TARGET, "features": FEATURES,
        "feature_source": "systems-level numerical columns only",
        "excluded": ["entrez", "symbols", "annotation/evidence columns", "both labels", "merge_status"],
        "n_labeled": int(len(y)), "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "n_unlabeled_excluded": int(df[TARGET].isna().sum()),
        "split": "single stratified gene-level 80/20 holdout; no gene repeated; seed 42",
        "n_train": int(len(train_idx)), "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "n_test": int(len(test_idx)), "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "numeric_feature_missing_values_train": {k: int(v) for k, v in x_train.isna().sum().items()},
        "models": [dummy_metrics, lr_metrics],
        "limitations": ["Single random holdout; estimates may vary with split.",
            "NCG systems-level cohort and its negative labels are source-defined, not a genome-wide unlabeled background.",
            "PPI-derived systems features may encode network ascertainment; this is not an independent STRING feature ablation.",
            "Random gene split does not prevent leakage through correlated genes or network structure."],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"entrez": labeled.iloc[test_idx]["entrez"].to_numpy(),
        "symbol": labeled.iloc[test_idx]["symbol_annotation"].to_numpy(),
        "y_true": y_test.to_numpy(), "dummy_prediction": dummy_pred,
        "dummy_probability": dummy_proba, "logistic_prediction": lr_pred,
        "logistic_probability": lr_proba}).to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump(lr, out / "logistic_regression.joblib")
    print(json.dumps({"output_directory": str(out), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
