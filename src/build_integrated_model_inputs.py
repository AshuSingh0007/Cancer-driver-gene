"""Align NCBI-crosswalked expression features, NCG labels, and STRING graph."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data/processed/validation_20261028_152218"
FEATURES = ROOT / "data/processed/expression_features_20261028_153824/gene_expression_features.csv"
MAPPING_DIR = max((p for p in (ROOT / "data/processed").glob("identifier_mapping_*") if (p / "identifier_mapping_summary.json").exists()), key=lambda p: p.stat().st_mtime)


def main():
    candidate = pd.read_csv(MAPPING_DIR / "unambiguous_expression_graph_ncg_candidate_genes.csv", dtype={"entrez": str})
    candidate["entrez"] = candidate.entrez.str.replace(r"\.0$", "", regex=True)
    labels = pd.read_csv(VALIDATION / "ncg_gene_label_table.csv", dtype={"entrez": str})
    labels["entrez"] = labels.entrez.str.replace(r"\.0$", "", regex=True)
    expression = pd.read_csv(FEATURES)
    edges = pd.read_csv(VALIDATION / "ppi_edges_string_v12_5.csv")
    features = expression.merge(candidate[["entrez", "ensembl_gene_id_versioned"]], on="ensembl_gene_id_versioned", how="inner", validate="one_to_one")
    features = features.merge(labels[["entrez", "symbol", "oncogene_status", "oncogene_label", "tsg_status", "tsg_label"]], on="entrez", how="left", validate="one_to_one")
    features = features.sort_values("entrez", key=lambda x: x.astype(int)).reset_index(drop=True)
    if features.entrez.duplicated().any() or features.ensembl_gene_id_versioned.duplicated().any():
        raise ValueError("Crosswalk produced duplicate Entrez or Ensembl feature keys")
    features.insert(0, "gene_index", np.arange(len(features), dtype=np.int32))
    index_by_entrez = {str(g): i for i, g in enumerate(features.entrez)}
    edges["source"] = edges.source.astype(int).astype(str)
    edges["target"] = edges.target.astype(int).astype(str)
    edges = edges[edges.source.isin(index_by_entrez) & edges.target.isin(index_by_entrez)].copy()
    edges["source_index"] = edges.source.map(index_by_entrez).astype(np.int32)
    edges["target_index"] = edges.target.map(index_by_entrez).astype(np.int32)
    if (edges.source_index == edges.target_index).any():
        raise ValueError("Self edge in integrated graph")
    src = edges.source_index.to_numpy(dtype=np.int64)
    dst = edges.target_index.to_numpy(dtype=np.int64)
    scores = edges.combined_score.to_numpy(dtype=np.float32) / 1000.0
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_weight = np.concatenate([scores, scores]).astype(np.float32)
    out = ROOT / "data/processed" / ("integrated_model_inputs_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    features_path = out / "gene_features_and_labels.csv"
    features.to_csv(features_path, index=False)
    x = features[["mean_log2_tpm_plus_1", "sd_log2_tpm_plus_1", "fraction_cases_tpm_positive"]].to_numpy(dtype=np.float32)
    y = features["oncogene_label"].to_numpy(dtype=np.float32)
    y_mask = np.isfinite(y)
    np.save(out / "expression_features_raw.npy", x, allow_pickle=False)
    np.save(out / "target_values.npy", np.nan_to_num(y, nan=-1).astype(np.int8), allow_pickle=False)
    np.save(out / "target_mask.npy", y_mask.astype(np.bool_), allow_pickle=False)
    np.savez_compressed(out / "graph_edge_index.npz", edge_index=edge_index, edge_weight=edge_weight,
        canonical_source_index=src, canonical_target_index=dst, canonical_edge_weight=scores)
    edges[["source_index", "target_index", "combined_score"]].to_csv(out / "graph_edges_indexed.csv", index=False)
    status_counts = features.oncogene_status.value_counts().to_dict()
    mapping_summary = json.loads((MAPPING_DIR / "identifier_mapping_summary.json").read_text(encoding="utf-8"))
    summary = {
        "prediction_unit": "gene",
        "target": "NCG NCG_oncogene; positive=1, explicit_zero=0, unknown=-1/masked; conflict excluded",
        "ncbi_crosswalk_source": str(MAPPING_DIR.relative_to(ROOT)),
        "gene_count": len(features),
        "expression_feature_shape": list(x.shape),
        "label_status_counts": status_counts,
        "known_label_count": int(y_mask.sum()),
        "positive_count": int((y[y_mask] == 1).sum()),
        "explicit_zero_count": int((y[y_mask] == 0).sum()),
        "unknown_count": int((~y_mask).sum()),
        "ambiguous_or_missing_mapping_genes_excluded_from_ncg_graph_intersection": int(mapping_summary["ncg_gene_count"] - len(features)),
        "graph_source": str((VALIDATION / "ppi_edges_string_v12_5.csv").relative_to(ROOT)),
        "canonical_undirected_edges": len(edges),
        "directed_message_edges": int(edge_index.shape[1]),
        "graph_confidence": "combined_score divided by 1000; edges at threshold >=700 only",
        "feature_transform_policy": "Raw cohort summary features are saved. Any scaling/imputation is fitted on training genes within each fold by the trainer.",
        "gene_index_key": "gene_index is shared by expression_features_raw.npy, target_values.npy, target_mask.npy, gene_features_and_labels.csv, and graph_edge_index.npz",
        "label_leakage_controls": "Labels are separate arrays, not included in node features; unknown labels are masked; COSMIC is not used.",
        "output_files": ["gene_features_and_labels.csv", "expression_features_raw.npy", "target_values.npy", "target_mask.npy", "graph_edge_index.npz", "graph_edges_indexed.csv"],
    }
    (out / "integrated_input_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert x.shape == (len(features), 3)
    assert np.array_equal(np.load(out / "expression_features_raw.npy", allow_pickle=False), x)
    assert edge_index.shape[0] == 2 and edge_index.min(initial=0) >= 0 and edge_index.max(initial=0) < len(features)
    assert features.loc[y_mask, "oncogene_status"].isin(["positive", "explicit_zero"]).all()
    print(json.dumps({"output_directory": str(out.relative_to(ROOT)), **summary}, indent=2))


if __name__ == "__main__":
    main()
