"""Export separated, versioned graph, expression, and target inputs."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data/processed/validation_20261028_152218"
FEATURE_DIR = ROOT / "data/processed/expression_features_20261028_153824"


def main():
    out = ROOT / "data/processed" / ("model_inputs_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    edges = pd.read_csv(VALIDATION / "ppi_edges_string_v12_5.csv")
    edge_nodes = set(edges.source.astype(int)) | set(edges.target.astype(int))
    labels = pd.read_csv(VALIDATION / "ncg_gene_label_table.csv")
    mapping = pd.read_csv(VALIDATION / "string_protein_entrez_mapping_v12_5_aliases.csv")
    ncg_ids = set(labels.entrez.astype(int))
    mapped_ids = set()
    for row in mapping.itertuples(index=False):
        if row.mapping_status == "unambiguous" and row.present_in_links_v12_5:
            mapped_ids.update(int(x) for x in str(row.entrez_ids).split(";") if x.isdigit())
    mapped_ncg = sorted(ncg_ids & mapped_ids)
    nodes = pd.DataFrame({"entrez": mapped_ncg})
    nodes["has_edge_at_score_ge_700"] = nodes.entrez.isin(edge_nodes)
    nodes.to_csv(out / "graph_nodes.csv", index=False)
    edges.to_csv(out / "graph_edges.csv", index=False)
    labels[["entrez", "symbol", "oncogene_status", "oncogene_label", "tsg_status", "tsg_label"]].to_csv(out / "gene_labels.csv", index=False)
    shutil.copy2(FEATURE_DIR / "gene_expression_features.csv", out / "gene_expression_features_ensembl_keys.csv")
    summary = {
        "prediction_unit": "gene",
        "target": "NCG oncogene annotation; positive and explicitly annotated zero are labeled, unknown stays unlabeled, conflict excluded",
        "graph_nodes_with_unambiguous_v12_5_mapping": len(mapped_ncg),
        "graph_nodes_with_score_ge_700_edge": len(edge_nodes),
        "graph_edge_count": len(edges),
        "target_label_file_rows": len(labels),
        "target_status_counts": labels.oncogene_status.value_counts().to_dict(),
        "expression_feature_rows": 60660,
        "expression_feature_key": "versioned Ensembl gene ID",
        "expression_to_target_join": "Not performed. The release-102 Ensembl BioMart endpoint returned HTML, so an authoritative Ensembl-to-Entrez bridge remains unresolved. The separate expression file and labels must not be joined by symbol without review.",
        "model_readiness": "Separated data components are exported; no integrated hybrid training matrix or split manifest is emitted until the authoritative crosswalk is obtained and reviewed.",
        "files": ["graph_nodes.csv", "graph_edges.csv", "gene_labels.csv", "gene_expression_features_ensembl_keys.csv"],
    }
    (out / "model_input_components_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_directory": str(out.relative_to(ROOT)), **summary}, indent=2))


if __name__ == "__main__":
    main()
