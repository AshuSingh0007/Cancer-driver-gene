"""Build an Ensembl stable-ID to Entrez map from NCBI human Gene cross-references."""
from __future__ import annotations

import csv
import gzip
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/reference/ncbi_gene_info_20261028/Homo_sapiens.gene_info.gz"
VALIDATION = ROOT / "data/processed/validation_20261028_152218"
FEATURES = ROOT / "data/processed/expression_features_20261028_153824/gene_expression_features.csv"
NCBI_ENSEMBL = re.compile(r"^ENSG\d+(?:\.\d+)?(?:_PAR_Y)?$")
EXPR_VERSION = re.compile(r"\.\d+(?:_PAR_Y)?$")


def stable_id(value: str) -> str:
    return EXPR_VERSION.sub("", value)


def main():
    entrez_by_ensg = defaultdict(set)
    symbol_by_pair = {}
    records = 0
    with gzip.open(SOURCE, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["#tax_id"] != "9606":
                continue
            records += 1
            entrez = row["GeneID"]
            for xref in row["dbXrefs"].split("|"):
                if not xref.startswith("Ensembl:"):
                    continue
                ensg = xref.split(":", 1)[1]
                if NCBI_ENSEMBL.fullmatch(ensg):
                    stable = stable_id(ensg)
                    entrez_by_ensg[stable].add(entrez)
                    symbol_by_pair[(stable, entrez)] = row["Symbol"]

    features = pd.read_csv(FEATURES, dtype={"ensembl_gene_id_versioned": str, "gene_symbol": str})
    by_entrez_expression = defaultdict(list)
    rows = []
    for row in features.itertuples(index=False):
        ens_stable = stable_id(row.ensembl_gene_id_versioned)
        ids = sorted(entrez_by_ensg.get(ens_stable, set()), key=int)
        status = "unmapped" if not ids else "unique_entrez" if len(ids) == 1 else "ambiguous_entrez"
        entrez = ids[0] if len(ids) == 1 else ""
        if entrez:
            by_entrez_expression[entrez].append(row.ensembl_gene_id_versioned)
        rows.append({
            "ensembl_gene_id_versioned": row.ensembl_gene_id_versioned,
            "ensembl_stable_id": ens_stable,
            "expression_gene_symbol": row.gene_symbol,
            "entrez_gene_id_candidates": ";".join(ids),
            "entrez_candidate_count": len(ids),
            "mapping_status": status,
            "ncbi_symbol_for_unique_entrez": symbol_by_pair.get((ens_stable, entrez), "") if entrez else "",
            "expression_ncbi_symbol_match": bool(entrez and row.gene_symbol.casefold() == symbol_by_pair.get((ens_stable, entrez), "").casefold()),
        })
    labels = pd.read_csv(VALIDATION / "ncg_gene_label_table.csv", dtype={"entrez": str})
    graph_nodes = pd.read_csv(ROOT / "data/processed/model_inputs_20261028_154017/graph_nodes.csv", dtype={"entrez": str})
    ncg_ids = set(labels.entrez.str.replace(r"\.0$", "", regex=True))
    graph_ids = set(graph_nodes.entrez.astype(str).str.replace(r"\.0$", "", regex=True))
    status_by_entrez = {k: v for k, v in zip(labels.entrez.astype(str).str.replace(r"\.0$", "", regex=True), labels.oncogene_status)}
    out = ROOT / "data/processed" / ("identifier_mapping_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    crosswalk_path = out / "ensembl_v36_to_entrez_ncbi_gene_info.csv"
    pd.DataFrame(rows).to_csv(crosswalk_path, index=False)
    integrated = []
    ensg_unique_by_entrez = {e: g for e, gs in by_entrez_expression.items() if len(gs) == 1 for g in gs}
    for entrez in sorted(ncg_ids & graph_ids & set(ensg_unique_by_entrez), key=int):
        integrated.append({"entrez": entrez, "ensembl_gene_id_versioned": ensg_unique_by_entrez[entrez], "oncogene_status": status_by_entrez.get(entrez, "unknown")})
    candidate_path = out / "unambiguous_expression_graph_ncg_candidate_genes.csv"
    pd.DataFrame(integrated).to_csv(candidate_path, index=False)
    counts = pd.Series([r["mapping_status"] for r in rows]).value_counts().to_dict()
    cross_ref_entrez = set().union(*entrez_by_ensg.values()) if entrez_by_ensg else set()
    expression_symbol_discordances = [r for r in rows if r["mapping_status"] == "unique_entrez" and not r["expression_ncbi_symbol_match"]]
    label_status = pd.Series([x["oncogene_status"] for x in integrated]).value_counts().to_dict()
    summary = {
        "source": "NCBI Gene FTP, Homo_sapiens.gene_info.gz",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_bytes": SOURCE.stat().st_size,
        "human_gene_records": records,
        "ncbi_ensembl_stable_id_records": len(entrez_by_ensg),
        "expression_gene_count": len(rows),
        "expression_mapping_status_counts": counts,
        "expression_genes_with_unique_entrez": counts.get("unique_entrez", 0),
        "expression_genes_with_ambiguous_entrez": counts.get("ambiguous_entrez", 0),
        "expression_genes_without_entrez_mapping": counts.get("unmapped", 0),
        "ncg_gene_count": len(ncg_ids),
        "ncg_genes_with_expression_and_graph_mapping": len(integrated),
        "eligible_label_status_counts": label_status,
        "expression_stable_id_to_entrez_one_to_many_collisions": {e: gs for e, gs in by_entrez_expression.items() if len(gs) > 1},
        "unique_ensembl_mappings_with_symbol_discordance_count": len(expression_symbol_discordances),
        "symbol_discordance_examples": expression_symbol_discordances[:100],
        "candidate_gene_table": candidate_path.name,
        "crosswalk_file": crosswalk_path.name,
        "mapping_policy": "Use exact Ensembl stable-ID cross-references in NCBI Gene dbXrefs, stripping only the GENCODE version suffix and recognized _PAR_Y suffix. Exclude ambiguous Ensembl-to-Entrez or Entrez-to-expression mappings; never choose by symbol.",
        "independent_reference_labels_used": False,
    }
    (out / "identifier_mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"expression_stable_id_to_entrez_one_to_many_collisions", "symbol_discordance_examples"}}, indent=2))


if __name__ == "__main__":
    main()
