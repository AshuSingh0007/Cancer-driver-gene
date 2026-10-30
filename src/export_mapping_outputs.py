"""Export a detailed STRING protein-to-Entrez mapping from local aliases."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = max((p for p in (ROOT / "data" / "processed").glob("validation_*") if (p / "data_validation_summary.json").exists()), key=lambda p: p.stat().st_mtime)
_MATCHED_ALIASES = sorted((ROOT / "data" / "reference").glob("metadata_*/9606.protein.aliases.v12.5.txt"))
ALIASES = _MATCHED_ALIASES[-1] if _MATCHED_ALIASES else ROOT / "9606.protein.aliases.v12.5.txt" / "9606.protein.aliases.v12.0.txt"
LINKS = ROOT / "9606.protein.links.v12.5.txt" / "9606.protein.links.v12.5.txt"

def main():
    protected = [OUT / name for name in ["string_protein_entrez_mapping_v12_5_aliases.csv",
        "expression_symbol_ncg_candidate_crosswalk.csv"]]
    if any(path.exists() for path in protected):
        raise FileExistsError(f"Validation outputs already exist in {OUT}; run validate_data.py to create a new dated run before exporting again")
    by_protein = defaultdict(set)
    with ALIASES.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("source") == "Ensembl_HGNC_entrez_id" and row.get("alias", "").isdigit():
                by_protein[row["#string_protein_id"]].add(int(row["alias"]))
    linked = set()
    with LINKS.open(encoding="ascii", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter=" ", skipinitialspace=True):
            linked.add(row["protein1"]); linked.add(row["protein2"])
    rows = []
    for protein, ids in sorted(by_protein.items()):
        rows.append({"string_protein_id": protein, "entrez_ids": ";".join(map(str, sorted(ids))),
            "entrez_id_count": len(ids), "mapping_status": "unambiguous" if len(ids) == 1 else "ambiguous",
            "present_in_links_v12_5": protein in linked})
    pd.DataFrame(rows).to_csv(OUT / "string_protein_entrez_mapping_v12_5_aliases.csv", index=False)
    # Emit an explicitly provisional symbol-based crosswalk, never an asserted
    # stable-ID mapping. All expression files were verified to share the same
    # gene-ID universe; one file is enough to enumerate it.
    expr_path = next((ROOT / "data" / "tcga" / "expression").rglob("*.rna_seq.augmented_star_gene_counts.tsv"))
    expr = pd.read_csv(expr_path, sep="\t", comment="#", usecols=["gene_id", "gene_name"], dtype=str)
    expr = expr[~expr.gene_id.str.startswith("N_", na=False)].drop_duplicates("gene_id")
    ann = pd.read_csv(ROOT / "NCG_cancerdrivers_annotation_supporting_evidence.tsv", sep="\t", usecols=["entrez", "symbol"], dtype=str)
    ann["entrez"] = pd.to_numeric(ann.entrez, errors="coerce")
    ann["symbol_norm"] = ann.symbol.str.strip().str.upper()
    symbol_to_ids = ann.dropna(subset=["entrez", "symbol_norm"]).groupby("symbol_norm").entrez.agg(lambda x: sorted(set(map(int, x))))
    expr["symbol_norm"] = expr.gene_name.str.strip().str.upper()
    expr_symbol_counts = expr.groupby("symbol_norm").gene_id.nunique()
    cross = []
    for row in expr.itertuples(index=False):
        ids = symbol_to_ids.get(row.symbol_norm, [])
        cross.append({"ensembl_gene_id_versioned": row.gene_id, "gene_symbol": row.gene_name,
            "candidate_entrez_ids_by_exact_symbol": ";".join(map(str, ids)),
            "candidate_entrez_count": len(ids), "expression_symbol_gene_id_count": int(expr_symbol_counts.get(row.symbol_norm, 0)),
            "mapping_status": "unique_symbol_candidate_only" if len(ids) == 1 and expr_symbol_counts.get(row.symbol_norm, 0) == 1 else "unmapped_or_ambiguous"})
    pd.DataFrame(cross).to_csv(OUT / "expression_symbol_ncg_candidate_crosswalk.csv", index=False)
    src = pd.read_csv(OUT / "ppi_edges_string_v12_5.csv")
    summary = json.loads((OUT / "string_validation_summary.json").read_text(encoding="utf-8"))
    summary["detailed_mapping_file"] = "string_protein_entrez_mapping_v12_5_aliases.csv"
    summary["release_matched_graph_file"] = "ppi_edges_string_v12_5.csv"
    (OUT / "string_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} protein mappings; {len(linked)} linked protein IDs; {len(cross)} expression-gene symbol candidates; graph copy has {len(src)} rows")

if __name__ == "__main__":
    main()
