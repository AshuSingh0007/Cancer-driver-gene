
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\Quantum")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NCG_FILE = OUTPUT_DIR / "NCG_preprocessed_merged.csv"

ALIASES_FILE = (
    BASE_DIR / "9606.protein.aliases.v12.5.txt"
    / "9606.protein.aliases.v12.0.txt"
)

LINKS_FILE = (
    BASE_DIR / "9606.protein.links.v12.5.txt"
    / "9606.protein.links.v12.5.txt"
)

CONFIDENCE_THRESHOLD = 700

# 1. Load the NCG gene set
ncg = pd.read_csv(NCG_FILE, low_memory=False)

ncg["entrez"] = pd.to_numeric(ncg["entrez"], errors="coerce")
ncg_ids = set(ncg["entrez"].dropna().astype(int))

print("NCG genes:", len(ncg_ids))

# 2. Load STRING aliases that specifically identify Entrez gene IDs
aliases = pd.read_csv(
    ALIASES_FILE,
    sep="\t",
    usecols=["#string_protein_id", "alias", "source"],
    dtype=str
)

entrez_aliases = aliases[
    aliases["source"] == "Ensembl_HGNC_entrez_id"
].copy()

entrez_aliases["alias"] = pd.to_numeric(
    entrez_aliases["alias"], errors="coerce"
)

entrez_aliases = entrez_aliases.dropna(subset=["alias"])
entrez_aliases["alias"] = entrez_aliases["alias"].astype(int)

# Keep only proteins that map unambiguously to one Entrez ID
entrez_aliases = entrez_aliases[
    entrez_aliases["alias"].isin(ncg_ids)
]

mapping_counts = (
    entrez_aliases.groupby("#string_protein_id")["alias"].nunique()
)

unambiguous_proteins = set(mapping_counts[mapping_counts == 1].index)

entrez_aliases = entrez_aliases[
    entrez_aliases["#string_protein_id"].isin(unambiguous_proteins)
].drop_duplicates(subset=["#string_protein_id"])

protein_to_gene = dict(zip(
    entrez_aliases["#string_protein_id"],
    entrez_aliases["alias"]
))

print("Unambiguous STRING proteins mapped to NCG genes:",
      len(protein_to_gene))

# 3. Read interactions in chunks and filter them
edge_parts = []
total_rows = 0
mapped_rows = 0

for chunk in pd.read_csv(
    LINKS_FILE,
    sep=r"\s+",
    usecols=["protein1", "protein2", "combined_score"],
    dtype={
        "protein1": str,
        "protein2": str,
        "combined_score": "int32"
    },
    chunksize=500_000
):
    total_rows += len(chunk)

    # Retain high-confidence interactions
    chunk = chunk[
        chunk["combined_score"] >= CONFIDENCE_THRESHOLD
    ].copy()

    # Map both proteins to Entrez gene IDs
    chunk["gene1"] = chunk["protein1"].map(protein_to_gene)
    chunk["gene2"] = chunk["protein2"].map(protein_to_gene)

    # Both endpoints must belong to the NCG gene set
    chunk = chunk.dropna(subset=["gene1", "gene2"])
    mapped_rows += len(chunk)

    chunk["gene1"] = chunk["gene1"].astype(int)
    chunk["gene2"] = chunk["gene2"].astype(int)

    # Remove self-interactions
    chunk = chunk[chunk["gene1"] != chunk["gene2"]]

    # Treat the PPI network as undirected
    chunk["source"] = chunk[["gene1", "gene2"]].min(axis=1)
    chunk["target"] = chunk[["gene1", "gene2"]].max(axis=1)

    edge_parts.append(
        chunk[["source", "target", "combined_score"]]
    )

    print("Processed link rows:", total_rows)

# 4. Combine chunks and remove duplicate gene-level edges
if edge_parts:
    edges = pd.concat(edge_parts, ignore_index=True)

    edges = (
        edges.groupby(["source", "target"], as_index=False)
        ["combined_score"].max()
    )

    edges.to_csv(
        OUTPUT_DIR / "NCG_STRING_PPI_edges.csv",
        index=False
    )

    mapped_genes = set(edges["source"]) | set(edges["target"])

    summary = {
        "ncg_gene_count": len(ncg_ids),
        "mapped_string_protein_count": len(protein_to_gene),
        "high_confidence_mapped_link_rows": mapped_rows,
        "unique_ppi_edges": len(edges),
        "genes_in_ppi_network": len(mapped_genes),
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

    pd.Series(summary).to_csv(
        OUTPUT_DIR / "ppi_network_summary.csv",
        header=["value"]
    )

    print("\nPPI NETWORK COMPLETED")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved: outputs/NCG_STRING_PPI_edges.csv")
else:
    print("No edges found. Check the ID mapping and input files.")
