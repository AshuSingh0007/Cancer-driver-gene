
import pandas as pd
import networkx as nx
from pathlib import Path

BASE_DIR = Path(r"D:\Quantum")
OUTPUT_DIR = BASE_DIR / "outputs"

NCG_FILE = OUTPUT_DIR / "NCG_preprocessed_merged.csv"
EDGE_FILE = OUTPUT_DIR / "NCG_STRING_PPI_edges.csv"

# Load the NCG gene table and filtered edges
ncg = pd.read_csv(NCG_FILE, low_memory=False)
edges = pd.read_csv(EDGE_FILE)

ncg["entrez"] = pd.to_numeric(ncg["entrez"], errors="coerce")
ncg_genes = set(ncg["entrez"].dropna().astype(int))

edges["source"] = edges["source"].astype(int)
edges["target"] = edges["target"].astype(int)

# Build an undirected graph
G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target",
    edge_attr="combined_score",
    create_using=nx.Graph()
)

# Basic checks
print("NCG genes:", len(ncg_genes))
print("Unique edge rows:", len(edges))
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())
print("Self-loops:", nx.number_of_selfloops(G))
print("Connected components:", nx.number_connected_components(G))

# Network coverage
network_genes = set(G.nodes())
print("NCG genes in graph:", len(network_genes & ncg_genes))
print("NCG genes absent from graph:", len(ncg_genes - network_genes))

# Degree statistics
degrees = dict(G.degree())
if degrees:
    print("Average degree:", round(sum(degrees.values()) / len(degrees), 3))
    print("Maximum degree:", max(degrees.values()))

# Save gene-level network statistics
stats = pd.DataFrame({
    "entrez": list(G.nodes()),
    "degree": [G.degree(n) for n in G.nodes()]
})
stats.to_csv(OUTPUT_DIR / "ppi_node_statistics.csv", index=False)

# Save connected component sizes
component_sizes = sorted(
    [len(c) for c in nx.connected_components(G)],
    reverse=True
)
pd.DataFrame({
    "component_id": range(1, len(component_sizes) + 1),
    "size": component_sizes
}).to_csv(OUTPUT_DIR / "ppi_component_sizes.csv", index=False)

print("\nValidation completed.")
print("Saved ppi_node_statistics.csv")
print("Saved ppi_component_sizes.csv")
