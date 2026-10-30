
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\Quantum")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ANNOTATION_FILE = (
    BASE_DIR / "NCG_cancerdrivers_annotation_supporting_evidence.tsv"
)
SYSTEMS_FILE = (
    BASE_DIR / "NCG_cancerdrivers_systemslevelproperties.tsv"
)

# Load the original datasets
annotation = pd.read_csv(ANNOTATION_FILE, sep="\t", low_memory=False)
systems = pd.read_csv(SYSTEMS_FILE, sep="\t", low_memory=False)

annotation.columns = annotation.columns.str.strip()
systems.columns = systems.columns.str.strip()

# Standardise gene identifiers
for df in (annotation, systems):
    df["entrez"] = pd.to_numeric(df["entrez"], errors="coerce").astype("Int64")
    df["symbol"] = df["symbol"].astype("string").str.strip()

# Remove records without usable gene identifiers
annotation = annotation.dropna(subset=["entrez", "symbol"])
systems = systems.dropna(subset=["entrez", "symbol"])

# Aggregate repeated annotation records into one row per Entrez ID
def combine_evidence(series):
    values = series.dropna().astype(str).str.strip()
    values = sorted(set(v for v in values if v and v.lower() != "nan"))
    return "; ".join(values) if values else pd.NA

# Preserve positive NCG annotations if any evidence record contains 1.
# Missing values remain missing rather than being treated as negative.
def aggregate_binary(series):
    values = pd.to_numeric(series, errors="coerce")
    if (values == 1).any():
        return 1
    if (values == 0).any():
        return 0
    return pd.NA

aggregation = {}

for column in annotation.columns:
    if column == "entrez":
        continue
    elif column == "NCG_oncogene" or column == "NCG_tsg":
        aggregation[column] = aggregate_binary
    else:
        aggregation[column] = combine_evidence

annotation_gene = (
    annotation.groupby("entrez", as_index=False)
    .agg(aggregation)
)

# Check whether each Entrez ID has conflicting gene symbols
symbol_counts = (
    annotation.groupby("entrez")["symbol"].nunique()
)
conflicting_ids = symbol_counts[symbol_counts > 1]

print("Genes with conflicting annotation symbols:", len(conflicting_ids))

# Systems table should have one record per Entrez ID
systems_gene = systems.drop_duplicates(subset=["entrez"], keep="first").copy()

# Merge annotation and systems-level features
merged = annotation_gene.merge(
    systems_gene,
    on="entrez",
    how="outer",
    suffixes=("_annotation", "_systems"),
    indicator=True
)

# Record whether the gene appeared in both source files
merged["merge_status"] = merged["_merge"].astype(str)
merged = merged.drop(columns=["_merge"])

# Save output
output_file = OUTPUT_DIR / "NCG_preprocessed_merged.csv"
merged.to_csv(output_file, index=False, encoding="utf-8-sig")

# Save matching report
print("\nMerge status:")
print(merged["merge_status"].value_counts().to_string())

print("\nAnnotation genes:", annotation_gene["entrez"].nunique())
print("Systems-level genes:", systems_gene["entrez"].nunique())
print("Merged rows:", len(merged))
print("Merged columns:", len(merged.columns))
print("\nSaved:", output_file)
print("Original files were not modified.")
