
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\Quantum")

ALIASES_FILE = (
    BASE_DIR
    / "9606.protein.aliases.v12.5.txt"
    / "9606.protein.aliases.v12.0.txt"
)

LINKS_FILE = (
    BASE_DIR
    / "9606.protein.links.v12.5.txt"
    / "9606.protein.links.v12.5.txt"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Verify the actual file paths
for path in [ALIASES_FILE, LINKS_FILE]:
    print("\nFile:", path)
    print("Exists:", path.exists())

    if not path.exists():
        raise FileNotFoundError(f"Check the file path: {path}")

# Inspect a small sample of the aliases file
aliases = pd.read_csv(
    ALIASES_FILE,
    sep="\t",
    nrows=10,
    header=0
)

print("\nALIASES SAMPLE")
print("Columns:", aliases.columns.tolist())
print(aliases.to_string(index=False))

# Inspect a small sample of the links file
links = pd.read_csv(
    LINKS_FILE,
    sep=r"\s+",
    nrows=10,
    header=0
)

print("\nLINKS SAMPLE")
print("Columns:", links.columns.tolist())
print(links.to_string(index=False))

print("\nSTRING inspection completed.")
