"""Acquire only the small reference metadata needed to unblock Phase 5.

This script does not download expression files or STRING interaction links. It
queries metadata for file IDs already listed in the local GDC manifest, gets
human-only STRING v12.5 aliases, and queries Ensembl release 102 BioMart. It
creates a unique output directory and refuses to overwrite existing files.
Run only after explicit user approval for network downloads.
"""
from __future__ import annotations

import csv
import argparse
import gzip
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(r"C:\Users\Hp\Downloads\gdc_manifest.2026-10-27.234917.txt")
GDC_URL = "https://api.gdc.cancer.gov/files"
STRING_URL = "https://stringdb-downloads.org/download/protein.aliases.v12.5/9606.protein.aliases.v12.5.txt.gz"
BIOMART_URL = "https://nov2020.archive.ensembl.org/biomart/martservice?query="
GDC_FIELDS = ",".join([
    "file_id", "file_name", "file_size", "md5sum", "data_category", "data_type",
    "analysis.workflow_type", "cases.case_id", "cases.submitter_id", "cases.samples.sample_id",
    "cases.samples.submitter_id", "cases.samples.sample_type", "cases.samples.tissue_type",
    "cases.samples.tumor_descriptor", "cases.samples.portions.analytes.aliquots.aliquot_id",
    "cases.samples.portions.analytes.aliquots.submitter_id",
])

def fetch(url: str, *, data: bytes | None = None, content_type: str | None = None) -> bytes:
    headers = {"User-Agent": "Quantum-CancerDriver-Research/1.0"}
    if content_type:
        headers["Content-Type"] = content_type
    req = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    # The compressed STRING alias archive can take longer than API metadata
    # requests on a constrained connection.
    with urlopen(req, timeout=300) as response:
        return response.read()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, help="Existing acquisition folder to complete")
    parser.add_argument("--only-ensembl", action="store_true", help="Fetch only the Ensembl 102 BioMart table")
    args = parser.parse_args()
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Local GDC manifest is required: {MANIFEST}")
    if args.output_dir:
        out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        if not out.is_dir():
            raise FileNotFoundError(f"Acquisition directory does not exist: {out}")
    else:
        out = ROOT / "data" / "reference" / ("metadata_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        out.mkdir(parents=True, exist_ok=False)
    if not args.only_ensembl:
        manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline=""), delimiter="\t"))
        ids = [row["id"] for row in manifest_rows]
        if len(ids) != len(set(ids)):
            raise ValueError("GDC manifest contains duplicate file IDs")

        # Query in small batches; this is metadata only, not the file contents.
        hits = []
        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            params = {"filters": {"op": "in", "content": {"field": "files.file_id", "value": batch}}}
            url = GDC_URL + "?fields=" + quote(GDC_FIELDS, safe=",") + "&format=JSON&size=100"
            response = json.loads(fetch(url, data=json.dumps(params).encode("utf-8"), content_type="application/json"))
            hits.extend(response.get("data", {}).get("hits", []))
            print(f"GDC metadata {min(start + 100, len(ids))}/{len(ids)}", flush=True)
            time.sleep(0.15)
        gdc_ids = {row.get("file_id") for row in hits}
        missing_gdc_ids = sorted(set(ids) - gdc_ids)
        gdc_path = out / "gdc_file_sample_case_metadata.json"
        gdc_path.write_text(json.dumps({"source": GDC_URL, "requested_file_ids": len(ids), "returned_records": len(hits),
            "missing_file_ids": missing_gdc_ids, "fields": GDC_FIELDS.split(","), "hits": hits}, indent=2), encoding="utf-8")

        # Fetch the human-filtered alias asset from versioned STRING 12.5.
        gz_path = out / "9606.protein.aliases.v12.5.txt.gz"
        if gz_path.exists(): raise FileExistsError(gz_path)
        payload = fetch(STRING_URL)
        gz_path.write_bytes(payload)
        aliases_path = out / "9606.protein.aliases.v12.5.txt"
        with gzip.open(gz_path, "rb") as source, aliases_path.open("wb") as dest:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(block)
        with aliases_path.open(encoding="utf-8", errors="strict") as f:
            alias_header = f.readline().rstrip("\r\n")
        if alias_header != "#string_protein_id\talias\tsource":
            raise ValueError(f"Unexpected STRING alias schema: {alias_header!r}")
    # Ensembl 102 corresponds to GENCODE v36. Keep the query and raw TSV.
    query = """<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE Query>
<Query virtualSchemaName='default' formatter='TSV' header='1' uniqueRows='1' count='' datasetConfigVersion='0.6'>
  <Dataset name='hsapiens_gene_ensembl' interface='default'>
    <Attribute name='ensembl_gene_id'/>
    <Attribute name='entrezgene_id'/>
    <Attribute name='hgnc_symbol'/>
  </Dataset>
</Query>"""
    mapping_bytes = fetch(BIOMART_URL + quote(query, safe=""))
    mapping_path = out / "ensembl102_gene_entrez_hgnc_mapping.tsv"
    first_line = mapping_bytes.splitlines()[0].decode("utf-8", errors="replace") if mapping_bytes else ""
    if "ensembl_gene_id" not in first_line or "entrezgene_id" not in first_line:
        error_path = out / ("ensembl102_biomart_error_response_" + datetime.now().strftime("%H%M%S") + ".html")
        if error_path.exists():
            raise FileExistsError(error_path)
        error_path.write_bytes(mapping_bytes)
        raise ValueError(f"Unexpected Ensembl BioMart response header: {first_line!r}")
    if mapping_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing mapping table: {mapping_path}")
    mapping_path.write_bytes(mapping_bytes)

    summary = {"output_directory": str(out.relative_to(ROOT)),
        "ensembl_mapping_source": "Ensembl archive Nov 2020 (release 102; GENCODE v36)",
        "ensembl_mapping_bytes": mapping_path.stat().st_size, "ensembl_mapping_header": first_line,
        "files": [mapping_path.name]}
    (out / "acquisition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__": main()
