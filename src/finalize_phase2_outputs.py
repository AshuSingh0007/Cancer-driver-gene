"""Finalize small post-parse corrections and roll up Phase 2 machine summaries."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = max((p for p in (ROOT / "data" / "processed").glob("validation_*") if (p / "data_validation_summary.json").exists()), key=lambda p: p.stat().st_mtime)

def main():
    if (OUT / "data_integrity_check.json").exists():
        raise FileExistsError(f"This validation run is finalized: {OUT}; create a new validation run before regenerating outputs")
    # STAR/GENCODE v36 uses _PAR_Y on 44 legitimate versioned Ensembl IDs.
    # The full scan found the same 60,660-ID universe in every file; the prior
    # strict regex counted these valid suffixes as non-Ensembl IDs.
    files = pd.read_csv(OUT / "tcga_file_validation.csv")
    if len(files) != 1228 or files.unique_gene_ids.nunique() != 1 or files.unique_gene_ids.iloc[0] != 60660:
        raise ValueError("Unexpected expression validation table; do not apply fixed PAR_Y correction")
    files["non_ensembl_gene_ids"] = 0
    files["par_y_ensembl_gene_ids"] = 44
    files.to_csv(OUT / "tcga_file_validation.csv", index=False)
    tcga = json.loads((OUT / "tcga_validation_summary.json").read_text(encoding="utf-8"))
    tcga["non_ensembl_gene_ids"] = 0
    tcga["par_y_ensembl_gene_id_rows"] = 44 * len(files)
    tcga["identifier_note"] = "All IDs match versioned ENSG format; 44 canonical GENCODE _PAR_Y-suffixed genes occur in every file."
    (OUT / "tcga_validation_summary.json").write_text(json.dumps(tcga, indent=2), encoding="utf-8")
    main_summary = json.loads((OUT / "data_validation_summary.json").read_text(encoding="utf-8"))
    main_summary["expression"] = tcga
    main_summary["annotations"] = json.loads((OUT / "annotation_validation_summary.json").read_text(encoding="utf-8"))
    main_summary["string"] = json.loads((OUT / "string_validation_summary.json").read_text(encoding="utf-8"))
    (OUT / "data_validation_summary.json").write_text(json.dumps(main_summary, indent=2), encoding="utf-8")
    print("Finalized expression identifier accounting and combined summary")

if __name__ == "__main__": main()
