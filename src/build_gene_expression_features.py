"""Build case-level BRCA primary-tumour gene expression summaries in a streaming pass.

Per sample, duplicate aliquot-file TPM values are averaged. If a case has multiple
primary-tumour samples, sample-level log2(TPM+1) values are median-combined per
gene. The output columns summarize those case-level values; no gene matrix is
materialized in RAM.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data/processed/validation_20261028_152218"
MANIFEST = VALIDATION / "tcga_sample_case_manifest.csv"
OUTROOT = ROOT / "data/processed"


def read_tpm(path: Path, ids: list[str]) -> np.ndarray:
    row_index = {gene_id: i for i, gene_id in enumerate(ids)}
    values = np.zeros(len(ids), dtype=np.float64)
    seen = np.zeros(len(ids), dtype=np.bool_)
    with path.open(encoding="utf-8", newline="") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f, delimiter="\t")
        if not {"gene_id", "tpm_unstranded"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing expression columns: {path}")
        for row in reader:
            gene_id = row["gene_id"]
            if gene_id.startswith("N_"):
                continue
            i = row_index.get(gene_id)
            if i is None:
                raise ValueError(f"Unexpected gene identifier {gene_id!r} in {path}")
            if seen[i]:
                raise ValueError(f"Duplicate gene ID {gene_id!r} in {path}")
            values[i] = float(row["tpm_unstranded"])
            seen[i] = True
    if not seen.all():
        raise ValueError(f"Incomplete gene rows in {path}: {int((~seen).sum())} missing")
    return values


def main():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    selected = [r for r in rows if r["status"] == "validated" and r["sample_type"] == "Primary Tumor"]
    sample_to_case: dict[str, str] = {}
    sample_to_files: dict[str, list[Path]] = defaultdict(list)
    for r in selected:
        samples = r["sample_id"].split("|") if r["sample_id"] else []
        cases = r["case_id"].split("|") if r["case_id"] else []
        if len(samples) != 1 or len(cases) != 1 or r["gdc_mapping_ambiguous"] == "True":
            raise ValueError(f"Ambiguous GDC mapping in {r['relative_path']}")
        sample, case = samples[0], cases[0]
        if sample in sample_to_case and sample_to_case[sample] != case:
            raise ValueError(f"Sample maps to multiple cases: {sample}")
        sample_to_case[sample] = case
        sample_to_files[sample].append(ROOT / r["relative_path"])

    first = selected[0]
    with (ROOT / first["relative_path"]).open(encoding="utf-8", newline="") as f:
        if f.readline().startswith("#"):
            pass
        else:
            f.seek(0)
        reader = csv.DictReader(f, delimiter="\t")
        gene_rows = [row for row in reader if not row["gene_id"].startswith("N_")]
    ids = [row["gene_id"] for row in gene_rows]
    symbols = [row["gene_name"] for row in gene_rows]
    id_index = {g: i for i, g in enumerate(ids)}
    cases_to_samples: dict[str, list[str]] = defaultdict(list)
    for sample, case in sample_to_case.items():
        cases_to_samples[case].append(sample)

    # A single vector per primary-tumour patient is accumulated at a time.
    sum_x = np.zeros(len(ids), dtype=np.float64)
    sum_x2 = np.zeros(len(ids), dtype=np.float64)
    positive = np.zeros(len(ids), dtype=np.uint32)
    case_count = 0
    for case, samples in sorted(cases_to_samples.items()):
        sample_logs = []
        for sample in sorted(samples):
            files = sorted(sample_to_files[sample])
            tpm_sum = np.zeros(len(ids), dtype=np.float64)
            for path in files:
                tpm_sum += read_tpm(path, ids)
            mean_tpm = tpm_sum / len(files)
            sample_logs.append(np.log2(mean_tpm + 1.0))
        case_values = np.median(np.stack(sample_logs, axis=0), axis=0)
        sum_x += case_values
        sum_x2 += case_values * case_values
        positive += (case_values > 0).astype(np.uint32)
        case_count += 1
        if case_count % 100 == 0 or case_count == len(cases_to_samples):
            print(f"primary-tumour cases {case_count}/{len(cases_to_samples)}", flush=True)
    mean = sum_x / case_count
    variance = np.maximum(sum_x2 / case_count - mean * mean, 0.0)
    sd = np.sqrt(variance)
    out = OUTROOT / ("expression_features_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    feature_path = out / "gene_expression_features.csv"
    with feature_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ensembl_gene_id_versioned", "gene_symbol", "mean_log2_tpm_plus_1", "sd_log2_tpm_plus_1", "fraction_cases_tpm_positive", "primary_tumour_case_count"])
        for i, gene_id in enumerate(ids):
            writer.writerow([gene_id, symbols[i], format(mean[i], ".9g"), format(sd[i], ".9g"), format(positive[i] / case_count, ".9g"), case_count])
    summary = {
        "feature_file": str(feature_path.relative_to(ROOT)),
        "prediction_unit": "gene; features summarize primary tumour cases, not individual patients",
        "selected_validated_primary_tumour_expression_files": len(selected),
        "primary_tumour_samples": len(sample_to_case),
        "primary_tumour_cases": case_count,
        "samples_with_multiple_expression_files": sum(len(v) > 1 for v in sample_to_files.values()),
        "cases_with_multiple_primary_samples": sum(len(v) > 1 for v in cases_to_samples.values()),
        "unique_versioned_ensembl_genes": len(ids),
        "features": ["mean(log2(TPM+1)) across cases", "population SD(log2(TPM+1)) across cases", "fraction of cases with TPM > 0"],
        "aggregation_policy": "Average raw TPM across files for a single GDC sample; transform sample TPM to log2(TPM+1); median-combine multiple primary-tumour samples within a case gene-wise; compute mean, population SD, and TPM-positive fraction across cases.",
        "unresolved_identifier_caveat": "Features are keyed by versioned Ensembl IDs. An authoritative Ensembl v102 to Entrez crosswalk was not obtained; symbol matches are candidate-only and are not used to silently join labels.",
        "peak_in_memory_expression_values": "one 60,660-value float64 vector per file/sample plus vectors for the current case; no patient-by-gene matrix is materialized",
    }
    (out / "expression_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
