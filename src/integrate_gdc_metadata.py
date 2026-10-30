"""Join GDC file metadata to locally validated TCGA expression files."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACQ = ROOT / "data/reference/metadata_20261028_151433"
VALIDATION = ROOT / "data/processed/validation_20261028"


def main():
    raw = json.loads((ACQ / "gdc_file_sample_case_metadata.json").read_text(encoding="utf-8"))
    by_id = {row["file_id"]: row for row in raw["hits"]}
    validations = list(csv.DictReader((VALIDATION / "tcga_file_validation.csv").open(encoding="utf-8", newline="")))
    records = []
    for item in validations:
        file_id = item.get("manifest_file_id")
        meta = by_id.get(file_id, {})
        cases = meta.get("cases", [])
        # Preserve a row per file and expose ambiguity rather than arbitrarily choosing.
        case_ids = sorted({c.get("case_id") for c in cases if c.get("case_id")})
        case_barcodes = sorted({c.get("submitter_id") for c in cases if c.get("submitter_id")})
        samples = [s for c in cases for s in c.get("samples", [])]
        sample_ids = sorted({s.get("sample_id") for s in samples if s.get("sample_id")})
        sample_barcodes = sorted({s.get("submitter_id") for s in samples if s.get("submitter_id")})
        sample_types = sorted({s.get("sample_type") for s in samples if s.get("sample_type")})
        records.append({
            **{k: item.get(k) for k in ("relative_path", "bytes", "status", "md5", "manifest_file_id", "manifest_checksum_match", "manifest_size_match")},
            "gdc_file_id": file_id,
            "gdc_file_metadata_found": bool(meta),
            "case_ids": "|".join(case_ids), "case_barcodes": "|".join(case_barcodes),
            "sample_ids": "|".join(sample_ids), "sample_barcodes": "|".join(sample_barcodes),
            "sample_types": "|".join(sample_types), "sample_count_for_file": len(sample_ids),
            "case_count_for_file": len(case_ids),
            "mapping_ambiguous": len(sample_ids) != 1 or len(case_ids) != 1 or len(sample_types) != 1,
        })
    local_ids = {r["gdc_file_id"] for r in records if r["gdc_file_id"]}
    type_by_sample = defaultdict(set)
    case_by_sample = defaultdict(set)
    case_by_barcode = {}
    files_by_sample = defaultdict(list)
    for r in records:
        for s in r["sample_ids"].split("|") if r["sample_ids"] else []:
            type_by_sample[s].update(r["sample_types"].split("|") if r["sample_types"] else [])
            case_by_sample[s].update(r["case_ids"].split("|") if r["case_ids"] else [])
            files_by_sample[s].append(r["relative_path"])
        for cb, ci in zip(r["case_barcodes"].split("|"), r["case_ids"].split("|")):
            if cb and ci: case_by_barcode[cb] = ci
    out = ROOT / "data/processed" / ("gdc_metadata_join_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    with (out / "tcga_sample_case_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), extrasaction="ignore")
        w.writeheader(); w.writerows(records)
    duplicate_samples = {s: paths for s, paths in files_by_sample.items() if len(paths) > 1}
    summary = {
        "gdc_file_metadata_records": len(by_id),
        "manifest_file_ids_requested": raw["requested_file_ids"],
        "gdc_file_ids_not_returned": raw["missing_file_ids"],
        "local_expression_files": len(records),
        "local_expression_file_ids_with_metadata": sum(r["gdc_file_metadata_found"] for r in records),
        "validated_local_expression_files": sum(r["status"] == "validated" for r in records),
        "metadata_unique_sample_count_all_gdc_files": len({s.get("sample_id") for h in raw["hits"] for c in h.get("cases", []) for s in c.get("samples", []) if s.get("sample_id")}),
        "metadata_unique_case_count_all_gdc_files": len({c.get("case_id") for h in raw["hits"] for c in h.get("cases", []) if c.get("case_id")}),
        "local_unique_sample_count": len(type_by_sample),
        "local_unique_case_count": len({ci for values in case_by_sample.values() for ci in values}),
        "local_sample_type_counts": dict(Counter(t for values in type_by_sample.values() for t in values)),
        "local_samples_with_conflicting_types": sorted(s for s, v in type_by_sample.items() if len(v) > 1),
        "local_duplicate_sample_file_groups": duplicate_samples,
        "local_files_with_ambiguous_or_missing_mapping": sum(bool(r["mapping_ambiguous"]) for r in records),
        "manifest": str((out / "tcga_sample_case_manifest.csv").relative_to(ROOT)),
    }
    (out / "gdc_metadata_join_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
