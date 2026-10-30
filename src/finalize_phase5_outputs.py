"""Summarize the completed cohort, mapping, and separated model inputs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data/reference/metadata_20261028_151433"
VALIDATION = ROOT / "data/processed/validation_20261028_152218"
FEATURES = ROOT / "data/processed/expression_features_20261028_153824"
MODEL_INPUTS = ROOT / "data/processed/model_inputs_20261028_154017"
JOIN = ROOT / "data/processed/gdc_metadata_join_20261028_152106"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    bad_tsv = REF / "ensembl102_gene_entrez_hgnc_mapping.tsv"
    if bad_tsv.exists():
        backup = REF / "ensembl102_gene_entrez_hgnc_mapping.extra-invalid-response.html"
        if backup.exists():
            raise FileExistsError(backup)
        bad_tsv.replace(backup)
    gdc = json.loads((REF / "gdc_file_sample_case_metadata.json").read_text(encoding="utf-8"))
    alias_gz = REF / "9606.protein.aliases.v12.5.txt.gz"
    alias_txt = REF / "9606.protein.aliases.v12.5.txt"
    with alias_txt.open(encoding="utf-8") as f:
        alias_header = f.readline().rstrip("\r\n")
    acquisition = {
        "gdc_metadata": {"requested_file_ids": gdc["requested_file_ids"], "returned_records": gdc["returned_records"], "missing_file_ids": gdc["missing_file_ids"]},
        "string_aliases": {"url": "https://stringdb-downloads.org/download/protein.aliases.v12.5/9606.protein.aliases.v12.5.txt.gz", "compressed_bytes": alias_gz.stat().st_size, "uncompressed_bytes": alias_txt.stat().st_size, "sha256_compressed": sha256(alias_gz), "header": alias_header, "status": "downloaded and header validated"},
        "ensembl102_mapping": {"status": "unresolved", "requested_endpoint": "https://nov2020.archive.ensembl.org/biomart/martservice", "result": "HTML archive help page; not tabular mapping data", "invalid_responses_preserved": [p.name for p in REF.glob("*invalid-response.html")]},
    }
    (REF / "acquisition_summary.json").write_text(json.dumps(acquisition, indent=2), encoding="utf-8")
    tcga = json.loads((VALIDATION / "tcga_validation_summary.json").read_text(encoding="utf-8"))
    string = json.loads((VALIDATION / "string_validation_summary.json").read_text(encoding="utf-8"))
    features = json.loads((FEATURES / "expression_feature_summary.json").read_text(encoding="utf-8"))
    model = json.loads((MODEL_INPUTS / "model_input_components_summary.json").read_text(encoding="utf-8"))
    phase5 = {
        "validated_expression_files": tcga["valid_file_count"],
        "unique_local_samples": tcga["unique_sample_count"],
        "unique_local_cases": tcga["unique_patient_count"],
        "local_sample_type_counts": tcga["sample_type_counts"],
        "ambiguous_gdc_mappings": tcga["ambiguous_gdc_file_mappings"],
        "string_release_matched": string["release_resolution"]["compatible_aliases_file_found"],
        "ncg_genes_with_unambiguous_string_mapping": string["ncg_genes_with_unambiguous_string_protein_mapping"],
        "release_matched_graph": string["release_matched_ncg_graph"],
        "primary_tumour_cases_used_for_expression_features": features["primary_tumour_cases"],
        "expression_feature_rows": features["unique_versioned_ensembl_genes"],
        "separated_model_inputs": model["files"],
        "integrated_expression_label_matrix_created": False,
        "blocker": "Ensembl v102 BioMart endpoint returned HTML rather than an authoritative Ensembl-to-Entrez table.",
        "training_started": False,
    }
    (VALIDATION / "phase5_validation_summary.json").write_text(json.dumps(phase5, indent=2), encoding="utf-8")
    data_path = VALIDATION / "data_validation_summary.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data["model_ready_expression_features_created"] = True
    data["model_ready_expression_feature_path"] = features["feature_file"]
    data["model_ready_integrated_training_matrix_created"] = False
    data["model_ready_reason"] = phase5["blocker"] + " Expression, graph, and labels are exported separately; no identifier join or train/test split was performed."
    data["gdc_metadata_join_summary"] = str((JOIN / "gdc_metadata_join_summary.json").relative_to(ROOT))
    data["phase5_validation_summary"] = "phase5_validation_summary.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(phase5, indent=2))


if __name__ == "__main__":
    main()
