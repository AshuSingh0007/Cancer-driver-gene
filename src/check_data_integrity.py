"""Assert critical Phase 2 invariants; writes a machine-readable check result."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = max((p for p in (ROOT / "data" / "processed").glob("validation_*") if (p / "data_validation_summary.json").exists()), key=lambda p: p.stat().st_mtime)

def main():
    tcga = json.loads((OUT / "tcga_validation_summary.json").read_text(encoding="utf-8"))
    string = json.loads((OUT / "string_validation_summary.json").read_text(encoding="utf-8"))
    ann = json.loads((OUT / "annotation_validation_summary.json").read_text(encoding="utf-8"))
    files = pd.read_csv(OUT / "tcga_file_validation.csv")
    manifest = pd.read_csv(OUT / "tcga_sample_case_manifest.csv")
    labels = pd.read_csv(OUT / "ncg_gene_label_table.csv")
    edges = pd.read_csv(OUT / "ppi_edges_string_v12_5.csv")
    features_dir = max((p for p in (ROOT / "data" / "processed").glob("expression_features_*") if (p / "expression_feature_summary.json").exists()), key=lambda p: p.stat().st_mtime)
    feature_summary = json.loads((features_dir / "expression_feature_summary.json").read_text(encoding="utf-8"))
    model_dir = max((p for p in (ROOT / "data" / "processed").glob("model_inputs_*") if (p / "model_input_components_summary.json").exists()), key=lambda p: p.stat().st_mtime)
    model_summary = json.loads((model_dir / "model_input_components_summary.json").read_text(encoding="utf-8"))
    graph_nodes = pd.read_csv(model_dir / "graph_nodes.csv")
    model_labels = pd.read_csv(model_dir / "gene_labels.csv")
    phase3 = max((p for p in (ROOT / "data" / "processed").glob("phase3_task_*") if (p / "phase3_task_summary.json").exists()), key=lambda p: p.stat().st_mtime)
    phase3_summary = json.loads((phase3 / "phase3_task_summary.json").read_text(encoding="utf-8"))
    breast_labels = pd.read_csv(phase3 / "ncg_breast_evidence_gene_labels.csv")
    checks = {
        "all_1228_expression_files_parse": len(files) == 1228 and tcga["valid_file_count"] == 1228 and tcga["files_with_issues"] == 0,
        "expression_checksum_and_size_matches": tcga["manifest_checksum_matches"] == 1228 and tcga["manifest_checksum_mismatches"] == 0 and tcga["manifest_size_matches"] == 1228,
        "three_manifest_files_absent": tcga["manifest_file_count"] == 1231 and len(tcga["manifest_entries_missing_from_project"]) == 3,
        "gene_universe_shared_and_unique": tcga["unique_gene_id_union"] == tcga["gene_ids_shared_by_every_file"] == 60660 and tcga["duplicate_gene_rows"] == 0,
        "expression_cells_and_rows_clean": tcga["malformed_rows"] == tcga["missing_numeric_cells"] == tcga["invalid_numeric_cells"] == 0,
        "all_gene_ids_valid_with_PAR_Y_support": tcga["non_ensembl_gene_ids"] == 0 and tcga["par_y_ensembl_gene_id_rows"] == 54032,
        "gdc_case_sample_metadata_joined": tcga["unique_sample_count"] == 1223 and tcga["unique_patient_count"] == 1093 and tcga["ambiguous_gdc_file_mappings"] == 0 and manifest["sample_type"].notna().all(),
        "primary_tumour_feature_cohort_documented": feature_summary["primary_tumour_samples"] == 1103 and feature_summary["primary_tumour_cases"] == 1092 and feature_summary["samples_with_multiple_expression_files"] == 5,
        "string_links_structurally_valid": string["link_rows"] == 10826054 and string["score_min"] >= 0 and string["score_max"] <= 1000 and string["raw_self_links"] == 0 and string["adjacent_duplicate_protein_pairs"] == 0,
        "string_mapping_releases_match": string["release_resolution"]["compatible_aliases_file_found"] is True and string["release_resolution"]["links_release"] == string["release_resolution"]["aliases_content_release"] == "12.5",
        "release_matched_graph_matches_statistics": len(edges) == string["release_matched_ncg_graph"]["edges"] == 24956 and string["release_matched_ncg_graph"]["self_loops"] == 0 and not (edges.source == edges.target).any(),
        "labels_preserve_unknown_and_have_no_conflicting_gene_rows": len(labels) == 3347 and int(labels.oncogene_status.eq("unknown").sum()) == 2756 and int(labels.oncogene_status.eq("conflict").sum()) == 0,
        "NCG_COSMIC_kept_separate": ann["cosmic_ncg_symbol_overlap"] == 656 and ann["oncogene_role_discordances_not_proven_conflicts"] == 63 and ann["tsg_role_discordances_not_proven_conflicts"] == 61,
        "phase3_breast_sensitivity_cohort_counts": len(breast_labels) == 298 and int(breast_labels.oncogene_label.notna().sum()) == 151 and int(breast_labels.oncogene_label.eq(1).sum()) == 52 and int(breast_labels.oncogene_label.eq(0).sum()) == 99 and int(phase3_summary["secondary_breast_evidence_sensitivity_task"]["oncogene_status_counts"]["unknown"]) == 147,
        "expression_gene_feature_table_integrity": feature_summary["unique_versioned_ensembl_genes"] == 60660 and feature_summary["feature_file"].endswith("gene_expression_features.csv"),
        "separated_model_input_components_integrity": len(graph_nodes) == 3325 and model_summary["graph_edge_count"] == len(edges) == 24956 and len(model_labels) == 3347 and model_summary["expression_to_target_join"].startswith("Not performed"),
        "required_reports_present": all((ROOT / "reports" / name).exists() for name in ["data_quality_report.md", "gene_identifier_mapping_report.md", "prediction_task_specification.md"]),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {"output_directory": str(OUT.relative_to(ROOT)), "checks": checks, "passed": all(checks.values())}
    check_path = OUT / "data_integrity_check.json"
    if check_path.exists():
        from datetime import datetime
        check_path = OUT / ("data_integrity_check_" + datetime.now().strftime("%H%M%S") + ".json")
    check_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]: raise SystemExit(1)

if __name__ == "__main__": main()
