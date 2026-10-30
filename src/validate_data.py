"""Memory-bounded Phase 2 validation for the existing D:\\Quantum project.

Reads source datasets only. Outputs are written to a new, versioned validation
directory and do not replace existing project artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "validation_20261028"
EXPR_DIR = ROOT / "data" / "tcga" / "expression"
MANIFEST = Path(r"C:\Users\Hp\Downloads\gdc_manifest.2026-10-27.234917.txt")
LINKS = ROOT / "9606.protein.links.v12.5.txt" / "9606.protein.links.v12.5.txt"
_MATCHED_ALIASES = sorted((ROOT / "data" / "reference").glob("metadata_*/9606.protein.aliases.v12.5.txt"))
ALIASES = _MATCHED_ALIASES[-1] if _MATCHED_ALIASES else ROOT / "9606.protein.aliases.v12.5.txt" / "9606.protein.aliases.v12.0.txt"
GDC_METADATA = ROOT / "data" / "reference" / "metadata_20261028_151433" / "gdc_file_sample_case_metadata.json"
EXPR_HEADER = ["gene_id", "gene_name", "gene_type", "unstranded", "stranded_first",
               "stranded_second", "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"]
ENSG_RE = re.compile(r"^ENSG\d+\.\d+(?:_PAR_Y)?$")
PAR_Y_RE = re.compile(r"^ENSG\d+\.\d+_PAR_Y$")


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def validate_expression():
    files = sorted(EXPR_DIR.rglob("*.rna_seq.augmented_star_gene_counts.tsv"))
    manifest_rows = {}
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                manifest_rows[row["filename"]] = row
    gdc_by_file = {}
    if GDC_METADATA.exists():
        gdc_payload = json.loads(GDC_METADATA.read_text(encoding="utf-8"))
        gdc_by_file = {row["file_id"]: row for row in gdc_payload.get("hits", [])}
    expected_by_name = {p.name: p for p in files}
    file_records = []
    global_gene_ids = set()
    common_gene_ids = None
    all_md5_seen = defaultdict(list)
    schema_counts = Counter()
    global_summary_labels = Counter()
    gene_type_counts = Counter()
    for i, path in enumerate(files, 1):
        started = time.time()
        md5 = hashlib.md5()
        genes = set()
        row_count = gene_rows = summary_rows = malformed = nonnumeric = missing_cells = dup_gene_rows = 0
        bad_ensg = par_y_ids = no_symbol = 0
        local_gene_type = Counter()
        header = None
        error = None
        try:
            with path.open("rb") as fh:
                first = fh.readline(); md5.update(first)
                first_text = first.decode("utf-8-sig", errors="strict").rstrip("\r\n")
                if first_text.startswith("# gene-model:"):
                    model = first_text.split(":", 1)[1].strip()
                    header_line = fh.readline(); md5.update(header_line)
                else:
                    model = None
                    header_line = first
                header = next(csv.reader([header_line.decode("utf-8", errors="strict").rstrip("\r\n")], delimiter="\t"))
                schema_counts[tuple(header)] += 1
                if header != EXPR_HEADER:
                    error = "unexpected_header"
                for raw in fh:
                    md5.update(raw)
                    row_count += 1
                    try:
                        row = next(csv.reader([raw.decode("utf-8", errors="strict").rstrip("\r\n")], delimiter="\t"))
                    except Exception:
                        malformed += 1
                        continue
                    if len(row) != len(EXPR_HEADER):
                        malformed += 1
                        continue
                    gene_id, symbol, gene_type = row[:3]
                    if gene_id.startswith("N_"):
                        summary_rows += 1; global_summary_labels[gene_id] += 1
                        continue
                    gene_rows += 1
                    if not ENSG_RE.fullmatch(gene_id): bad_ensg += 1
                    if PAR_Y_RE.fullmatch(gene_id): par_y_ids += 1
                    if not symbol: no_symbol += 1
                    if gene_id in genes: dup_gene_rows += 1
                    genes.add(gene_id)
                    local_gene_type[gene_type or "<missing>"] += 1
                    numeric = row[3:]
                    for j, value in enumerate(numeric):
                        if value == "":
                            missing_cells += 1
                            continue
                        try:
                            number = float(value)
                            if not (number >= 0 and number < float("inf")):
                                nonnumeric += 1
                            elif j < 3 and not number.is_integer():
                                nonnumeric += 1
                        except ValueError:
                            nonnumeric += 1
            file_md5 = md5.hexdigest()
            all_md5_seen[file_md5].append(path)
            global_gene_ids.update(genes)
            common_gene_ids = genes.copy() if common_gene_ids is None else common_gene_ids.intersection(genes)
            gene_type_counts.update(local_gene_type)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            model = None
            file_md5 = md5.hexdigest()
        exp = manifest_rows.get(path.name)
        parent_id = path.parent.name
        filename_id = path.name.split(".", 1)[0]
        gdc = gdc_by_file.get(exp.get("id")) if exp else None
        gdc_cases = gdc.get("cases", []) if gdc else []
        case_ids = sorted({c.get("case_id") for c in gdc_cases if c.get("case_id")})
        case_barcodes = sorted({c.get("submitter_id") for c in gdc_cases if c.get("submitter_id")})
        gdc_samples = [s for c in gdc_cases for s in c.get("samples", [])]
        sample_ids = sorted({s.get("sample_id") for s in gdc_samples if s.get("sample_id")})
        sample_barcodes = sorted({s.get("submitter_id") for s in gdc_samples if s.get("submitter_id")})
        sample_types = sorted({s.get("sample_type") for s in gdc_samples if s.get("sample_type")})
        record = {
            "relative_path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "gene_model": model, "header": header, "row_count_excluding_header": row_count,
            "gene_rows": gene_rows, "summary_rows": summary_rows,
            "summary_ids": sorted({k for k, v in global_summary_labels.items() if v and k.startswith("N_")}),
            "unique_gene_ids": len(genes), "duplicate_gene_rows": dup_gene_rows,
            "non_ensembl_gene_ids": bad_ensg, "par_y_ensembl_gene_ids": par_y_ids,
            "missing_gene_symbols": no_symbol,
            "missing_numeric_cells": missing_cells, "invalid_numeric_cells": nonnumeric,
            "malformed_rows": malformed, "gene_type_counts": dict(local_gene_type),
            "md5": file_md5, "manifest_entry_found": exp is not None,
            "manifest_file_id": exp.get("id") if exp else None,
            "manifest_checksum_match": (file_md5.lower() == exp.get("md5", "").lower()) if exp else None,
            "manifest_size_match": (path.stat().st_size == int(exp["size"])) if exp else None,
            "outer_directory_uuid": parent_id, "filename_uuid": filename_id,
            "sample_id": "|".join(sample_ids) or None, "case_id": "|".join(case_ids) or None,
            "sample_barcode": "|".join(sample_barcodes) or None,
            "case_barcode": "|".join(case_barcodes) or None,
            "sample_type": "|".join(sample_types) or None,
            "gdc_sample_count": len(sample_ids), "gdc_case_count": len(case_ids),
            "gdc_mapping_ambiguous": len(sample_ids) != 1 or len(case_ids) != 1 or len(sample_types) != 1,
            "status": "validated" if not error and malformed == 0 and nonnumeric == 0 and dup_gene_rows == 0 else "issues",
            "error": error, "validation_seconds": round(time.time() - started, 3),
        }
        file_records.append(record)
        if i % 100 == 0 or i == len(files):
            print(f"expression {i}/{len(files)}", flush=True)
    duplicates = {p: [str(x.relative_to(ROOT)) for x in group] for p, group in all_md5_seen.items() if len(group) > 1}
    manifest_names = set(manifest_rows)
    present = set(expected_by_name)
    sample_files = defaultdict(list)
    for x in file_records:
        for sample_id in (x.get("sample_id") or "").split("|"):
            if sample_id:
                sample_files[sample_id].append(x["relative_path"])
    duplicate_sample_files = {sample: paths for sample, paths in sample_files.items() if len(paths) > 1}
    meta = {
        "expression_file_count": len(files), "valid_file_count": sum(x["status"] == "validated" for x in file_records),
        "files_with_issues": sum(x["status"] != "validated" for x in file_records),
        "total_bytes": sum(p.stat().st_size for p in files), "manifest_path": str(MANIFEST) if MANIFEST.exists() else None,
        "manifest_file_count": len(manifest_rows), "manifest_files_present": len(manifest_names & present),
        "manifest_entries_missing_from_project": sorted(manifest_names - present),
        "project_files_absent_from_manifest": sorted(present - manifest_names),
        "manifest_checksum_matches": sum(x["manifest_checksum_match"] is True for x in file_records),
        "manifest_checksum_mismatches": sum(x["manifest_checksum_match"] is False for x in file_records),
        "manifest_size_matches": sum(x["manifest_size_match"] is True for x in file_records),
        "unique_sample_count": len({s for x in file_records for s in (x.get("sample_id") or "").split("|") if s}),
        "unique_patient_count": len({c for x in file_records for c in (x.get("case_id") or "").split("|") if c}),
        "sample_type_counts": dict(Counter(t for s in {s for x in file_records for s in (x.get("sample_id") or "").split("|") if s}
            for t in {t for x in file_records if s in (x.get("sample_id") or "").split("|") for t in (x.get("sample_type") or "").split("|") if t})),
        "duplicate_biological_sample_file_groups": duplicate_sample_files,
        "ambiguous_gdc_file_mappings": sum(x.get("gdc_mapping_ambiguous", True) for x in file_records),
        "sample_case_status": "GDC Files API metadata queried by manifest file UUID; file-to-case/sample/sample-type links are recorded in the sample/case manifest.",
        "schema_counts": {"\t".join(k): v for k, v in schema_counts.items()},
        "summary_row_occurrences": dict(global_summary_labels), "gene_type_occurrences": dict(gene_type_counts),
        "unique_gene_id_union": len(global_gene_ids), "gene_ids_shared_by_every_file": len(common_gene_ids or set()),
        "duplicate_file_content_groups": duplicates,
        "file_status_counts": dict(Counter(x["status"] for x in file_records)),
        "total_gene_rows": sum(x["gene_rows"] for x in file_records),
        "duplicate_gene_rows": sum(x["duplicate_gene_rows"] for x in file_records),
        "non_ensembl_gene_ids": sum(x["non_ensembl_gene_ids"] for x in file_records),
        "par_y_ensembl_gene_id_rows": sum(x["par_y_ensembl_gene_ids"] for x in file_records),
        "missing_gene_symbols": sum(x["missing_gene_symbols"] for x in file_records),
        "missing_numeric_cells": sum(x["missing_numeric_cells"] for x in file_records),
        "invalid_numeric_cells": sum(x["invalid_numeric_cells"] for x in file_records),
        "malformed_rows": sum(x["malformed_rows"] for x in file_records),
        "summary_row_count_per_file_expected": 4,
    }
    pd.DataFrame(file_records).drop(columns=["header", "gene_type_counts", "summary_ids"]).to_csv(OUT / "tcga_file_validation.csv", index=False)
    pd.DataFrame(file_records)[["relative_path", "manifest_file_id", "manifest_checksum_match", "manifest_size_match", "outer_directory_uuid", "filename_uuid", "sample_id", "sample_barcode", "case_id", "case_barcode", "sample_type", "gdc_sample_count", "gdc_case_count", "gdc_mapping_ambiguous", "status"]].to_csv(OUT / "tcga_sample_case_manifest.csv", index=False)
    write_json(OUT / "tcga_validation_summary.json", meta)
    return meta, file_records


def validate_string():
    # Resolve STRING protein -> Entrez IDs from whatever alias release is present.
    alias_map = defaultdict(set)
    alias_rows = 0
    with ALIASES.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        alias_cols = reader.fieldnames
        for row in reader:
            alias_rows += 1
            if row.get("source") == "Ensembl_HGNC_entrez_id":
                prot = row.get("#string_protein_id", "")
                val = row.get("alias", "")
                if val.isdigit(): alias_map[prot].add(int(val))
    unique_entrez = {p: next(iter(ids)) for p, ids in alias_map.items() if len(ids) == 1}
    ambiguous = sum(len(v) > 1 for v in alias_map.values())
    matched, score_hist = 0, Counter()
    min_score, max_score = None, None
    self_loops = duplicate_pairs = 0
    previous = None
    proteins = set()
    endpoint_rows = mapped_endpoint_rows = 0
    pair_order_monotonic = True
    with LINKS.open(encoding="ascii", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=" ", skipinitialspace=True)
        header = reader.fieldnames
        for row in reader:
            a, b = row["protein1"], row["protein2"]
            try: score = int(row["combined_score"])
            except Exception: continue
            matched += 1
            min_score = score if min_score is None else min(min_score, score)
            max_score = score if max_score is None else max(max_score, score)
            if score < 0 or score > 1000: score_hist["outside_0_1000"] += 1
            if score < 400: score_hist["<400"] += 1
            elif score < 700: score_hist["400-699"] += 1
            elif score < 900: score_hist["700-899"] += 1
            else: score_hist[">=900"] += 1
            if a == b: self_loops += 1
            proteins.update((a, b)); endpoint_rows += 2
            mapped_endpoint_rows += int(a in unique_entrez) + int(b in unique_entrez)
            pair = (a, b)
            if previous is not None:
                if pair == previous: duplicate_pairs += 1
                if pair < previous: pair_order_monotonic = False
            previous = pair
            if matched % 2_000_000 == 0: print(f"STRING links {matched:,}", flush=True)
    mapped_link_endpoints = sum(1 for p in proteins if p in unique_entrez)
    ncg = pd.read_csv(ROOT / "outputs" / "NCG_preprocessed_merged.csv", low_memory=False)
    ncg_ids = set(pd.to_numeric(ncg.entrez, errors="coerce").dropna().astype(int))
    ncg_proteins = {p: gid for p, gid in unique_entrez.items() if gid in ncg_ids}
    # Calculate candidate-induced edges directly from full links with the verified existing mapping.
    gene_edges = set(); gene_edge_scores = {}; mapped_candidate_rows = 0; candidate_loops = 0; candidate_edge_dupes = 0
    with LINKS.open(encoding="ascii", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=" ", skipinitialspace=True)
        for row in reader:
            try: score = int(row["combined_score"])
            except Exception: continue
            if score < 700: continue
            a, b = row["protein1"], row["protein2"]
            ga, gb = ncg_proteins.get(a), ncg_proteins.get(b)
            if ga is None or gb is None: continue
            mapped_candidate_rows += 1
            if ga == gb: candidate_loops += 1; continue
            key = (min(ga, gb), max(ga, gb))
            if key in gene_edges: candidate_edge_dupes += 1
            gene_edges.add(key)
            gene_edge_scores[key] = max(score, gene_edge_scores.get(key, 0))
    existing = pd.read_csv(ROOT / "outputs" / "NCG_STRING_PPI_edges.csv")
    graph = nx.from_pandas_edgelist(existing, "source", "target", edge_attr="combined_score", create_using=nx.Graph())
    node_count = graph.number_of_nodes(); edge_count = graph.number_of_edges()
    network_stats = {
        "source_links_path": str(LINKS.relative_to(ROOT)), "source_links_header": header,
        "link_rows": matched, "protein_universe_size": len(proteins),
        "score_min": min_score, "score_max": max_score, "score_bins": dict(score_hist),
        "raw_self_links": self_loops, "adjacent_duplicate_protein_pairs": duplicate_pairs,
        "source_pair_order_monotonic": pair_order_monotonic,
        "alias_path": str(ALIASES.relative_to(ROOT)), "alias_header": alias_cols,
        "alias_rows": alias_rows, "proteins_with_entrez_alias": len(alias_map),
        "proteins_with_unambiguous_entrez": len(unique_entrez), "ambiguous_protein_mappings": ambiguous,
        "unique_alias_mapped_protein_ids_covered_by_links": mapped_link_endpoints,
        "link_endpoint_mapping_rate": mapped_endpoint_rows / endpoint_rows if endpoint_rows else None,
        "unique_linked_protein_mapping_rate": mapped_link_endpoints / len(proteins) if proteins else None,
        "ncg_gene_count": len(ncg_ids), "ncg_genes_with_unambiguous_string_protein_mapping": len(set(ncg_proteins.values())),
        "ncg_gene_mapping_rate": len(set(ncg_proteins.values())) / len(ncg_ids),
        "ncg_candidate_links_score_ge_700_mapped_before_self_link_removal": mapped_candidate_rows,
        "ncg_candidate_gene_self_links": candidate_loops,
        "ncg_candidate_duplicate_gene_edges_after_canonicalization": candidate_edge_dupes,
        "ncg_candidate_unique_gene_edges": len(gene_edges),
        "existing_saved_graph": {"nodes": node_count, "edges": edge_count,
            "self_loops": nx.number_of_selfloops(graph), "components": nx.number_connected_components(graph),
            "largest_component": max(map(len, nx.connected_components(graph)), default=0),
            "density": nx.density(graph) if node_count > 1 else 0,
            "mean_degree": sum(dict(graph.degree()).values()) / node_count if node_count else 0,
            "max_degree": max(dict(graph.degree()).values(), default=0)},
        "release_resolution": {"links_release": "12.5", "aliases_content_release": "12.5" if "v12.5" in ALIASES.name else "12.0",
            "compatible_aliases_file_found": "v12.5" in ALIASES.name,
            "project_search_result": "Version-matched human STRING v12.5 aliases were acquired and header-validated." if "v12.5" in ALIASES.name else "Only v12.0 aliases found.",
            "interpretation": "STRING-to-Entrez mappings use the version-matched STRING v12.5 human aliases." if "v12.5" in ALIASES.name else "Mapping remains provisional across releases."}
    }
    graph_path = OUT / "ppi_edges_string_v12_5.csv"
    pd.DataFrame([{"source": a, "target": b, "combined_score": score}
        for (a, b), score in sorted(gene_edge_scores.items())]).to_csv(graph_path, index=False)
    graph = nx.from_pandas_edgelist(pd.read_csv(graph_path), "source", "target", edge_attr="combined_score", create_using=nx.Graph())
    network_stats["release_matched_ncg_graph"] = {"path": graph_path.name, "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(), "self_loops": nx.number_of_selfloops(graph),
        "components": nx.number_connected_components(graph),
        "largest_component": max(map(len, nx.connected_components(graph)), default=0),
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0,
        "mean_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() else 0,
        "max_degree": max(dict(graph.degree()).values(), default=0)}
    write_json(OUT / "string_validation_summary.json", network_stats)
    return network_stats


def validate_labels():
    ann_path = ROOT / "NCG_cancerdrivers_annotation_supporting_evidence.tsv"
    ann = pd.read_csv(ann_path, sep="\t", low_memory=False)
    ann["entrez_num"] = pd.to_numeric(ann.entrez, errors="coerce")
    ann["symbol_norm"] = ann.symbol.astype("string").str.strip().str.upper()
    sys = pd.read_csv(ROOT / "NCG_cancerdrivers_systemslevelproperties.tsv", sep="\t", low_memory=False)
    id_to_symbol = ann.dropna(subset=["entrez_num"]).groupby("entrez_num").symbol_norm.agg(lambda x: sorted(set(x.dropna())))
    cosmic = {}
    for build in ("GRCh37", "GRCh38"):
        path = ROOT / f"Cosmic_CancerGeneCensus_Tsv_v104_{build}" / f"Cosmic_CancerGeneCensus_v104_{build}.tsv" / f"Cosmic_CancerGeneCensus_v104_{build}.tsv"
        df = pd.read_csv(path, sep="\t", dtype=str)
        cosmic[build] = df
    cos = cosmic["GRCh38"].copy()
    ncg_symbols = set(ann.symbol_norm.dropna())
    cosmic_symbols = set(cos.GENE_SYMBOL.dropna().str.strip().str.upper())
    # Gene-level target policy: source annotations, with mixed 0/1 records excluded as conflicts.
    label_rows = []
    for entrez, group in ann.groupby("entrez_num", dropna=True):
        symbols = sorted(set(group.symbol_norm.dropna()))
        states = {}
        for target in ("NCG_oncogene", "NCG_tsg"):
            vals = set(pd.to_numeric(group[target], errors="coerce").dropna().astype(int))
            states[target] = ("conflict" if vals == {0, 1} else "positive" if vals == {1} else "explicit_zero" if vals == {0} else "unknown")
        label_rows.append({"entrez": int(entrez), "symbol": ";".join(symbols),
            "oncogene_status": states["NCG_oncogene"], "oncogene_label": 1 if states["NCG_oncogene"] == "positive" else 0 if states["NCG_oncogene"] == "explicit_zero" else None,
            "tsg_status": states["NCG_tsg"], "tsg_label": 1 if states["NCG_tsg"] == "positive" else 0 if states["NCG_tsg"] == "explicit_zero" else None,
            "annotation_record_count": len(group), "symbol_conflict": len(symbols) > 1})
    labels = pd.DataFrame(label_rows)
    labels.to_csv(OUT / "ncg_gene_label_table.csv", index=False)
    # Repeat-level disagreement details
    conflict_records = []
    for target in ("NCG_oncogene", "NCG_tsg"):
        vals = ann.assign(_v=pd.to_numeric(ann[target], errors="coerce")).dropna(subset=["entrez_num", "_v"])
        for eid, g in vals.groupby("entrez_num"):
            observed = sorted(set(g._v.astype(int)))
            if len(observed) > 1:
                conflict_records.append({"source":"NCG", "entrez":int(eid), "symbol":";".join(sorted(set(g.symbol_norm.dropna()))), "target":target, "observed_values":json.dumps(observed), "record_count":len(g)})
    roles = cos["ROLE_IN_CANCER"].fillna("").str.lower().str.replace(" ", "", regex=False)
    role_sets = {s: set(filter(None, role.split(","))) for s, role in zip(cos.GENE_SYMBOL.str.upper(), roles)}
    for _, row in cos.iterrows():
        role_value = "" if pd.isna(row.ROLE_IN_CANCER) else str(row.ROLE_IN_CANCER)
        rr = set(filter(None, role_value.lower().replace(" ", "").split(",")))
        if "oncogene" in rr and "tsg" in rr:
            conflict_records.append({"source":"COSMIC", "entrez":None, "symbol":row.GENE_SYMBOL, "target":"ROLE_IN_CANCER", "observed_values":str(row.ROLE_IN_CANCER), "record_count":1})
    pd.DataFrame(conflict_records).to_csv(OUT / "annotation_conflicts.csv", index=False)
    # NCG-COSMIC comparisons for gene-level roles, keeping COSMIC entirely out of target/input construction.
    ncg_by_symbol = labels.assign(symbol=labels.symbol.str.split(";")).explode("symbol")
    nmap = {r.symbol: (r.oncogene_status, r.tsg_status, r.entrez) for r in ncg_by_symbol.itertuples() if r.symbol}
    compare=[]
    for _, row in cos.iterrows():
        s = str(row.GENE_SYMBOL).upper(); roleset=role_sets.get(s,set()); n=nmap.get(s)
        if n:
            n_on, n_tsg, eid = n
            c_on = "positive" if "oncogene" in roleset else "role_does_not_list_oncogene" if roleset else "unknown"
            c_tsg = "positive" if "tsg" in roleset else "role_does_not_list_tsg" if roleset else "unknown"
            compare.append({"symbol":s,"entrez":eid,"cosmic_role":row.ROLE_IN_CANCER,"cosmic_tier":row.TIER,
                "ncg_oncogene_status":n_on,"cosmic_oncogene_role_status":c_on,"oncogene_role_discordance":(n_on == "positive" and c_on == "role_does_not_list_oncogene") or (n_on == "explicit_zero" and c_on == "positive"),
                "ncg_tsg_status":n_tsg,"cosmic_tsg_role_status":c_tsg,"tsg_role_discordance":(n_tsg == "positive" and c_tsg == "role_does_not_list_tsg") or (n_tsg == "explicit_zero" and c_tsg == "positive")})
    compare_df=pd.DataFrame(compare); compare_df.to_csv(OUT / "ncg_cosmic_gene_comparison.csv",index=False)
    summary={
        "ncg_annotation_rows":len(ann),"ncg_unique_entrez":int(ann.entrez_num.nunique()),"ncg_unique_symbols":int(ann.symbol_norm.nunique()),
        "ncg_system_rows":len(sys),"ncg_system_unique_entrez":int(pd.to_numeric(sys.entrez,errors="coerce").nunique()),
        "ncg_repeat_record_conflicts":int(sum(x["source"]=="NCG" for x in conflict_records)),
        "ncg_symbol_conflicting_entrez":int((labels.symbol_conflict).sum()),
        "oncogene_gene_status_counts":labels.oncogene_status.value_counts().to_dict(),
        "tsg_gene_status_counts":labels.tsg_status.value_counts().to_dict(),
        "candidate_gene_universe": "3,347 NCG Entrez IDs (curated cohort only), subject to ID mapping and expression availability; not genome-wide.",
        "cosmic_builds":{build:{"rows":len(df),"unique_symbols":int(df.GENE_SYMBOL.nunique()),"columns":df.columns.tolist()} for build,df in cosmic.items()},
        "cosmic_ncg_symbol_overlap":len(ncg_symbols & cosmic_symbols),
        "cosmic_only_symbols":len(cosmic_symbols-ncg_symbols),"ncg_only_symbols":len(ncg_symbols-cosmic_symbols),
        "ncg_cosmic_comparison_rows":len(compare_df),"oncogene_role_discordances_not_proven_conflicts":int(compare_df.oncogene_role_discordance.sum()) if len(compare_df) else 0,
        "tsg_role_discordances_not_proven_conflicts":int(compare_df.tsg_role_discordance.sum()) if len(compare_df) else 0,
        "role_categories":cos.ROLE_IN_CANCER.fillna("<missing>").value_counts(dropna=False).to_dict(),
        "label_policy":{"positive":"NCG gene-level target == 1 in any repeated source row, provided no conflicting 0 exists.",
          "explicit_zero":"NCG gene-level target == 0 and no conflicting 1 exists; only source-explicit zero is used as comparison label.",
          "conflict":"Gene has both 0 and 1 in repeated rows for a target; exclude from that target's supervised set and report separately.",
          "unknown":"All missing for that target; remains unlabeled and is never converted to negative.",
          "cosmic":"Independent reference comparison only, never used as target or predictor in the initial NCG-label task."},
        "cosmic_use_note":"Local v104 README documents the table and source but does not state a redistribution license. Do not redistribute COSMIC tables; use only within applicable COSMIC terms and verify those terms before publishing derived/redistributed data."
    }
    write_json(OUT / "annotation_validation_summary.json",summary)
    return summary


def main():
    global OUT
    if OUT.exists() and any(OUT.iterdir()):
        OUT = ROOT / "data" / "processed" / ("validation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    OUT.mkdir(parents=True, exist_ok=True)
    print("Output:", OUT, flush=True)
    expression, _ = validate_expression()
    print("Expression validation finished", flush=True)
    string = validate_string()
    print("STRING validation finished", flush=True)
    labels = validate_labels()
    print("Annotation validation finished", flush=True)
    summary={"run_date":"2026-10-28","python":sys.version,"expression":expression,"string":string,"annotations":labels,
        "model_ready_expression_features_created":False,
        "model_ready_reason":"No sample/case IDs or sample-type metadata can be linked to the expression files from the available manifest/annotation files; cohort selection cannot be defined without misclassifying samples.",
        "prediction_task_specification":"See reports/prediction_task_specification.md",
        "integrity_checks":{"all_raw_inputs_read_only":True,"outputs_in_new_directory":str(OUT.relative_to(ROOT)),"training_run":False}}
    write_json(OUT / "data_validation_summary.json",summary)
    print(json.dumps({"expression_files":expression["expression_file_count"],"expression_valid":expression["valid_file_count"],"manifest_checksum_matches":expression["manifest_checksum_matches"],"string_links":string["link_rows"],"annotation_overlap":labels["cosmic_ncg_symbol_overlap"],"output":str(OUT)},indent=2),flush=True)


if __name__ == "__main__":
    main()
