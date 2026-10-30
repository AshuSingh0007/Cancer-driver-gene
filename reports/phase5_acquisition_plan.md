# Phase 5 input acquisition plan

**Execution status:** GDC case/sample metadata and the STRING v12.5 human aliases were acquired after approval and are recorded under `data/reference/metadata_20261028_151433/`. The Ensembl 102 BioMart request did not return a mapping table; it returned the current HTML archive/help page. That response is preserved as an invalid-response artifact and excluded from data processing. The acquisition JSON summary records this outcome.

This is the concrete set of small reference files needed to unblock the data pipeline. The acquisition script uses Python's standard library, writes into a new unique directory under `data/reference/`, refuses to overwrite existing outputs, and downloads no expression files or interaction links.

| Source | Requested content | Expected size / scope | Purpose |
|---|---|---:|---|
| NCI Genomic Data Commons Files API | Case/sample/sample type/barcode metadata for only the 1,231 file UUIDs already in the local GDC manifest | JSON metadata only; no expression data | Link each local file ID to case/sample and identify cohort types |
| STRING version 12.5, Homo sapiens | `9606.protein.aliases.v12.5.txt.gz` | 18.9 MB compressed; extracted locally for mapping | Match aliases to the existing v12.5 links file |
| Ensembl archive, October 2020, release 102 | BioMart Ensembl gene ID, Entrez gene ID, HGNC symbol mapping | One gene-reference table | Match the GENCODE v36 expression genes to NCG Entrez IDs using a version-matched cross-reference |

The GDC API's official [search/retrieval guide](https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/) documents file metadata associated with cases/samples. The [GENCODE release history](https://www.gencodegenes.org/human/releases.html) maps GENCODE v36 to Ensembl 102. The official STRING 12.5 download page documents the human aliases file and its size.

## Execution

The prepared acquisition script is `src/acquire_missing_metadata.py`. After approval, run:

```powershell
.\.venv\Scripts\python.exe src\acquire_missing_metadata.py
```

The approved GDC and STRING acquisition was validated in a new run, with output details in `data/reference/metadata_20261028_151433/acquisition_summary.json`. Phase 2 validation was rerun using the matched aliases and GDC metadata. Sample selection and expression aggregation are documented in `reports/data_quality_report.md` and implemented by `src/build_gene_expression_features.py`. The Ensembl-to-Entrez mapping remains the identifier blocker for a joined hybrid matrix.

## Reproducible Ensembl mapping fallback

The archived BioMart endpoint currently returns the archive landing page. If BioMart remains unavailable, request approval to obtain the release-102 human core xref tables from Ensembl's versioned FTP/MySQL release (core database name `homo_sapiens_core_102_38`). Join `gene` to `object_xref` (`object_type='Gene'`), `xref`, and `external_db`, retaining Entrez Gene references, and export Ensembl stable ID, Entrez ID, and display symbol. Strip only the Ensembl version suffix when joining stable IDs; retain and report all one-to-many and missing mappings. Verify the downloaded release and file checksums, then compare mappings to GENCODE v36 IDs and local NCG symbols before freezing the eligible gene universe. Ensembl documents release-versioned MySQL/FTP data and the release-numbered database naming scheme in its [FTP guide](https://www.ensembl.org/info/data/ftp/index.html) and [versioning guide](https://static.ensembl.org/info/docs/api/versions.html). At the time of this plan, no alternative mapping dataset had been downloaded; the later approved NCBI Gene mapping outcome is recorded in the superseding section below.

## Superseding mapping outcome

The planned release-102 BioMart query did not return a mapping table. A reproducible fallback was completed using the official NCBI Gene FTP human `gene_info` cross-references, under user approval. Script: `src/build_ncbi_identifier_crosswalk.py`; outputs: `data/processed/identifier_mapping_20261028_155713/`. It joins exact Ensembl stable IDs from NCBI `dbXrefs`, with no symbol-only resolution. The complete pipeline then produced aligned model inputs and a model comparison; see `reports/phase6_model_evaluation.md`. The NCBI source size and mapping counts are recorded in machine-readable summaries.

